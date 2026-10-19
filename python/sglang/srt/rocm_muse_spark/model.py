# ruff: noqa: BLE001, RUF012, SIM102
"""Torch-native Muse 1.2 / Muse Spark text model for SGLang.

BF16 checkpoints run a plain ``torch.nn.functional.linear`` expert loop on CUDA
and a fused grouped GEMM on ROCm (``ops/muse_moe_triton.py``, or
``ops/muse_moe_flydsl.py`` when ``MUSE_MOE_BACKEND=flydsl``).
Serialized ModelOpt NVFP4 expert weights stay packed and are executed by the
public FlashInfer TRT-LLM MoE kernel via the public ModelOpt quant method. Muse
applies its post-expert norm between GEMM2 and the top-k combine, which that
kernel fuses together, so the combine is finished separately by the CuTeDSL op
in ``sglang_meta_common/ops`` (with a torch fallback).
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from sglang.srt.configs.muse_spark import (
    MuseSpark1xTextConfig,
    should_skip_weight_loading,
)
from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.environ import envs
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor as OSSLogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.rocm_muse_spark.fp4_moe import MuseSpark1xFp4Experts
from sglang.srt.rocm_muse_spark.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.srt.rocm_muse_spark.fused_norm_gate import FusedRMSNormGate
from sglang.srt.rocm_muse_spark.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.srt.rocm_muse_spark.layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xResidualStream,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
    QKVOGateParallelLinear,
)
from sglang.srt.rocm_muse_spark.muse_qk_norm import fused_qk_norm, qk_norm_supported
from sglang.srt.rocm_muse_spark.muse_router_gemv import (
    router_gemv,
    router_gemv_supported,
)
from sglang.srt.runtime_context import get_context, get_exec, get_parallel
from sglang.srt.utils import add_prefix
from torch import nn

logger = logging.getLogger(__name__)


def _original_layout_selected() -> bool:
    """Select the original TP8/W4A4 layout, independently of K1/K2 dispatch.

    ``SGLANG_MUSE_ORIGINAL_NATIVE_ONLY=1`` is the same-weight native baseline;
    selecting the FlyDSL backend enables the layout and installs K1/K2 later.
    Both are explicit and default off.
    """
    native_only = os.getenv("SGLANG_MUSE_ORIGINAL_NATIVE_ONLY", "0")
    if native_only not in ("0", "1"):
        raise ValueError("SGLANG_MUSE_ORIGINAL_NATIVE_ONLY must be 0 or 1")
    if native_only == "1":
        return True
    return (
        get_context().is_config_namespace_published("exec")
        and get_exec().kernel.monokernel_backend == "flydsl"
    )


def _tp_expert_stack_slice(
    loaded: torch.Tensor,
    target_shape: torch.Size,
    name: str,
    rank: int,
    tp_size: int,
) -> torch.Tensor:
    """Shard all 256 experts over their intermediate dimension for TP8."""
    if tuple(loaded.shape) == tuple(target_shape):
        return loaded
    if loaded.ndim != 3 or loaded.shape[0] != target_shape[0]:
        return loaded
    if "w13_" in name and loaded.shape[1] == target_shape[1] * tp_size:
        loaded_half = loaded.shape[1] // 2
        target_half = target_shape[1] // 2
        start = rank * target_half
        return torch.cat(
            (
                loaded.narrow(1, start, target_half),
                loaded.narrow(1, loaded_half + start, target_half),
            ),
            dim=1,
        ).contiguous()
    if "w2_" in name and loaded.shape[2] == target_shape[2] * tp_size:
        return loaded.narrow(2, rank * target_shape[2], target_shape[2]).contiguous()
    return loaded


def _original_boundary(
    boundary: MuseSpark1xResidualStream,
    pre_norm: MuseSpark1xRMSNorm,
    out_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Original FP32 post-norm -> blend -> FP32 pre-norm boundary."""
    if boundary.branch.is_cuda and boundary.connection._coefficients_ready:
        from sglang.srt.rocm_muse_spark.native_ops.fused_rmsnorm import (
            fused_norm_residual_norm_fwd,
            supports_norm_residual_norm,
        )

        if supports_norm_residual_norm(boundary.branch, boundary.stream):
            post = boundary.post_norm
            return fused_norm_residual_norm_fwd(
                boundary.branch,
                boundary.stream,
                post.weight,
                pre_norm.weight,
                post.eps,
                pre_norm.eps,
                alpha=boundary.connection.alpha,
                beta=boundary.connection.beta,
                post_gain_center=(
                    post.gain_center
                    if post.weight is not None and post.zero_centered_gamma
                    else 0.0
                ),
                pre_gain_center=(
                    pre_norm.gain_center
                    if pre_norm.weight is not None and pre_norm.zero_centered_gamma
                    else 0.0
                ),
                post_input_scale=isinstance(post, MuseSpark1xRMSNormWithInputScale),
                pre_input_scale=isinstance(pre_norm, MuseSpark1xRMSNormWithInputScale),
                out_dtype=out_dtype,
            )
    branch = boundary.post_norm(boundary.branch, out_dtype=torch.float32)
    stream = boundary.connection(boundary.stream, branch)
    return pre_norm(stream, out_dtype=out_dtype), stream


def _original_tp_finalize(
    route_out: torch.Tensor,
    scores: torch.Tensor,
    post_expert_norm: nn.Module,
) -> torch.Tensor:
    """Per-route TP all-reduce before original weight-first FP32 normalization."""
    # The original native path reduces the BF16 route partials.  K2 performs
    # the same rank-ordered FP32 accumulation followed by a BF16 seam before
    # applying the weight-first post-expert norm.
    route = tensor_model_parallel_all_reduce(route_out.to(torch.bfloat16))
    tokens, topk = scores.shape
    if route.is_cuda:
        from sglang.srt.rocm_muse_spark.native_ops.fused_finalize import (
            fused_post_expert_finalize,
        )

        output = torch.empty(
            tokens, route.shape[-1], dtype=torch.bfloat16, device=route.device
        )
        weight = getattr(post_expert_norm, "weight", None)
        gain = (
            post_expert_norm.gain_center
            if weight is not None
            and getattr(post_expert_norm, "zero_centered_gamma", False)
            else 0.0
        )
        return fused_post_expert_finalize(
            route.view(tokens, topk, route.shape[-1]),
            weight,
            scores,
            float(post_expert_norm.eps),
            float(gain),
            True,
            output,
        )
    normalized = post_expert_norm(route, out_dtype=torch.float32)
    return (
        normalized.view(tokens, topk, normalized.shape[-1])
        .mul(scores.float().unsqueeze(-1))
        .sum(dim=1)
        .to(torch.bfloat16)
    )


class MuseSpark1xLogitsProcessor(OSSLogitsProcessor):
    """Public logits pipeline with the checkpoint-requested accumulator dtype."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # Same as models/muse_spark.py (#146): the lm_head output and the final-logit
        # softcap run in fp32; the released checkpoints carry no use_fp32_for_output flag.
        self.use_fp32_output = True

    def _compute_lm_head(self, hidden_states, lm_head, embedding_bias=None):
        if not hasattr(lm_head, "weight"):
            return super()._compute_lm_head(hidden_states, lm_head, embedding_bias)
        output_dtype = torch.float32 if self.use_fp32_output else lm_head.weight.dtype
        logits = torch.mm(
            hidden_states.to(lm_head.weight.dtype),
            lm_head.weight.T,
            out_dtype=output_dtype,
        )
        if embedding_bias is not None:
            logits = logits + embedding_bias
        return logits


def _is_rocm() -> bool:
    return getattr(torch.version, "hip", None) is not None


def _moe_backend() -> str:
    backend = os.environ.get("MUSE_MOE_BACKEND", "auto").lower()
    if backend not in {"auto", "native", "triton", "flydsl"}:
        raise ValueError(
            "MUSE_MOE_BACKEND must be one of auto, native, triton, or flydsl; "
            f"got {backend!r}"
        )
    if backend == "auto":
        return "triton" if _is_rocm() else "native"
    return backend


def _want_triton_moe() -> bool:
    """ROCm defaults to the fused Triton expert path; CUDA stays on the PyTorch loop.

    Explicit ``native``, ``triton``, and ``flydsl`` values select that backend.
    Triton and FlyDSL are graph-capturable, so they are used for both eager and
    captured forwards.
    """
    return _moe_backend() == "triton"


def _want_fused_moe() -> bool:
    """Grouped GEMM expert path that keeps Triton weight layouts (Triton or FlyDSL)."""
    return _moe_backend() in {"triton", "flydsl"}


def _run_fused_moe(
    hidden_states: torch.Tensor,
    topk_scores: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    local_expert_start: int,
    post_expert_norm: nn.Module | None,
    w13_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    if _moe_backend() == "flydsl":
        from sglang.srt.rocm_muse_spark.muse_moe_flydsl import (
            muse_moe_flydsl as fused_moe,
        )
    else:
        from sglang.srt.rocm_muse_spark.muse_moe_triton import (
            muse_moe_triton as fused_moe,
        )

    weight, gain, has_norm = _post_expert_norm_args(post_expert_norm)
    eps = float(post_expert_norm.eps) if post_expert_norm is not None else 0.0
    kwargs = {}
    if w13_scale is not None or w2_scale is not None:
        kwargs["w13_scale"] = w13_scale
        kwargs["w2_scale"] = w2_scale
    return fused_moe(
        hidden_states,
        topk_scores,
        topk_ids,
        w13,
        w2,
        int(local_expert_start),
        weight,
        eps,
        post_norm_gain=gain,
        has_norm=has_norm,
        **kwargs,
    )


def _post_expert_norm_args(
    post_expert_norm: nn.Module | None,
) -> tuple[torch.Tensor | None, float, bool]:
    """Weight / gain / has_norm without allocating tensors (unsafe under capture)."""
    if post_expert_norm is None:
        return None, 0.0, False
    weight = getattr(post_expert_norm, "weight", None)
    gain = 0.0
    if weight is not None and getattr(post_expert_norm, "zero_centered_gamma", False):
        gain = float(post_expert_norm.gain_center)
    return weight, gain, True


def _quantization_name(quant_config: QuantizationConfig | None) -> str | None:
    if quant_config is None:
        return None
    get_name = getattr(quant_config, "get_name", None)
    return get_name() if get_name is not None else type(quant_config).__name__


def _is_quark_mxfp4(quant_config: QuantizationConfig | None) -> bool:
    """Return whether this is the AMD Quark OCP-MX FP4 expert format."""
    if _quantization_name(quant_config) != "quark":
        return False
    config = getattr(quant_config, "quant_config", None) or {}
    global_config = (
        config.get("global_quant_config", {}) if isinstance(config, dict) else {}
    )
    weight = (global_config or {}).get("weight") or {}
    return (
        weight.get("dtype") == "fp4"
        and weight.get("scale_format") == "e8m0"
        and weight.get("group_size") == 32
    )


def _uses_quark_mxfp4_experts(
    quant_config: QuantizationConfig | None, moe_prefix: str
) -> bool:
    """
    Whether this layer's experts are packed MXFP4 rather than left in BF16.

    Quark exports can be partially quantized: Muse Spark 1.2 MXFP4 keeps two of
    the 62 MoE layers in BF16. The exclude list names them the way its own
    checkpoint generation spelled them, so both spellings are tried below.
    """
    if not _is_quark_mxfp4(quant_config):
        return False
    layer_prefix = moe_prefix.rpartition(".")[0]
    if _is_nvfp4_source(quant_config):
        # Requantized ModelOpt export: Quark turns the producer's exclude list
        # (substrings such as "layers.0." / "layers.61.", ModelOpt semantics) into
        # anchored fnmatch regexes, which the generic module-path matcher would
        # never match against the full module name. Search them unanchored.
        target = add_prefix("mlp.experts", layer_prefix)
        for pattern in getattr(quant_config, "exclude_layers", None) or []:
            if pattern.startswith("re:"):
                body = pattern[3:].removesuffix("\\Z")
                if re.search(body, target):
                    return False
            elif pattern in target:
                return False
        return True
    excluded = getattr(quant_config, "exclude_layers", None) or []
    return not is_layer_skipped(add_prefix("mlp.experts", layer_prefix), excluded)


def _is_nvfp4_source(quant_config: QuantizationConfig | None) -> bool:
    """Whether a Quark MXFP4 config requantizes from an NVFP4 export."""
    source = getattr(quant_config, "dequantization_config", None)
    if source is None:
        return False
    try:
        from sglang.srt.layers.quantization.quark.utils import Nvfp4SourceConfig
    except ImportError:  # pragma: no cover
        return False
    return isinstance(source, Nvfp4SourceConfig)


def _uses_nvfp4_experts(
    quant_config: QuantizationConfig | None, layer_prefix: str
) -> bool:
    if _quantization_name(quant_config) != "modelopt_fp4":
        return False
    excluded = getattr(quant_config, "exclude_modules", None) or []
    return not is_layer_skipped(add_prefix("mlp.experts", layer_prefix), excluded)


# The released config states the per-layer schedule outright: a zero rope base marks a
# NoPE layer and layer_types marks the sliding ones. Reading the lists keeps this in step
# with the checkpoint instead of re-deriving the pattern from every_n_layers_nope.
def _compute_use_rope(config, layer_idx: int) -> bool:
    """A zero rope base marks a NoPE layer."""
    return bool(config.layer_rope_theta[layer_idx])


def _compute_sliding_window(config, layer_idx: int) -> int | None:
    """The config states the span; the kernels want the causal offset."""
    if config.layer_types[layer_idx] != "sliding_attention":
        return None
    return config.sliding_window - 1


# One-launch per-head q/k RMSNorm (+ query scale) on ROCm; MUSE_FUSE_QK_NORM=0 restores the eager path.
_USE_FUSED_QK_NORM = (
    os.environ.get("MUSE_FUSE_QK_NORM", "1") == "1" and torch.version.hip is not None
)
# Decode-sized fp32 router as one split-K Triton GEMV (+reduce); MUSE_ROUTER_GEMV=0 restores torch.mm.
_USE_ROUTER_GEMV = (
    os.environ.get("MUSE_ROUTER_GEMV", "1") == "1" and torch.version.hip is not None
)

# Replicated pre_expert_proj/post_expert_proj GEMMs at decode-sized M stay on nn.Linear (hipBLASLt). aiter's tuned
# GEMM is opt-in via MUSE_AITER_DENSE=1: with the pinned aiter its table resolves these shapes to kernels that are
# slower than hipBLASLt on MI355X (pre_expert_proj M=1..32: 21 us vs 15-19 us; post_expert_proj M<=64: 42 us vs
# 14-18 us).
_USE_AITER_DENSE = (
    os.environ.get("MUSE_AITER_DENSE", "0") == "1" and torch.version.hip is not None
)
_USE_SPLITK_POOLER = envs.SGLANG_MUSE_SPLITK_POOLER.get()
_tgemm = None


def _dense_linear(x: torch.Tensor, linear: nn.Linear, max_m: int) -> torch.Tensor:
    global _tgemm
    if _USE_SPLITK_POOLER and linear.bias is None:
        from sglang.kernels.ops.muse_spark.pooler import try_splitk_pooler

        result = try_splitk_pooler(x, linear.weight)
        if result is not None:
            return result
    if (
        _USE_AITER_DENSE
        and linear.bias is None
        and x.dtype == torch.bfloat16
        and linear.weight.dtype == torch.bfloat16
        and 0 < x.shape[0] <= max_m
    ):
        if _tgemm is None:
            try:
                from aiter.tuned_gemm import tgemm as _t

                _tgemm = _t
            except Exception:  # aiter without tuned_gemm: stay on torch
                _tgemm = False
        if _tgemm:
            return _tgemm.mm(x, linear.weight, None, otype=torch.bfloat16)
    return linear(x)


class MuseSpark1xRouter(nn.Module):
    """Router with an opt-in original K2-compatible weight layout."""

    def __init__(self, hidden_size: int, num_experts: int, dtype: torch.dtype):
        super().__init__()
        self.original_layout = _original_layout_selected()
        shape = (
            (num_experts, hidden_size)
            if self.original_layout
            else (hidden_size, num_experts)
        )
        self.weight = nn.Parameter(
            torch.empty(shape, dtype=dtype), requires_grad=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.original_layout:
            return F.linear(hidden_states, self.weight)
        return torch.mm(hidden_states, self.weight)


class MuseSpark1xAttention(nn.Module):
    def __init__(self, config: MuseSpark1xTextConfig, layer_idx: int, prefix: str):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.use_o_gate = True
        self.use_o_norm = True
        self.use_qk_norm = True
        self.use_fused_qkv_prepare = envs.SGLANG_MUSE_FUSED_QKV_PREPARE.get()
        self.use_rope = _compute_use_rope(config, layer_idx)
        sliding_window = _compute_sliding_window(config, layer_idx)
        # Attention temperature tuning is not part of the released architecture.
        self.use_attn_temperature_tuning = False
        self.temperature_floor = None
        self.temperature_q_scale = None

        parallel = get_parallel()
        tp_size = parallel.attn_tp_size
        if config.num_attention_heads % tp_size:
            raise ValueError("num_attention_heads must be divisible by attention TP")
        if config.num_key_value_heads >= tp_size:
            if config.num_key_value_heads % tp_size:
                raise ValueError(
                    "num_key_value_heads must be divisible by attention TP"
                )
        elif tp_size % config.num_key_value_heads:
            raise ValueError("attention TP must divide replicated KV heads")
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.scale_query_by = config.qk_scale_factor
        if _original_layout_selected() and self.use_qk_norm:
            # Original K1 normalizes Q without a BF16 query-scale round trip;
            # native attention must use the same scale after the QK product.
            self.scaling *= self.scale_query_by
            self.scale_query_by = 1.0

        if self.use_o_gate:
            self.qkv_proj = QKVOGateParallelLinear(
                hidden_size=config.hidden_size,
                head_size=self.head_dim,
                total_num_heads=config.num_attention_heads,
                total_num_kv_heads=config.num_key_value_heads,
                bias=False,
                prefix=add_prefix("qkv_proj", prefix),
                parallel_group="attn_tp",
            )
        else:
            self.qkv_proj = QKVParallelLinear(
                hidden_size=config.hidden_size,
                head_size=self.head_dim,
                total_num_heads=config.num_attention_heads,
                total_num_kv_heads=config.num_key_value_heads,
                bias=False,
                quant_config=None,
                prefix=add_prefix("qkv_proj", prefix),
                parallel_group="attn_tp",
            )
        self.o_proj = RowParallelLinear(
            input_size=config.num_attention_heads * self.head_dim,
            output_size=config.hidden_size,
            bias=False,
            reduce_results=True,
            quant_config=None,
            prefix=add_prefix("o_proj", prefix),
            parallel_group="attn_tp",
        )
        self.rotary_emb = (
            get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=getattr(
                    config, "context_len", config.max_position_embeddings
                ),
                base=int(config.layer_rope_theta[layer_idx]),
                is_neox_style=False,
                dtype=torch.float32,
            )
            if self.use_rope
            else None
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_idx,
            prefix=add_prefix("attn", prefix),
            use_irope=self.use_rope,
            sliding_window_size=sliding_window or -1,
        )
        self.norm_eps = config.rms_norm_eps
        # Names used by the immutable original K1 host contract.
        self.rms_norm_eps = self.norm_eps
        self.fused_qk_norm_rope_kv_insert = True
        self.fuse_o_norm_gate = True
        # The output norm and the sigmoid gate are one fused kernel. The norm is
        # weightless here, so this module carries no parameters -- it exists to
        # hand the op its eps.
        self._o_norm_module = (
            MuseSpark1xRMSNorm(self.head_dim, eps=self.norm_eps, has_weight=False)
            if self.use_o_norm
            else None
        )
        self._fused_o_norm_gate = FusedRMSNormGate(
            self._o_norm_module, self.use_o_gate, self.head_dim
        )

    def _per_head_norm(self, value: torch.Tensor) -> torch.Tensor:
        dtype = value.dtype
        return (
            F.rms_norm(
                value.reshape(-1, self.head_dim),
                (self.head_dim,),
                None,
                self.norm_eps,
            )
            .reshape_as(value)
            .to(dtype)
        )

    @property
    def pre_qkv_proj_norm(self):
        return self.pre_attn_norm

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        qkv, _ = self.qkv_proj(hidden_states)
        split_sizes = [self.q_size, self.kv_size, self.kv_size]
        if self.use_o_gate:
            q, k, v, o_gate = qkv.split(split_sizes + [self.q_size], dim=-1)
        else:
            q, k, v = qkv.split(split_sizes, dim=-1)
            o_gate = None
        prepared = None
        if self.use_fused_qkv_prepare:
            prepare = getattr(get_attn_backend(), "prepare_muse_qkv", None)
            if prepare is not None:
                prepared = prepare(self, q, k, v, positions, forward_batch)
        if prepared is not None:
            q, k = prepared
        elif self.use_qk_norm:
            if _USE_FUSED_QK_NORM and qk_norm_supported(q, k, self.head_dim):
                q, k = fused_qk_norm(
                    q, k, self.head_dim, self.norm_eps, self.scale_query_by
                )
            else:
                q = self._per_head_norm(q) * self.scale_query_by
                k = self._per_head_norm(k)
        if self.use_attn_temperature_tuning:
            temperature = (
                torch.log((positions.float() + 1.0) / self.temperature_floor)
                * self.temperature_q_scale
                + 1.0
            ).clamp_min(1.0)
            q = (q * temperature.unsqueeze(-1)).to(q.dtype)
        if self.rotary_emb is not None and prepared is None:
            q, k = self.rotary_emb(positions, q, k)
        output = self.attn(q, k, v, forward_batch, save_kv_cache=prepared is None)
        output = self._fused_o_norm_gate.forward(output, o_gate)
        output, _ = self.o_proj(output)
        return output


class MuseSpark1xMLP(nn.Module):
    def __init__(self, config: MuseSpark1xTextConfig, prefix: str):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=None,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        output, _ = self.down_proj(F.silu(gate) * up)
        return output


class MuseSpark1xNativeExperts(nn.Module):
    """Local expert weights: PyTorch loop on CUDA, fused Triton or FlyDSL MoE on ROCm."""

    def __init__(
        self,
        num_local_experts: int,
        expert_hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.expert_hidden_size = expert_hidden_size
        self.intermediate_size = intermediate_size
        self.w13_weight = nn.Parameter(
            torch.empty(
                num_local_experts,
                2 * intermediate_size,
                expert_hidden_size,
            ),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.empty(
                num_local_experts,
                expert_hidden_size,
                intermediate_size,
            ),
            requires_grad=False,
        )

    @torch.no_grad()
    def prepare_original_tp_weights(self) -> None:
        """Shuffle the live BF16 TP shard in place for grouped AITER MoE."""
        if (
            not getattr(self, "original_tp", False)
            or not self.w13_weight.is_cuda
            or getattr(self.w13_weight, "is_shuffled", False)
        ):
            return
        from aiter.ops.shuffle import shuffle_weight

        self.w13_weight.copy_(shuffle_weight(self.w13_weight.contiguous(), (16, 16)))
        self.w2_weight.copy_(shuffle_weight(self.w2_weight.contiguous(), (16, 16)))
        self.w13_weight.is_shuffled = True
        self.w2_weight.is_shuffled = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_ids: torch.Tensor,
        local_expert_start: int,
        post_expert_norm: nn.Module | None,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        if getattr(self, "original_tp", False):
            ids = topk_ids.reshape(-1).long()
            from sglang.srt.rocm_muse_spark.native_ops.fused_norm_expand import (
                take_expanded,
            )

            expanded = take_expanded(hidden_states, topk_ids.shape[1])
            if expanded is None:
                expanded = (
                    hidden_states[:, None, :]
                    .expand(-1, topk_ids.shape[1], -1)
                    .reshape(-1, hidden_states.shape[-1])
                )
            if hidden_states.is_cuda and _is_rocm():
                # A grouped top-1 invocation retains one output per route
                # without materializing [T * topk, 2I, H] selected weights.
                # That selected-weight tensor is multi-GiB for native prefill.
                from aiter import ActivationType, QuantType
                from aiter.fused_moe import fused_moe
                from aiter.ops.flydsl.moe_common import GateMode

                route_ids = ids.reshape(-1, 1).to(torch.int32).contiguous()
                route_out = fused_moe(
                    hidden_states=expanded.contiguous(),
                    w1=self.w13_weight,
                    w2=self.w2_weight,
                    topk_weight=torch.ones_like(route_ids, dtype=torch.float32),
                    topk_ids=route_ids,
                    quant_type=QuantType.No,
                    activation=ActivationType.Silu,
                    doweight_stage1=False,
                    hidden_pad=0,
                    intermediate_pad=0,
                    gate_mode=GateMode.SEPARATED.value,
                    dtype=torch.bfloat16,
                )
            else:
                # Small CPU reference used by semantic contract tests.
                w13 = self.w13_weight.index_select(0, ids)
                gate_up = torch.bmm(w13, expanded.unsqueeze(-1)).squeeze(-1)
                gate, up = gate_up.chunk(2, dim=-1)
                inter = F.silu(gate) * up
                w2 = self.w2_weight.index_select(0, ids)
                route_out = torch.bmm(w2, inter.unsqueeze(-1)).squeeze(-1)
            return _original_tp_finalize(route_out, topk_scores, post_expert_norm)
        if _want_fused_moe():
            return _run_fused_moe(
                hidden_states,
                topk_scores,
                topk_ids,
                self.w13_weight,
                self.w2_weight,
                int(local_expert_start),
                post_expert_norm,
            )
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return self._forward_capturable(
                hidden_states,
                topk_scores,
                topk_ids,
                local_expert_start,
                post_expert_norm,
            )
        output = torch.zeros_like(hidden_states)
        for local_id in range(self.num_local_experts):
            global_id = local_expert_start + local_id
            token_slot = (topk_ids == global_id).nonzero(as_tuple=False)
            if token_slot.numel() == 0:
                continue
            token_ids = token_slot[:, 0]
            slots = token_slot[:, 1]
            selected = hidden_states.index_select(0, token_ids)
            gate_up = F.linear(selected, self.w13_weight[local_id])
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = F.linear(F.silu(gate) * up, self.w2_weight[local_id])
            if post_expert_norm is not None:
                expert_output = post_expert_norm(expert_output)
            expert_output = expert_output * topk_scores[token_ids, slots].unsqueeze(-1)
            output.index_add_(0, token_ids, expert_output.to(output.dtype))
        return output

    def _forward_capturable(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_ids: torch.Tensor,
        local_expert_start: int,
        post_expert_norm: nn.Module | None,
    ) -> torch.Tensor:
        """Fixed-shape equivalent of ``forward`` for CUDA graph capture.

        ``forward`` picks out each expert's tokens with ``nonzero`` and skips
        the experts that drew none. Both read device data on the host, which
        stream capture rejects. Here every local expert instead runs over all
        tokens and is scaled by its routing weight, which is zero for a token
        that did not select it -- the same result with no host synchronisation,
        paid for by running each local expert densely.
        """
        output = torch.zeros_like(hidden_states)
        for local_id in range(self.num_local_experts):
            global_id = local_expert_start + local_id
            # A token's top-k ids are distinct, so at most one slot matches and
            # this sum is exactly that slot's score (zero when unrouted).
            weight = (topk_scores * (topk_ids == global_id)).sum(dim=-1, keepdim=True)
            gate_up = F.linear(hidden_states, self.w13_weight[local_id])
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = F.linear(F.silu(gate) * up, self.w2_weight[local_id])
            if post_expert_norm is not None:
                expert_output = post_expert_norm(expert_output)
            output = output + (expert_output * weight).to(output.dtype)
        return output


class MuseSpark1xMxfp4Experts(nn.Module):
    """AMD Quark MXFP4 experts with packed E2M1 weights and E8M0 scales."""

    def __init__(
        self,
        num_local_experts: int,
        expert_hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self._prepacked_moe = None
        self.num_local_experts = num_local_experts
        self.expert_hidden_size = expert_hidden_size
        self.intermediate_size = intermediate_size
        self.w13_weight = nn.Parameter(
            torch.empty(
                num_local_experts,
                2 * intermediate_size,
                expert_hidden_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.w13_weight_scale = nn.Parameter(
            torch.empty(
                num_local_experts,
                2 * intermediate_size,
                expert_hidden_size // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.empty(
                num_local_experts,
                expert_hidden_size,
                intermediate_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.w2_weight_scale = nn.Parameter(
            torch.empty(
                num_local_experts,
                expert_hidden_size,
                intermediate_size // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )

    def prepare_prepacked_weights(self) -> None:
        if getattr(self, "original_tp", False):
            self._prepare_original_tp_weights()
            self._prepacked_moe = None
            return
        if (
            not envs.SGLANG_MUSE_MOE_PREPACKED.get()
            or _moe_backend() != "flydsl"
            or not self.w13_weight.is_cuda
            or not getattr(
                torch.cuda.get_device_properties(self.w13_weight.device),
                "gcnArchName",
                "",
            ).startswith("gfx950")
            or self.num_local_experts != 32
            or self.expert_hidden_size != 4096
            or self.intermediate_size != 4096
        ):
            self._prepacked_moe = None
            return
        from sglang.kernels.ops.muse_spark.moe_prepacked import PrepackedMuseMoE

        weights = (
            self.w13_weight,
            self.w13_weight_scale,
            self.w2_weight,
            self.w2_weight_scale,
        )
        if self._prepacked_moe is None:
            self._prepacked_moe = PrepackedMuseMoE(*weights)
        else:
            self._prepacked_moe.update_weights(*weights)

    @torch.no_grad()
    def _prepare_original_tp_weights(self) -> None:
        """Convert the sole TP shard in place to original AITER layouts.

        The parameters remain the live model weights used by K2 and the native
        baseline.  No persistent raw/shuffled duplicate is retained.
        """
        from aiter.ops.shuffle import shuffle_weight
        from aiter.utility.fp4_utils import e8m0_shuffle

        if not getattr(self, "original_layout_dirty", True):
            return
        self.w13_weight.copy_(shuffle_weight(self.w13_weight.contiguous(), (16, 16)))
        self.w2_weight.copy_(shuffle_weight(self.w2_weight.contiguous(), (16, 16)))
        for scale in (self.w13_weight_scale, self.w2_weight_scale):
            shape = scale.shape
            scale.copy_(e8m0_shuffle(scale.flatten(0, 1)).view(shape))
        self.w13_weight.is_shuffled = True
        self.w2_weight.is_shuffled = True
        self.original_layout_dirty = False

    def _forward_original_tp(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_ids: torch.Tensor,
        post_expert_norm: nn.Module,
    ) -> torch.Tensor:
        """Production AITER W4A4 fallback with route-preserving output."""
        from aiter import ActivationType, QuantType
        from aiter.fused_moe import fused_moe
        from aiter.ops.flydsl.moe_common import GateMode

        num_tokens = hidden_states.shape[0]
        topk = topk_ids.shape[1]
        from sglang.srt.rocm_muse_spark.native_ops.fused_norm_expand import (
            take_expanded,
        )

        expanded = take_expanded(hidden_states, topk)
        if expanded is None:
            expanded = (
                hidden_states[:, None, :]
                .expand(num_tokens, topk, self.expert_hidden_size)
                .reshape(num_tokens * topk, self.expert_hidden_size)
                .contiguous()
            )
        ids = topk_ids.reshape(-1, 1).to(torch.int32).contiguous()
        scores = topk_scores.to(torch.float32).contiguous()
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        w13 = (
            self.w13_weight.view(fp4_dtype)
            if fp4_dtype is not None
            else self.w13_weight
        )
        w2 = self.w2_weight.view(fp4_dtype) if fp4_dtype is not None else self.w2_weight
        # Tensor views do not inherit Python attributes from their Parameters.
        # AITER reads the layout flag from the actual tensors passed below.
        w13.is_shuffled = self.w13_weight.is_shuffled
        w2.is_shuffled = self.w2_weight.is_shuffled
        route_out = fused_moe(
            hidden_states=expanded,
            w1=w13,
            w2=w2,
            topk_weight=torch.ones_like(ids, dtype=torch.float32),
            topk_ids=ids,
            quant_type=QuantType.per_1x32,
            activation=ActivationType.Silu,
            w1_scale=self.w13_weight_scale,
            w2_scale=self.w2_weight_scale,
            doweight_stage1=False,
            hidden_pad=0,
            intermediate_pad=0,
            gate_mode=GateMode.SEPARATED.value,
            dtype=torch.bfloat16,
        )
        return _original_tp_finalize(route_out, scores, post_expert_norm)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_ids: torch.Tensor,
        local_expert_start: int,
        post_expert_norm: nn.Module | None,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        if getattr(self, "original_tp", False):
            return self._forward_original_tp(
                hidden_states, topk_scores, topk_ids, post_expert_norm
            )
        if not _want_fused_moe():
            raise RuntimeError(
                "Quark MXFP4 experts require MUSE_MOE_BACKEND=triton or flydsl"
            )
        if self._prepacked_moe is not None and _moe_backend() == "flydsl":
            weight, gain, has_norm = _post_expert_norm_args(post_expert_norm)
            # Match the released Muse post-expert norm: zero-centered gamma
            # with unit gain and epsilon 1e-8. Other contracts keep the fallback.
            if (
                16 <= hidden_states.shape[0] <= 256
                and hidden_states.shape[0] % 8 == 0
                and hidden_states.shape[1] == 4096
                and hidden_states.dtype == torch.bfloat16
                and hidden_states.is_contiguous()
                and topk_ids.shape == topk_scores.shape == (hidden_states.shape[0], 8)
                and has_norm
                and weight is not None
                and weight.shape == (4096,)
                and weight.dtype == torch.bfloat16
                and weight.is_contiguous()
                and gain == 1.0
                and float(post_expert_norm.eps) == 1e-8
            ):
                return self._prepacked_moe(
                    hidden_states,
                    topk_scores,
                    topk_ids,
                    int(local_expert_start),
                    weight,
                )
        return _run_fused_moe(
            hidden_states,
            topk_scores,
            topk_ids,
            self.w13_weight,
            self.w2_weight,
            int(local_expert_start),
            post_expert_norm,
            w13_scale=self.w13_weight_scale,
            w2_scale=self.w2_weight_scale,
        )

    _forward_capturable = forward

    @property
    def routed_experts(self):
        return self


class MuseSpark1xMoE(nn.Module):
    def __init__(
        self,
        config: MuseSpark1xTextConfig,
        layer_idx: int,
        nvfp4_checkpoint: bool,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        if config.num_shared_experts:
            raise ValueError("Muse Spark OSS supports routed experts only")
        parallel = get_parallel()
        self.original_mono_tp = _original_layout_selected()
        if self.original_mono_tp:
            if parallel.tp_size != 8 or parallel.moe_ep_size != 1:
                raise ValueError(
                    "The original Muse MonoKernel requires TP8 with EP disabled"
                )
        elif parallel.moe_tp_size != 1:
            raise ValueError(
                "The torch-native Muse MoE requires expert TP1; use EP equal to TP"
            )
        if config.num_local_experts % parallel.moe_ep_size:
            raise ValueError("num_experts must be divisible by expert parallel size")
        self.layer_idx = layer_idx
        self.tp_size = parallel.tp_size
        self.use_token_sharded_pooler = envs.SGLANG_MUSE_TOKEN_SHARDED_POOLER.get()
        self.ep_size = parallel.moe_ep_size
        self.ep_rank = parallel.moe_ep_rank
        self.num_experts = config.num_local_experts
        self.num_local_experts = (
            self.num_experts
            if self.original_mono_tp
            else self.num_experts // self.ep_size
        )
        self.local_expert_start = (
            0 if self.original_mono_tp else self.ep_rank * self.num_local_experts
        )
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.routed_scaling_factor
        self.route_eps = 1e-15
        self.router_multiplier = config.output_multiplier
        self.router_in_fp32 = True
        self.hidden_size = config.hidden_size
        self.expert_hidden_size = config.moe_hidden_size
        self.intermediate_size = (
            config.moe_intermediate_size // self.tp_size
            if self.original_mono_tp
            else config.moe_intermediate_size
        )
        router_dtype = (
            torch.float32 if self.router_in_fp32 else torch.get_default_dtype()
        )
        self.router = MuseSpark1xRouter(
            self.hidden_size, self.num_experts, dtype=router_dtype
        )
        self.register_buffer(
            "expert_bias", torch.zeros(self.num_experts, dtype=torch.float32)
        )
        self.pre_expert_proj = (
            nn.Linear(self.hidden_size, self.expert_hidden_size, bias=False)
            if self.expert_hidden_size != self.hidden_size
            else None
        )
        self.post_expert_proj = (
            nn.Linear(self.expert_hidden_size, self.hidden_size, bias=False)
            if self.expert_hidden_size != self.hidden_size
            else None
        )
        self.pre_expert_norm = MuseSpark1xRMSNorm(
            self.expert_hidden_size,
            eps=config.rms_norm_eps,
            has_weight=True,
            zero_centered_gamma=True,
            residual_id=2 * layer_idx + 2,
        )
        self.post_expert_norm = MuseSpark1xRMSNormWithInputScale(
            self.expert_hidden_size,
            eps=config.post_norm_eps,
            has_weight=True,
            is_post_norm=True,
            zero_centered_gamma=True,
            post_norm_gain_center_type="one",
            residual_id=2 * layer_idx + 2,
        )
        self.nvfp4_checkpoint = nvfp4_checkpoint
        self.mxfp4_checkpoint = _uses_quark_mxfp4_experts(quant_config, prefix)
        if nvfp4_checkpoint:
            # Packed NVFP4: the FlashInfer TRT-LLM kernel runs the experts and we
            # finalize in torch so the post-expert norm lands before the top-k sum.
            self.experts = MuseSpark1xFp4Experts(
                num_experts=self.num_experts,
                num_local_experts=self.num_local_experts,
                local_expert_offset=self.local_expert_start,
                hidden_size=self.expert_hidden_size,
                intermediate_size=self.intermediate_size,
                top_k=self.top_k,
                quant_config=quant_config,
            )
        elif self.mxfp4_checkpoint:
            self.experts = MuseSpark1xMxfp4Experts(
                self.num_local_experts,
                self.expert_hidden_size,
                self.intermediate_size,
            )
        else:
            self.experts = MuseSpark1xNativeExperts(
                self.num_local_experts,
                self.expert_hidden_size,
                self.intermediate_size,
            )
        if isinstance(
            self.experts, (MuseSpark1xMxfp4Experts, MuseSpark1xNativeExperts)
        ):
            self.experts.original_tp = self.original_mono_tp

    def _routing_op(self) -> FusedMuseSparkRouting:
        """The fused router: sigmoid, biased top-k, gather, renormalise, scale.

        Built lazily and rebound if ``expert_bias`` is replaced, because
        ``Module._apply`` (``.to()``, ``.cuda()``) swaps the buffer for a new
        tensor and the op holds a direct reference to it.
        """
        op = getattr(self, "_fused_routing_op", None)
        if op is None or op.expert_bias is not self.expert_bias:
            op = FusedMuseSparkRouting(
                expert_bias=self.expert_bias,
                route_scale=self.route_scale,
                router_output_multiplier=self.router_multiplier,
                router_score_division_eps=self.route_eps,
            )
            object.__setattr__(self, "_fused_routing_op", op)
        return op

    @property
    def pooler(self):
        return self.pre_expert_proj

    @property
    def unpooler(self):
        return self.post_expert_proj

    def _pool_inputs(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch | None = None
    ) -> torch.Tensor:
        if (
            forward_batch is not None
            and forward_batch.forward_mode.is_extend_without_speculative()
            and self.use_token_sharded_pooler
            and self.pre_expert_proj is not None
            and get_parallel().attn_tp_size == self.tp_size
        ):
            from sglang.kernels.ops.muse_spark.pooler import (
                try_token_sharded_pooler,
            )

            pooled = try_token_sharded_pooler(
                hidden_states,
                self.pre_expert_proj,
                self.pre_expert_norm,
                get_parallel().tp_group,
            )
            if pooled is not None:
                return pooled
        expert_input = (
            _dense_linear(hidden_states, self.pre_expert_proj, max_m=128)
            if self.pre_expert_proj is not None
            else hidden_states
        )
        if self.pre_expert_norm is not None:
            if self.original_mono_tp and expert_input.is_cuda:
                from sglang.srt.rocm_muse_spark.native_ops.fused_norm_expand import (
                    fused_rmsnorm_expand,
                )

                norm = self.pre_expert_norm
                expert_input = fused_rmsnorm_expand(
                    expert_input,
                    norm.weight,
                    float(norm.eps),
                    float(norm.gain_center if norm.zero_centered_gamma else 0.0),
                    isinstance(norm, MuseSpark1xRMSNormWithInputScale),
                    self.top_k,
                )
            else:
                expert_input = self.pre_expert_norm(expert_input)
        return expert_input

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch | None = None
    ) -> torch.Tensor:
        if (
            _USE_ROUTER_GEMV
            and self.router_in_fp32
            and router_gemv_supported(hidden_states, self.router.weight)
        ):
            # bf16 -> fp32 conversion happens inside the kernel (exact), IEEE fp32 accumulate
            router_logits = router_gemv(hidden_states, self.router.weight)
        else:
            router_input = (
                hidden_states.float() if self.router_in_fp32 else hidden_states
            )
            router_logits = self.router(router_input)
        topk_scores, topk_ids = self._routing_op().forward(
            None, router_logits, self.top_k, True
        )

        expert_input = self._pool_inputs(hidden_states, forward_batch)
        if self.nvfp4_checkpoint:
            output = self.experts(
                expert_input, topk_scores, topk_ids, self.post_expert_norm
            )
        else:
            output = self.experts(
                expert_input,
                topk_scores,
                topk_ids,
                self.local_expert_start,
                self.post_expert_norm,
            )
        if self.tp_size > 1 and not self.original_mono_tp:
            output = tensor_model_parallel_all_reduce(output)
        if self.post_expert_proj is not None:
            output = _dense_linear(output, self.post_expert_proj, max_m=64)
        return output


class MuseSpark1xDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MuseSpark1xTextConfig,
        layer_idx: int,
        nvfp4_checkpoint: bool,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = MuseSpark1xAttention(
            config, layer_idx, add_prefix("self_attn", prefix)
        )
        self.post_attn_norm = MuseSpark1xRMSNorm(
            config.hidden_size,
            eps=config.post_norm_eps,
            has_weight=False,
            is_post_norm=True,
            zero_centered_gamma=True,
            post_norm_gain_center_type="one",
            residual_id=2 * layer_idx + 1,
        )
        self.pre_feedforward_layernorm = MuseSpark1xRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            has_weight=True,
            zero_centered_gamma=True,
            residual_id=2 * layer_idx + 2,
        )
        self.post_feedforward_layernorm = MuseSpark1xRMSNormWithInputScale(
            config.hidden_size,
            eps=config.post_norm_eps,
            has_weight=True,
            is_post_norm=True,
            zero_centered_gamma=True,
            post_norm_gain_center_type="one",
            residual_id=2 * layer_idx + 2,
        )
        self.is_moe_layer = self._is_moe_layer(config, layer_idx)
        self.mlp = (
            MuseSpark1xMoE(
                config,
                layer_idx,
                nvfp4_checkpoint,
                add_prefix("mlp", prefix),
                quant_config,
            )
            if self.is_moe_layer
            else MuseSpark1xMLP(config, add_prefix("mlp", prefix))
        )
        self.post_attention_residual_gate = MuseSpark1xNormPreservingResidualConnection(
            config.hidden_size, temperature=config.residual_gate_temperature
        )
        self.post_feedforward_residual_gate = (
            MuseSpark1xNormPreservingResidualConnection(
                config.hidden_size, temperature=config.residual_gate_temperature
            )
        )
        # The attention boundary -- post-norm, residual add, then the FFN
        # pre-norm -- is one fused kernel. The FFN boundary is not fused: its
        # pre-norm belongs to the next layer, so it cannot be expressed here.
        self._fused_attn_boundary = FusedPostNormResidualPreNorm(
            self.post_attn_norm,
            self.post_attention_residual_gate,
            self.pre_feedforward_layernorm,
        )

    @staticmethod
    def _is_moe_layer(config: MuseSpark1xTextConfig, layer_idx: int) -> bool:
        return config.mlp_layer_types[layer_idx] == "sparse"

    def forward(
        self,
        positions: torch.Tensor,
        stream: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_input: torch.Tensor | None = None,
        ffn_boundary: FusedPostNormResidualPreNorm | None = None,
        pre_norm_dtype: torch.dtype | None = None,
        defer_ffn_boundary: bool = False,
    ):
        attn_input = (
            self.self_attn_input_norm(stream) if attn_input is None else attn_input
        )
        attn_output = self.self_attn(positions, attn_input, forward_batch)
        if _original_layout_selected():
            ffn_input, stream = _original_boundary(
                MuseSpark1xResidualStream(
                    stream=stream,
                    branch=attn_output,
                    post_norm=self.post_attn_norm,
                    connection=self.post_attention_residual_gate,
                ),
                self.pre_feedforward_layernorm,
            )
        else:
            ffn_input, stream, _ = self._fused_attn_boundary.forward(
                stream, attn_output, branch_dtype=torch.bfloat16
            )
        ffn_output = (
            self.mlp(ffn_input, forward_batch=forward_batch)
            if self.is_moe_layer
            else self.mlp(ffn_input)
        )
        if defer_ffn_boundary:
            return MuseSpark1xResidualStream(
                stream=stream,
                branch=ffn_output,
                post_norm=self.post_feedforward_layernorm,
                connection=self.post_feedforward_residual_gate,
            )
        if ffn_boundary is None:
            return self.post_feedforward_residual_gate(
                stream, self.post_feedforward_layernorm(ffn_output)
            )
        # Cross-layer fusion: post_feedforward_layernorm + gated residual + the NEXT layer's pre_attn_norm
        # (or the final norm) in one kernel; returns the new stream and the next attention input.
        next_attn_input, stream, _ = ffn_boundary.forward(
            stream,
            ffn_output,
            branch_dtype=torch.bfloat16,
            pre_norm_dtype=pre_norm_dtype,
        )
        return stream, next_attn_input

    def self_attn_input_norm(self, stream: torch.Tensor) -> torch.Tensor:
        return self.self_attn.pre_attn_norm(stream).to(torch.bfloat16)

    # Compatibility names are properties, not duplicate module registrations.
    @property
    def feed_forward(self):
        return self.mlp

    @property
    def pre_ffn_norm(self):
        return self.pre_feedforward_layernorm

    @property
    def post_ffn_norm(self):
        return self.post_feedforward_layernorm

    @property
    def add_attn(self):
        return self.post_attention_residual_gate

    @property
    def add_ffn(self):
        return self.post_feedforward_residual_gate


# Keep the pre-attention norm under self_attn to match Muse checkpoint names.
def _install_pre_attn_norm(
    config: MuseSpark1xTextConfig, layer: MuseSpark1xDecoderLayer
) -> None:
    layer.self_attn.pre_attn_norm = MuseSpark1xRMSNorm(
        config.hidden_size,
        eps=config.rms_norm_eps,
        has_weight=True,
        zero_centered_gamma=True,
        residual_id=2 * layer.layer_idx + 1,
    )


class MuseSpark1xModel(nn.Module):
    def __init__(
        self,
        config: MuseSpark1xTextConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ):
        super().__init__()
        self.config = config
        self.residual_dtype = torch.float32
        # Layer indices whose output residual stream feeds the DFLASH draft
        # context (set through MuseSpark1xForCausalLM.set_dflash_layers_to_capture).
        self.layers_to_capture: list[int] = []
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.embed_norm = MuseSpark1xRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, has_weight=False
        )
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            layer_prefix = add_prefix(f"layers.{layer_idx}", prefix)
            layer = MuseSpark1xDecoderLayer(
                config,
                layer_idx,
                _uses_nvfp4_experts(quant_config, layer_prefix),
                layer_prefix,
                quant_config,
            )
            _install_pre_attn_norm(config, layer)
            self.layers.append(layer)
        self.norm = MuseSpark1xRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            has_weight=True,
            zero_centered_gamma=True,
            output_norm_gain_center_type="zero",
            residual_id=-1,
        )
        # Cross-layer FFN boundaries: layer i's post_feedforward_layernorm + post_feedforward_residual_gate + layer i+1's
        # pre_attn_norm (the final norm for the last layer) as one fused kernel each.
        # MUSE_FUSE_FFN_BOUNDARY=0 restores the per-module eager path.
        self._ffn_boundaries = None
        if (
            os.environ.get("MUSE_FUSE_FFN_BOUNDARY", "1") == "1"
            and len(self.layers) > 0
        ):
            pre_norms = [layer.self_attn.pre_attn_norm for layer in self.layers[1:]] + [
                self.norm
            ]
            self._ffn_boundaries = [
                FusedPostNormResidualPreNorm(
                    layer.post_feedforward_layernorm,
                    layer.post_feedforward_residual_gate,
                    pre_norm,
                )
                for layer, pre_norm in zip(self.layers, pre_norms)
            ]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        if input_embeds is None and self.embed_norm is not None:
            hidden_states = self.embed_norm(
                hidden_states,
                out_dtype=(torch.float32 if _original_layout_selected() else None),
            )
        stream = hidden_states.to(self.residual_dtype)
        # DFLASH draft context: the residual stream after each listed layer, read
        # in operator dtype (bf16) like the CUDA model does.
        aux_hidden_states: list[torch.Tensor] = []
        if (
            _original_layout_selected()
            and self._ffn_boundaries is not None
            and stream.shape[0] > 0
        ):
            final_pre_norm_dtype = torch.bfloat16
            attn_input = self.layers[0].self_attn_input_norm(stream)
            last = len(self.layers) - 1
            for i, layer in enumerate(self.layers):
                boundary = layer(
                    positions,
                    stream,
                    forward_batch,
                    attn_input=attn_input,
                    defer_ffn_boundary=True,
                )
                next_norm = (
                    self.norm
                    if i == last
                    else self.layers[i + 1].self_attn.pre_attn_norm
                )
                attn_input, stream = _original_boundary(
                    boundary, next_norm, final_pre_norm_dtype
                )
                if i in self.layers_to_capture:
                    aux_hidden_states.append(stream.to(torch.bfloat16))
            hidden_states = attn_input
        elif self._ffn_boundaries is not None and stream.shape[0] > 0:
            # Keep the accepted default path unchanged when original layout is off.
            final_pre_norm_dtype = torch.bfloat16
            attn_input = self.layers[0].self_attn_input_norm(stream)
            last = len(self.layers) - 1
            for i, layer in enumerate(self.layers):
                stream, attn_input = layer(
                    positions,
                    stream,
                    forward_batch,
                    attn_input=attn_input,
                    ffn_boundary=self._ffn_boundaries[i],
                    pre_norm_dtype=final_pre_norm_dtype if i == last else None,
                )
                if i in self.layers_to_capture:
                    aux_hidden_states.append(stream.to(torch.bfloat16))
            hidden_states = attn_input
        else:
            for i, layer in enumerate(self.layers):
                stream = layer(positions, stream, forward_batch)
                if i in self.layers_to_capture:
                    aux_hidden_states.append(stream.to(torch.bfloat16))
            norm_input = stream.to(torch.bfloat16)
            hidden_states = self.norm(norm_input).to(torch.bfloat16)
        if self.layers_to_capture:
            return hidden_states, aux_hidden_states
        return hidden_states


class MuseSparkModel(nn.Module):
    """Holds the decoder under the released ``model.language_model.*`` parameter names.

    The release introduced this level so the text stack sits beside the vision and audio
    towers. Attribute reads fall through to the decoder, so the rest of this file keeps
    addressing ``self.model.layers`` and ``self.model.embed_tokens`` as before.
    """

    def __init__(
        self,
        config: MuseSpark1xTextConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ):
        super().__init__()
        self.language_model = MuseSpark1xModel(
            config, quant_config, add_prefix("language_model", prefix)
        )

    def __getattr__(self, name):
        # nn.Module resolves real submodules/parameters first; anything else is the
        # decoder's, which keeps `self.model.layers` and friends working.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.language_model, name)

    def forward(self, *args, **kwargs):
        return self.language_model(*args, **kwargs)


class MuseSpark1xForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: MuseSpark1xTextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        if getattr(config, "inference_moe_w13_interleaved", False):
            raise ValueError("Interleaved expert checkpoints are not supported")
        quant_name = _quantization_name(quant_config)
        if quant_name not in (None, "modelopt_fp4", "quark"):
            raise ValueError(
                "Only BF16, serialized ModelOpt NVFP4, and Quark MXFP4 are "
                f"supported, got {quant_name}"
            )
        if quant_name == "quark" and not _is_quark_mxfp4(quant_config):
            raise ValueError(
                "Unsupported Quark format; expected FP4/E8M0/group-size 32"
            )
        self.config = config
        self.quant_config = quant_config
        self.nvfp4_checkpoint = quant_name == "modelopt_fp4"
        self.mxfp4_checkpoint = _is_quark_mxfp4(quant_config)
        # ``--quantization quark_mxfp4`` on a ModelOpt NVFP4 checkpoint: Quark builds
        # an online MXFP4 config with the NVFP4 export as its dequantization source.
        # The generic requantizer only drives FusedMoE, so this model converts its
        # stacked NVFP4 expert tensors itself while loading (see
        # _stash_nvfp4_expert_tensor / _requantize_nvfp4_stack).
        self.requant_nvfp4_experts = self.mxfp4_checkpoint and _is_nvfp4_source(
            quant_config
        )
        self._nvfp4_pending: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
        self.model = MuseSparkModel(config, quant_config, add_prefix("model", prefix))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=False,
        )
        if config.tie_word_embeddings:
            self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = MuseSpark1xLogitsProcessor(
            config, logit_scale=config.output_multiplier
        )
        self.capture_aux_hidden_states = False

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ):
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds=input_embeds
        )
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def set_dflash_layers_to_capture(self, layer_ids: list[int]) -> None:
        """Capture the residual stream after each listed decoder layer.

        DFLASH drafts consume the target hidden state produced by layer
        ``layer_ids[k]`` (HF ``hidden_states[layer_ids[k] + 1]``). Mirrors the
        CUDA model's hook so the generic DFLASH worker can drive this copy.
        """
        if layer_ids is None:
            raise ValueError(
                "DFLASH requires explicit layer_ids for aux hidden capture."
            )
        num_layers = len(self.model.language_model.layers)
        bad = [i for i in layer_ids if not 0 <= i < num_layers]
        if bad:
            raise ValueError(
                f"DFLASH target layer ids {bad} are out of range for a "
                f"{num_layers}-layer Muse Spark target."
            )
        self.capture_aux_hidden_states = True
        # MuseSparkModel.__getattr__ only reads through to the decoder; write the
        # list on the decoder itself.
        self.model.language_model.layers_to_capture = list(layer_ids)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_layers(self):
        return self.model.layers

    def get_attention_sliding_window_size(self) -> int | None:
        if "sliding_attention" in self.config.layer_types:
            return self.config.sliding_window - 1
        return None

    # The release stacks the experts into one tensor per projection and renames the
    # router; shapes are unchanged, so these are pure name moves onto the same params.
    # Checkpoint spellings this model maps onto its own module names.
    _WEIGHT_REMAPS = (
        (".mlp.gate.e_score_correction_bias", ".mlp.expert_bias"),
        (".mlp.gate.weight", ".mlp.router.weight"),
        # this model folds the pre-attention norm into the attention module
        (".input_layernorm.", ".self_attn.pre_attn_norm."),
        (".mlp.experts.post_expert_norm", ".mlp.post_expert_norm"),
        (".mlp.experts.gate_up_proj", ".mlp.experts.w13_weight"),
        (".mlp.experts.down_proj", ".mlp.experts.w2_weight"),
    )

    @staticmethod
    def _deinterleave_rope_halves(loaded_weight, heads: int):
        """Undo the released q/k rope-pair interleave.

        The checkpoint stores each head's rope pairs interleaved, the layout stock HF
        rope expects; this model's rotary embedding wants the two halves contiguous.
        """
        shape = loaded_weight.shape
        return (
            loaded_weight.reshape(heads, 2, shape[0] // heads // 2, *shape[1:])
            .transpose(1, 2)
            .reshape(shape)
            .contiguous()
        )

    _NVFP4_STACKED_EXPERT_RE = re.compile(
        r"^(.*\.mlp\.experts)\.(gate_up_proj|down_proj)"
        r"(_weight_scale_2|_weight_scale|_input_scale)?$"
    )

    def _stash_nvfp4_expert_tensor(
        self,
        params: dict[str, torch.Tensor],
        name: str,
        loaded_weight: torch.Tensor,
    ) -> bool:
        """Collect a layer's stacked NVFP4 expert tensors and requantize them to
        MXFP4 once weight, block scale and global scale have all arrived.

        ModelOpt exports a layer's routed experts as ``gate_up_proj`` (uint8 packed
        e2m1 [E, 2I, H/2]), ``gate_up_proj_weight_scale`` (fp8 e4m3 [E, 2I, H/16]),
        ``gate_up_proj_weight_scale_2`` (fp32 [E, 2] = gate/up global scales) and
        ``gate_up_proj_input_scale`` (unused here), likewise for ``down_proj`` with
        per-expert scalars. Excluded layers ship BF16 experts without scales; those
        keep the regular path. Returns True when the tensor was consumed.
        """
        match = self._NVFP4_STACKED_EXPERT_RE.match(name)
        if match is None:
            return False
        prefix, proj, suffix = match.groups()
        if suffix is None and loaded_weight.dtype != torch.uint8:
            return False  # BF16 experts of an excluded layer
        if suffix == "_input_scale":
            return True
        if loaded_weight.shape[0] == self.config.num_local_experts:
            parallel = get_parallel()
            local = loaded_weight.shape[0] // parallel.moe_ep_size
            loaded_weight = loaded_weight.narrow(0, parallel.moe_ep_rank * local, local)
        key = (prefix, proj)
        pieces = self._nvfp4_pending.setdefault(key, {})
        pieces["weight" if suffix is None else suffix.lstrip("_")] = (
            loaded_weight.contiguous()
        )
        if len(pieces) == 3:
            self._requantize_nvfp4_stack(params, prefix, proj, pieces)
            del self._nvfp4_pending[key]
        return True

    @torch.no_grad()
    def _requantize_nvfp4_stack(
        self,
        params: dict[str, torch.Tensor],
        prefix: str,
        proj: str,
        pieces: dict[str, torch.Tensor],
    ) -> None:
        """NVFP4 (e2m1 x fp8 block-16 scale x fp32 global scale) -> MXFP4 (e2m1 x
        e8m0 block-32 scale), one expert at a time on the GPU, written straight into
        the Quark-layout parameters the MXFP4 kernels read."""
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from sglang.srt.layers.quantization.dequantization import dequantize_nvfp4

        tag = "w13" if proj == "gate_up_proj" else "w2"
        weight = params[f"{prefix}.{tag}_weight"]
        scale = params[f"{prefix}.{tag}_weight_scale"]
        device = weight.device
        w_q, w_s, s2 = (
            pieces["weight"],
            pieces["weight_scale"],
            pieces["weight_scale_2"],
        )
        if w_q.shape[0] != weight.shape[0]:
            raise RuntimeError(
                f"Expert count mismatch for {prefix}.{proj}: "
                f"parameter={weight.shape[0]}, checkpoint={w_q.shape[0]}"
            )
        for expert in range(w_q.shape[0]):
            if tag == "w13":
                # [gate rows; up rows] share one global scale per half.
                half = w_q.shape[1] // 2
                s2_rows = torch.cat(
                    [s2[expert, 0].repeat(half), s2[expert, 1].repeat(half)]
                ).view(-1, 1)
            else:
                s2_rows = s2[expert].reshape(())
            dequant = dequantize_nvfp4(
                w_q[expert].to(device, non_blocking=True),
                w_s[expert].to(device, non_blocking=True),
                s2_rows.to(device=device, dtype=torch.float32),
                out_dtype=torch.float32,
            )
            q_weight, q_scale = dynamic_mxfp4_quant(dequant)
            if q_weight.shape != weight.shape[1:] or q_scale.shape != scale.shape[1:]:
                raise RuntimeError(
                    f"Requantized {prefix}.{proj} shape mismatch: "
                    f"{tuple(q_weight.shape)}/{tuple(q_scale.shape)} vs "
                    f"{tuple(weight.shape[1:])}/{tuple(scale.shape[1:])}"
                )
            weight.data[expert].copy_(q_weight)
            scale.data[expert].copy_(q_scale.to(scale.dtype))

    def _copy_expert_stack(
        self, param: torch.Tensor, loaded_weight: torch.Tensor, name: str
    ) -> None:
        if _original_layout_selected():
            loaded_weight = _tp_expert_stack_slice(
                loaded_weight,
                param.shape,
                name,
                get_parallel().tp_rank,
                get_parallel().tp_size,
            )
        if param.shape == loaded_weight.shape:
            param.data.copy_(loaded_weight)
            return
        parallel = get_parallel()
        if loaded_weight.shape[0] == self.config.num_local_experts:
            local = loaded_weight.shape[0] // parallel.moe_ep_size
            start = parallel.moe_ep_rank * local
            loaded_weight = loaded_weight.narrow(0, start, local)
        if param.shape != loaded_weight.shape:
            raise RuntimeError(
                f"Expert weight shape mismatch for {name}: "
                f"parameter={tuple(param.shape)}, checkpoint={tuple(loaded_weight.shape)}"
            )
        param.data.copy_(loaded_weight)

    def _load_single_expert(
        self,
        params: dict[str, torch.Tensor],
        name: str,
        loaded_weight: torch.Tensor,
    ) -> bool:
        match = re.match(
            r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\."
            r"(weight|weight_scale|weight_scale_2|input_scale)$",
            name,
        )
        if match is None:
            return False
        prefix, expert_raw, projection, suffix = match.groups()
        if suffix == "input_scale":
            return True
        expert_id = int(expert_raw)
        parallel = get_parallel()
        mono_tp = _original_layout_selected()
        local_count = (
            self.config.num_local_experts
            if mono_tp
            else self.config.num_local_experts // parallel.moe_ep_size
        )
        start = 0 if mono_tp else parallel.moe_ep_rank * local_count
        if not start <= expert_id < start + local_count:
            return True
        local_id = expert_id - start
        is_w13 = projection in ("gate_proj", "up_proj")
        param_name = f"{prefix}.{'w13' if is_w13 else 'w2'}_{suffix}"
        param = params.get(param_name)
        if param is None:
            raise ValueError(
                f"Checkpoint weight '{name}' has no parameter '{param_name}'"
            )
        if suffix == "weight_scale_2":
            if is_w13 and param.ndim == 2:
                shard = 0 if projection == "gate_proj" else 1
                param.data[local_id, shard].copy_(loaded_weight.reshape(()))
            else:
                param.data[local_id].copy_(loaded_weight.reshape(()))
            return True
        target = param.data[local_id]
        if is_w13:
            half = target.shape[0] // 2
            offset = 0 if projection == "gate_proj" else half
            target = target.narrow(0, offset, half)
        if mono_tp and target.shape != loaded_weight.shape:
            if is_w13 and loaded_weight.shape[0] == target.shape[0] * parallel.tp_size:
                loaded_weight = loaded_weight.narrow(
                    0, parallel.tp_rank * target.shape[0], target.shape[0]
                )
            elif (
                not is_w13
                and loaded_weight.shape[-1] == target.shape[-1] * parallel.tp_size
            ):
                loaded_weight = loaded_weight.narrow(
                    -1, parallel.tp_rank * target.shape[-1], target.shape[-1]
                )
        if target.shape != loaded_weight.shape:
            raise RuntimeError(
                f"Expert weight shape mismatch for {name}: "
                f"parameter={tuple(target.shape)}, checkpoint={tuple(loaded_weight.shape)}"
            )
        target.copy_(loaded_weight)
        return True

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        if _original_layout_selected():
            for layer in self.model.layers:
                experts = getattr(layer.mlp, "experts", None)
                if isinstance(experts, MuseSpark1xMxfp4Experts):
                    experts.original_layout_dirty = True
        params = dict(self.named_parameters())
        params.update(
            {name: value for name, value in self.named_buffers() if name not in params}
        )
        stacked = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        # The o-gate ships in every Muse Spark 1.2 checkpoint and the release spells
        # it self_attn.gate_proj. The dense .gate_up_proj entries above are tried first,
        # so an MLP gate_proj never reaches this mapping.
        stacked.append((".qkv_proj", ".gate_proj", "g"))

        skipped = 0
        for raw_name, loaded_weight in weights:
            name = raw_name
            if self.requant_nvfp4_experts and self._stash_nvfp4_expert_tensor(
                params, name, loaded_weight
            ):
                continue
            if name.endswith((".self_attn.q_proj.weight", ".self_attn.k_proj.weight")):
                loaded_weight = self._deinterleave_rope_halves(
                    loaded_weight,
                    self.config.num_attention_heads
                    if name.endswith(".q_proj.weight")
                    else self.config.num_key_value_heads,
                )
            for old, new in self._WEIGHT_REMAPS:
                if old in name:
                    name = name.replace(old, new)
                    break
            if should_skip_weight_loading(name, self.config):
                skipped += 1
                continue
            if name.endswith(
                (".self_attn.q_scale", ".self_attn.k_scale", ".self_attn.v_scale")
            ):
                skipped += 1
                continue
            # Expert activation scales are only meaningful when the experts stay
            # packed: the FP4 quant method turns them into w13/w2_input_scale_quant
            # and g1_scale_c. A BF16 expert loop has no use for them.
            if name.endswith(
                (
                    ".w13_input_scale",
                    ".w2_input_scale",
                    ".gate_up_proj_input_scale",
                    ".down_proj_input_scale",
                )
            ):
                if not self.nvfp4_checkpoint:
                    skipped += 1
                    continue
            if self._load_single_expert(params, name, loaded_weight):
                continue

            handled = False
            original_name = name
            for param_pattern, weight_pattern, shard_id in stacked:
                if weight_pattern not in name:
                    continue
                candidate = name.replace(weight_pattern, param_pattern)
                param = params.get(candidate)
                if param is None:
                    continue
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight, shard_id)
                handled = True
                break
            if handled:
                continue
            name = original_name
            param = params.get(name)
            if param is None:
                raise ValueError(
                    f"Checkpoint weight '{raw_name}' is outside the Muse Spark text model"
                )
            if "router.weight" in name and param.shape != loaded_weight.shape:
                loaded_weight = loaded_weight.t().contiguous()
            if ".experts.w" in name:
                self._copy_expert_stack(param, loaded_weight, name)
            else:
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)
        if skipped:
            logger.info("Skipped %d non-text or unused checkpoint tensors", skipped)
        if self._nvfp4_pending:
            raise RuntimeError(
                "Incomplete NVFP4 expert tensors for "
                + ", ".join(f"{p}.{proj}" for p, proj in sorted(self._nvfp4_pending))
            )
        self.post_load_weights()

    @torch.no_grad()
    def post_load_weights(self) -> None:
        for layer in self.model.layers:
            for residual in (
                layer.post_attention_residual_gate,
                layer.post_feedforward_residual_gate,
            ):
                if isinstance(residual, MuseSpark1xNormPreservingResidualConnection):
                    residual.precompute_coefficients()
        # Derived layouts are prepared before graph capture and refreshed in
        # place after weight loading, preserving captured buffer addresses.
        for layer in self.model.layers:
            experts = getattr(layer.mlp, "experts", None)
            if isinstance(experts, MuseSpark1xMxfp4Experts):
                experts.prepare_prepacked_weights()
            elif isinstance(experts, MuseSpark1xNativeExperts):
                experts.prepare_original_tp_weights()
        for module in self.model.modules():
            if isinstance(module, MuseSpark1xRMSNorm):
                module.cache_effective_weight()
        boundaries = [layer._fused_attn_boundary for layer in self.model.layers]
        boundaries += list(getattr(self.model, "_ffn_boundaries", None) or [])
        for op in boundaries:
            refresh = getattr(op, "refresh_cached_weights", None)
            if refresh is not None:
                refresh(torch.bfloat16)


EntryClass = [MuseSpark1xForCausalLM]

__all__ = [
    "MuseSpark1xForCausalLM",
    "MuseSpark1xTextConfig",
]
