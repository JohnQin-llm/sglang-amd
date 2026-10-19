"""Small, torch-native layer primitives used by the Muse Spark model."""

from __future__ import annotations

import enum
import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from sglang.srt.distributed import divide
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    LinearParallelGroup,
    resolve_linear_parallel_group,
)
from sglang.srt.runtime_context import get_context, get_exec
from torch import nn


def _original_precision_selected() -> bool:
    if os.getenv("SGLANG_MUSE_ORIGINAL_NATIVE_ONLY", "0") == "1":
        return True
    return (
        get_context().is_config_namespace_published("exec")
        and get_exec().kernel.monokernel_backend == "flydsl"
    )


class GainCenterType(enum.Enum):
    ONE = "one"
    ZERO = "zero"
    INVERSE_RESIDUAL_INDEX = "inverse_residual_index"
    INVERSE_RESIDUAL_INDEX_V2 = "inverse_residual_index_v2"
    INVERSE_SQRT_RESIDUAL_INDEX = "inverse_sqrt_residual_index"
    INVERSE_SQRT_LAYERS = "inverse_sqrt_layers"


def calculate_gain_center(
    gain_center_type: str, residual_id: int, n_layers: int | None = None
) -> float:
    gain_center = GainCenterType(gain_center_type)
    if gain_center == GainCenterType.ONE:
        return 1.0
    if gain_center == GainCenterType.ZERO:
        return 0.0
    if gain_center == GainCenterType.INVERSE_RESIDUAL_INDEX:
        return 1.0 / (2.0 * math.ceil(residual_id / 2.0))
    if gain_center == GainCenterType.INVERSE_RESIDUAL_INDEX_V2:
        return 1.0 / (residual_id + 1.0)
    if gain_center == GainCenterType.INVERSE_SQRT_RESIDUAL_INDEX:
        return 1.0 / math.sqrt(2.0 * math.ceil(residual_id / 2.0))
    if gain_center == GainCenterType.INVERSE_SQRT_LAYERS:
        if n_layers is None:
            raise ValueError("n_layers is required for inverse_sqrt_layers")
        return 1.0 / math.sqrt(n_layers)
    raise ValueError(f"Unknown gain_center_type: {gain_center_type}")


class MuseSpark1xRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        has_weight: bool = True,
        is_post_norm: bool = False,
        zero_centered_gamma: bool = False,
        post_norm_gain_center_type: str = "one",
        output_norm_gain_center_type: str = "zero",
        residual_id: int | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.has_weight = has_weight
        self.zero_centered_gamma = zero_centered_gamma
        self.weight = nn.Parameter(torch.empty(hidden_size)) if has_weight else None
        self.gain_center = 1.0
        self.original_precision = _original_precision_selected()
        if residual_id is not None and is_post_norm:
            self.gain_center = calculate_gain_center(
                post_norm_gain_center_type, residual_id
            )
        elif residual_id == -1:
            self.gain_center = calculate_gain_center(
                output_norm_gain_center_type, residual_id
            )

    def _get_effective_weight(self) -> torch.Tensor | None:
        if self.weight is None:
            return None
        cached = getattr(self, "_effective_weight_cache", None)
        if cached is not None:
            return cached
        if self.zero_centered_gamma:
            return self.weight + self.gain_center
        return self.weight

    @property
    def variance_epsilon(self) -> float:
        """Original Muse operator ABI spelling."""
        return self.eps

    def cache_effective_weight(self) -> None:
        """Materialize ``weight + gain_center`` once (call after weights are loaded, never
        inside a CUDA-graph capture) so forward() stops launching an add kernel per call."""
        if self.weight is None:
            return
        with torch.no_grad():
            cached = getattr(self, "_effective_weight_cache", None)
            # Temporarily hide the old cache so the source is recomputed from
            # the live parameter, then preserve the captured address on update.
            self._effective_weight_cache = None
            value = self._get_effective_weight().detach()
            if cached is None:
                self._effective_weight_cache = value.clone()
            else:
                cached.copy_(value)
                self._effective_weight_cache = cached

    def forward(
        self, x: torch.Tensor, out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        if x.numel() == 0:
            return x
        dtype = x.dtype
        if self.original_precision:
            out_dtype = out_dtype or dtype
            if x.is_cuda:
                from sglang.srt.rocm_muse_spark.native_ops.fused_rmsnorm import (
                    fused_rmsnorm_fwd,
                )

                return fused_rmsnorm_fwd(
                    x,
                    self.weight,
                    self.eps,
                    gain_center=(self.gain_center if self.zero_centered_gamma else 0.0),
                    out_dtype=out_dtype,
                )
            value = x.float()
            variance = value.pow(2).mean(dim=-1, keepdim=True)
            value = value * torch.rsqrt(variance + self.eps)
            if self.weight is not None:
                weight = self.weight.float()
                if self.zero_centered_gamma:
                    weight = weight + self.gain_center
                value = value * weight
            return value.to(out_dtype)
        return F.rms_norm(
            x, (self.hidden_size,), self._get_effective_weight(), self.eps
        ).to(dtype)


class MuseSpark1xRMSNormWithInputScale(MuseSpark1xRMSNorm):
    def forward(
        self, x: torch.Tensor, out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        if x.numel() == 0:
            return x
        dtype = x.dtype
        if self.original_precision:
            out_dtype = out_dtype or dtype
            if x.is_cuda:
                from sglang.srt.rocm_muse_spark.native_ops.fused_rmsnorm import (
                    fused_rmsnorm_fwd,
                )

                return fused_rmsnorm_fwd(
                    x,
                    self.weight,
                    self.eps,
                    gain_center=(self.gain_center if self.zero_centered_gamma else 0.0),
                    input_scale=True,
                    out_dtype=out_dtype,
                )
            value = x.float()
            if self.weight is not None:
                weight = self.weight.float()
                if self.zero_centered_gamma:
                    weight = weight + self.gain_center
                value = value * weight
            return F.rms_norm(value, (self.hidden_size,), None, self.eps).to(out_dtype)
        weight = self._get_effective_weight()
        if weight is not None:
            x = weight * x
        return F.rms_norm(x, (self.hidden_size,), None, self.eps).to(dtype)


class MuseSpark1xResidualConnection(nn.Module):
    def forward(self, stream: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        return stream + branch.to(stream.dtype)


class MuseSpark1xNormPreservingResidualConnection(nn.Module):
    """Training-compatible gated residual implemented only with torch ops."""

    def __init__(
        self,
        dim: int,
        temperature: float = 0.3,
        channel_wise: bool = True,
        use_stream_gate: bool = True,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.temperature = temperature
        self.use_stream_gate = use_stream_gate
        self.eps = eps
        self.gate = nn.Parameter(torch.empty(dim if channel_wise else 1))
        self.register_buffer("_alpha", None, persistent=False)
        self.register_buffer("_beta", None, persistent=False)

    def _coefficients(
        self, w: torch.Tensor
    ) -> tuple[torch.Tensor | float, torch.Tensor]:
        beta = torch.sigmoid(w)
        if not self.use_stream_gate:
            return 1.0, beta
        alpha_sq = torch.sigmoid(-w) * (1.0 + beta)
        return torch.sqrt(torch.clamp(alpha_sq, min=self.eps)), beta

    def precompute_coefficients(self) -> None:
        with torch.no_grad():
            alpha, beta = self._coefficients(
                self.gate.view(1, -1).float() / self.temperature
            )
            alpha = alpha if isinstance(alpha, torch.Tensor) else torch.tensor(alpha)
            if self._alpha is None:
                self._alpha = alpha
                self._beta = beta
            else:
                # Preserve addresses captured by K1/K2 while refreshing live gates.
                self._alpha.copy_(alpha)
                self._beta.copy_(beta)

    @property
    def alpha(self) -> torch.Tensor | None:
        return self._alpha

    @property
    def beta(self) -> torch.Tensor | None:
        return self._beta

    @property
    def _coefficients_ready(self) -> bool:
        return self._alpha is not None and self._beta is not None

    def forward(self, stream: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        if stream.numel() == 0:
            return stream
        if self._alpha is None:
            alpha, beta = self._coefficients(
                self.gate.view(1, -1).float() / self.temperature
            )
        else:
            alpha, beta = self._alpha, self._beta
        return (alpha * stream + beta * branch.to(stream.dtype)).to(stream.dtype)


@dataclass(slots=True)
class MuseSpark1xResidualStream:
    """An FFN residual boundary deferred to its next reader.

    The following layer's pre-attention norm (or the model's final norm) is the
    first consumer of the blended stream.  Keeping the branch separate lets the
    native fused boundary and the original K1 close exactly the same state.
    """

    stream: torch.Tensor
    branch: torch.Tensor
    post_norm: MuseSpark1xRMSNorm
    connection: MuseSpark1xNormPreservingResidualConnection


class QKVOGateParallelLinear(ColumnParallelLinear):
    """Tensor-parallel Q/K/V projection with one additional query-sized gate."""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        prefix: str = "",
        parallel_group: LinearParallelGroup = "attn_tp",
    ):
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        tp_rank, tp_size = resolve_linear_parallel_group(parallel_group)
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.num_heads = divide(total_num_heads, tp_size)
        if tp_size >= total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, total_num_kv_heads)
        else:
            self.num_kv_heads = divide(total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        self.q_proj_shard_size = self.num_heads * head_size
        self.kv_proj_shard_size = self.num_kv_heads * head_size
        self.g_proj_shard_size = self.q_proj_shard_size
        local = (
            self.q_proj_shard_size
            + 2 * self.kv_proj_shard_size
            + self.g_proj_shard_size
        )
        super().__init__(
            input_size=hidden_size,
            output_size=local * tp_size,
            bias=bias,
            gather_output=False,
            quant_config=None,
            prefix=prefix,
            parallel_group=parallel_group,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ):
        if loaded_shard_id not in ("q", "k", "v", "g"):
            raise ValueError(f"Unsupported QKV gate shard: {loaded_shard_id}")
        sizes = {
            "q": self.q_proj_shard_size,
            "k": self.kv_proj_shard_size,
            "v": self.kv_proj_shard_size,
            "g": self.g_proj_shard_size,
        }
        offsets = {
            "q": 0,
            "k": self.q_proj_shard_size,
            "v": self.q_proj_shard_size + self.kv_proj_shard_size,
            "g": self.q_proj_shard_size + 2 * self.kv_proj_shard_size,
        }
        output_dim = getattr(param, "output_dim", 0)
        shard_size = sizes[loaded_shard_id]
        param_data = param.data.narrow(output_dim, offsets[loaded_shard_id], shard_size)
        if loaded_shard_id in ("q", "g"):
            start = self.tp_rank * shard_size
        else:
            start = (self.tp_rank // self.num_kv_head_replicas) * shard_size
        param_data.copy_(loaded_weight.narrow(output_dim, start, shard_size))
