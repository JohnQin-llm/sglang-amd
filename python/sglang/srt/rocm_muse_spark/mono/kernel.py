# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The layer kernels (``MONO_SPEC.build``): each takes and returns what
``MuseSpark1xDecoderLayer.forward`` does -- ``(positions, x)`` with ``x`` the
previous layer's deferred FFN boundary (``MuseSpark1xResidualStream``) -- and
runs the layer as two persistent launches:

- K1 (``ops/k1.py``): closes the previous layer's FFN boundary (post-FFN norm,
  gated blend into the fp32 stream, pre-QKV norm), the QKV + output-gate
  projection, QK-norm + RoPE + paged KV insert, the attention over the paged
  cache and the gated o-norm -> o_proj's input. Rank-local.
- K2 (``ops/k2.py``): o_proj and its TP all-reduce, the attention boundary
  (post-attention norm, gated blend -> the returned stream, pre-FFN norm), the
  router and the pooler (a slice a rank, all-gathered), routing, the
  pre-expert norm, the MXFP4 experts, the per-route TP all-reduce, the
  post-expert-norm finalize and the unpooler (a slice a rank, all-gathered).

The result is the layer's own deferred FFN boundary, so mono and vLLM layers
interleave freely. Weights are views of the loaded parameters, taken at every
launch; the attention metadata, KV cache and slot mapping come from the
forward context, as the fused QK-norm / RoPE / KV-insert op takes them.
"""

import dataclasses
import hashlib
import json
from typing import Any

import torch

from sglang.srt.monokernels.spec import Plan
from sglang.srt.rocm_muse_spark.mono.runner import MonoRunner


def _ptr_checked(t: torch.Tensor, name: str) -> torch.Tensor:
    if not t.is_contiguous():
        raise ValueError(f"MonoKernel input {name} must be contiguous")
    return t


def layer_keys(
    layer,
    width: int,
    q_len: int,
    fault: dict | None = None,
    *,
    post_eps: float | None = None,
    experimental_two_cta: bool = False,
):
    """The K1 and K2 build keys of ``layer`` at ``width`` rows."""
    from sglang.srt.rocm_muse_spark.mono.ops import k1, k2

    fault = fault or {}
    a = layer.self_attn
    m = layer.feed_forward
    k1_key = k1.make_key(
        tokens=width,
        q_len=q_len,
        window=max(0, int(a.attn.sliding_window_size)),
        rope=bool(a.use_rope),
        scale=float(a.scaling),
        eps=a.pre_qkv_proj_norm.variance_epsilon,
        # K1 closes the *previous* layer's FFN boundary.  Do not infer this
        # epsilon from the current layer even though the target checkpoint
        # happens to use one value everywhere.
        post_eps=(
            layer.post_ffn_norm.variance_epsilon if post_eps is None else post_eps
        ),
        qk_eps=a.rms_norm_eps,
        **fault,
    )
    k2_key = k2.make_key(
        tokens=width,
        experimental_two_cta=experimental_two_cta,
        eps_post_attn=layer.post_attn_norm.variance_epsilon,
        eps_ffn=layer.pre_ffn_norm.variance_epsilon,
        eps_pre_expert=m.pre_expert_norm.variance_epsilon,
        eps_post_expert=m.post_expert_norm.variance_epsilon,
        # The original adapter folds this scale into its router weights;
        # SGLang keeps it as a separate operation before sigmoid.
        router_output_multiplier=(
            1.0
            if getattr(m, "router_multiplier", None) is None
            else float(m.router_multiplier)
        ),
        **fault,
    )
    return k1_key, k2_key


def compile_checked(key) -> list[str]:
    """Compiles one build without launching it, then the residency proof: the
    runtime's occupancy must hold the selected whole grid resident, with no
    private scratch. Returns problems (empty: launchable).

    SGPR-to-VGPR spills can still occur and are reported separately in the
    compiled kernel resources.
    """
    from sglang.srt.rocm_muse_spark.mono.ops import k1, k2
    from sglang.srt.rocm_muse_spark.mono.ops.common.plan import BLOCKS, THREADS
    from sglang.srt.rocm_muse_spark.mono.ops.common.residency import residency

    mod = k1 if isinstance(key, k1.K1Build) else k2
    mod.compile_only(key)
    cus = torch.cuda.get_device_properties().multi_processor_count
    found = residency(mod.build(key), THREADS)
    if not found:
        return [f"{type(key).__name__}: no code object to check"]
    blocks = BLOCKS if mod is k1 else k2.grid_blocks(key)
    return [p for r in found for p in r.problems(blocks, cus)]


class MuseLayerKernel:
    """One sparse layer at one width (rows a launch)."""

    def __init__(
        self, layer, runner: MonoRunner, width: int, *, native_attention: bool = False
    ):
        self.layer = layer
        self.runner = runner
        self.width = width
        # Full attention keeps its native cache and attention implementation.
        # Only its output projection and MoE use the fused K2 kernel.
        self.native_attention = native_attention
        self.name = f"layer.{layer.layer_idx}.attn"
        self.cache_layers = () if native_attention else (self.name,)
        # the step's first launch bumps the epoch (a graph captures it)
        self.first = layer.layer_idx == runner.first_layer
        self.k1_key, self.k2_key = layer_keys(
            layer,
            width,
            runner.q_len,
            runner.fault,
            post_eps=runner.prev_post_eps.get(layer.layer_idx),
            experimental_two_cta=runner.k2_keys[width].experimental_two_cta,
        )

    # ---- LayerKernel protocol

    def health(self) -> torch.Tensor:
        """The output rank's flags, a word a TP rank."""
        return self.runner.flags

    def warmup_keys(self) -> list:
        return [self.k2_key] if self.native_attention else [self.k1_key, self.k2_key]

    def compile(self, key) -> None:
        from sglang.srt.rocm_muse_spark.mono.ops import k1, k2

        (k1 if key == self.k1_key else k2).compile_only(key)

    def launch(self, step: Plan, positions: torch.Tensor, x: Any, forward_batch):
        return self.forward(positions, x, forward_batch, new_step=self.first)

    # ---- the two launches

    def forward(self, positions: torch.Tensor, x: Any, forward_batch, new_step: bool):
        from sglang.srt.rocm_muse_spark.layers import MuseSpark1xResidualStream
        from sglang.srt.rocm_muse_spark.mono.cache import (
            attention_inputs,
            prepare_seq_lens,
        )
        from sglang.srt.rocm_muse_spark.mono.ops import k1, k2

        layer, runner, t = self.layer, self.runner, self.width
        a = layer.self_attn
        m = layer.feed_forward
        ex = m.experts.routed_experts
        lengths = forward_batch.seq_lens
        if lengths.numel() > runner.seq_lens.numel():
            raise ValueError("MonoKernel sequence-length buffer capacity exceeded")
        seq_lens = runner.seq_lens[: lengths.numel()]
        if new_step:
            seq_lens = prepare_seq_lens(forward_batch, runner.seq_lens, runner.q_len)
        stream, branch = x.stream, x.branch
        if stream.shape[0] != t or branch.shape[0] != t:
            raise ValueError(f"MonoKernel width {t}: got {stream.shape[0]} rows")
        if float(x.post_norm.variance_epsilon) != self.k1_key.post_eps:
            raise ValueError(
                "MonoKernel predecessor post-FFN epsilon changed after build: "
                f"{x.post_norm.variance_epsilon} != {self.k1_key.post_eps}"
            )
        if new_step:
            runner.step_begin()
        stream_a = runner.stream_a[:t]
        core = runner.core[:t]
        if self.native_attention:
            from sglang.srt.rocm_muse_spark.model import _original_boundary

            attn_input, stream_a = _original_boundary(x, a.pre_qkv_proj_norm)
            core = a.forward_attention_core(positions, attn_input, forward_batch)
        else:
            cache = attention_inputs(layer, forward_batch, seq_lens=seq_lens)
            k1.k1_launch(
                self.k1_key,
                stream_in=_ptr_checked(stream, "stream"),
                branch_in=_ptr_checked(branch, "branch"),
                w_pf=x.post_norm.weight,
                alpha_f=x.connection.alpha,
                beta_f=x.connection.beta,
                w_qn=a.pre_qkv_proj_norm.weight,
                w_qkv=a.qkv_proj.weight,
                positions=positions,
                cos_sin=a.rotary_emb.cos_sin_cache if a.use_rope else None,
                slot_mapping=cache.slot_mapping,
                kv_cache=cache.kv_cache,
                block_table=cache.block_table,
                seq_lens=cache.seq_lens,
                stream_a=stream_a,
                core=core,
                layer=layer.layer_idx,
                rank=runner.rank,
                **runner.launch_args(),
            )
        stream_out = torch.empty_like(stream)
        branch_out = torch.empty_like(branch)
        k2.k2_launch(
            self.k2_key,
            core=core,
            stream_a=stream_a,
            w_o=a.o_proj.weight,
            alpha_a=layer.add_attn.alpha,
            beta_a=layer.add_attn.beta,
            w_fn=layer.pre_ffn_norm.weight,
            w_router=m.router.weight,
            expert_bias=m.expert_bias,
            w_pool=m.pooler.weight,
            w_pre=m.pre_expert_norm.weight,
            w13=ex.w13_weight,
            w13s=ex.w13_weight_scale,
            w2=ex.w2_weight,
            w2s=ex.w2_weight_scale,
            w_post=m.post_expert_norm.weight,
            w_unpool=m.unpooler.weight,
            stream_out=stream_out,
            branch_out=branch_out,
            layer=layer.layer_idx,
            **runner.launch_args(),
            **runner.peer_args(),
        )
        return MuseSpark1xResidualStream(
            stream=stream_out,
            branch=branch_out,
            post_norm=layer.post_ffn_norm,
            connection=layer.add_ffn,
        )


# ---- standalone use (drivers, tests): one runner a process, kernels by
# (layer, width); every call is its own step (bumps the epoch)

_RUNNER: MonoRunner | None = None
_KERNELS: dict[tuple[int, int], MuseLayerKernel] = {}


def prepare_layer_monokernel(
    layer,
    widths: tuple[int, ...] = (1,),
    q_len: int = 1,
    fault: dict | None = None,
    prev_post_eps: float | None = None,
) -> MonoRunner:
    """Builds this rank's runner (collective over the TP group: every rank must
    call it, outside CUDA graph capture) and ``layer``'s kernels at ``widths``."""
    global _RUNNER
    if _RUNNER is None:
        device = torch.device("cuda", torch.cuda.current_device())
        _RUNNER = MonoRunner(device)
    _RUNNER.q_len = q_len
    _RUNNER.first_layer = layer.layer_idx
    _RUNNER.prev_post_eps[layer.layer_idx] = float(
        layer.post_ffn_norm.variance_epsilon if prev_post_eps is None else prev_post_eps
    )
    if fault is not None:
        _RUNNER.fault = dict(fault)
    for w in widths:
        k = MuseLayerKernel(layer, _RUNNER, w)
        problems = [p for key in k.warmup_keys() for p in compile_checked(key)]
        if problems:
            raise RuntimeError("MonoKernel not launchable: " + "; ".join(problems))
        _KERNELS[layer.layer_idx, w] = k
    return _RUNNER


def run_layer_monokernel(layer, positions: torch.Tensor, x: Any, new_step: bool = True):
    """``layer(positions, x)`` on its two persistent launches (the kernels
    ``prepare_layer_monokernel`` built), under the caller's forward context."""
    k = _KERNELS.get((layer.layer_idx, positions.size(-1)))
    if k is None:
        raise RuntimeError(
            f"no MonoKernel for layer {layer.layer_idx} at width {positions.size(-1)}"
        )
    raise RuntimeError(
        "standalone run_layer_monokernel requires ForwardBatch in SGLang"
    )


def layer_monokernel_health() -> torch.Tensor | None:
    """This process's runner's flags (meaningful on TP rank 0) and its own
    health words."""
    return None if _RUNNER is None else _RUNNER.flags


def layer_monokernel_identity(layer, width: int) -> dict[str, Any]:
    """Return the exact source/build identity used by a prepared candidate.

    The source digests are part of FlyDSL's cache key.  Emitting the same data
    with every result prevents a timing or SNR record from being attributed to
    a later dirty-worktree candidate.
    """
    from sglang.srt.rocm_muse_spark.mono.ops import k1, k2

    kernel = _KERNELS.get((layer.layer_idx, width))
    if kernel is None:
        raise RuntimeError(
            "no prepared MonoKernel identity for layer "
            f"{layer.layer_idx}, width {width}"
        )
    k1_key = dataclasses.asdict(kernel.k1_key)
    k2_key = dataclasses.asdict(kernel.k2_key)

    def build_identity(source: str, key: dict[str, Any]) -> dict[str, Any]:
        build = {"source_sha256": source, "key": key}
        encoded = json.dumps(build, sort_keys=True, separators=(",", ":")).encode()
        build["build_sha256"] = hashlib.sha256(encoded).hexdigest()
        return build

    payload = {
        "schema": "muse_layer_monokernel_candidate_v1",
        "layer": int(layer.layer_idx),
        "width": int(width),
        "module_realpath": str(__file__),
        "gpu_arch": torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).gcnArchName,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "k1": build_identity(k1._SOURCES, k1_key),
        "k2": build_identity(k2._SOURCES, k2_key),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["candidate_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload
