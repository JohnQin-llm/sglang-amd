# SPDX-License-Identifier: Apache-2.0
"""Registered SGLang contract for the original Muse TP8 K1+K2 kernels."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist
from sglang.srt.monokernels.spec import (
    MONOKERNEL_API_VERSION,
    BuildContext,
    DeviceLimits,
    Feature,
    MonoSpec,
    Numerics,
    NumericsKind,
    Plan,
    Requirements,
)
from sglang.srt.rocm_muse_spark.model import _original_boundary
from sglang.srt.rocm_muse_spark.mono.kernel import (
    MuseLayerKernel,
    compile_checked,
    layer_keys,
)
from sglang.srt.rocm_muse_spark.mono.runner import MonoRunner, fail_if_unhealthy_
from sglang.srt.rocm_muse_spark.mono.spec import (
    BLOCKS,
    EXPERTS,
    HEAD_DIM,
    HIDDEN,
    INTER,
    KV_BLOCK,
    MOE_HIDDEN,
    SERVING_WIDTHS,
    TOP_K,
    TP,
    WIDE_ENABLED,
    WIDE_TOKENS,
    WINDOW,
    kernel_layers,
    native_attention_enabled,
    native_attention_layers,
)
from sglang.srt.runtime_context import get_parallel, get_spec


def _text_config(config):
    return getattr(config, "text_config", config)


def _layer_selection(config) -> tuple[list[int], set[int]]:
    native = (
        set(native_attention_layers(config))
        if os.getenv("SGLANG_MUSE_ORIGINAL_FULL_K2", "0") == "1"
        else set()
    )
    selected = sorted(set(kernel_layers(config)) | native)
    if native_attention_enabled():
        native = set(selected)
    return selected, native


def config_disable_reasons(cfg: Any, config: Any) -> list[str]:
    config = _text_config(config)
    reasons: list[str] = []
    expected = {
        "hidden_size": HIDDEN,
        "num_attention_heads": 128,
        "num_key_value_heads": 16,
        "head_dim": HEAD_DIM,
        "num_local_experts": EXPERTS,
        "num_experts_per_tok": TOP_K,
        "moe_hidden_size": MOE_HIDDEN,
        "moe_intermediate_size": INTER * TP,
        "sliding_window": WINDOW,
    }
    for name, want in expected.items():
        got = getattr(config, name, None)
        if got != want:
            reasons.append(f"config.{name} {got!r} is not {want!r}")
    if getattr(cfg, "page_size", None) != KV_BLOCK:
        reasons.append(f"page_size must be {KV_BLOCK}")
    kv_dtype = getattr(cfg, "kv_cache_dtype", None)
    if kv_dtype not in ("bf16", "bfloat16", "auto") and not (
        native_attention_enabled() and kv_dtype == "fp8_e4m3"
    ):
        reasons.append("K1 requires BF16 KV; FP8 E4M3 requires native attention")
    if os.getenv("SGLANG_MUSE_ORIGINAL_NATIVE_ONLY", "0") == "1":
        reasons.append(
            "SGLANG_MUSE_ORIGINAL_NATIVE_ONLY=1 selects the baseline; unset it "
            "to launch K1/K2"
        )
    # Kernel-level rows=8 numerical checks pass, but full-model quality
    # qualification is still pending. Target verification requires an explicit
    # experimental flag until that qualification is complete.
    algorithm = getattr(cfg, "speculative_algorithm", None)
    if WIDE_ENABLED and (
        algorithm != "DFLASH" or cfg.speculative_num_draft_tokens != 8
    ):
        reasons.append("wide staged verification requires DFlash block8")
    native_probe = any(
        os.getenv(name, "0") == "1"
        for name in (
            "SGLANG_MUSE_ORIGINAL_FULL_K2",
            "SGLANG_MUSE_ORIGINAL_NATIVE_ATTENTION",
        )
    )
    if native_probe and (
        algorithm != "DFLASH" or cfg.speculative_num_draft_tokens != 8
    ):
        reasons.append("native-attention K2 is an explicit DFlash block8 timing probe")
    if algorithm is not None:
        if algorithm != "DFLASH" or cfg.speculative_num_draft_tokens != 8:
            reasons.append("the timing probe supports only DFlash block8")
        if os.getenv("SGLANG_MUSE_ORIGINAL_UNQUALIFIED_DFLASH", "0") != "1":
            reasons.append("rows=8 target-verify is not production-qualified")
    return reasons


def model_disable_reasons(model, config, copies: dict) -> list[str]:
    config = _text_config(config)
    reasons: list[str] = []
    if copies:
        reasons.append("the original port does not retain checkpoint copies")
    if not getattr(model, "mxfp4_checkpoint", False):
        reasons.append("experts are not Quark MXFP4 W4A4")
    decoder = model.model.language_model
    selected, _ = _layer_selection(config)
    if not selected:
        reasons.append("no eligible sparse sliding layers")
    for i in selected:
        layer = decoder.layers[i]
        attention = layer.self_attn
        if attention.scaling != HEAD_DIM**-0.5 * config.qk_scale_factor:
            reasons.append(f"layer {i}: attention scale does not include QK scaling")
        if attention.scale_query_by != 1.0:
            reasons.append(f"layer {i}: native Q scaling differs from original K1")
        router = layer.mlp.router.weight
        if (
            router.shape != (EXPERTS, HIDDEN)
            or router.dtype != torch.float32
            or not router.is_contiguous()
        ):
            reasons.append(
                f"layer {i}: router must be contiguous FP32 [experts,hidden]"
            )
        expert = getattr(getattr(layer.mlp, "experts", None), "routed_experts", None)
        if expert is None:
            reasons.append(f"layer {i}: missing routed MXFP4 experts")
            continue
        expected = {
            "w13_weight": (EXPERTS, 2 * INTER, MOE_HIDDEN // 2),
            "w13_weight_scale": (EXPERTS, 2 * INTER, MOE_HIDDEN // 32),
            "w2_weight": (EXPERTS, MOE_HIDDEN, INTER // 2),
            "w2_weight_scale": (EXPERTS, MOE_HIDDEN, INTER // 32),
        }
        for name, shape in expected.items():
            tensor = getattr(expert, name, None)
            if tensor is None or tuple(tensor.shape) != shape:
                reasons.append(
                    f"layer {i}: {name} shape "
                    f"{None if tensor is None else tuple(tensor.shape)} is not {shape}"
                )
        if not getattr(expert.w13_weight, "is_shuffled", False):
            reasons.append(f"layer {i}: W4A4 weights are not AITER-shuffled")
        for residual in (layer.add_attn, layer.add_ffn):
            if not residual._coefficients_ready:
                reasons.append(f"layer {i}: residual coefficients are not prepared")
    return reasons


def _gather_build_errors(errors: list[str]) -> list[str]:
    group = get_parallel().tp_group
    if group.world_size == 1:
        return errors
    gathered: list[list[str] | None] = [None] * group.world_size
    dist.all_gather_object(gathered, errors, group=group.cpu_group)
    return [
        f"rank {rank}: {error}"
        for rank, rank_errors in enumerate(gathered)
        for error in rank_errors or ()
    ]


class BuiltMuseOriginal:
    def __init__(self, model, runner: MonoRunner, kernels: dict[tuple[int, int], Any]):
        self.model = model
        self.decoder = model.model.language_model
        self.runner = runner
        self.kernels = kernels

    def close(self) -> None:
        self.runner.close()

    def forward(
        self,
        plan: Plan,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        input_embeds: torch.Tensor | None = None,
    ):
        if plan.key.rows != positions.numel():
            raise ValueError(
                f"routed width {plan.key.rows} does not match {positions.numel()} rows"
            )
        if plan.key.tokens_per_request != self.runner.q_len:
            raise ValueError("MonoKernel query length differs from its compiled plan")
        decoder = self.decoder
        if input_embeds is None:
            embedding = self.model.get_input_embeddings()
            hidden = embedding(input_ids)
            # The multimodal wrapper returns already-normalized embeddings.
            # Match its native forward, including its BF16 output boundary.
            if embedding is decoder.embed_tokens and decoder.embed_norm is not None:
                hidden = decoder.embed_norm(hidden, out_dtype=torch.float32)
        else:
            hidden = input_embeds
        stream = hidden.to(decoder.residual_dtype)
        attn_input = decoder.layers[0].self_attn_input_norm(stream)
        boundary = None
        aux: list[torch.Tensor] = []
        for i, layer in enumerate(decoder.layers):
            kernel = self.kernels.get((i, plan.key.rows))
            if kernel is not None:
                if boundary is None:
                    raise RuntimeError(f"layer {i}: K1 has no deferred predecessor")
                boundary = kernel.launch(plan, positions, boundary, forward_batch)
            else:
                if boundary is not None:
                    attn_input, stream = _original_boundary(
                        boundary,
                        layer.self_attn.pre_attn_norm,
                    )
                boundary = layer(
                    positions,
                    stream,
                    forward_batch,
                    attn_input=attn_input,
                    defer_ffn_boundary=True,
                )
            if i in decoder.layers_to_capture:
                next_norm = (
                    decoder.norm
                    if i == len(decoder.layers) - 1
                    else decoder.layers[i + 1].self_attn.pre_attn_norm
                )
                # DFlash consumes the post-layer residual, before the next
                # layer's input norm. Use the same fused boundary arithmetic
                # as the native model, including its FP32 accumulation.
                _, captured_stream = _original_boundary(boundary, next_norm)
                aux.append(captured_stream.to(torch.bfloat16))
        hidden, _ = _original_boundary(
            boundary,
            decoder.norm,
            torch.bfloat16,
        )
        # One terminal check per model step, after the final K2 imported every
        # rank's alarms.  Do not add this after each covered layer.
        fail_if_unhealthy_(self.runner.health)
        aux_hidden = aux if self.model.capture_aux_hidden_states else None
        return self.model.logits_processor(
            input_ids,
            hidden,
            self.model.lm_head,
            forward_batch,
            aux_hidden,
        )


def build(model, config, context: BuildContext) -> BuiltMuseOriginal:
    decoder = model.model.language_model
    indices, native_layers = _layer_selection(config)
    spec_config = get_spec()
    q_len = (
        1
        if spec_config.speculative_algorithm is None
        else int(spec_config.speculative_num_draft_tokens)
    )
    supported_widths = (
        SERVING_WIDTHS if WIDE_ENABLED else ((8,) if q_len == 8 else (1,))
    )
    if q_len not in (1, 8) or any(
        w not in supported_widths or w % q_len for w in context.widths
    ):
        raise ValueError(
            "Original Muse kernels require widths 1/8/16/24 containing whole "
            "decode tokens or DFlash8 blocks"
        )
    compile_errors: list[str] = []
    built_keys = set()
    for i in indices:
        previous_eps = decoder.layers[i - 1].post_ffn_norm.variance_epsilon
        for width in context.widths:
            native_attention = width in WIDE_TOKENS or i in native_layers
            keys = layer_keys(
                decoder.layers[i],
                width,
                q_len,
                post_eps=previous_eps,
                native_attention=native_attention,
            )
            if native_attention:
                keys = keys[1:]
            for key in keys:
                if key in built_keys:
                    continue
                built_keys.add(key)
                try:
                    compile_errors.extend(compile_checked(key))
                except Exception as error:  # noqa: BLE001 - agree all build failures
                    compile_errors.append(
                        f"{type(key).__name__}: {type(error).__name__}: {error}"
                    )
    failures = _gather_build_errors(compile_errors)
    if failures:
        raise RuntimeError("K1/K2 compile/residency refusal: " + "; ".join(failures))

    runner = MonoRunner(torch.device("cuda", torch.cuda.current_device()))
    runner.q_len = q_len
    runner.first_layer = min(indices)
    kernels: dict[tuple[int, int], MuseLayerKernel] = {}
    for i in indices:
        runner.prev_post_eps[i] = float(
            decoder.layers[i - 1].post_ffn_norm.variance_epsilon
        )
        runner.layer_names[i] = f"model.layers.{i}"
        for width in context.widths:
            native_attention = width in WIDE_TOKENS or i in native_layers
            kernels[i, width] = MuseLayerKernel(
                decoder.layers[i],
                runner,
                width,
                native_attention=native_attention,
            )
    built = BuiltMuseOriginal(model, runner, kernels)
    # Keep an explicit owner for deterministic, idempotent server teardown.
    model._muse_original_monokernels = built
    return built


MONO_SPEC = MonoSpec(
    name="Muse original TP8 W4A4 K1+K2",
    devices={
        "gfx950": DeviceLimits(
            widths=SERVING_WIDTHS,
            tp_sizes=(TP,),
            resident_blocks=BLOCKS,
        )
    },
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"aiter"}),
        server_args={
            "page_size": frozenset({KV_BLOCK}),
            "kv_cache_dtype": frozenset({"auto", "bf16", "bfloat16", "fp8_e4m3"}),
        },
    ),
    numerics=Numerics(
        kind=NumericsKind.TOLERANCE,
        bounds={"forward.hidden.cosine": 0.999},
    ),
    model_disable_reasons=model_disable_reasons,
    build=build,
    api_version=MONOKERNEL_API_VERSION,
    config_disable_reasons=config_disable_reasons,
)
