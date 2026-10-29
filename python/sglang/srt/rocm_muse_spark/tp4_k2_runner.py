# SPDX-License-Identifier: Apache-2.0
"""Opt-in TP4/EP4 K2 for Muse Spark target verify (``SGLANG_MUSE_TP4_K2=1``).

One persistent launch per MXFP4 MoE layer and rank (``tp4mono.ops.tpm_k2``) replaces the native chain from
o_proj through the unpool: o_proj and its all-reduce, the attention boundary, the joint router/pooler, the
top-8 route, the compact-A4 experts, the finalize, the MoE all-reduce, and the unpool with its gather. It runs
at the verify widths in ``WIDTHS`` (rows = batch x 8 draft tokens, batch 1..5), where it was measured faster
than the native chain; every other width keeps the native path.

With ``ffn_boundaries`` (``SGLANG_MUSE_TP4_K2_FFN_BOUNDARY=1``) the launch also applies the layer's FFN
boundary -- post-FFN norm, gated residual add and the next layer's attention norm, the work of
``FusedPostNormResidualPreNorm`` -- in place of its gather, and ``forward_through_ffn_boundary`` returns the
next layer's stream and attention input.

The runner owns one scratch, epoch, health and peer allocation per process, sized for the widest width. Each
layer's existing weights are bound in place (no copies); ``refresh`` re-checks the bound storage after a
reload. The model bumps the epoch once a forward (``step_begin``) and fail-stops on a health alarm after its
last layer (``step_end``); both are captured into the CUDA graph with the launches.
"""

import logging
import os
from dataclasses import dataclass, replace

import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.rocm_muse_spark.layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.rocm_muse_spark.tp4mono.ops import tpm_k2
from sglang.srt.rocm_muse_spark.tp4mono.ops.common.epoch import advance_epoch_
from sglang.srt.rocm_muse_spark.tp4mono.ops.common.residency import residency
from sglang.srt.rocm_muse_spark.tp4mono.runtime.abi import (
    FAILED,
    PEER_HEAD,
    health_words,
)
from sglang.srt.rocm_muse_spark.tp4mono.runtime.peer import (
    PeerBuffer,
    bind_agreed,
    tp_agree,
)

logger = logging.getLogger(__name__)

TP = 4
HIDDEN = 8192
CORE = 2048  # this rank's o_proj input: 32 heads x 64
EXPERT_HIDDEN = 4096
LOCAL_EXPERTS = 64
WIDTHS = (8, 16, 24, 32, 40)


class Tp4K2Unsupported(ValueError):
    """A collectively declined model contract; native execution is still valid."""


def widths_from_env() -> tuple:
    """``WIDTHS`` up to ``SGLANG_MUSE_TP4_K2_MAX_ROWS`` (default 40, one of ``WIDTHS``)."""
    value = os.environ.get("SGLANG_MUSE_TP4_K2_MAX_ROWS", str(WIDTHS[-1]))
    if not value.isdigit() or int(value) not in WIDTHS:
        raise Tp4K2Unsupported(f"SGLANG_MUSE_TP4_K2_MAX_ROWS must be one of {WIDTHS}")
    return tuple(m for m in WIDTHS if m <= int(value))


def _signature(tensor: torch.Tensor) -> tuple:
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _tensor(name, value, shape, dtype, device) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or value.dtype != dtype
        or value.device != device
        or not value.is_contiguous()
    ):
        raise Tp4K2Unsupported(f"TP4 K2: unsupported {name} layout")
    return value


def _layer_parameters(layer, rank: int, device: torch.device) -> dict:
    """Validate the layer contract the K2 implements and return the tensors it reads in place."""
    attention, moe, boundary = layer.self_attn, layer.mlp, layer._fused_attn_boundary
    if not (
        moe.tp_size == moe.ep_size == TP
        and moe.ep_rank == rank
        and moe.num_local_experts == LOCAL_EXPERTS
        and moe.local_expert_start == LOCAL_EXPERTS * rank
        and moe.num_experts == TP * LOCAL_EXPERTS
        and moe.top_k == 8
        and moe.hidden_size == HIDDEN
        and moe.expert_hidden_size == EXPERT_HIDDEN
        and moe.intermediate_size == EXPERT_HIDDEN
        and moe.route_eps == 1e-15
        and moe.router_in_fp32
        and moe.mxfp4_checkpoint
        and not moe.nvfp4_checkpoint
        and moe.pre_expert_proj is not None
        and moe.pre_expert_proj.bias is None
        and moe.post_expert_proj is not None
        and moe.post_expert_proj.bias is None
        and attention.q_size == CORE
        and getattr(attention.o_proj, "bias", None) is None
    ):
        raise Tp4K2Unsupported("TP4 K2: unsupported layer contract")
    op = moe._routing_op()
    if not (
        op.router_score_normalization == "sigmoid"
        and op.expert_bias is moe.expert_bias
        and op.router_output_multiplier == moe.router_multiplier
        and op.route_scale == moe.route_scale
        and op.router_score_division_eps == moe.route_eps
        and op.num_valid_tokens is None
        and op.local_expert_mapping is None
        and op.sol_topk_out is None
    ):
        raise Tp4K2Unsupported("TP4 K2: unsupported routing")
    # Attention boundary: weightless post-norm, channel-wise stream gate, weighted pre-FFN norm.
    gate = layer.post_attention_residual_gate
    if not (
        boundary._post_norm is layer.post_attn_norm
        and boundary._residual_add is gate
        and boundary._pre_norm is layer.pre_feedforward_layernorm
        and not boundary._post_norm_is_input_scale
        and boundary._use_residual_gate
        and not boundary._pre_norm_is_input_scale
        and not layer.post_attn_norm.has_weight
        and layer.pre_feedforward_layernorm.has_weight
        and getattr(boundary, "_weights_cached", False)
        and isinstance(gate, MuseSpark1xNormPreservingResidualConnection)
        and gate.use_stream_gate
        and gate.gate.numel() == HIDDEN
    ):
        raise Tp4K2Unsupported("TP4 K2: unsupported attention boundary")
    # The K2 reads the pre-expert norm's cached effective weight and applies the post-expert
    # norm as bf16(y * bf16(w + 1)) with an input scale.
    pre, post = moe.pre_expert_norm, moe.post_expert_norm
    if not (
        pre.has_weight
        and pre.zero_centered_gamma
        and isinstance(post, MuseSpark1xRMSNormWithInputScale)
        and post.has_weight
        and post.zero_centered_gamma
        and post.gain_center == 1.0
    ):
        raise Tp4K2Unsupported("TP4 K2: unsupported expert norms")
    experts = moe.experts
    if not (
        getattr(experts, "_single_layout_mode", False)
        and getattr(experts, "_single_layout_state", None) == "shuffled"
    ):
        raise Tp4K2Unsupported("TP4 K2 requires the shuffled MXFP4 single layout")
    packed = getattr(moe, "router_pooler_packed_weight", None)
    if packed is None or packed.data_ptr() != moe.pre_expert_proj.weight.data_ptr():
        raise Tp4K2Unsupported("TP4 K2 requires the joint router/pooler weight")
    bf16, f32, u8 = torch.bfloat16, torch.float32, torch.uint8
    e = LOCAL_EXPERTS
    return dict(
        w_o=_tensor("o_proj", attention.o_proj.weight, (HIDDEN, CORE), bf16, device),
        alpha_a=_tensor("gate alpha", boundary._cc_alpha, (HIDDEN,), f32, device),
        beta_a=_tensor("gate beta", boundary._cc_beta, (HIDDEN,), f32, device),
        w_fn=_tensor("pre-FFN norm", boundary._cc_pre_weight, (HIDDEN,), bf16, device),
        w_joint=_tensor("joint", packed, (EXPERT_HIDDEN + 512, HIDDEN), bf16, device),
        expert_bias=_tensor("expert bias", moe.expert_bias, (TP * e,), f32, device),
        w_pre=_tensor(
            "pre-expert norm",
            getattr(pre, "_effective_weight_cache", None),
            (EXPERT_HIDDEN,),
            bf16,
            device,
        ),
        w13=_tensor("w13", experts.w13_weight, (e, 2 * EXPERT_HIDDEN, 2048), u8, device),
        w13s=_tensor("w13 scale", experts.w13_weight_scale, (e, 2 * EXPERT_HIDDEN, 128), u8, device),
        w2=_tensor("w2", experts.w2_weight, (e, EXPERT_HIDDEN, 2048), u8, device),
        w2s=_tensor("w2 scale", experts.w2_weight_scale, (e, EXPERT_HIDDEN, 128), u8, device),
        w_post=_tensor("post-expert norm", post.weight, (EXPERT_HIDDEN,), bf16, device),
        w_unpool=_tensor("unpool", moe.post_expert_proj.weight, (HIDDEN, EXPERT_HIDDEN), bf16, device),
    )


def _ffn_boundary_parameters(layer, boundary, device: torch.device) -> dict:
    """Validate the FFN boundary the K2 tail implements (``tpm_front.ffn_boundary_token``) and return the
    cached tensors ``FusedPostNormResidualPreNorm`` passes to its kernel."""
    post, gate = layer.post_feedforward_layernorm, layer.post_feedforward_residual_gate
    if not (
        boundary._post_norm is post
        and boundary._residual_add is gate
        and boundary._post_norm_is_input_scale
        and post.has_weight
        and boundary._use_residual_gate
        and isinstance(gate, MuseSpark1xNormPreservingResidualConnection)
        and gate.gate.numel() == HIDDEN
        and boundary._pre_norm.has_weight
        and not boundary._pre_norm_is_input_scale
        and getattr(boundary, "_weights_cached", False)
    ):
        raise Tp4K2Unsupported("TP4 K2: unsupported FFN boundary")
    bf16, f32 = torch.bfloat16, torch.float32
    return dict(
        w_post_ffn=_tensor("post-FFN norm", boundary._cc_post_weight, (HIDDEN,), bf16, device),
        alpha_f=_tensor("FFN gate alpha", boundary._cc_alpha, (HIDDEN,), f32, device),
        beta_f=_tensor("FFN gate beta", boundary._cc_beta, (HIDDEN,), f32, device),
        w_next_attn=_tensor("next attention norm", boundary._cc_pre_weight, (HIDDEN,), bf16, device),
    )


def latency_trim_fields(last_layer: int) -> dict:
    """The K2Build fields SGLANG_MUSE_TP4_K2_LATENCY_TRIM sets; ``last_layer``: the step's last K2 layer, the
    only launch that ends in the cross-rank finish consensus. Every field keeps the arithmetic and its order
    (bitwise); c16-claude/k2-lowm/RESULTS.md has each one's measured saving."""
    if not (isinstance(last_layer, int) and last_layer >= 0):
        raise ValueError(f"TP4 K2 latency trim needs the last bound layer, got {last_layer!r}")
    return dict(latency_trim=True, poll_health_grace=True, wave_major_down=True, moe_rs_from=16,
                rendezvous_layer=last_layer)


def _layer_key(layer, ffn_boundary=None, last_layer=None) -> tpm_k2.K2Build:
    """The build for this layer's constants; ``tokens`` is set per width. ``last_layer``: the highest bound
    layer index (the runner's), for SGLANG_MUSE_TP4_K2_LATENCY_TRIM."""
    moe = layer.mlp
    tail = {}
    if ffn_boundary is not None:
        tail = dict(
            ffn_boundary=True,
            eps_post_ffn=float(ffn_boundary._post_norm.eps),
            eps_next_attn=float(ffn_boundary._pre_norm.eps),
        )
    return tpm_k2.K2Build(
        tokens=WIDTHS[0],
        eps_post_attn=float(layer.post_attn_norm.eps),
        eps_ffn=float(layer.pre_feedforward_layernorm.eps),
        eps_pre_expert=float(moe.pre_expert_norm.eps),
        eps_post_expert=float(moe.post_expert_norm.eps),
        router_output_multiplier=float(moe.router_multiplier),
        route_scale=float(moe.route_scale),
        **tail,
        **(latency_trim_fields(last_layer) if envs.SGLANG_MUSE_TP4_K2_LATENCY_TRIM.get() else {}),
    )


def _bound_parameters(layer, ffn_boundary, rank: int, device: torch.device) -> dict:
    params = _layer_parameters(layer, rank, device)
    if ffn_boundary is not None:
        params.update(_ffn_boundary_parameters(layer, ffn_boundary, device))
    return params


@dataclass
class _LayerBinding:
    layer: object
    ffn_boundary: object  # the layer's FusedPostNormResidualPreNorm when the tail applies it, else None
    key: tpm_k2.K2Build
    parameters: dict
    signatures: dict
    launches: dict  # rows -> compiled launch


class Tp4K2Runner:
    """Construct collectively after post-load preparation, before graph capture.

    ``layers`` are the MXFP4 MoE decoder layers; each must satisfy ``_layer_parameters``.
    ``ffn_boundaries`` (layer index -> its FFN boundary, for every layer) moves those boundaries into the launch.
    """

    def __init__(self, layers, parallel, widths=None, ffn_boundaries=None):
        self.peers = None
        self.layers = {}
        self._reported = set()
        self.fuses_ffn_boundary = ffn_boundaries is not None
        self.group = parallel.tp_group
        self.rank = self.group.rank_in_group
        self.device = layers[0].self_attn.o_proj.weight.device if layers else None
        unsupported = None
        properties = None

        def validate_local():
            nonlocal unsupported, properties
            try:
                self.widths = tuple(widths) if widths is not None else widths_from_env()
                if not self.widths or any(m not in WIDTHS for m in self.widths):
                    raise Tp4K2Unsupported(f"TP4 K2 widths must be within {WIDTHS}")
                if not (
                    parallel.tp_size == TP
                    and parallel.attn_tp_size == TP
                    and parallel.moe_ep_size == TP
                    and parallel.moe_tp_size == 1
                    and parallel.moe_ep_rank == self.rank
                    and parallel.tp_rank == self.rank
                    and self.group.world_size == TP
                ):
                    raise Tp4K2Unsupported("TP4 K2 requires aligned TP4/EP4 rank order")
                if not layers:
                    raise Tp4K2Unsupported("TP4 K2 has no MXFP4 MoE layers")
                self.last_layer = max(layer.layer_idx for layer in layers)
                properties = torch.cuda.get_device_properties(self.device)
                if not getattr(properties, "gcnArchName", "").startswith("gfx950") or (
                    properties.multi_processor_count != tpm_k2.BLOCKS
                ):
                    raise Tp4K2Unsupported("TP4 K2 requires a 256-CU gfx950")
                for layer in layers:
                    boundary = None
                    if ffn_boundaries is not None:
                        boundary = ffn_boundaries.get(layer.layer_idx)
                        if boundary is None:
                            raise Tp4K2Unsupported("TP4 K2: a layer has no FFN boundary")
                    params = _bound_parameters(layer, boundary, self.rank, self.device)
                    self.layers[layer.layer_idx] = _LayerBinding(
                        layer,
                        boundary,
                        _layer_key(layer, boundary, self.last_layer),
                        params,
                        {name: _signature(value) for name, value in params.items()},
                        {},
                    )
            except Tp4K2Unsupported as error:
                unsupported = error

        # Declined contracts are distinguished from compiler, HIP and allocation
        # failures: only the former let every rank keep the native path.
        bind_agreed(validate_local, self.group.cpu_group)
        if not tp_agree(unsupported is None, self.group.cpu_group):
            raise Tp4K2Unsupported(str(unsupported or "another TP rank declined TP4 K2"))
        contract = (self.widths, tuple((i, b.key) for i, b in self.layers.items()))
        contracts = [None] * TP
        dist.all_gather_object(contracts, contract, group=self.group.cpu_group)
        if any(other != contract for other in contracts):
            raise Tp4K2Unsupported("TP4 K2 layer/key/width selection differs across TP ranks")

        def prepare():
            launches = {}
            for binding in self.layers.values():
                for m in self.widths:
                    key = replace(binding.key, tokens=m)
                    if key not in launches:
                        tpm_k2.compile_only(key)
                        launch = tpm_k2.build(key)
                        resources = residency(launch, tpm_k2.THREADS)
                        if not resources or any(
                            r.problems(tpm_k2.BLOCKS, properties.multi_processor_count)
                            for r in resources
                        ):
                            raise RuntimeError(f"TP4 K2 M{m} failed residency admission")
                        launches[key] = launch
                    binding.launches[m] = launches[key]
            widest = max(self.widths)
            self.scratch = torch.zeros(
                max(tpm_k2.scratch_bytes(m) for m in self.widths),
                dtype=torch.uint8,
                device=self.device,
            )
            self.epoch = torch.zeros(1, dtype=torch.int32, device=self.device)
            self.health = torch.zeros(
                health_words(tpm_k2.BLOCKS), dtype=torch.int32, device=self.device
            )
            self.stream_out = torch.empty((widest, HIDDEN), dtype=torch.float32, device=self.device)
            self.branch_out = torch.empty((widest, HIDDEN), dtype=torch.bfloat16, device=self.device)
            torch.cuda.synchronize(self.device)

        bind_agreed(prepare, self.group.cpu_group)
        self.peers = PeerBuffer(
            PEER_HEAD + max(tpm_k2.peer_bytes(m) for m in self.widths),
            self.group.cpu_group,
            self.rank,
            TP,
            self.device,
        )

        def bind_peers():
            self.peer_table = self.peers.addresses + PEER_HEAD
            # The alarm words live in rank 0's peer header, a word a rank.
            self.alarm = int(self.peers.addresses[0].item()) + 4 * self.rank
            torch.cuda.synchronize(self.device)

        try:
            bind_agreed(bind_peers, self.group.cpu_group)
        except Exception:
            self.close()
            raise

    def eligible(self, forward_batch, rows: int) -> bool:
        """Host metadata only; the decision holds for the whole model step on every rank."""
        return rows in self.widths and forward_batch.forward_mode.is_target_verify()

    def step_begin(self) -> None:
        """Once a forward, before its first launch (captured into the graph)."""
        advance_epoch_(self.epoch)

    def step_end(self) -> None:
        """After the last launch of a forward: a failed in-kernel wait on any rank stops the step.

        Every launch returns only after all CTAs and ranks reached its terminal rendezvous and
        imported the shared alarm words, so this captured assertion makes a failure fatal on
        every rank; no rank continues with partially written state. With
        SGLANG_MUSE_TP4_K2_LATENCY_TRIM only the launch of the last bound layer has that rendezvous
        (K2Build.rendezvous_layer): a failure on any rank still reaches every rank's health there,
        before this assertion, and the step's outputs are never used.
        """
        torch._assert_async(
            self.health[FAILED] == 0,
            "Muse TP4 K2 fail-stop: a bounded device wait failed",
        )

    def forward(self, layer, core: torch.Tensor, stream: torch.Tensor):
        """(stream after the attention boundary, unpooled MoE branch) of ``layer`` for these rows."""
        if self.fuses_ffn_boundary:
            raise RuntimeError("TP4 K2: this runner applies the FFN boundary; use forward_through_ffn_boundary")
        m = core.shape[0]
        if stream.untyped_storage().data_ptr() == self.stream_out.untyped_storage().data_ptr():
            raise RuntimeError("TP4 K2: the input stream aliases the runner's output")
        stream_out, branch_out = self.stream_out[:m], self.branch_out[:m]
        self._launch(layer, core, stream, stream_out=stream_out, branch_out=branch_out, next_in=0)
        return stream_out, branch_out

    def forward_through_ffn_boundary(self, layer, core: torch.Tensor, stream: torch.Tensor):
        """(stream after the FFN boundary, the next layer's attention input) of ``layer`` for these rows: what
        ``forward`` followed by the layer's ``FusedPostNormResidualPreNorm`` returns, in one launch."""
        if not self.fuses_ffn_boundary:
            raise RuntimeError("TP4 K2: this runner was built without the FFN boundary")
        m = core.shape[0]
        # Fresh outputs a layer: the next layer's launch reads this stream while it writes its own.
        stream_out = torch.empty((m, HIDDEN), dtype=torch.float32, device=self.device)
        next_in = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=self.device)
        self._launch(layer, core, stream, stream_out=stream_out, branch_out=0, next_in=next_in)
        return stream_out, next_in

    def _launch(self, layer, core, stream, **outputs) -> None:
        binding = self.layers[layer.layer_idx]
        m = core.shape[0]
        _tensor("core", core, (m, CORE), torch.bfloat16, self.device)
        _tensor("stream", stream, (m, HIDDEN), torch.float32, self.device)
        unused = dict.fromkeys(("w_post_ffn", "alpha_f", "beta_f", "w_next_attn"), 0)
        args = dict(
            unused,
            **binding.parameters,
            **outputs,
            core=core,
            stream_a=stream,
            x_in=0,
            scratch=self.scratch,
            peers=self.peer_table,
            rank=self.rank,
            epoch=self.epoch,
            layer=layer.layer_idx,
            health=self.health,
            alarm=self.alarm,
            tl=0,
        )
        args = {
            name: value.data_ptr() if isinstance(value, torch.Tensor) else value
            for name, value in args.items()
        }
        binding.launches[m](*tpm_k2.ABI.pack(args), stream=torch.cuda.current_stream())
        if m not in self._reported:
            self._reported.add(m)
            logger.info(
                "Muse TP4 K2 first launch: rows=%d layer=%d ffn_boundary=%d",
                m,
                layer.layer_idx,
                self.fuses_ffn_boundary,
            )

    def refresh(self) -> None:
        """Verify the graph-bound storage after a weight reload (``post_load_weights``).

        In-place reloads are read by the next launch as they are. A replaced tensor is
        rejected on every rank before another request runs: captured graphs hold the old
        addresses, so the model and its graphs must be rebuilt. Health and epoch are never reset.
        """

        def verify():
            for binding in self.layers.values():
                current = _bound_parameters(binding.layer, binding.ffn_boundary, self.rank, self.device)
                if (
                    _layer_key(binding.layer, binding.ffn_boundary, self.last_layer) != binding.key
                    or {k: _signature(v) for k, v in current.items()} != binding.signatures
                ):
                    raise RuntimeError("TP4 K2: reload changed bound storage or configuration")

        bind_agreed(verify, self.group.cpu_group)

    def close(self) -> None:
        """Release peer storage once every rank's GPU work is quiescent; idempotent."""
        peers, self.peers = self.peers, None
        if peers is not None:
            peers.close()
