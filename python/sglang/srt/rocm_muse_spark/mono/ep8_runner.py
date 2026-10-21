# SPDX-License-Identifier: Apache-2.0
"""Opt-in full K2 for Muse DFlash q8 with TP8 and complete EP8 experts.

The model owns one scratch/peer allocation. Layer bindings reference existing
parameters and prepacked expert weights. A stable expert-major router cache
is allocated per eligible layer; post-load refresh preserves captured addresses.
Native attention and the decoder's FFN boundaries remain in charge.
"""

from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from sglang.srt.rocm_muse_spark.mono.ops import ep8_k2
from sglang.srt.rocm_muse_spark.mono.ops.common.epoch import advance_epoch_
from sglang.srt.rocm_muse_spark.mono.ops.common.residency import residency
from sglang.srt.rocm_muse_spark.mono.runner import fail_if_unhealthy_
from sglang.srt.rocm_muse_spark.mono.runtime.abi import PEER_HEAD
from sglang.srt.rocm_muse_spark.mono.runtime.peer import (
    PeerBuffer,
    bind_agreed,
    tp_agree,
)


class Ep8FullK2Unsupported(ValueError):
    """A collectively declined model contract; native execution is still valid."""


def _signature(tensor: torch.Tensor) -> tuple:
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _tensor_version(tensor):
    # Some checkpoint loaders construct inference tensors without a version
    # counter. Refresh never relies on this optional diagnostic to skip copies.
    try:
        return tensor._version
    except RuntimeError:
        return None


def _check_tensor(name, tensor, shape, dtype, device):
    if (
        not isinstance(tensor, torch.Tensor)
        or tuple(tensor.shape) != shape
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
    ):
        raise Ep8FullK2Unsupported(f"EP8 full K2: unsupported {name} layout")


def _layer_parameters(layer, rank: int, device: torch.device) -> dict:
    """Validate the full boundary before any persistent kernel can launch."""
    attention, moe = layer.self_attn, layer.mlp
    if (
        moe.original_mono_tp
        or moe.ep_size != 8
        or moe.ep_rank != rank
        or moe.local_expert_start != rank * 32
        or moe.num_local_experts != 32
        or moe.num_experts != 256
        or moe.top_k != 8
        or moe.hidden_size != 8192
        or moe.expert_hidden_size != 4096
        or moe.intermediate_size != 4096
        or moe.route_scale != 1.0
        or moe.route_eps != 1e-15
        or not moe.router_in_fp32
        or attention.q_size != 1024
        or attention.o_proj.bias is not None
        or layer.post_attn_norm.weight is not None
    ):
        raise Ep8FullK2Unsupported("EP8 full K2: unsupported layer contract")
    norms = (layer.pre_feedforward_layernorm, moe.pre_expert_norm, moe.post_expert_norm)
    if any(
        not n.zero_centered_gamma or n.gain_center != 1.0 or n.original_precision
        for n in norms
    ):
        raise Ep8FullK2Unsupported(
            "EP8 full K2 requires native BF16 zero-centered norms"
        )
    gate = layer.post_attention_residual_gate
    if not gate.use_stream_gate:
        raise Ep8FullK2Unsupported("EP8 full K2 requires channel-wise residual gates")
    packed = moe.experts._prepacked_moe
    if packed is None:
        raise Ep8FullK2Unsupported(
            "EP8 full K2 requires existing prepacked expert weights"
        )
    values = dict(
        w_o=attention.o_proj.weight,
        alpha_a=gate.alpha,
        beta_a=gate.beta,
        w_fn=norms[0].weight,
        w_router=moe.router.weight,
        expert_bias=moe.expert_bias,
        w_pool=moe.pre_expert_proj.weight,
        w_pre=norms[1].weight,
        w13=packed.w13,
        w13s=packed.w13_scale,
        w2=packed.w2,
        w2s=packed.w2_scale,
        w_post=norms[2].weight,
        w_unpool=moe.post_expert_proj.weight,
    )
    shapes = dict(
        w_o=(8192, 1024),
        w_fn=(8192,),
        w_router=(8192, 256),
        expert_bias=(256,),
        w_pool=(4096, 8192),
        w_pre=(4096,),
        w_post=(4096,),
        w_unpool=(8192, 4096),
    )
    for name, shape in shapes.items():
        dtype = torch.float32 if name in ("w_router", "expert_bias") else torch.bfloat16
        _check_tensor(name, values[name], shape, dtype, device)
    for name in ("alpha_a", "beta_a"):
        value = values[name]
        if (
            value is None
            or value.numel() != 8192
            or value.dtype != torch.float32
            or value.device != device
            or not value.is_contiguous()
        ):
            raise Ep8FullK2Unsupported(f"EP8 full K2: unsupported {name}")
    # PrepackedMuseMoE owns the precise shuffled layout. Sizes catch accidental
    # EP1/intermediate-TP bindings without copying or interpreting its storage.
    for name, size in (
        ("w13", 32 * 8192 * 2048),
        ("w13s", 32 * 8192 * 128),
        ("w2", 32 * 4096 * 2048),
        ("w2s", 32 * 4096 * 128),
    ):
        value = values[name]
        if (
            value.numel() * value.element_size() != size
            or value.dtype != torch.uint8
            or value.device != device
            or not value.is_contiguous()
        ):
            raise Ep8FullK2Unsupported(f"EP8 full K2: unsupported prepacked {name}")
    return values


def _layer_key(layer):
    return ep8_k2.K2Build(
        tokens=8,
        eps_post_attn=float(layer.post_attn_norm.eps),
        eps_ffn=float(layer.pre_feedforward_layernorm.eps),
        eps_pre_expert=float(layer.mlp.pre_expert_norm.eps),
        eps_post_expert=float(layer.mlp.post_expert_norm.eps),
        router_output_multiplier=float(layer.mlp.router_multiplier),
    )


@dataclass
class _LayerBinding:
    layer: object
    key: object
    launch: object
    parameters: dict
    signatures: dict
    router_source_version: int | None = None


class Ep8FullK2Runner:
    """Construct collectively after post-load preparation, before graph capture."""

    def __init__(self, layers, parallel):
        self.peers = None
        self.layers = {}
        self.router_weights = {}
        self.parallel = parallel
        self.group = parallel.tp_group
        self.rank = self.group.rank_in_group
        self.device = next(layers[0].parameters()).device

        unsupported = None
        properties = None

        def validate_local():
            nonlocal unsupported, properties
            try:
                if (
                    parallel.tp_size != 8
                    or parallel.attn_tp_size != 8
                    or parallel.moe_ep_size != 8
                    or parallel.moe_tp_size != 1
                    or parallel.moe_ep_rank != self.rank
                    or parallel.tp_rank != self.rank
                    or self.group.world_size != 8
                ):
                    raise Ep8FullK2Unsupported(
                        "EP8 full K2 requires aligned TP8/EP8 rank order"
                    )
                properties = torch.cuda.get_device_properties(self.device)
                if not getattr(properties, "gcnArchName", "").startswith("gfx950"):
                    raise Ep8FullK2Unsupported("EP8 full K2 requires gfx950")
                for layer in layers:
                    # BF16 boundary experts keep the complete native path.
                    if getattr(layer.mlp.experts, "_prepacked_moe", None) is None:
                        continue
                    params = _layer_parameters(layer, self.rank, self.device)
                    self.layers[layer.layer_idx] = _LayerBinding(
                        layer,
                        _layer_key(layer),
                        None,
                        params,
                        {name: _signature(value) for name, value in params.items()},
                    )
                if not self.layers:
                    raise Ep8FullK2Unsupported(
                        "EP8 full K2 has no compatible prepacked layers"
                    )
            except Ep8FullK2Unsupported as error:
                unsupported = error

        # Declined contracts are distinguished from compiler, HIP and allocation
        # failures. Only the former allow every rank to retain native execution.
        bind_agreed(validate_local, self.group.cpu_group)
        if not tp_agree(unsupported is None, self.group.cpu_group):
            raise Ep8FullK2Unsupported(
                str(unsupported or "another TP rank declined EP8 full K2")
            )
        contract = tuple((index, binding.key) for index, binding in self.layers.items())
        contracts = [None] * 8
        dist.all_gather_object(contracts, contract, group=self.group.cpu_group)
        if any(other != contract for other in contracts):
            raise Ep8FullK2Unsupported(
                "EP8 full K2 layer/key selection differs across TP ranks"
            )

        def prepare():
            keys = {}
            for binding in self.layers.values():
                key = binding.key
                if key not in keys:
                    ep8_k2.compile_only(key)
                    launch = ep8_k2.build(key)
                    resources = residency(launch, 512)
                    if not resources or any(
                        r.problems(256, properties.multi_processor_count)
                        for r in resources
                    ):
                        raise RuntimeError("EP8 full K2 failed residency admission")
                    keys[key] = launch
                binding.launch = keys[key]
                # Native EP8 stores [hidden, experts]; K2 loads expert-major
                # rows. Preserve the native parameter and cache only this small
                # transpose (8 MiB/layer), never another expert-weight copy.
                source = binding.parameters["w_router"]
                packed_router = source.t().contiguous()
                self.router_weights[binding.layer.layer_idx] = packed_router
                binding.parameters["w_router"] = packed_router
                binding.router_source_version = _tensor_version(source)
            self.scratch = torch.zeros(
                max(ep8_k2.scratch_bytes(k) for k in keys),
                dtype=torch.uint8,
                device=self.device,
            )
            self.epoch = torch.zeros(1, dtype=torch.int32, device=self.device)
            self.health = torch.zeros(257 * 8, dtype=torch.int32, device=self.device)
            self.stream_out = torch.empty(
                (8, 8192), dtype=torch.float32, device=self.device
            )
            self.branch_out = torch.empty(
                (8, 8192), dtype=torch.bfloat16, device=self.device
            )
            torch.cuda.synchronize(self.device)

        bind_agreed(prepare, self.group.cpu_group)
        self.peers = PeerBuffer(
            PEER_HEAD + ep8_k2.peer_bytes(),
            self.group.cpu_group,
            self.rank,
            8,
            self.device,
        )

        def bind_peers():
            self.peer_table = self.peers.addresses + PEER_HEAD
            self.alarm = int(self.peers.addresses[0].item()) + 4 * self.rank
            torch.cuda.synchronize(self.device)

        try:
            bind_agreed(bind_peers, self.group.cpu_group)
        except Exception:
            self.close()
            raise

    @staticmethod
    def eligible(forward_batch, input_ids) -> bool:
        """Host metadata only; this decision is shared by the whole model step."""
        return (
            forward_batch.forward_mode.is_target_verify()
            and forward_batch.batch_size == 1
            and forward_batch.seq_lens.numel() == 1
            and input_ids.numel() == 8
            and getattr(forward_batch.spec_info, "draft_token_num", None) == 8
        )

    def step_begin(self):
        advance_epoch_(self.epoch)

    def step_end(self):
        # Each layer's terminal rendezvous has imported every rank's alarm.
        fail_if_unhealthy_(self.health)

    def forward(self, layer, core, stream):
        binding = self.layers[layer.layer_idx]
        _check_tensor("core", core, (8, 1024), torch.bfloat16, self.device)
        _check_tensor("stream", stream, (8, 8192), torch.float32, self.device)
        args = dict(
            binding.parameters,
            core=core,
            stream_a=stream,
            stream_out=self.stream_out,
            branch_out=self.branch_out,
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
        binding.launch(*ep8_k2.ABI.pack(args), stream=torch.cuda.current_stream())
        return self.stream_out, self.branch_out

    def refresh(self):
        """Verify graph-bound addresses after in-place weight/cache refresh.

        Rebinding a parameter is rejected before another request can execute;
        callers must rebuild the model and captured graphs for changed storage.
        Health and epochs are deliberately never reset on reload.
        """

        def verify():
            for binding in self.layers.values():
                current = _layer_parameters(binding.layer, self.rank, self.device)
                if (
                    _layer_key(binding.layer) != binding.key
                    or {k: _signature(v) for k, v in current.items()}
                    != binding.signatures
                ):
                    raise RuntimeError(
                        "EP8 full K2 reload changed captured storage or configuration"
                    )

        bind_agreed(verify, self.group.cpu_group)

        def refresh_router_weights():
            # Every supported weight reload invokes post_load_weights. Refresh
            # even if a loader bypassed Tensor's version counter; graph-bound
            # caches retain their addresses. Pointer changes were collectively
            # rejected before any rank mutates its cached router.
            for binding in self.layers.values():
                source = binding.layer.mlp.router.weight
                binding.parameters["w_router"].copy_(source.t())
                binding.router_source_version = _tensor_version(source)
            torch.cuda.synchronize(self.device)

        bind_agreed(refresh_router_weights, self.group.cpu_group)

    def close(self):
        """Release peer storage after all ranks' GPU work is quiescent."""
        if self.peers is None:
            return
        # All ranks must have completed device work before any peer allocation
        # is freed. The graceful scheduler hook runs before process-group teardown.
        failure = None
        try:
            torch.cuda.synchronize(self.device)
        except Exception as error:
            failure = error
        # A failed or missing rank must not leave the other shutdown paths in an
        # unbounded wait. No peer storage is freed when quiescence is unproven.
        dist.monitored_barrier(
            group=self.group.cpu_group,
            timeout=timedelta(seconds=30),
            wait_all_ranks=True,
        )
        if not tp_agree(failure is None, self.group.cpu_group):
            raise RuntimeError(
                "EP8 full K2 shutdown could not quiesce every rank"
            ) from failure
        self.peers.close()
        self.peers = None
