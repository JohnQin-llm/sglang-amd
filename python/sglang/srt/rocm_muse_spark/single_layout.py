"""Host lifecycle and byte addressing for the opt-in official1020 Muse layout.

Derived from AITER ops/shuffle.py (SHA256
38f41cd233c529cba6d18f0654c267c3279df75d50295263ffa3f0df8f427b0b).
All four buffers are one-byte elements. Gate/up uses shuffle_weight(16,16)
and e8m0_shuffle; down uses shuffle_{weight,scale}_a16w4(..., False).
Despite different reshape axes these give the same flattened byte mappings.
No gate/up interleave or padding applies to the supported Muse dimensions.
"""

NAMES = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
ERROR = (
    "Muse single-layout weights cannot be updated online or exported as a native "
    "checkpoint. Use quiescent model.load_weights for reload, or restart with "
    "SGLANG_MUSE_MOE_SINGLE_LAYOUT=0 for export/online updates."
)


def weight_offset(n, k_byte, k_bytes=2048):
    """Native (N, packed K byte) -> shuffled byte, relative to one expert.

    A native aligned 16-byte fragment remains contiguous and aligned. A 4-byte
    fragment within it does too. This does NOT hold for scale dwords.
    """
    return n // 16 * (k_bytes * 16) + k_byte // 16 * 256 + n % 16 * 16 + k_byte % 16


def scale_offset(n, k_scale, k_scales=128):
    return (
        n // 32 * (k_scales * 32)
        + k_scale // 8 * 256
        + k_scale % 4 * 64
        + n % 16 * 4
        + k_scale % 8 // 4 * 2
        + n % 32 // 16
    )


def initialize(experts, enabled):
    experts._single_layout_mode = bool(enabled)
    experts._single_layout_state = "native"
    experts._single_layout_validated = False
    if enabled:
        experts._derived_weight_cache_error = ERROR
        experts._weight_export_error = ERROR


def check_options(experts, *, enabled, a4w4, backend, compact_unchanged=True):
    """Host-only fail-closed checks; safe during graph capture (no device reads)."""
    mode = getattr(experts, "_single_layout_mode", False)
    if mode != enabled or (
        mode and (not a4w4 or backend != "flydsl" or not compact_unchanged)
    ):
        raise RuntimeError(
            "Muse single-layout options changed or are unsupported; restart the model"
        )
    if mode and experts._single_layout_state == "failed":
        raise RuntimeError(
            "Interrupted Muse single-layout load/transform; discard this model and reload"
        )
    return mode


def require_ready(experts):
    if (
        getattr(experts, "_single_layout_mode", False)
        and experts._single_layout_state != "shuffled"
    ):
        raise RuntimeError("Muse single-layout weights are not ready for inference")


def validate_scope(experts, *, tp, ep, moe_tp, arch):
    """Validate all layers before the first parameter is changed."""
    experts._single_layout_validated = False
    e = experts.num_local_experts
    if (tp, ep, e) not in ((4, 4, 64), (8, 8, 32)) or moe_tp != 1 or arch != "gfx950":
        raise ValueError(
            "Muse single-layout requires gfx950 TP4EP4/E64 or TP8EP8/E32, moeTP1"
        )
    if experts.expert_hidden_size != 4096 or experts.intermediate_size != 4096:
        raise ValueError("Muse single-layout requires H=I=4096")
    for name, shape in zip(
        NAMES, ((e, 8192, 2048), (e, 4096, 2048), (e, 8192, 128), (e, 4096, 128))
    ):
        t = getattr(experts, name)
        # Compare dtype name to avoid importing torch in the pure host contract.
        if (
            tuple(t.shape) != shape
            or str(t.dtype) != "torch.uint8"
            or not t.is_contiguous()
        ):
            raise ValueError(f"Unsupported Muse single-layout buffer: {name}")
        if t.device.type != "cuda" or t.element_size() != 1:
            raise ValueError(
                f"Muse single-layout requires one-byte GPU storage: {name}"
            )

    experts._single_layout_validated = True


def _permuted(t, scale, inverse):
    """One tensor temporary, no model-sized native/shuffled cache.

    Down's [N/16, K/64, 4, 16, 16] and up's [N/16, K/32, 2,
    16,16] coalesce to the same [N/16, K/16, 16,16]. Scales likewise
    coalesce across the expert axis, whose size is a multiple of 256 rows.
    """
    e, n, k = t.shape
    if scale:
        if inverse:
            return (
                t.view(e, n // 32, k // 8, 4, 16, 2, 2)
                .permute(0, 1, 6, 4, 2, 5, 3)
                .contiguous()
                .view(t.shape)
            )
        return (
            t.view(e, n // 32, 2, 16, k // 8, 2, 4)
            .permute(0, 1, 4, 6, 3, 5, 2)
            .contiguous()
            .view(t.shape)
        )
    if inverse:
        return (
            t.view(e, n // 16, k // 16, 16, 16)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(t.shape)
        )
    return (
        t.view(e, n // 16, 16, k // 16, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(t.shape)
    )


def transform(experts, *, inverse=False):
    """Caller must hold quiescence and torch.no_grad(); preserve Parameter addresses.

    Any interrupted copy is terminal: neither native nor shuffled interpretation
    is safe, and a partial retry cannot repair an unknown partially copied tensor.
    """
    if not getattr(experts, "_single_layout_validated", False):
        raise RuntimeError(
            "Muse single-layout scope must be validated before transformation"
        )
    state = experts._single_layout_state
    wanted, result = ("shuffled", "loading") if inverse else ("native", "shuffled")
    if not inverse and state == "shuffled":
        return
    if state != wanted:
        raise RuntimeError(f"Cannot transform Muse single-layout weights from {state}")
    experts.a4w4_weights = None
    experts._single_layout_state = "transforming"
    try:
        for name in NAMES:
            t = getattr(experts, name)
            temporary = _permuted(t, name.endswith("scale"), inverse)
            t.copy_(temporary)
            del temporary
        experts._single_layout_state = result
    except BaseException:
        experts._single_layout_state = "failed"
        raise


def begin_load(experts):
    if experts._single_layout_state == "shuffled":
        transform(experts, inverse=True)
    elif experts._single_layout_state == "native":
        experts._single_layout_state = "loading"
    else:
        raise RuntimeError(
            "Muse single-layout reload requires an intact quiescent model"
        )


def finish_load(experts):
    if experts._single_layout_state != "loading":
        raise RuntimeError("Muse single-layout load was interrupted")
    experts._single_layout_state = "native"


class SingleLayoutStateDictGuard:
    """Reject native-format serialization and load_state_dict before tensor writes."""

    def _save_to_state_dict(self, *args, **kwargs):
        if getattr(self, "_single_layout_mode", False):
            raise RuntimeError(ERROR)
        return super()._save_to_state_dict(*args, **kwargs)

    def _load_from_state_dict(self, *args, **kwargs):
        if getattr(self, "_single_layout_mode", False):
            raise RuntimeError(ERROR)
        return super()._load_from_state_dict(*args, **kwargs)
