"""Split-KV target-verify attention for a speculative draft tree.

One program owns a block of the (node, query head) rows that share a KV head, for
one request and one key range, so each KV tile it reads serves the whole GQA
group of every tree node. Keys come from the KV pool through the request table:
the prefix (``seq_lens`` keys), then the tree nodes in node order. A node sees
prefix key p only inside the sliding window measured from its own position, and
tree node j only if bit j of its ancestor mask is set. Key ranges reduce through
an LSE merge.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_hip

_LOG2E = 1.4426950408889634

# Max tree nodes: one int64 ancestor mask per node.
MAX_TREE_NODES = 64
SUPPORTED_HEAD_DIMS = (64, 128, 256)

_AMD_LAUNCH_KWARGS = {"matrix_instr_nonkdim": 16} if is_hip() else {}


# The masked request count varies per replay; specializing on it recompiles.
@triton.jit(do_not_specialize=["num_masked_reqs"])
def _tree_ancestor_bits_kernel(
    MASK,
    OUT,
    num_masked_reqs,
    NODES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    j = tl.arange(0, BLOCK)
    self_bit = tl.full((BLOCK,), 1, tl.int64) << i.to(tl.int64)
    if req < num_masked_reqs:
        cells = tl.load(
            MASK + req * NODES * NODES + i[:, None] * NODES + j[None, :],
            mask=(i[:, None] < NODES) & (j[None, :] < NODES),
            other=0,
        )
        bits = tl.sum((cells != 0).to(tl.int64) << j[None, :].to(tl.int64), axis=1)
        bits = bits | self_bit
    else:
        bits = self_bit
    tl.store(OUT + req * NODES + i, bits, mask=i < NODES)


def tree_ancestor_bits(
    *,
    tree_mask: torch.Tensor,
    out: torch.Tensor,
    num_nodes: int,
    num_masked_reqs: int,
) -> torch.Tensor:
    """Fill ``out`` ([reqs * num_nodes] int64) with each node's visible tree nodes
    as a bitmask, read from a QLEN_ONLY tree mask ([reqs, nodes, nodes] bool).
    Requests at or past ``num_masked_reqs`` (graph padding) see only themselves."""
    num_reqs = out.numel() // num_nodes
    if num_reqs == 0:
        return out
    _tree_ancestor_bits_kernel[(num_reqs,)](
        tree_mask.view(torch.uint8),
        out,
        num_masked_reqs,
        NODES=num_nodes,
        BLOCK=max(16, triton.next_power_of_2(num_nodes)),
        num_warps=4,
    )
    return out


@triton.jit
def _gather_slots(
    req_row,
    SLOT_MAP,
    positions,
    key_ok,
    HAS_SLOT_MAP: tl.constexpr,
    MASKED: tl.constexpr,
):
    if MASKED:
        slots = tl.load(req_row + positions, mask=key_ok, other=0).to(tl.int64)
        if HAS_SLOT_MAP:
            slots = tl.load(SLOT_MAP + slots, mask=key_ok, other=0).to(tl.int64)
    else:
        slots = tl.load(req_row + positions).to(tl.int64)
        if HAS_SLOT_MAP:
            slots = tl.load(SLOT_MAP + slots).to(tl.int64)
    return slots


@triton.jit
def _attend_tile(
    q,
    maximum,
    denom,
    accum,
    K,
    V,
    slots,
    key_ok,
    visible,
    kvhead,
    d,
    k_stride_0,
    k_stride_1,
    v_stride_0,
    v_stride_1,
    qk_scale,
    SOFTCAP: tl.constexpr,
    MASKED: tl.constexpr,
):
    """One online-softmax step over a key tile. Scores are in log2 units;
    ``visible`` ([rows, keys]) is read only when MASKED."""
    k_ptrs = K + slots[None, :] * k_stride_0 + kvhead * k_stride_1 + d[:, None]
    v_ptrs = V + slots[:, None] * v_stride_0 + kvhead * v_stride_1 + d[None, :]
    if MASKED:
        k = tl.load(k_ptrs, mask=key_ok[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=key_ok[:, None], other=0.0)
    else:
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
    scores = tl.dot(q, k.to(q.dtype)) * qk_scale
    if SOFTCAP > 0:
        # SOFTCAP is the cap in log2 units: cap * tanh(s / cap) in natural units.
        scores = SOFTCAP * (2 * tl.sigmoid(2 * scores / SOFTCAP) - 1)
    if MASKED:
        scores = tl.where(visible, scores, float("-inf"))
        new_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        # A row with nothing visible so far keeps -inf; offset it by 0 instead.
        offset = tl.where(new_maximum > float("-inf"), new_maximum, 0.0)
    else:
        new_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        offset = new_maximum
    probabilities = tl.exp2(scores - offset[:, None])
    rescale = tl.exp2(maximum - offset)
    denom = denom * rescale + tl.sum(probabilities, axis=1)
    accum = tl.dot(
        probabilities.to(q.dtype), v.to(q.dtype), acc=accum * rescale[:, None]
    )
    return new_maximum, denom, accum


@triton.jit
def _prefix_tiles(prefix, WINDOW: tl.constexpr, BN: tl.constexpr):
    """First visible prefix key and the [lo, hi) key tiles over it; the root
    (position = prefix) sees the most prefix keys."""
    if WINDOW > 0:
        low = tl.maximum(prefix - WINDOW + 1, 0)
    else:
        low = 0
    return low, low // BN, tl.cdiv(prefix, BN)


@triton.jit
def _active_splits(tile_lo, tile_hi, SPLITS: tl.constexpr, MIN_TILES: tl.constexpr):
    """Key ranges a request uses: SPLITS at most, MIN_TILES tiles each at least."""
    return tl.minimum(tl.maximum(tl.cdiv(tile_hi - tile_lo, MIN_TILES), 1), SPLITS)


@triton.jit
def _finish_rows(
    maximum,
    denom,
    accum,
    head,
    SINKS,
    v_scale,
    HAS_SINKS: tl.constexpr,
):
    """Normalized rows from one (maximum, denom, accum) state, 0 where nothing is visible."""
    if HAS_SINKS:
        sink = tl.load(SINKS + head).to(tl.float32) * 1.4426950408889634
        top = tl.maximum(maximum, sink)
        offset = tl.where(top > float("-inf"), top, 0.0)
        rescale = tl.exp2(maximum - offset)
        denom = denom * rescale + tl.exp2(sink - offset)
        accum = accum * rescale[:, None]
    denom = tl.where(denom > 0, denom, 1.0)
    return accum * (v_scale / denom)[:, None]


@triton.jit
def _tree_verify_partials_kernel(
    Q,
    K,
    V,
    OUT,
    REQ_TO_TOKEN,
    REQ_POOL_INDICES,
    SEQ_LENS,
    POSITIONS,
    ANCESTORS,
    SLOT_MAP,
    K_SCALE,
    V_SCALE,
    SINKS,
    PARTIAL,
    MAXIMUM,
    DENOM,
    q_stride_0,
    q_stride_1,
    k_stride_0,
    k_stride_1,
    v_stride_0,
    v_stride_1,
    o_stride_0,
    o_stride_1,
    req_to_token_stride,
    sm_scale,
    HEADS: tl.constexpr,
    GQA: tl.constexpr,
    NODES: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WINDOW: tl.constexpr,
    SPLITS: tl.constexpr,
    MIN_TILES: tl.constexpr,
    ROW_BLOCKS: tl.constexpr,
    SOFTCAP: tl.constexpr,
    HAS_SLOT_MAP: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BNODE: tl.constexpr,
):
    kvhead = tl.program_id(0) // ROW_BLOCKS
    row_block = tl.program_id(0) % ROW_BLOCKS
    seq = tl.program_id(1) // SPLITS
    split = tl.program_id(1) % SPLITS

    prefix = tl.load(SEQ_LENS + seq).to(tl.int32)
    low, tile_lo, tile_hi = _prefix_tiles(prefix, WINDOW, BN)
    active = _active_splits(tile_lo, tile_hi, SPLITS, MIN_TILES)
    if split < active:
        req_row = (
            REQ_TO_TOKEN
            + tl.load(REQ_POOL_INDICES + seq).to(tl.int64) * req_to_token_stride
        )
        rows = row_block * BM + tl.arange(0, BM)
        row_ok = rows < NODES * GQA
        node = rows // GQA
        head = kvhead * GQA + rows % GQA
        token = seq * NODES + node
        d = tl.arange(0, HEAD_DIM)
        q = tl.load(
            Q + token[:, None] * q_stride_0 + head[:, None] * q_stride_1 + d[None, :],
            mask=row_ok[:, None],
            other=0.0,
        )
        qk_scale = sm_scale * 1.4426950408889634
        if K_SCALE is not None:
            qk_scale *= tl.load(K_SCALE).to(tl.float32)
        if WINDOW > 0:
            query_position = tl.load(POSITIONS + token, mask=row_ok, other=0).to(
                tl.int32
            )
        else:
            query_position = node  # Unread without a window.

        maximum, denom, accum = _attend_prefix(
            q, K, V, req_row, SLOT_MAP, query_position, prefix, low, tile_lo,
            tile_hi, split, active, kvhead, d, k_stride_0, k_stride_1, v_stride_0,
            v_stride_1, qk_scale, NODES, WINDOW, SOFTCAP, HAS_SLOT_MAP, BM, BN,
            HEAD_DIM,
        )  # fmt: skip
        # Tree nodes sit at [prefix, prefix + NODES), inside every node's window.
        if split == active - 1:
            nodes = tl.arange(0, BNODE)
            key_ok = nodes < NODES
            ancestors = tl.load(ANCESTORS + token, mask=row_ok, other=0).to(tl.int64)
            is_ancestor = (ancestors[:, None] >> nodes[None, :].to(tl.int64)) & 1
            visible = key_ok[None, :] & (
                (is_ancestor != 0) | (nodes[None, :] == node[:, None])
            )
            slots = _gather_slots(
                req_row, SLOT_MAP, prefix + nodes, key_ok, HAS_SLOT_MAP, True
            )
            maximum, denom, accum = _attend_tile(
                q, maximum, denom, accum, K, V, slots, key_ok, visible, kvhead, d,
                k_stride_0, k_stride_1, v_stride_0, v_stride_1, qk_scale, SOFTCAP,
                True,
            )  # fmt: skip

        if active == 1:
            v_scale = 1.0
            if V_SCALE is not None:
                v_scale = tl.load(V_SCALE).to(tl.float32)
            out = _finish_rows(maximum, denom, accum, head, SINKS, v_scale, HAS_SINKS)
            tl.store(
                OUT
                + token[:, None] * o_stride_0
                + head[:, None] * o_stride_1
                + d[None, :],
                out.to(OUT.dtype.element_ty),
                mask=row_ok[:, None],
            )
        if SPLITS > 1:
            if active > 1:
                state = (token * HEADS + head) * SPLITS + split
                tl.store(MAXIMUM + state, maximum, mask=row_ok)
                tl.store(DENOM + state, denom, mask=row_ok)
                tl.store(
                    PARTIAL + state[:, None] * HEAD_DIM + d[None, :],
                    accum,
                    mask=row_ok[:, None],
                )


@triton.jit
def _attend_prefix(
    q,
    K,
    V,
    req_row,
    SLOT_MAP,
    query_position,
    prefix,
    low,
    tile_lo,
    tile_hi,
    split,
    active,
    kvhead,
    d,
    k_stride_0,
    k_stride_1,
    v_stride_0,
    v_stride_1,
    qk_scale,
    NODES: tl.constexpr,
    WINDOW: tl.constexpr,
    SOFTCAP: tl.constexpr,
    HAS_SLOT_MAP: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """This split's share of the prefix tiles. Edge tiles (window start, prefix
    end) mask; interior tiles, which every node sees whole, do not."""
    tiles_per_split = tl.cdiv(tile_hi - tile_lo, active)
    begin = tile_lo + split * tiles_per_split
    end = tl.minimum(begin + tiles_per_split, tile_hi)
    if WINDOW > 0:
        full_low = tl.maximum(prefix + NODES - WINDOW, 0)
        full_begin = tl.minimum(tl.maximum(tl.cdiv(full_low, BN), tile_lo), tile_hi)
    else:
        full_begin = tile_lo
    full_end = tl.maximum(prefix // BN, full_begin)

    maximum = tl.full((BM,), float("-inf"), tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    accum = tl.zeros((BM, HEAD_DIM), tl.float32)
    keys = tl.arange(0, BN)
    for tile in range(begin, tl.minimum(end, full_begin)):
        positions = tile * BN + keys
        key_ok = (positions >= low) & (positions < prefix)
        visible = key_ok[None, :]
        if WINDOW > 0:
            visible = visible & (query_position[:, None] - positions[None, :] < WINDOW)
        slots = _gather_slots(req_row, SLOT_MAP, positions, key_ok, HAS_SLOT_MAP, True)
        maximum, denom, accum = _attend_tile(
            q, maximum, denom, accum, K, V, slots, key_ok, visible, kvhead, d,
            k_stride_0, k_stride_1, v_stride_0, v_stride_1, qk_scale, SOFTCAP, True,
        )  # fmt: skip
    for tile in range(tl.maximum(begin, full_begin), tl.minimum(end, full_end)):
        positions = tile * BN + keys
        slots = _gather_slots(req_row, SLOT_MAP, positions, None, HAS_SLOT_MAP, False)
        maximum, denom, accum = _attend_tile(
            q, maximum, denom, accum, K, V, slots, None, None, kvhead, d,
            k_stride_0, k_stride_1, v_stride_0, v_stride_1, qk_scale, SOFTCAP, False,
        )  # fmt: skip
    for tile in range(tl.maximum(begin, full_end), end):
        positions = tile * BN + keys
        key_ok = (positions >= low) & (positions < prefix)
        visible = key_ok[None, :]
        if WINDOW > 0:
            visible = visible & (query_position[:, None] - positions[None, :] < WINDOW)
        slots = _gather_slots(req_row, SLOT_MAP, positions, key_ok, HAS_SLOT_MAP, True)
        maximum, denom, accum = _attend_tile(
            q, maximum, denom, accum, K, V, slots, key_ok, visible, kvhead, d,
            k_stride_0, k_stride_1, v_stride_0, v_stride_1, qk_scale, SOFTCAP, True,
        )  # fmt: skip
    return maximum, denom, accum


@triton.jit
def _tree_verify_reduce_kernel(
    PARTIAL,
    MAXIMUM,
    DENOM,
    OUT,
    SEQ_LENS,
    V_SCALE,
    SINKS,
    o_stride_0,
    o_stride_1,
    HEADS: tl.constexpr,
    NODES: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WINDOW: tl.constexpr,
    SPLITS: tl.constexpr,
    MIN_TILES: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    BN: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    prefix = tl.load(SEQ_LENS + token // NODES).to(tl.int32)
    _, tile_lo, tile_hi = _prefix_tiles(prefix, WINDOW, BN)
    active = _active_splits(tile_lo, tile_hi, SPLITS, MIN_TILES)
    # One active range wrote the output itself.
    if active > 1:
        split = tl.arange(0, SPLITS)
        split_ok = split < active
        state = (token * HEADS + head) * SPLITS + split
        d = tl.arange(0, HEAD_DIM)
        maximum = tl.load(MAXIMUM + state, mask=split_ok, other=float("-inf"))
        denom = tl.load(DENOM + state, mask=split_ok, other=0.0)
        accum = tl.load(
            PARTIAL + state[:, None] * HEAD_DIM + d[None, :],
            mask=split_ok[:, None],
            other=0.0,
        )
        top = tl.max(maximum, axis=0)
        offset = tl.where(top > float("-inf"), top, 0.0)
        weight = tl.exp2(maximum - offset)
        total = tl.sum(denom * weight, axis=0)
        merged = tl.sum(accum * weight[:, None], axis=0)
        if HAS_SINKS:
            sink = tl.load(SINKS + head).to(tl.float32) * 1.4426950408889634
            top = tl.maximum(offset, sink)
            rescale = tl.exp2(offset - top)
            total = total * rescale + tl.exp2(sink - top)
            merged = merged * rescale
        v_scale = 1.0
        if V_SCALE is not None:
            v_scale = tl.load(V_SCALE).to(tl.float32)
        total = tl.where(total > 0, total, 1.0)
        tl.store(
            OUT + token * o_stride_0 + head * o_stride_1 + d,
            (merged * (v_scale / total)).to(OUT.dtype.element_ty),
        )


def tree_verify_supported(*, head_dim: int, num_nodes: int) -> bool:
    return head_dim in SUPPORTED_HEAD_DIMS and 1 <= num_nodes <= MAX_TREE_NODES


def tree_verify_config(*, head_dim: int, rows: int) -> tuple[int, int, int]:
    """(row block, key tile, warps). Measured on MI355X at the Muse-Glimmer
    (head_dim 128, GQA 16) and Qwen3.8-27B (head_dim 256, GQA 6) shapes, bs 1-16;
    head_dim 64 takes the head_dim 128 tile untuned. A 256-row block spills at
    head_dim 128, and a 64-key tile overflows LDS at head_dim 256."""
    if head_dim == 256:
        block_m, block_n, num_warps = 128, 32, 4
    else:
        block_m, block_n, num_warps = 128, 64, 8
    return min(block_m, max(16, triton.next_power_of_2(rows))), block_n, num_warps


def pick_tree_splits(*, num_programs: int) -> tuple[int, int]:
    """(key ranges per request at most, min keys per range) for ``num_programs``
    (request, KV head, row block) programs. Measured on MI355X (256 CUs) at bs
    1-16, contexts 1K-8K: about 256 programs in all, with ranges of at least 128
    keys while the batch alone fills under 8 CUs, 256 keys past that."""
    splits = 1
    while splits * 2 * num_programs <= 256 and splits < 64:
        splits *= 2
    return splits, 128 if num_programs <= 8 else 256


def tree_verify_attention(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    ancestors: torch.Tensor,
    sm_scale: float,
    window: int = 0,
    slot_map: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    logit_cap: float = 0.0,
    sinks: Optional[torch.Tensor] = None,
    splits: Optional[int] = None,
    min_split_keys: Optional[int] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: int = 2,
) -> torch.Tensor:
    """Attention for every request's tree nodes against its prefix and the tree.

    q, out: [reqs * nodes, heads, head_dim], node-major per request.
    k_cache, v_cache: [slots, kv_heads, head_dim]; the tree's K/V are already
    written, at request-table columns [seq_lens, seq_lens + nodes).
    seq_lens: [reqs] prefix lengths. positions: [reqs * nodes] each node's
    position (prefix + depth), read only when ``window`` > 0.
    ancestors: [reqs * nodes] int64, bit j set when the node sees node j.
    splits: key ranges per request at most; a request uses fewer when its prefix
    gives a range under ``min_split_keys`` keys.
    window: keys visible to a query at position p are (p - window, p]; 0 = all.
    slot_map: full-pool slot -> window-pool slot, for a sliding-window pool.
    k_scale, v_scale: [1] fp32 dequant scales of an fp8 cache.
    """
    num_reqs = seq_lens.numel()
    num_tokens, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    num_nodes = num_tokens // max(1, num_reqs)
    gqa = num_heads // num_kv_heads
    assert num_tokens == num_reqs * num_nodes and num_heads == num_kv_heads * gqa
    assert tree_verify_supported(head_dim=head_dim, num_nodes=num_nodes)
    assert k_cache.shape[-1] == v_cache.shape[-1] == head_dim
    assert k_cache.stride(-1) == v_cache.stride(-1) == q.stride(-1) == 1
    if num_tokens == 0:
        return out

    rows = num_nodes * gqa
    default_m, default_n, default_warps = tree_verify_config(
        head_dim=head_dim, rows=rows
    )
    block_m = block_m or default_m
    block_n = block_n or default_n
    num_warps = num_warps or default_warps
    row_blocks = triton.cdiv(rows, block_m)
    default_splits, default_min_keys = pick_tree_splits(
        num_programs=num_reqs * num_kv_heads * row_blocks
    )
    splits = splits or default_splits
    min_tiles = max(1, (min_split_keys or default_min_keys) // block_n)
    if splits > 1:
        state_shape = (num_tokens, num_heads, splits)
        partial = torch.empty(
            (*state_shape, head_dim), dtype=torch.float32, device=q.device
        )
        maximum = torch.empty(state_shape, dtype=torch.float32, device=q.device)
        denom = torch.empty(state_shape, dtype=torch.float32, device=q.device)
    else:
        partial = maximum = denom = None

    _tree_verify_partials_kernel[(num_kv_heads * row_blocks, num_reqs * splits)](
        q,
        k_cache,
        v_cache,
        out,
        req_to_token,
        req_pool_indices,
        seq_lens,
        positions,
        ancestors,
        slot_map,
        k_scale,
        v_scale,
        sinks,
        partial,
        maximum,
        denom,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        out.stride(0),
        out.stride(1),
        req_to_token.stride(0),
        float(sm_scale),
        HEADS=num_heads,
        GQA=gqa,
        NODES=num_nodes,
        HEAD_DIM=head_dim,
        WINDOW=int(window),
        SPLITS=splits,
        MIN_TILES=min_tiles,
        ROW_BLOCKS=row_blocks,
        SOFTCAP=float(logit_cap) * _LOG2E,
        HAS_SLOT_MAP=slot_map is not None,
        HAS_SINKS=sinks is not None,
        BM=block_m,
        BN=block_n,
        BNODE=max(16, triton.next_power_of_2(num_nodes)),
        num_warps=num_warps,
        num_stages=num_stages,
        **_AMD_LAUNCH_KWARGS,
    )
    if splits > 1:
        _tree_verify_reduce_kernel[(num_tokens, num_heads)](
            partial,
            maximum,
            denom,
            out,
            seq_lens,
            v_scale,
            sinks,
            out.stride(0),
            out.stride(1),
            HEADS=num_heads,
            NODES=num_nodes,
            HEAD_DIM=head_dim,
            WINDOW=int(window),
            SPLITS=splits,
            MIN_TILES=min_tiles,
            HAS_SINKS=sinks is not None,
            BN=block_n,
            num_warps=2,
        )
    return out
