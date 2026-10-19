# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Executable Gate-0 proof for the Muse layer MonoKernel.

The device kernels use raw addresses and persistent CTA-to-CTA waits.  These
checks deliberately duplicate the *contract* (region extents, task sets and
producer/consumer order), not device implementation code, so a changed layout
or specialization must update and re-prove both sides before it can launch.
"""

from collections.abc import Iterable
from itertools import pairwise

from sglang.srt.rocm_muse_spark.mono.ops import k1, k2
from sglang.srt.rocm_muse_spark.mono.ops.common.plan import (
    BLOCKS,
    PAIR_BYTES,
    Record,
    Space,
)
from sglang.srt.rocm_muse_spark.mono.ops.stages import attn, moe, oproj, seam


def _assert_layout(layout: dict[str, tuple[int, int]], total: int) -> None:
    spans = []
    for name, (offset, size) in layout.items():
        assert offset >= 0 and size > 0, (name, offset, size)
        assert offset + size <= total, (name, offset, size, total)
        spans.append((offset, offset + size, name))
    spans.sort()
    for (_, end, left), (start, _, right) in pairwise(spans):
        assert end <= start, (left, right, end, start)


def _assert_pairs(layout: dict[str, tuple[int, int]], expected: dict[str, int]) -> None:
    for name, pairs in expected.items():
        assert layout[name][1] == pairs * PAIR_BYTES, (
            name,
            layout[name][1],
            pairs,
        )


def _owners(count: int, base: int = 0, *, reverse: bool = False) -> list[int]:
    """Persistent ``first_task`` ownership, enumerated rather than assumed."""
    tasks = []
    for bid in range(BLOCKS):
        first = BLOCKS - 1 - bid if reverse else (bid - base) % BLOCKS
        tasks.extend(range(first, count, BLOCKS))
    return tasks


def _assert_owned(count: int, base: int = 0, *, reverse: bool = False) -> None:
    got = _owners(count, base, reverse=reverse)
    assert len(got) == len(set(got)) == count
    assert sorted(got) == list(range(count))


def _assert_dag(nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> None:
    left = set(nodes)
    incoming = {n: set() for n in left}
    outgoing = {n: set() for n in left}
    for producer, consumer in edges:
        assert producer in left and consumer in left, (producer, consumer)
        incoming[consumer].add(producer)
        outgoing[producer].add(consumer)
    ready = [n for n in left if not incoming[n]]
    seen = []
    while ready:
        n = ready.pop()
        seen.append(n)
        for consumer in outgoing[n]:
            incoming[consumer].remove(n)
            if not incoming[consumer]:
                ready.append(consumer)
    assert len(seen) == len(left), {n: v for n, v in incoming.items() if v}


def _assert_trace(
    trace: Record,
    stages: tuple[str, ...],
    scratch: dict[str, tuple[int, int]],
    peer: dict[str, tuple[int, int]],
    allowed: set[tuple[str, str, Space, str]],
) -> bool:
    """Match a cold JIT trace to the validated memory/synchronization plan.

    FlyDSL does not execute the Python kernel body on a disk-cache hit, so an
    empty record is reported to the caller as ``False``.  Static validation is
    still run before this function, while release validation forces one cold
    compile for each source digest.  A non-empty (cold) trace is checked
    completely, including every address's concrete offset and extent.
    """
    assert not trace.unnamed, trace.unnamed
    if not trace.stages and not trace.accesses:
        return False
    assert tuple(trace.stages) == stages, trace.stages
    order = {stage: i for i, stage in enumerate(stages)}
    previous = -1
    seen = set()
    layouts = {Space.SCRATCH: scratch, Space.PEER: peer}
    for access in trace.accesses:
        signature = (access.stage, access.region, access.space, access.kind)
        assert signature in allowed, access
        assert order[access.stage] >= previous, access
        previous = order[access.stage]
        expected = layouts[access.space][access.region]
        assert (access.offset, access.extent) == expected, (access, expected)
        seen.add(signature)
    assert seen == allowed, (allowed - seen, seen - allowed)
    return True


def validate_k1(key: k1.K1Build) -> dict[str, int]:
    """Prove one width/query/layer-kind K1 packed layout and task plan."""
    k1._check_key(key)
    s, q, units = key.tokens, key.q_len, key.units
    layout = k1.scratch_layout(key)
    assert set(layout) == {*k1.REGIONS, "x"}
    _assert_layout(layout, k1.scratch_bytes(key))
    _assert_pairs(
        layout,
        {
            "p1": seam.NS * s,
            "p2": seam.NS * s,
            "xrdy": seam.NS,
            "qkv": s * k1.QKV_ROWS,
            "apart": units * attn.GQA * q * attn.PART,
        },
    )
    assert layout["x"][1] == s * k1.HIDDEN * 2

    # Every direct or first_task-mapped task has exactly one resident owner.
    _assert_owned(k1.GEMV_TASKS, k1.GEMV0)
    _assert_owned(seam.NS, k1.SEAM0)
    _assert_owned(units, k1.ATT0)
    _assert_owned(s * attn.NQ, k1.COMB0)
    assert k1.SEAM0 + seam.NS <= BLOCKS and units <= BLOCKS

    # The compiled capacity covers every tile at the declared boundary.  For a
    # sliding build the 2K window is the actual maximum even at long positions.
    context = key.window or key.max_len
    tiles = (context + attn.TILE - 1) // attn.TILE
    assert key.chunks * attn.per_max(key.max_len, key.chunks, key.q_len) >= tiles

    nodes = (
        "seam_p1_put",
        "seam_p1_sum",
        "seam_p2_put",
        "seam_p2_sum",
        "x_publish",
        "qkv_publish",
        "kv_insert_attention",
        "partial_publish",
        "combine",
    )
    _assert_dag(
        nodes,
        (
            ("seam_p1_put", "seam_p1_sum"),
            ("seam_p1_sum", "seam_p2_put"),
            ("seam_p2_put", "seam_p2_sum"),
            ("seam_p2_sum", "x_publish"),
            ("x_publish", "qkv_publish"),
            ("qkv_publish", "kv_insert_attention"),
            ("kv_insert_attention", "partial_publish"),
            ("qkv_publish", "combine"),
            ("partial_publish", "combine"),
        ),
    )
    return {"scratch_bytes": k1.scratch_bytes(key), "units": units}


def validate_k2(key: k2.K2Build) -> dict[str, int]:
    """Prove one K2 width's scratch/peer layouts and persistent task plan."""
    s = key.tokens
    assert 1 <= s <= k2.MAX_TOKENS
    layout = k2.scratch_layout(s)
    expected_names = {
        "msq1",
        "msq2",
        "xrdy",
        *moe.scratch_pairs(s),
        *moe.scratch_plain(s),
        "xrow",
        "finish",
    }
    assert set(layout) == expected_names
    _assert_layout(layout, k2.scratch_bytes(key))
    _assert_pairs(
        layout,
        {
            "msq1": k2.NSL * s,
            "msq2": k2.NSL * s,
            "xrdy": k2.NSL,
            **moe.scratch_pairs(s),
            "finish": BLOCKS + k2.FINISH_GROUPS + k2.FINISH_ROOTS,
        },
    )
    for name, size in moe.scratch_plain(s).items():
        assert layout[name][1] == size
    assert layout["xrow"][1] == s * k2.seam.H * 2
    _assert_layout(
        {"done": (k2.K2_DONE_OFFSET, k2.K2_DONE_BYTES)},
        256,
    )

    peer = k2.peer_layout()
    assert set(peer) == {"ar", *moe.peer_pairs(k2.MAX_TOKENS)}
    _assert_layout(peer, k2._half())
    _assert_pairs(
        peer,
        {
            "ar": oproj.ar_pairs(k2.MAX_TOKENS),
            **moe.peer_pairs(k2.MAX_TOKENS),
        },
    )
    assert k2.peer_bytes() == 2 * k2._half()

    _assert_owned(oproj.OTASKS)
    _assert_owned(moe.R_TASKS, k2.R0)
    _assert_owned(moe.P_TASKS, k2.P0)
    _assert_owned(k2.NSL)
    _assert_owned(s)
    _assert_owned(s, moe.XQ0)
    max_unique = min(moe.MAX_U, s * moe.TOPK)
    ug_cases = (
        (
            (1, min(max_unique, moe.UG_Q8_ONE_GROUP_MAX_U)),
            (moe.UG_UNITS, max_unique),
        )
        if s == 8
        else ((moe.ug_units(s), max_unique),)
    )
    for units, route_bound in ug_cases:
        ug_tasks = route_bound * moe.UG_GROUPS // units
        _assert_owned(ug_tasks, reverse=True)
        assert ug_tasks <= moe.ug_task_capacity(s) <= moe.UG_TASKS_MAX
    _assert_owned(moe.DOWN_TASKS)
    if s == 8:
        # Every global slice has exactly one owner rank; every down tile targets
        # the same rank as the finalizer of its containing 128-column slice.
        slices = [
            rank * moe.OWNED_FIN_TASKS + local
            for rank in range(k2.TP)
            for local in range(moe.OWNED_FIN_TASKS)
        ]
        assert sorted(slices) == list(range(moe.FIN_TASKS))
        assert len(set(slices)) == moe.FIN_TASKS
        owner_work = [
            (rank * moe.OWNED_FIN_TASKS + local // s, local % s)
            for rank in range(k2.TP)
            for local in range(moe.OWNER_FIN_CTAS)
        ]
        expected_work = [
            (j, t) for j in range(moe.FIN_TASKS) for t in range(s)
        ]
        assert sorted(owner_work) == expected_work
        assert len(set(owner_work)) == moe.FIN_TASKS * s
        for tile in range(moe.DOWN_TASKS):
            assert (
                tile // (moe.DOWN_TASKS // k2.TP)
                == (tile * moe.ROWS // moe.FIN_COLS) // moe.OWNED_FIN_TASKS
            )
        _assert_owned(moe.OWNER_FIN_CTAS, k2.FIN0)
    _assert_owned(moe.FIN_TASKS, k2.FIN0)  # local FIN gather/materialization
    _assert_owned(moe.UP_TASKS, k2.U0)
    _assert_owned(moe.GATHER_TASKS, k2.G0)
    assert moe.MAX_U * moe.UG_GROUPS // moe.UG_UNITS == moe.UG_TASKS_MAX

    # Global W13 descriptors are byte bounded.  The final vector/dword of the
    # final expert reaches, but never crosses, its layout-derived extent.
    assert moe.W13_BYTES == moe.E * (2 * moe.RI) * moe.W13_K // 2
    last_w13_dword_end = (
        (moe.E - 1) * (2 * moe.RI * moe.W13_K // 8)
        + ((2 * moe.RI) // moe.ROWS - 1) * (2 * moe.W13_K)
        + (moe.W13_STEPS - 1) * 256
        + 63 * 4
        + 4
    )
    assert last_w13_dword_end * 4 == moe.W13_BYTES
    assert moe.W13_SCALE_BYTES == moe.E * (2 * moe.RI) * moe.W13_SCALE_G
    last_scale_dword = moe.scale_dword(
        moe.E * (2 * moe.RI) - 1, moe.W13_SCALE_G - 1, moe.W13_SCALE_G
    )
    assert (last_scale_dword + 1) * 4 == moe.W13_SCALE_BYTES
    assert max(moe.W13_BYTES, moe.W13_SCALE_BYTES) <= 0x7FFFFFFF

    nodes = (
        "oproj_ar_publish",
        "seam_msq1_put",
        "seam_msq1_sum",
        "seam_msq2_put",
        "seam_msq2_sum",
        "ffn_publish",
        "router_publish",
        "pool_publish",
        "route_publish",
        "xq_publish",
        "route_table",
        "ug_publish",
        "down_publish",
        "fin_msq_put",
        "fin_publish",
        *(("fin_receive",) if s == 8 else ()),
        "branch_publish",
        "gather",
    )
    _assert_dag(
        nodes,
        (
            ("oproj_ar_publish", "seam_msq1_put"),
            ("seam_msq1_put", "seam_msq1_sum"),
            ("seam_msq1_sum", "seam_msq2_put"),
            ("seam_msq2_put", "seam_msq2_sum"),
            ("seam_msq2_sum", "ffn_publish"),
            ("ffn_publish", "router_publish"),
            ("ffn_publish", "pool_publish"),
            ("router_publish", "route_publish"),
            ("pool_publish", "xq_publish"),
            ("route_publish", "route_table"),
            ("route_table", "ug_publish"),
            ("xq_publish", "ug_publish"),
            ("ug_publish", "down_publish"),
            ("down_publish", "fin_msq_put"),
            ("fin_msq_put", "fin_publish"),
            *(
                (
                    ("fin_publish", "fin_receive"),
                    ("fin_receive", "branch_publish"),
                )
                if s == 8
                else (("fin_publish", "branch_publish"),)
            ),
            ("branch_publish", "gather"),
        ),
    )
    return {
        "scratch_bytes": k2.scratch_bytes(key),
        "peer_bytes": k2.peer_bytes(),
    }


def validate_k1_trace(key: k1.K1Build, trace: Record) -> bool:
    """Validate every named mailbox access emitted by a K1 cold compile."""
    validate_k1(key)
    scratch = k1.scratch_layout(key)
    return _assert_trace(
        trace,
        ("qkv", "seam", "attention", "combine"),
        scratch,
        {},
        {
            ("qkv", "xrdy", Space.SCRATCH, "poll"),
            ("qkv", "qkv", Space.SCRATCH, "put"),
            ("seam", "p1", Space.SCRATCH, "put"),
            ("seam", "p1", Space.SCRATCH, "poll"),
            ("seam", "p2", Space.SCRATCH, "put"),
            ("seam", "p2", Space.SCRATCH, "poll"),
            ("seam", "xrdy", Space.SCRATCH, "put"),
            ("attention", "qkv", Space.SCRATCH, "poll"),
            ("attention", "apart", Space.SCRATCH, "put"),
            ("combine", "qkv", Space.SCRATCH, "poll"),
            ("combine", "apart", Space.SCRATCH, "poll"),
        },
    )


def validate_k2_trace(key: k2.K2Build, trace: Record) -> bool:
    """Validate every named mailbox access emitted by a K2 cold compile."""
    validate_k2(key)
    return _assert_trace(
        trace,
        (
            "oproj",
            "router",
            "pool",
            "seam",
            "route",
            "xq",
            "xq_wait",
            "route_wait",
            "ug",
            "inter_wait",
            "down",
            "finalize",
            *(("fin_gather",) if key.tokens == 8 else ()),
            "up",
            "gather",
            "finish",
        ),
        k2.scratch_layout(key.tokens),
        {
            **k2.peer_layout(),
            "done": (k2.K2_DONE_OFFSET, k2.K2_DONE_BYTES),
        },
        {
            ("oproj", "ar", Space.PEER, "put"),
            ("router", "xrdy", Space.SCRATCH, "poll"),
            ("router", "rpart", Space.SCRATCH, "put"),
            ("router", "rpart", Space.SCRATCH, "poll"),
            ("router", "rlog", Space.PEER, "put"),
            ("pool", "xrdy", Space.SCRATCH, "poll"),
            ("pool", "ppart", Space.SCRATCH, "put"),
            ("pool", "ppart", Space.SCRATCH, "poll"),
            ("pool", "pool", Space.PEER, "put"),
            ("seam", "ar", Space.PEER, "poll"),
            ("seam", "msq1", Space.SCRATCH, "put"),
            ("seam", "msq1", Space.SCRATCH, "poll"),
            ("seam", "msq2", Space.SCRATCH, "put"),
            ("seam", "msq2", Space.SCRATCH, "poll"),
            ("seam", "xrdy", Space.SCRATCH, "put"),
            ("route", "rlog", Space.PEER, "poll"),
            ("route", "route", Space.SCRATCH, "put"),
            ("xq", "pool", Space.PEER, "poll"),
            ("xq", "xqrdy", Space.SCRATCH, "put"),
            ("xq_wait", "xqrdy", Space.SCRATCH, "poll"),
            ("route_wait", "route", Space.SCRATCH, "poll"),
            ("ug", "ugrdy", Space.SCRATCH, "put"),
            ("inter_wait", "ugrdy", Space.SCRATCH, "poll"),
            ("down", "yr", Space.PEER, "put"),
            ("finalize", "yr", Space.PEER, "poll"),
            *(
                {
                    ("finalize", "owned_fmsq", Space.PEER, "put"),
                    ("finalize", "owned_fmsq", Space.PEER, "poll"),
                    ("finalize", "owned_fin", Space.PEER, "put"),
                    ("fin_gather", "owned_fin", Space.PEER, "poll"),
                    ("fin_gather", "frdy", Space.SCRATCH, "put"),
                }
                if key.tokens == 8
                else {
                    ("finalize", "fmsq", Space.SCRATCH, "put"),
                    ("finalize", "fmsq", Space.SCRATCH, "poll"),
                    ("finalize", "frdy", Space.SCRATCH, "put"),
                }
            ),
            ("up", "frdy", Space.SCRATCH, "poll"),
            ("up", "br", Space.PEER, "put"),
            ("gather", "br", Space.PEER, "poll"),
            ("finish", "finish", Space.SCRATCH, "put"),
            ("finish", "finish", Space.SCRATCH, "poll"),
            ("finish", "done", Space.PEER, "put"),
            ("finish", "done", Space.PEER, "poll"),
        },
    )


def validate_all() -> dict[str, int]:
    """Gate 0 matrix: rows 1..8, every uniform q_len and both K1 kinds."""
    builds = 0
    max_scratch = 0
    for s in range(1, k1.MAX_TOKENS + 1):
        validate_k2(k2.K2Build(tokens=s))
        builds += 1
        for q in range(1, s + 1):
            if s % q:
                continue
            for window, rope in ((k1.WINDOW, True), (0, False)):
                result = validate_k1(
                    k1.K1Build(
                        tokens=s,
                        q_len=q,
                        window=window,
                        rope=rope,
                        scale=1.0,
                    )
                )
                max_scratch = max(max_scratch, result["scratch_bytes"])
                builds += 1
    assert max_scratch <= k1.max_scratch_bytes(k1.MAX_TOKENS)
    assert k2.max_scratch_bytes() >= max(
        k2.scratch_bytes(k2.K2Build(tokens=s)) for s in range(1, k2.MAX_TOKENS + 1)
    )
    return {"builds": builds, "max_k1_scratch_bytes": max_scratch}
