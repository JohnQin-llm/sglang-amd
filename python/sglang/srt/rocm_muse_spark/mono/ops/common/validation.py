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
    WAVES,
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


def _owners(
    count: int, base: int = 0, *, reverse: bool = False, blocks: int = BLOCKS
) -> list[int]:
    """Persistent ``first_task`` ownership, enumerated rather than assumed."""
    tasks = []
    for bid in range(blocks):
        first = blocks - 1 - bid if reverse else (bid - base) % blocks
        tasks.extend(range(first, count, blocks))
    return tasks


def _assert_owned(
    count: int, base: int = 0, *, reverse: bool = False, blocks: int = BLOCKS
) -> None:
    got = _owners(count, base, reverse=reverse, blocks=blocks)
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
    quarter = key.experimental_two_cta and s == 8
    assert 1 <= s <= k2.MAX_TOKENS
    blocks = k2.grid_blocks(key)
    finish_groups, finish_roots = k2.finish_counts(blocks)
    layout = k2.scratch_layout(s, experimental_two_cta=key.experimental_two_cta)
    moe_pairs = moe.scratch_pairs(s, wave_splitk=key.experimental_two_cta)
    expected_names = {
        "msq1",
        "msq2",
        "xrdy",
        *moe_pairs,
        *moe.scratch_plain(s),
        "xrow",
        "finish",
        *(("rstd1", "rstd2") if quarter else ()),
        *(("expert_tail_done", "up_partial", "up_partial_ready") if quarter else ()),
    }
    assert set(layout) == expected_names
    _assert_layout(layout, k2.scratch_bytes(key))
    _assert_pairs(
        layout,
        {
            "msq1": k2.NSL * s,
            "msq2": k2.NSL * s,
            "xrdy": k2.NSL,
            **moe_pairs,
            "finish": blocks + finish_groups + finish_roots,
            **({"rstd1": 8, "rstd2": 8} if quarter else {}),
            **(
                {"expert_tail_done": k2.EXPERT_TAIL_PAIRS, "up_partial_ready": 256}
                if quarter
                else {}
            ),
        },
    )
    if quarter:
        # Freeze the pre-handoff experimental tail: the new mailbox must be an
        # append-only allocation and may not move either quarter-seam region.
        assert layout["rstd1"] == (416128, 64)
        assert layout["rstd2"] == (416192, 64)
        rstd2_end = layout["rstd2"][0] + layout["rstd2"][1]
        assert layout["expert_tail_done"][0] == (rstd2_end + 15) // 16 * 16
        assert layout["expert_tail_done"][1] == 512
        assert layout["expert_tail_done"] == (416256, 512)
        assert layout["up_partial"] == (416768, 262144)
        assert layout["up_partial_ready"] == (678912, 2048)
        assert k2.scratch_bytes(key) == 680960
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

    # The candidate restores the dense kernel's original 256 producers: each
    # producer owns two of the 512 o_proj tasks even though the grid has 512
    # CTAs.  The default build has the same producer/task geometry.
    _assert_owned(oproj.OTASKS)
    if key.experimental_two_cta:
        assert blocks == 512
        assert list(range(BLOCKS)) == [bid for bid in range(blocks) if bid < BLOCKS]
    _assert_owned(moe.R_TASKS, k2.R0)
    _assert_owned(moe.P_TASKS, k2.P0)
    _assert_owned(32 if quarter else k2.NSL)
    _assert_owned(s)
    _assert_owned(s, moe.XQ0)
    max_unique = min(moe.MAX_U, s * moe.TOPK)
    if not key.experimental_two_cta and s == 8:
        ug_cases = (
            (1, min(max_unique, moe.UG_Q8_ONE_GROUP_MAX_U)),
            (moe.UG_UNITS, max_unique),
        )
    elif not key.experimental_two_cta:
        ug_cases = ((moe.ug_units(s), max_unique),)
    ug_capacity = moe.ug_task_capacity(s, wave_splitk=key.experimental_two_cta)
    ug_capacity_limit = (
        moe.UG_TASKS_WAVE_SPLITK if key.experimental_two_cta else moe.UG_TASKS_MAX
    )
    if key.experimental_two_cta:
        # team = bid // 16, member = bid % 16.  A team advances by 32
        # experts; each member publishes exactly one of the expert's 16 flags.
        ug_work = [
            (slot, member, slot * moe.UG_GROUPS + member)
            for team in range(32)
            for member in range(16)
            for slot in range(team, max_unique, 32)
        ]
        expected_ug = [
            (slot, member, slot * moe.UG_GROUPS + member)
            for slot in range(max_unique)
            for member in range(16)
        ]
        assert sorted(ug_work) == expected_ug
        assert len({task for _, _, task in ug_work}) == max_unique * moe.UG_GROUPS
        assert max_unique * moe.UG_GROUPS <= ug_capacity <= ug_capacity_limit
        assert ug_capacity == moe.UG_TASKS_WAVE_SPLITK == 1024
        # Every member publishes its own task before entering load_expert_inter;
        # each member's wait set is exactly the team's 16 publications.
        for slot in range(max_unique):
            publications = {slot * moe.UG_GROUPS + member for member in range(16)}
            assert publications == set(
                range(slot * moe.UG_GROUPS, (slot + 1) * moe.UG_GROUPS)
            )
    else:
        for units, route_bound in ug_cases:
            ug_tasks = route_bound * moe.UG_GROUPS // units
            _assert_owned(ug_tasks, reverse=True, blocks=blocks)
            assert ug_tasks <= ug_capacity <= ug_capacity_limit
    if key.experimental_two_cta:
        down_work = [
            (slot, member * 16 + wave + 8 * iteration, member // 2)
            for team in range(32)
            for member in range(16)
            for slot in range(team, max_unique, 32)
            for iteration in range(2)
            for wave in range(WAVES)
        ]
        expected_down = [
            (slot, tile, tile // (moe.DOWN_TASKS // k2.TP))
            for slot in range(max_unique)
            for tile in range(moe.DOWN_TASKS)
        ]
        assert sorted(down_work) == expected_down
        assert len(set(down_work)) == max_unique * moe.DOWN_TASKS
    else:
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
        expected_work = [(j, t) for j in range(moe.FIN_TASKS) for t in range(s)]
        assert sorted(owner_work) == expected_work
        assert len(set(owner_work)) == moe.FIN_TASKS * s
        for tile in range(moe.DOWN_TASKS):
            assert (
                tile // (moe.DOWN_TASKS // k2.TP)
                == (tile * moe.ROWS // moe.FIN_COLS) // moe.OWNED_FIN_TASKS
            )
        if key.experimental_two_cta:
            # The extra 32 CTAs are the candidate finalizers; CTAs 0..31 stay
            # reserved for the unchanged tagged FIN receive/materialization.
            fin_base = blocks - moe.OWNER_FIN_CTAS
            assert fin_base == 480
            assert list(range(fin_base, blocks)) == [
                fin_base + owner for owner in range(moe.OWNER_FIN_CTAS)
            ]
        else:
            _assert_owned(moe.OWNER_FIN_CTAS, k2.FIN0)
    _assert_owned(moe.FIN_TASKS, k2.FIN0)  # local FIN gather/materialization
    _assert_owned(moe.UP_TASKS, k2.U0)
    _assert_owned(moe.GATHER_TASKS, k2.G0)
    assert moe.MAX_U * moe.UG_GROUPS // moe.UG_UNITS == moe.UG_TASKS_MAX
    if key.experimental_two_cta:
        assert k2.health_words_for_key(key) == (blocks + 1) * k2.TP == 4104
        tail_producers = list(range(k2.EXPERT_TAIL0, blocks))
        assert tail_producers == list(range(448, 512))
        assert k2.EXPERT_TAIL_PAIRS == moe.UP_TASKS == 64
        assert (moe.UP_PARALLEL_SHARDS, moe.UP_PARALLEL_TASKS) == (4, 256)
        assert moe.UP_PARTIAL_BYTES == 262144
        assert moe.UP_PARTIAL_READY_PAIRS == 256
        # Enumerate independently: each output tile has all eight original
        # wave partials, and each shard publishes one unique readiness flag.
        wave_owners = set()
        partial_flags = set()
        for bid in range(256):
            shard, tile = divmod(bid, 64)
            partial_flags.add(shard * 64 + tile)
            assert 448 <= 448 + bid % 64 < 512
            for wave in range(2):
                item = (tile, 2 * shard + wave)
                assert item not in wave_owners
                wave_owners.add(item)
        assert partial_flags == set(range(256))
        assert wave_owners == {(tile, wave) for tile in range(64) for wave in range(8)}
        assert set(range(k2.U0, k2.U0 + moe.UP_TASKS)) == set(range(192, 256))
        # Compact FIN indices must cover original K once across four shards.
        mapped = []
        for shard in range(4):
            for chunk in range(8):
                mapped.append(2 * shard + chunk % 2 + 8 * (chunk // 2))
        assert sorted(mapped) == list(range(32))
        assert 64 * 8 * s * 16 * 4 == layout["up_partial"][1]

    if quarter:
        # Independently enumerate the quarter-token CTA/wave/lane mapping: each token
        # owns all 64 original partial slots and every output column once.
        slices = set()
        columns = set()
        ready = set()
        for bid in range(32):
            token, q = divmod(bid, 4)
            ready.add((token, q))
            for wave in range(8):
                for i in range(2):
                    j = q * 16 + wave * 2 + i
                    assert (token, j) not in slices
                    slices.add((token, j))
                    for lane in range(64):
                        for element in range(2):
                            item = (token, j * 128 + lane * 2 + element)
                            assert item not in columns
                            columns.add(item)
        assert len(slices) == 8 * 64
        assert len(columns) == 8 * k2.seam.H
        assert ready == {(token, q) for token in range(8) for q in range(4)}

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
    assert moe.W2_BYTES == moe.E * moe.MH * moe.RI // 2
    last_w2_dword_end = (
        (moe.E - 1) * (moe.MH * moe.RI // 8)
        + (moe.MH // moe.ROWS - 1) * (2 * moe.RI)
        + (moe.W2_STEPS - 1) * 256
        + 63 * 4
        + 4
    )
    assert last_w2_dword_end * 4 == moe.W2_BYTES
    assert moe.W2_SCALE_BYTES == moe.E * moe.MH * moe.W2_SCALE_G
    last_w2_scale_dword = moe.scale_dword(
        moe.E * moe.MH - 1, moe.W2_SCALE_G - 1, moe.W2_SCALE_G
    )
    assert (last_w2_scale_dword + 1) * 4 == moe.W2_SCALE_BYTES

    seam_nodes = (
        (
            "seam_msq1_put",
            "seam_rstd1_put",
            "seam_rstd1_load",
            "seam_msq2_put",
            "seam_rstd2_put",
            "seam_rstd2_load",
        )
        if quarter
        else (
            "seam_msq1_put",
            "seam_msq1_sum",
            "seam_msq2_put",
            "seam_msq2_sum",
        )
    )
    nodes = (
        "oproj_ar_publish",
        *seam_nodes,
        "ffn_publish",
        "router_publish",
        "pool_publish",
        "route_publish",
        "xq_publish",
        "route_table",
        "ug_publish",
        "down_publish",
        *(("late_expert_ready",) if quarter else ()),
        "fin_msq_put",
        "fin_publish",
        *(("fin_receive",) if s == 8 else ()),
        "up_weight_load",
        "up_compute",
        *(("up_partial_publish", "up_combine") if quarter else ()),
        "branch_publish",
        "gather",
    )
    seam_edges = (
        (
            ("oproj_ar_publish", "seam_msq1_put"),
            ("seam_msq1_put", "seam_rstd1_put"),
            ("seam_rstd1_put", "seam_rstd1_load"),
            ("seam_rstd1_load", "seam_msq2_put"),
            ("seam_msq2_put", "seam_rstd2_put"),
            ("seam_rstd2_put", "seam_rstd2_load"),
            ("seam_rstd2_load", "ffn_publish"),
        )
        if quarter
        else (
            ("oproj_ar_publish", "seam_msq1_put"),
            ("seam_msq1_put", "seam_msq1_sum"),
            ("seam_msq1_sum", "seam_msq2_put"),
            ("seam_msq2_put", "seam_msq2_sum"),
            ("seam_msq2_sum", "ffn_publish"),
        )
    )
    _assert_dag(
        nodes,
        (
            *seam_edges,
            ("ffn_publish", "router_publish"),
            ("ffn_publish", "pool_publish"),
            ("router_publish", "route_publish"),
            ("pool_publish", "xq_publish"),
            ("route_publish", "route_table"),
            ("route_table", "ug_publish"),
            ("xq_publish", "ug_publish"),
            ("ug_publish", "down_publish"),
            ("down_publish", "fin_msq_put"),
            *(
                (
                    ("down_publish", "late_expert_ready"),
                    ("late_expert_ready", "fin_msq_put"),
                    ("late_expert_ready", "up_weight_load"),
                )
                if quarter
                else ()
            ),
            ("fin_msq_put", "fin_publish"),
            *(
                (
                    ("fin_publish", "fin_receive"),
                    ("fin_receive", "up_compute"),
                )
                if s == 8
                else (("fin_publish", "up_compute"),)
            ),
            ("down_publish", "up_weight_load"),
            ("up_weight_load", "up_compute"),
            *(
                (
                    ("fin_receive", "up_weight_load"),
                    ("up_compute", "up_partial_publish"),
                    ("up_partial_publish", "up_combine"),
                    ("up_combine", "branch_publish"),
                )
                if quarter
                else (("up_compute", "branch_publish"),)
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
            *(("expert_ready",) if key.experimental_two_cta else ()),
            "finalize",
            *(("fin_gather",) if key.tokens == 8 else ()),
            "up",
            *(("up",) if key.experimental_two_cta else ()),
            "gather",
            "finish",
        ),
        k2.scratch_layout(key.tokens, experimental_two_cta=key.experimental_two_cta),
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
            *(
                {
                    ("seam", "rstd1", Space.SCRATCH, "put"),
                    ("seam", "rstd1", Space.SCRATCH, "poll"),
                    ("seam", "rstd2", Space.SCRATCH, "put"),
                    ("seam", "rstd2", Space.SCRATCH, "poll"),
                }
                if key.experimental_two_cta
                else set()
            ),
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
            *(
                {
                    ("expert_ready", "expert_tail_done", Space.SCRATCH, "put"),
                    ("expert_ready", "expert_tail_done", Space.SCRATCH, "poll"),
                }
                if key.experimental_two_cta
                else set()
            ),
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
            *(
                {
                    ("up", "up_partial_ready", Space.SCRATCH, "put"),
                    ("up", "up_partial_ready", Space.SCRATCH, "poll"),
                }
                if key.experimental_two_cta
                else set()
            ),
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
    validate_k2(k2.K2Build(tokens=8, experimental_two_cta=True))
    builds += 1
    assert max_scratch <= k1.max_scratch_bytes(k1.MAX_TOKENS)
    assert k2.max_scratch_bytes() >= max(
        k2.scratch_bytes(k2.K2Build(tokens=s)) for s in range(1, k2.MAX_TOKENS + 1)
    )
    candidate = k2.K2Build(tokens=8, experimental_two_cta=True)
    assert k2.max_scratch_bytes(experimental_two_cta=True) >= k2.scratch_bytes(
        candidate
    )
    return {"builds": builds, "max_k1_scratch_bytes": max_scratch}
