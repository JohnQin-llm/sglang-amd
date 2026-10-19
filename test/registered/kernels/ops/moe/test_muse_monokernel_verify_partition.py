# SPDX-License-Identifier: Apache-2.0
"""Exercise K1 when a verification block crosses a KV partition."""

import pytest
import torch
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=45, stage="jit-kernel-unit", runner_config="amd")


@pytest.mark.parametrize("window", [2047, 2048])
def test_verify_partition_boundary_graph_replay(window):
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("requires MI355X")
    properties = torch.cuda.get_device_properties(0)
    if not properties.gcnArchName.startswith("gfx950"):
        pytest.skip("requires gfx950")

    from sglang.srt.rocm_muse_spark.mono.kernel import compile_checked
    from sglang.srt.rocm_muse_spark.mono.ops import k1
    from sglang.srt.rocm_muse_spark.mono.ops.common.epoch import advance_epoch_
    from sglang.srt.rocm_muse_spark.mono.ops.common.plan import HEALTH_WORDS

    key = k1.make_key(
        tokens=8,
        q_len=8,
        window=window,
        rope=True,
        scale=0.125,
        eps=1e-5,
        post_eps=1e-8,
        qk_eps=1e-5,
    )
    assert not compile_checked(key)
    device = torch.device("cuda", 0)
    stream = torch.ones(8, 8192, device=device)
    stream[-1, 0] = -1
    branch = torch.zeros_like(stream, dtype=torch.bfloat16)
    norm_weight = torch.zeros(8192, device=device, dtype=torch.bfloat16)
    alpha = torch.ones(8192, device=device)
    beta = torch.zeros_like(alpha)
    weight = torch.zeros(2304, 8192, device=device, dtype=torch.bfloat16)
    # Q/K and the output gate are zero. V reads two input columns; only the
    # last query has a negative first half. This makes dropping that token
    # observable in the output direction, even after the output RMS norm.
    for head in range(2):
        start = 1152 + 64 * head
        weight[start : start + 32, 0] = 1
        weight[start + 32 : start + 64, 1] = 1
    pages = 160
    cache = torch.zeros(pages, 2, 16, 128, device=device, dtype=torch.bfloat16)
    table = ((torch.arange(pages, device=device, dtype=torch.int32) * 7) % pages)[
        None, :
    ]
    lengths = torch.empty(1, device=device, dtype=torch.int32)
    positions = torch.empty(8, device=device, dtype=torch.int64)
    slots = torch.empty_like(positions)
    rotary = torch.zeros(pages * 16, 64, device=device)
    rotary[:, :32] = 1
    output = torch.empty(8, 1024, device=device, dtype=torch.bfloat16)
    residual = torch.empty_like(stream)
    scratch = torch.zeros(k1.scratch_bytes(key), device=device, dtype=torch.uint8)
    epoch = torch.zeros(1, device=device, dtype=torch.int32)
    health = torch.zeros(HEALTH_WORDS, device=device, dtype=torch.int32)
    alarm = torch.zeros(8, device=device, dtype=torch.int32)

    def set_prefix(prefix):
        positions.copy_(torch.arange(prefix, prefix + 8, device=device))
        slots.copy_(table[0, positions // 16].long() * 16 + positions % 16)
        lengths.fill_(prefix + 8)
        cache.zero_()
        if prefix >= 2040:
            # A strong value at key zero makes the left-window cutoff observable.
            # The final query at prefix2040 drops it only for the 2047 window.
            cache[0, :, 0, 64:96] = 64

    def launch():
        advance_epoch_(epoch)
        k1.k1_launch(
            key,
            stream_in=stream,
            branch_in=branch,
            w_pf=norm_weight,
            alpha_f=alpha,
            beta_f=beta,
            w_qn=norm_weight,
            w_qkv=weight,
            positions=positions,
            cos_sin=rotary,
            slot_mapping=slots,
            kv_cache=cache,
            block_table=table,
            seq_lens=lengths,
            stream_a=residual,
            core=output,
            scratch=scratch,
            epoch=epoch,
            layer=3,
            health=health,
            alarm=alarm.data_ptr(),
            rank=0,
        )

    set_prefix(184)
    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    values = torch.ones(8, 64, device=device)
    values[-1, :32] = -1
    # Cover an empty prefix, 64-key segments, the previous 192-key boundary,
    # and the transition to 128-key segments at total length 1025. The same
    # graph must observe the live lengths and insert every verification token.
    for prefix in (
        0,
        56,
        57,
        63,
        64,
        184,
        185,
        191,
        192,
        1016,
        1017,
        1023,
        1024,
        2040,
        2041,
        2046,
        2047,
        2048,
        2050,
    ):
        numerator = values.cumsum(0)
        if prefix >= 2040:
            includes_key_zero = torch.arange(prefix, prefix + 8, device=device) < window
            numerator[:, :32] += includes_key_zero[:, None] * 64
        count = torch.arange(prefix + 1, prefix + 9, device=device).clamp_max(window)
        expected = (numerator / count[:, None]).to(torch.bfloat16).float()
        expected = (
            0.5
            * expected
            / torch.sqrt(expected.square().mean(-1, keepdim=True) + key.qk_eps)
        ).to(torch.bfloat16)
        for _ in range(3):
            set_prefix(prefix)
            graph.replay()
            torch.testing.assert_close(
                output.view(8, 16, 64),
                expected[:, None, :].expand(8, 16, 64),
                rtol=0,
                atol=0.004,
                msg=f"verification block with prefix length {prefix}",
            )
            torch.testing.assert_close(residual, stream, rtol=0, atol=0)
            assert not health.any().item()
            assert not alarm.any().item()
    graph.reset()
