import unittest

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _check_prepare(m, use_rope, swa, scaled):
    from sglang.kernels.ops.kvcache.cache_ops import launch_reshape_and_cache_flash
    from sglang.kernels.ops.muse_spark.qkv_prepare import fused_qkv_prepare
    from sglang.srt.layers.rotary_embedding import get_rope
    from sglang.srt.rocm_muse_spark.muse_qk_norm import fused_qk_norm

    rope = get_rope(
        64,
        rotary_dim=64,
        max_position=131072,
        base=10000,
        is_neox_style=False,
        dtype=torch.float32,
    ).cuda()
    qkv = torch.randn(m, 2304, device="cuda", dtype=torch.bfloat16)
    q, k, v, _gate = qkv.split([1024, 128, 128, 1024], dim=-1)
    pos = torch.randint(0, 131072, (m,), device="cuda", dtype=torch.int64)
    loc = torch.randperm(256, device="cuda")[:m].contiguous()
    mapping = torch.randperm(256, device="cuda") if swa else None
    ks = torch.tensor([0.75], device="cuda") if scaled else None
    vs = torch.tensor([1.25], device="cuda") if scaled else None
    caches = [
        torch.zeros((16, 16, 2, 64), device="cuda", dtype=torch.float8_e4m3fn)
        for _ in range(4)
    ]

    def baseline():
        qo, ko = fused_qk_norm(q, k, 64, 1e-5, 5.4730064863875)
        if use_rope:
            qo, ko = rope(pos, qo, ko)
        launch_reshape_and_cache_flash(
            ko.view(m, 2, 64),
            v.view(m, 2, 64),
            caches[0],
            caches[1],
            loc,
            mapping,
            ks,
            vs,
        )
        return qo, ko

    def candidate():
        return fused_qkv_prepare(
            q,
            k,
            v,
            pos,
            rope.cos_sin_cache if use_rope else None,
            caches[2],
            caches[3],
            loc,
            eps=1e-5,
            q_scale=5.4730064863875,
            swa_mapping=mapping,
            k_scale=ks,
            v_scale=vs,
        )

    graphs = []
    for fn in (baseline, candidate):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fn()
        graphs.append((graph, captured))
    for repeat in range(3):
        qkv.normal_()
        pos.random_(0, 131072)
        loc.copy_(torch.randperm(256, device="cuda")[:m])
        if not swa and repeat == 2:
            loc[-1] = -1
        for c in caches:
            c.zero_()
        for graph, _ in graphs:
            graph.replay()
        torch.cuda.synchronize()
        for actual, expected in zip(graphs[1][1], graphs[0][1]):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for actual, expected in zip(caches[2:], caches[:2]):
            torch.testing.assert_close(actual.float(), expected.float(), atol=0, rtol=0)


def _check_integration():
    from types import SimpleNamespace
    from unittest.mock import patch

    # Registration precedes the ROCm model import, as in server startup.
    from sglang.kernels.ops.muse_spark import fused_muse_spark_routing  # noqa: F401
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
    from sglang.srt.layers.rotary_embedding import get_rope
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.rocm_muse_spark.model import MuseSpark1xAttention

    class PairIdentity(torch.nn.Module):
        def forward(self, x):
            return x, None

    class RecordingAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_id = 0
            self.is_cross_attention = False
            self.sliding_window_size = -1
            self.k_scale = None
            self.v_scale = None

        def forward(self, q, k, v, batch, save_kv_cache=True):
            self.saved = save_kv_cache
            return q

    attention = MuseSpark1xAttention.__new__(MuseSpark1xAttention)
    torch.nn.Module.__init__(attention)
    attention.qkv_proj = PairIdentity()
    attention.o_proj = PairIdentity()
    attention.q_size = 1024
    attention.kv_size = 128
    attention.head_dim = 64
    attention.num_heads = 16
    attention.num_kv_heads = 2
    attention.use_o_gate = True
    attention.use_qk_norm = True
    attention.use_fused_qkv_prepare = True
    attention.use_attn_temperature_tuning = False
    attention.norm_eps = 1e-5
    attention.scale_query_by = 5.4730064863875
    attention.rotary_emb = get_rope(
        64,
        rotary_dim=64,
        max_position=128,
        base=10000,
        is_neox_style=False,
        dtype=torch.float32,
    ).cuda()
    attention.attn = RecordingAttention()
    attention._fused_o_norm_gate = SimpleNamespace(forward=lambda out, gate: out)
    backend = AiterAttnBackend.__new__(AiterAttnBackend)
    backend.use_mla = False
    backend.kv_cache_is_vectorized_5d = False
    backend.kv_cache_dtype = torch.float8_e4m3fn
    backend.use_triton_unified_attention = True
    backend.use_sliding_window_kv_pool = False
    backend.k_scale = torch.tensor([0.7], device="cuda")
    backend.v_scale = torch.tensor([1.3], device="cuda")
    kc = torch.zeros(512, 2, 64, device="cuda", dtype=torch.float8_e4m3fn)
    vc = torch.zeros_like(kc)
    backend.token_to_kv_pool = SimpleNamespace(get_kv_buffer=lambda layer: (kc, vc))
    for m, mode, vectorized, expected in [
        (8, ForwardMode.TARGET_VERIFY, False, False),
        (16, ForwardMode.TARGET_VERIFY, False, False),
        (24, ForwardMode.TARGET_VERIFY, False, False),
        (32, ForwardMode.TARGET_VERIFY, False, False),
        (256, ForwardMode.TARGET_VERIFY, False, True),
        (8, ForwardMode.EXTEND, False, True),
        (1, ForwardMode.DECODE, False, True),
        (8, ForwardMode.TARGET_VERIFY, True, True),
    ]:
        backend.kv_cache_is_vectorized_5d = vectorized
        batch = SimpleNamespace(
            forward_mode=mode,
            attn_attend_prefix_cache=False,
            out_cache_loc=torch.arange(m, device="cuda", dtype=torch.int64),
        )
        x = torch.randn(m, 2304, device="cuda", dtype=torch.bfloat16)
        pos = torch.arange(m, device="cuda", dtype=torch.int64) % 128
        with patch(
            "sglang.srt.rocm_muse_spark.model.get_attn_backend", return_value=backend
        ):
            actual = attention(pos, x, batch)
            assert attention.attn.saved == expected, (m, mode, vectorized)
            attention.use_fused_qkv_prepare = False
            reference = attention(pos, x, batch)
            attention.use_fused_qkv_prepare = True
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "Requires MI355X")
class TestMuseQKVPrepare(CustomTestCase):
    @torch.no_grad()
    def test_changed_graph_inputs(self):
        torch.manual_seed(123)
        for case in (
            (8, True, False, True),
            (8, True, True, True),
            (8, False, False, True),
            (1, True, False, False),
            (32, True, True, True),
            (9, False, False, True),
        ):
            with self.subTest(case=case):
                _check_prepare(*case)

    @torch.no_grad()
    def test_model_and_backend_dispatch(self):
        # Seeds 5 and 16 exercise BF16 midpoints affected by the reduction layout.
        for seed in (0, 5, 16):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                _check_integration()


if __name__ == "__main__":
    unittest.main()
