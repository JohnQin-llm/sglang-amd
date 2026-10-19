import unittest

import torch
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _check_window(length_list, heads, softmax_scale=64**-0.5):
    from aiter.ops.triton.attention.unified_attention import unified_attention
    from sglang.kernels.ops.muse_spark.swa_verify import (
        muse_swa_verify_attention,
        prepare_muse_swa_verify,
    )

    batch = len(length_list)
    m = 8
    kv = heads // 8
    dim = 64
    width = (max(length_list) + 15) // 16
    q = torch.randn(batch * m, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch * width, 16, kv, dim, device="cuda").to(torch.float8_e4m3fn)
    v = torch.randn_like(k.float()).to(k.dtype)
    table = torch.randperm(batch * width, device="cuda", dtype=torch.int32).reshape(
        batch, width
    )
    lengths = torch.tensor(length_list, device="cuda", dtype=torch.int32)
    cu = torch.arange(0, (batch + 1) * m, m, device="cuda", dtype=torch.int32)
    ks = torch.tensor([0.7], device="cuda")
    vs = torch.tensor([1.3], device="cuda")
    out = torch.empty_like(q)
    base = torch.empty_like(q)
    args = {
        "q": q,
        "k": k,
        "v": v,
        "out": base,
        "cu_seqlens_q": cu,
        "seqused_k": lengths,
        "max_seqlen_q": m,
        "max_seqlen_k": max(length_list),
        "softmax_scale": softmax_scale,
        "causal": True,
        "window_size": (2046, 0),
        "block_table": table,
        "softcap": 0.0,
        "q_descale": None,
        "k_descale": ks,
        "v_descale": vs,
    }

    def candidate():
        prepared = prepare_muse_swa_verify(lengths, table)
        for _ in range(2):
            muse_swa_verify_attention(
                q,
                k,
                v,
                out,
                cu,
                lengths,
                table,
                softmax_scale,
                ks,
                vs,
                prepared=prepared,
            )

    def baseline():
        for _ in range(2):
            unified_attention(**args)

    def reference():
        answers = []
        for b, length in enumerate(lengths.tolist()):
            for i in range(m):
                end = length - m + i + 1
                pos = torch.arange(max(0, end - 2047), end, device="cuda")
                pages = table[b, pos // 16].long()
                kk = k[pages, pos % 16].float().repeat_interleave(8, 1) * ks
                vv = v[pages, pos % 16].float().repeat_interleave(8, 1) * vs
                scores = (
                    torch.einsum("hd,lhd->hl", q[b * m + i].float(), kk) * softmax_scale
                )
                answers.append(torch.einsum("hl,lhd->hd", scores.softmax(-1), vv))
        return torch.stack(answers)

    graphs = {}
    for tag, fn in [("baseline", baseline), ("candidate", candidate)]:
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        graphs[tag] = g
    ref = reference()
    torch.testing.assert_close(out.float(), ref, atol=0.004, rtol=0.04)
    # Replay with changed page contents, lengths and queries to exercise graph metadata updates.
    lengths.add_(torch.where(lengths > 32, -7, 0))
    q.mul_(0.8)
    table.copy_(table.flip(1))
    ks.fill_(1.2)
    vs.fill_(0.9)
    graphs["candidate"].replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), reference(), atol=0.004, rtol=0.04)


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "Requires MI355X")
class TestMuseSWAVerify(CustomTestCase):
    @torch.no_grad()
    def test_original_tp_score_scale(self):
        # Original TP moves the model's query multiplier after the QK product.
        torch.manual_seed(123)
        _check_window([1024, 3008], 16, softmax_scale=64**-0.5 * 5.4730064863875)

    @torch.no_grad()
    def test_backend_dispatch_boundaries(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        backend = AiterAttnBackend.__new__(AiterAttnBackend)
        backend.use_mla = False
        backend._use_unified_verify = True
        backend.page_size = 16
        backend.k_scale = backend.v_scale = None
        layer = SimpleNamespace(
            layer_id=0,
            is_cross_attention=False,
            qk_head_dim=64,
            v_head_dim=64,
            tp_q_head_num=16,
            tp_k_head_num=2,
            tp_v_head_num=2,
            scaling=0.125,
            logit_cap=0.0,
            k_scale=None,
            v_scale=None,
        )
        for batch, enabled, fp8, window, expected in (
            (4, True, True, 2047, True),
            (5, True, True, 2047, True),
            (31, True, True, 2047, True),
            (32, True, True, 2047, True),
            (33, True, True, 2047, False),
            (32, False, True, 2047, False),
            (32, True, False, 2047, False),
            (32, True, True, -1, False),
        ):
            with self.subTest(batch=batch, enabled=enabled, fp8=fp8, window=window):
                backend.use_muse_swa_verify = enabled
                backend.kv_cache_dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
                cache = torch.empty(
                    16, 2, 64, device="cuda", dtype=backend.kv_cache_dtype
                )
                backend.token_to_kv_pool = SimpleNamespace(
                    get_kv_buffer=lambda _, cache=cache: (cache, cache)
                )
                backend.forward_metadata = SimpleNamespace(
                    max_q_len=8,
                    kv_indices=torch.zeros(batch, 1, device="cuda", dtype=torch.int32),
                    swa_page_table=None,
                    qo_indptr=torch.arange(
                        0, (batch + 1) * 8, 8, device="cuda", dtype=torch.int32
                    ),
                    muse_swa_verify=None,
                )
                layer.sliding_window_size = window
                forward_batch = SimpleNamespace(
                    attn_attend_prefix_cache=False,
                    out_cache_loc=None,
                    forward_mode=ForwardMode.TARGET_VERIFY,
                    seq_lens=torch.zeros(batch, device="cuda", dtype=torch.int32),
                )
                q = torch.empty(batch * 8, 1024, device="cuda", dtype=torch.bfloat16)
                with (
                    patch(
                        "sglang.kernels.ops.muse_spark.swa_verify.muse_swa_verify_attention"
                    ) as optimized,
                    patch(
                        "sglang.srt.layers.attention.aiter_backend.unified_attention"
                    ) as fallback,
                ):
                    backend.forward_extend(q, None, None, layer, forward_batch)
                self.assertEqual(optimized.called, expected)
                self.assertEqual(fallback.called, not expected)

    @torch.no_grad()
    def test_window_boundaries_and_graph_replay(self):
        torch.manual_seed(2026)
        for lengths, heads in (
            ([8], 16),
            ([2048], 16),
            ([65543], 16),
            ([8, 2055, 131071], 16),
            ([65551], 32),
            ([8, 127, 2055, 65543], 16),
        ):
            with self.subTest(lengths=lengths, heads=heads):
                _check_window(lengths, heads)

    @torch.no_grad()
    def test_large_batch_window_boundaries_and_graph_replay(self):
        torch.manual_seed(20261018)
        for batch in (5, 15, 16, 31, 32):
            # Short rows also exercise the values used by padded graph batches.
            lengths = ([8, 15, 127, 2047, 2048, 2055, 8199, 65543] * 4)[:batch]
            with self.subTest(batch=batch):
                _check_window(lengths, 16)


if __name__ == "__main__":
    unittest.main()
