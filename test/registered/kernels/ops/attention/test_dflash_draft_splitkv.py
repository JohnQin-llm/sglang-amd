"""DFlash draft split-KV attention (TP4 Muse Spark draft: 16 Q / 2 KV heads x 128, an
8-token block per request over a 2048-slot window of the FP8 draft cache) against
extend_attention_fwd, the generic kernel it replaces for these blocks."""

import unittest

import torch
import torch.nn.functional as F

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

HEADS, KV_HEADS, HEAD_DIM, BLOCK, WINDOW, PAGE = 16, 2, 128, 8, 2048, 16
POOL_SLOTS = 1 << 17
SCALE = HEAD_DIM**-0.5


def make_case(contexts, seed, cache_dtype=torch.float8_e4m3fn, q_mult=1.0):
    """One draft layer's inputs at decode: each request's window is the last
    min(compact length, WINDOW) committed tokens, where the compact cache starts on
    a page boundary; its pages are random pages of the pool (page 0 stays unused)."""
    g = torch.Generator().manual_seed(seed)
    batch = len(contexts)
    perm = torch.randperm(POOL_SLOTS // PAGE - 1, generator=g) + 1
    slots, lens, offsets, cursor = [], [], [], 0
    for ctx in contexts:
        start = max(ctx - WINDOW, 0)
        compact = ctx - (start - start % PAGE)
        window = min(compact, WINDOW)
        first_page = (ctx - window) // PAGE
        pages = perm[cursor : cursor + (ctx - 1) // PAGE - first_page + 1]
        cursor += pages.numel()
        pos = torch.arange(ctx - window, ctx)
        slots.append(pages[pos // PAGE - first_page] * PAGE + pos % PAGE)
        lens.append(window)
        offsets.append(compact - window)
    tokens = batch * BLOCK
    q = torch.randn(tokens, HEADS, HEAD_DIM, generator=g) * q_mult
    k_ext, v_ext = (torch.randn(tokens, KV_HEADS, HEAD_DIM, generator=g) for _ in "kv")
    cache = [
        torch.randn(POOL_SLOTS, KV_HEADS, HEAD_DIM, generator=g).to(cache_dtype)
        for _ in "kv"
    ]
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32)
    kv_indptr[1:] = torch.tensor(lens).cumsum(0)
    case = dict(
        q=q.to(torch.bfloat16),
        k_ext=k_ext.to(torch.bfloat16),
        v_ext=v_ext.to(torch.bfloat16),
        k_cache=cache[0],
        v_cache=cache[1],
        qo_indptr=torch.arange(0, (batch + 1) * BLOCK, BLOCK),
        kv_indptr=kv_indptr,
        kv_indices=torch.cat(slots),
        window_kv_offsets=torch.tensor(offsets, dtype=torch.int32),
        mask_indptr=torch.zeros(batch + 1, dtype=torch.int64),
    )
    return {name: x.cuda() for name, x in case.items()}


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "gfx950 DFlash draft kernel")
class TestDflashDraftSplitKv(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.attention import dflash_draft_splitkv
        from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd

        cls.ops = dflash_draft_splitkv
        cls.extend_attention_fwd = staticmethod(extend_attention_fwd)

    def run_new(self, c, k_scale=1.0, v_scale=1.0, logit_cap=0.0, out=None):
        out = torch.full_like(c["q"], float("nan")) if out is None else out
        self.assertTrue(
            self.ops.dflash_draft_splitkv_supported(
                c["q"],
                c["k_ext"],
                c["v_ext"],
                c["k_cache"],
                c["v_cache"],
                BLOCK,
                WINDOW,
            )
        )
        self.ops.dflash_draft_splitkv_fwd(
            c["q"], c["k_ext"], c["v_ext"], out, c["k_cache"], c["v_cache"],
            c["qo_indptr"], c["kv_indptr"], c["kv_indices"], BLOCK, SCALE,
            k_scale, v_scale, sliding_window_size=WINDOW, logit_cap=logit_cap,
        )  # fmt: skip
        return out

    def run_generic(self, c, k_scale=1.0, v_scale=1.0, logit_cap=0.0):
        """The draft call's extend_attention_fwd arguments (TARGET_VERIFY, no mask)."""
        out = torch.full_like(c["q"], float("nan"))
        self.extend_attention_fwd(
            c["q"], c["k_ext"], c["v_ext"], out, c["k_cache"], c["v_cache"],
            c["qo_indptr"], c["kv_indptr"], c["kv_indices"], None, False,
            c["mask_indptr"], BLOCK, k_scale, v_scale, sm_scale=SCALE,
            logit_cap=logit_cap, sliding_window_size=WINDOW,
            window_kv_offsets=c["window_kv_offsets"], page_size=PAGE,
        )  # fmt: skip
        return out

    def assert_close(self, out, ref, min_cos, msg):
        self.assertTrue(bool(torch.isfinite(out).all()), msg)
        cos = F.cosine_similarity(
            out.float().reshape(-1, HEAD_DIM), ref.float().reshape(-1, HEAD_DIM), dim=1
        )
        self.assertGreaterEqual(float(cos.min()), min_cos, msg)

    def test_long_contexts(self):
        """Compact 2048..2063-token caches of 100K+ contexts, batch 1..16. Both kernels
        round Q and P to FP8; the split kernel rounds P against each split's running
        maximum, so the two agree to FP8 precision, not bitwise."""
        for batch in (1, 2, 3, 4, 5, 8, 10, 12, 16):
            contexts = [100_000 + 37 * i + 1001 * batch for i in range(batch)]
            for q_mult in (1.0, 4.0):
                c = make_case(contexts, seed=batch, q_mult=q_mult)
                with self.subTest(batch=batch, q_mult=q_mult):
                    self.assert_close(
                        self.run_new(c), self.run_generic(c), 0.998, f"{contexts}"
                    )

    def test_short_and_ragged_windows(self):
        """Windows shorter than a tile or a page, and one request per page boundary."""
        contexts = [1, 7, 8, 9, 15, 16, 17, 100, 2047, 2048, 2049, 2063, 2064, 150_001]
        for seed in range(3):
            c = make_case(contexts, seed=seed)
            self.assert_close(self.run_new(c), self.run_generic(c), 0.998, "fp8")

    def test_descales_and_logit_cap(self):
        contexts = [100_003, 2050, 640, 120_000]
        for k_scale, v_scale, cap in ((0.5, 2.0, 0.0), (1.0, 1.0, 5.0)):
            c = make_case(contexts, seed=3)
            with self.subTest(k_scale=k_scale, v_scale=v_scale, logit_cap=cap):
                self.assert_close(
                    self.run_new(c, k_scale, v_scale, cap),
                    self.run_generic(c, k_scale, v_scale, cap),
                    0.998,
                    "scales",
                )

    def test_bf16_cache(self):
        """Without FP8 rounding the two kernels agree to BF16 precision."""
        c = make_case(
            [100_000, 2049, 300, 130_000, 7], seed=4, cache_dtype=torch.bfloat16
        )
        self.assert_close(self.run_new(c), self.run_generic(c), 0.9999, "bf16 cache")

    def test_graph_replay_with_new_inputs(self):
        contexts = [120_000, 2055, 140_000, 900]
        first, second = make_case(contexts, seed=5), make_case(contexts, seed=6)
        c = {name: x.clone() for name, x in first.items()}
        out = torch.empty_like(c["q"])
        self.run_new(c, out=out)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.run_new(c, out=out)
        for name in c:
            c[name].copy_(second[name])
        graph.replay()
        self.assertTrue(torch.equal(out, self.run_new(second)))

    def test_supported_guard(self):
        c = make_case([3000, 100], seed=7)
        q, k_ext, v_ext, kc, vc = (
            c[n] for n in ("q", "k_ext", "v_ext", "k_cache", "v_cache")
        )
        supported = self.ops.dflash_draft_splitkv_supported
        self.assertTrue(supported(q, k_ext, v_ext, kc, vc, BLOCK, WINDOW))
        # head_dim 64, three query heads per KV head, a block longer than the window,
        # and FP32 queries keep the generic kernel.
        narrow = [x[..., :64] for x in (q, k_ext, v_ext, kc, vc)]
        self.assertFalse(supported(*narrow, BLOCK, WINDOW))
        self.assertFalse(supported(q[:, :6], k_ext, v_ext, kc, vc, BLOCK, WINDOW))
        self.assertFalse(supported(q, k_ext, v_ext, kc, vc, WINDOW + 1, WINDOW))
        self.assertFalse(supported(q.float(), k_ext, v_ext, kc, vc, BLOCK, WINDOW))


if __name__ == "__main__":
    unittest.main()
