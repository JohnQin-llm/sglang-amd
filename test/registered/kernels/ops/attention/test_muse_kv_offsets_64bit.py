"""Muse Spark verify and DFlash draft attention over a KV cache layer whose pages lie
past 32-bit element offsets: 8,912,896 tokens x 256 B = 2.28 GB per K or V buffer,
beyond 2^31 elements from token 8,388,608 on (TP8 verify layout: token 16,777,216).

Each kernel runs on pages beyond that offset and must give the same bits as on the
same K/V moved to the start of a small pool, and match an FP32 reference."""

import math
import random
import unittest

import torch
import torch.nn.functional as F

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

PAGE, TOKENS, HEAD_DIM, WINDOW = 16, 8, 64, 2047
POOL_BYTES = (2**23 + 2**19) * 256
FP8 = torch.float8_e4m3fn


def first_page_past_32_bits(pool):
    """The first page (or slot) whose element offset is 2^31 or more."""
    return 2**31 // pool.stride(0)


def pick_pages(rng, count, pool):
    """``count`` distinct pages in random order, all past 32-bit offsets but one."""
    first = first_page_past_32_bits(pool)
    assert pool.shape[0] - first > count
    pages = rng.sample(range(first, pool.shape[0]), count - 1)
    pages.insert(rng.randrange(count), rng.randrange(first))
    return pages


def fill(pool, pages, generator):
    rows = torch.tensor(pages, device="cuda")
    shape = (len(pages), *pool.shape[1:])
    values = torch.randn(shape, generator=generator, device="cuda").to(FP8)
    pool.view(torch.uint8)[rows] = values.view(torch.uint8)


def relocate(k, v, index, used):
    """``k``/``v`` rows ``used`` moved to rows 0.. of a small pool, and ``index`` (page
    table or slot list) renumbered to match."""
    rows = torch.tensor(used, device="cuda")
    lookup = torch.zeros(k.shape[0], dtype=index.dtype, device="cuda")
    lookup[rows] = torch.arange(len(used), dtype=index.dtype, device="cuda")
    k, v = (x.view(torch.uint8)[rows].contiguous().view(FP8) for x in (k, v))
    return k, v, lookup[index.long()]


def verify_inputs(rng, lens, k, v, q_heads, seed):
    """BF16 Q, int64 KV lengths and an int32 page table over pages past 32-bit offsets;
    those pages of ``k``/``v`` are filled with random FP8 values."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    need = [math.ceil(n / PAGE) for n in lens]
    pages = pick_pages(rng, sum(need), k)
    fill(k, pages, g)
    fill(v, pages, g)
    table = torch.zeros((len(lens), max(need) + 3), dtype=torch.int32)
    offset = 0
    for b, n in enumerate(need):
        table[b, :n] = torch.tensor(pages[offset : offset + n])
        table[b, n:] = pages[offset]  # never read: past the request's last key
        offset += n
    q = torch.randn(len(lens) * TOKENS, q_heads, HEAD_DIM, generator=g, device="cuda")
    lens = torch.tensor(lens, dtype=torch.int64, device="cuda")
    return q.to(torch.bfloat16), lens, table.cuda(), pages


def verify_reference(q, k, v, lens, table, window, k_scale, v_scale):
    """FP32 attention of the last TOKENS positions of each request, causal among them."""
    q_heads, kv_heads = q.shape[1], k.shape[2]
    group = q_heads // kv_heads
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for b, kv_len in enumerate(lens.tolist()):
        pages = table[b, : math.ceil(kv_len / PAGE)].long()
        kk = k[pages].float().reshape(-1, kv_heads, HEAD_DIM)[:kv_len] * k_scale
        vv = v[pages].float().reshape(-1, kv_heads, HEAD_DIM)[:kv_len] * v_scale
        key_pos = torch.arange(kv_len, device=q.device)
        query_pos = kv_len - TOKENS + torch.arange(TOKENS, device=q.device)
        visible = key_pos[None, :] <= query_pos[:, None]
        if window:
            visible &= query_pos[:, None] - key_pos[None, :] < window
        qq = q[b * TOKENS : (b + 1) * TOKENS].float().view(TOKENS, kv_heads, group, -1)
        s = torch.einsum("tghd,ngd->tghn", qq, kk) * HEAD_DIM**-0.5
        s = s.masked_fill(~visible[:, None, None, :], float("-inf"))
        out[b * TOKENS : (b + 1) * TOKENS] = torch.einsum(
            "tghn,ngd->tghd", torch.softmax(s, dim=-1), vv
        ).reshape(TOKENS, q_heads, HEAD_DIM)
    return out


def draft_reference(q, k_ext, v_ext, k_cache, v_cache, kv_indptr, kv_indices, scale):
    """FP32 DFlash draft block: each request's TOKENS queries attend, non-causally, to
    their window of cached K/V (Q rounded to the cache dtype, as the kernel does) and
    to the block's own K/V."""
    heads, kv_heads = q.shape[1], k_ext.shape[1]
    group = heads // kv_heads
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for b in range(kv_indptr.numel() - 1):
        rows = slice(b * TOKENS, (b + 1) * TOKENS)
        slots = kv_indices[kv_indptr[b] : kv_indptr[b + 1]].long()
        prefix = slots.numel()
        qb = q[rows].float().view(TOKENS, kv_heads, group, -1)
        q8 = q[rows].to(k_cache.dtype).float().view(TOKENS, kv_heads, group, -1)
        s_prefix = torch.einsum("tghd,ngd->tghn", q8, k_cache[slots].float()) * scale
        query_pos = prefix + torch.arange(TOKENS, device=q.device)
        visible = (
            query_pos[:, None] <= torch.arange(prefix, device=q.device) + WINDOW + 1
        )
        s_prefix = s_prefix.masked_fill(~visible[:, None, None, :], float("-inf"))
        s_block = torch.einsum("tghd,ngd->tghn", qb, k_ext[rows].float()) * scale
        p = torch.softmax(torch.cat([s_prefix, s_block], dim=-1), dim=-1)
        values = torch.cat([v_cache[slots].float(), v_ext[rows].float()])
        out[rows] = torch.einsum("tghn,ngd->tghd", p, values).reshape(TOKENS, heads, -1)
    return out


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "gfx950 Muse attention kernels")
class TestMuseKvOffsets64Bit(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.attention import (
            dflash_draft_splitkv,
            gqa_verify_splitkv_2kv,
            verify_tp4,
        )

        cls.draft = dflash_draft_splitkv
        cls.vv2 = gqa_verify_splitkv_2kv
        cls.tp4 = verify_tp4
        # One layer's K and V buffers, viewed in each kernel's cache layout.
        cls.k_bytes = torch.zeros(POOL_BYTES, dtype=torch.uint8, device="cuda")
        cls.v_bytes = torch.zeros(POOL_BYTES, dtype=torch.uint8, device="cuda")

    @classmethod
    def tearDownClass(cls):
        del cls.k_bytes, cls.v_bytes
        torch.cuda.empty_cache()

    def pools(self, *row_shape):
        return [x.view(FP8).view(-1, *row_shape) for x in (self.k_bytes, self.v_bytes)]

    def assert_close(self, out, ref, min_cos, msg):
        self.assertTrue(bool(torch.isfinite(out).all()), msg)
        cos = F.cosine_similarity(
            out.float().reshape(-1, out.shape[-1]),
            ref.reshape(-1, ref.shape[-1]),
            dim=1,
        )
        self.assertGreaterEqual(float(cos.min()), min_cos, msg)

    def check_verify(self, run, q_heads, kv_heads, cases, window=0):
        k, v = self.pools(PAGE, kv_heads, HEAD_DIM)
        rng = random.Random(q_heads + window)
        k_scale, v_scale = 0.5, 0.75
        descales = [torch.tensor([x], device="cuda") for x in (k_scale, v_scale)]
        for seed, lens in enumerate(cases):
            q, s, table, pages = verify_inputs(rng, lens, k, v, q_heads, seed)
            far = torch.full_like(q, float("nan"))
            run(q, k, v, far, s, table, *descales)
            k_near, v_near, table_near = relocate(k, v, table, pages)
            near = torch.full_like(q, float("nan"))
            run(q, k_near, v_near, near, s, table_near, *descales)
            with self.subTest(lens=lens):
                self.assertTrue(torch.equal(far, near), f"bits differ at {lens}")
                ref = verify_reference(q, k, v, s, table, window, k_scale, v_scale)
                self.assert_close(far, ref, 0.9999, f"{lens}")

    def test_verify_tp4_full(self):
        def run(q, k, v, out, s, table, kd, vd):
            self.assertTrue(self.tp4.can_verify_tp4(q, k, v, out, s, table, kd, vd))
            self.tp4.verify_full_tp4(q, k, v, out, s, table, HEAD_DIM**-0.5, kd, vd)

        self.check_verify(run, 32, 4, [[100003], [8191, 3000, 40000], [5000] * 16])

    def test_verify_tp4_sliding(self):
        def run(q, k, v, out, s, table, kd, vd):
            self.assertTrue(self.tp4.can_verify_tp4(q, k, v, out, s, table, kd, vd))
            self.tp4.verify_swa_tp4(
                q, k, v, out, s, table, HEAD_DIM**-0.5, kd, vd, window=WINDOW
            )

        cases = [[100003], [2054, 9000, 40000], [5000] * 16]
        self.check_verify(run, 32, 4, cases, window=WINDOW)

    def test_verify_4kv(self):
        def run(q, k, v, out, s, table, kd, vd):
            self.assertTrue(
                self.vv2.can_gqa_verify_splitkv_4kv(q, k, v, out, s, table, kd, vd)
            )
            self.vv2.gqa_verify_splitkv_4kv(
                q, k, v, out, s, table, HEAD_DIM**-0.5, kd, vd
            )

        self.check_verify(run, 32, 4, [[100003], [8191, 3000, 40000], [5000] * 16])

    def test_verify_2kv(self):
        def run(q, k, v, out, s, table, kd, vd):
            self.vv2.gqa_verify_splitkv_2kv(
                q, k, v, out, s, table, HEAD_DIM**-0.5, kd, vd
            )

        self.check_verify(run, 16, 2, [[100003], [8191, 3000, 40000], [5000] * 16])

    def test_draft_splitkv(self):
        heads, kv_heads, dim = 16, 2, 128
        k, v = self.pools(kv_heads, dim)
        rng = random.Random(1)
        scale = dim**-0.5
        for index_dtype in (torch.int32, torch.int64):
            for seed, prefixes in enumerate(
                [[2048], [2048, 1000, 2056, 37], [2048] * 16]
            ):
                g = torch.Generator(device="cuda").manual_seed(seed)
                slots = pick_pages(rng, sum(prefixes), k)
                fill(k, slots, g)
                fill(v, slots, g)
                kv_indices = torch.tensor(slots, dtype=index_dtype, device="cuda")
                kv_indptr = torch.tensor([0, *prefixes], device="cuda").cumsum(0)
                bs = len(prefixes)
                qo_indptr = torch.arange(0, (bs + 1) * TOKENS, TOKENS, device="cuda")
                q, k_ext, v_ext = (
                    torch.randn(bs * TOKENS, n, dim, generator=g, device="cuda").to(
                        torch.bfloat16
                    )
                    for n in (heads, kv_heads, kv_heads)
                )

                def run(k_cache, v_cache, indices):
                    o = torch.full_like(q, float("nan"))
                    args = (q, k_ext, v_ext, o, k_cache, v_cache, qo_indptr, kv_indptr)
                    self.assertTrue(
                        self.draft.dflash_draft_splitkv_supported(
                            q, k_ext, v_ext, k_cache, v_cache, TOKENS, WINDOW + 1
                        )
                    )
                    self.draft.dflash_draft_splitkv_fwd(
                        *args, indices, TOKENS, scale, sliding_window_size=WINDOW + 1
                    )
                    return o

                far = run(k, v, kv_indices)
                near = run(*relocate(k, v, kv_indices, slots))
                with self.subTest(index_dtype=index_dtype, prefixes=prefixes):
                    self.assertTrue(
                        torch.equal(far, near), f"bits differ at {prefixes}"
                    )
                    ref = draft_reference(
                        q, k_ext, v_ext, k, v, kv_indptr, kv_indices, scale
                    )
                    self.assert_close(far, ref, 0.995, f"{prefixes}")


if __name__ == "__main__":
    unittest.main()
