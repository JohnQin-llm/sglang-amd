"""TP4 Muse Spark target-verify attention (verify_full_tp4 / verify_swa_tp4)
against an FP32 reference over the same FP8 KV cache, and the merge with the
attention output norm and gate against the model's separate kernel."""

import math
import random
import unittest

import torch
import torch.nn.functional as F

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

Q_HEADS, KV_HEADS, HEAD_DIM, PAGE, TOKENS, WINDOW = 32, 4, 64, 16, 8, 2047
SCALE = HEAD_DIM**-0.5
# Page index written past each request's last page; reading it would fault.
OUT_OF_RANGE_PAGE = 10**7
LAYERS = {"full": 0, "swa": WINDOW}
EPS = 1e-5  # Muse Spark rms_norm_eps
# Verify graph widths: batch 1..8, then even batches to 16.
BATCHES = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16)


def make_inputs(lens, seed, q_mult=1.0, k_mult=1.0, total_pages=None, aligned=False):
    """Q, FP8 K/V pages, int64 KV lengths and a shuffled int32 page table."""
    g = torch.Generator().manual_seed(seed)
    need = [math.ceil(n / PAGE) for n in lens]
    total_pages = total_pages or sum(need) + 7
    shape = (total_pages, PAGE, KV_HEADS, HEAD_DIM)
    if aligned:
        # Every key equals every query direction: all scores are q_mult * k_mult * 64 * SCALE.
        k = torch.full(shape, k_mult)
        q = torch.full((len(lens) * TOKENS, Q_HEADS, HEAD_DIM), q_mult)
    else:
        k = torch.randn(shape, generator=g) * k_mult
        q = torch.randn(len(lens) * TOKENS, Q_HEADS, HEAD_DIM, generator=g) * q_mult
    v = torch.randn(shape, generator=g)
    perm = torch.randperm(total_pages, generator=g)
    table = torch.full((len(lens), max(need) + 3), OUT_OF_RANGE_PAGE, dtype=torch.int32)
    offset = 0
    for b, n in enumerate(need):
        table[b, :n] = perm[offset : offset + n].int()
        offset += n
    return (
        q.to(torch.bfloat16).cuda(),
        k.to(torch.float8_e4m3fn).cuda(),
        v.to(torch.float8_e4m3fn).cuda(),
        torch.tensor(lens, dtype=torch.int64).cuda(),
        table.cuda(),
    )


def reference(q, k, v, lens, table, window, k_scale, v_scale):
    """FP32 attention of the last TOKENS positions of each request, causal among them."""
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    group = Q_HEADS // KV_HEADS
    for b, kv_len in enumerate(lens.tolist()):
        pages = table[b, : math.ceil(kv_len / PAGE)].long()
        kk = k[pages].float().reshape(-1, KV_HEADS, HEAD_DIM)[:kv_len] * k_scale
        vv = v[pages].float().reshape(-1, KV_HEADS, HEAD_DIM)[:kv_len] * v_scale
        key_pos = torch.arange(kv_len, device=q.device)
        query_pos = kv_len - TOKENS + torch.arange(TOKENS, device=q.device)
        visible = key_pos[None, :] <= query_pos[:, None]
        if window:
            visible &= query_pos[:, None] - key_pos[None, :] < window
        qq = q[b * TOKENS : (b + 1) * TOKENS].float().view(TOKENS, KV_HEADS, group, -1)
        s = torch.einsum("tghd,ngd->tghn", qq, kk) * SCALE
        s = s.masked_fill(~visible[:, None, None, :], float("-inf"))
        p = torch.softmax(s, dim=-1)
        out[b * TOKENS : (b + 1) * TOKENS] = torch.einsum(
            "tghn,ngd->tghd", p, vv
        ).reshape(TOKENS, Q_HEADS, HEAD_DIM)
    return out


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "gfx950 TP4 verify kernels")
class TestMuseVerifyTp4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.attention import verify_tp4

        cls.ops = verify_tp4

    def launch(
        self, layer, q, k, v, out, lens, table, k_descale, v_descale, output_gate=None
    ):
        self.assertTrue(
            self.ops.can_verify_tp4(q, k, v, out, lens, table, k_descale, v_descale)
        )
        if layer == "full":
            self.ops.verify_full_tp4(
                q,
                k,
                v,
                out,
                lens,
                table,
                SCALE,
                k_descale,
                v_descale,
                output_gate=output_gate,
            )
        else:
            self.ops.verify_swa_tp4(
                q,
                k,
                v,
                out,
                lens,
                table,
                SCALE,
                k_descale,
                v_descale,
                window=WINDOW,
                output_gate=output_gate,
            )

    def run_layer(self, layer, q, k, v, lens, table, k_scale=1.0, v_scale=1.0):
        out = torch.full_like(q, float("nan"))
        k_descale = torch.tensor([k_scale], device="cuda")
        v_descale = torch.tensor([v_scale], device="cuda")
        self.launch(layer, q, k, v, out, lens, table, k_descale, v_descale)
        return out

    def assert_matches(self, out, ref, msg):
        self.assertTrue(bool(torch.isfinite(out).all()), msg)
        cos = F.cosine_similarity(
            out.float().reshape(-1, HEAD_DIM), ref.reshape(-1, HEAD_DIM), dim=1
        )
        self.assertGreaterEqual(float(cos.min()), 0.9999, msg)

    def test_mixed_lengths(self):
        """Page and tile tails, contexts shorter than the window, long contexts and
        page-table entries past each request's end that must never be read."""
        rng = random.Random(0)
        pool = [8, 9, 15, 16, 17, 63, 64, 65, 71, 72, 127, 129, 200, 1000, 2047]
        pool += [2054, 2055, 2056, 2100, 4097, 12345, 33333]
        for layer, window in LAYERS.items():
            for bs in (1, 2, 3, 5, 8, 13, 16):
                long = [100003, 262151] if bs <= 5 else []
                lens = [rng.choice(pool + long) for _ in range(bs)]
                q, k, v, s, t = make_inputs(lens, seed=bs)
                with self.subTest(layer=layer, lens=lens):
                    self.assert_matches(
                        self.run_layer(layer, q, k, v, s, t, 0.5, 0.5),
                        reference(q, k, v, s, t, window, 0.5, 0.5),
                        f"{layer} {lens}",
                    )

    def test_every_short_length(self):
        """Batch 1 runs the most splits, so most of them hold no key at these lengths."""
        for layer, window in LAYERS.items():
            for kv_len in range(TOKENS, 209):
                q, k, v, s, t = make_inputs([kv_len], seed=kv_len)
                self.assert_matches(
                    self.run_layer(layer, q, k, v, s, t),
                    reference(q, k, v, s, t, window, 1.0, 1.0),
                    f"{layer} kv_len={kv_len}",
                )

    def test_scores_outside_fixed_reference(self):
        """Row sums that overflow or underflow exp2 against the fixed reference
        take the online-maximum recomputation."""
        cases = {
            "overflow, random": (40.0, 4.0, False),
            "overflow, aligned": (8.0, 4.0, True),  # +369 in log2 units
            "underflow, aligned": (-8.0, 4.0, True),  # -369 in log2 units
            "peaky": (2.0, 1.5, False),
        }
        lens = [5000, 70000, 129]
        for layer, window in LAYERS.items():
            for name, (q_mult, k_mult, aligned) in cases.items():
                q, k, v, s, t = make_inputs(
                    lens, seed=7, q_mult=q_mult, k_mult=k_mult, aligned=aligned
                )
                with self.subTest(layer=layer, case=name):
                    self.assert_matches(
                        self.run_layer(layer, q, k, v, s, t),
                        reference(q, k, v, s, t, window, 1.0, 1.0),
                        f"{layer} {name}",
                    )

    def test_graph_replay_with_new_inputs(self):
        """A captured call reads Q, K, V, lengths and page table at replay time."""
        lens_a, lens_b = [5000, 64, 131, 70000], [69999, 9, 4100, 2055]
        total = sum(math.ceil(n / PAGE) for n in lens_a + lens_b) + 64
        for layer, window in LAYERS.items():
            a = make_inputs(lens_a, 101, total_pages=total)
            b = make_inputs(lens_b, 202, total_pages=total)
            q, k, v, s = (x.clone() for x in a[:4])
            width = max(a[4].shape[1], b[4].shape[1])
            t = torch.zeros(len(lens_a), width, dtype=torch.int32, device="cuda")
            t[:, : a[4].shape[1]] = a[4]
            out = torch.empty_like(q)
            one = torch.ones(1, device="cuda")
            self.launch(layer, q, k, v, out, s, t, one, one)  # compile before capture
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.launch(layer, q, k, v, out, s, t, one, one)
            for name, src in (("A", a), ("B", b), ("A again", a)):
                for dst, x in zip((q, k, v, s), src[:4]):
                    dst.copy_(x)
                t.zero_()
                t[:, : src[4].shape[1]] = src[4]
                out.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                self.assert_matches(
                    out,
                    reference(q, k, v, s, t, window, 1.0, 1.0),
                    f"{layer} replay {name}",
                )

    def separate_norm_gate(self, out, gate):
        """The model's output norm and gate kernel on the merged output."""
        from sglang.srt.rocm_muse_spark.fused_norm_gate import FusedRMSNormGate
        from sglang.srt.rocm_muse_spark.layers import MuseSpark1xRMSNorm

        norm = MuseSpark1xRMSNorm(HEAD_DIM, eps=EPS, has_weight=False)
        op = FusedRMSNormGate(norm, True, HEAD_DIM)
        return op.forward_triton(out.view(out.shape[0], -1), gate)

    @staticmethod
    def projection_gate(tokens, seed):
        """The output gate as the QKV projection leaves it: its last 2048 columns."""
        g = torch.Generator(device="cuda").manual_seed(seed)
        qkv = torch.randn(
            tokens, 4608, generator=g, device="cuda", dtype=torch.bfloat16
        )
        return qkv[:, -Q_HEADS * HEAD_DIM :]

    def test_output_gate_matches_separate_kernel(self):
        """At every verify graph width, merge + norm + gate in one kernel is bitwise
        equal to the merge followed by the model's norm-gate kernel. Batch 1 with a
        short context leaves most splits empty."""
        rng = random.Random(1)
        pool = [9, 100, 2047, 2056, 5000, 70000, 131072]
        for layer in LAYERS:
            for bs in BATCHES:
                lens = [rng.choice(pool) for _ in range(bs)]
                if bs == 1:
                    lens = [100]
                q, k, v, s, t = make_inputs(lens, seed=10 + bs)
                gate = self.projection_gate(q.shape[0], seed=bs)
                with self.subTest(layer=layer, lens=lens):
                    expected = self.separate_norm_gate(
                        self.run_layer(layer, q, k, v, s, t, 0.5, 0.25), gate
                    )
                    out = torch.full_like(q, float("nan"))
                    output_gate = self.ops.OutputNormGate(gate, EPS)
                    scales = [torch.tensor([x], device="cuda") for x in (0.5, 0.25)]
                    self.launch(layer, q, k, v, out, s, t, *scales, output_gate)
                    self.assertTrue(output_gate.applied)
                    self.assertTrue(torch.equal(out.view(q.shape[0], -1), expected))

    def test_output_gate_layout_guard(self):
        """A gate the merge cannot read is left to the model: the merge output is
        unchanged and the gate is not marked applied."""
        q, k, v, s, t = make_inputs([3000, 64], seed=5)
        gate = self.projection_gate(q.shape[0], seed=5)
        rejected = {
            "column-major": gate.t().contiguous().t(),
            "FP32": gate.float(),
            "head-grouped": gate.reshape(q.shape[0], 4, -1),
        }
        for layer in LAYERS:
            plain = self.run_layer(layer, q, k, v, s, t)
            for name, bad in rejected.items():
                with self.subTest(layer=layer, gate=name):
                    out = torch.full_like(q, float("nan"))
                    output_gate = self.ops.OutputNormGate(bad, EPS)
                    one = torch.ones(1, device="cuda")
                    self.launch(layer, q, k, v, out, s, t, one, one, output_gate)
                    self.assertFalse(output_gate.applied)
                    self.assertTrue(torch.equal(out, plain))

    def test_output_gate_graph_replay(self):
        """A captured gated call reads the gate and attention inputs at replay time."""
        lens_a, lens_b = [5000, 64, 131, 70000], [69999, 9, 4100, 2055]
        total = sum(math.ceil(n / PAGE) for n in lens_a + lens_b) + 64
        for layer in LAYERS:
            a = make_inputs(lens_a, 303, total_pages=total)
            b = make_inputs(lens_b, 404, total_pages=total)
            gates = {"A": self.projection_gate(32, 1), "B": self.projection_gate(32, 2)}
            q, k, v, s = (x.clone() for x in a[:4])
            width = max(a[4].shape[1], b[4].shape[1])
            t = torch.zeros(len(lens_a), width, dtype=torch.int32, device="cuda")
            t[:, : a[4].shape[1]] = a[4]
            gate = gates["A"].clone()
            out = torch.empty_like(q)
            one = torch.ones(1, device="cuda")
            output_gate = self.ops.OutputNormGate(gate, EPS)
            self.launch(layer, q, k, v, out, s, t, one, one, output_gate)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.launch(layer, q, k, v, out, s, t, one, one, output_gate)
            for name, src in (("A", a), ("B", b), ("A", a)):
                for dst, x in zip((q, k, v, s), src[:4]):
                    dst.copy_(x)
                t.zero_()
                t[:, : src[4].shape[1]] = src[4]
                gate.copy_(gates[name])
                out.fill_(float("nan"))
                graph.replay()
                expected = self.separate_norm_gate(
                    self.run_layer(layer, *src), gates[name]
                )
                with self.subTest(layer=layer, replay=name):
                    self.assertTrue(torch.equal(out.view(32, -1), expected))

    def test_guard_rejects_other_layouts(self):
        """Layouts the kernels do not handle stay on the existing verify paths."""
        q, k, v, s, t = make_inputs([100, 2000], seed=3)
        out = torch.empty_like(q)
        one = torch.ones(1, device="cuda")
        self.assertTrue(self.ops.can_verify_tp4(q, k, v, out, s, t, one, one))
        self.assertTrue(self.ops.can_verify_tp4(q, k, v, out, s, t, None, None))
        rejected = {
            "int64 page table": (q, k, v, out, s, t.long(), one, one),
            "16 Q / 2 KV heads": (
                q[:, :16],
                k[:, :, :2],
                v[:, :, :2],
                out[:, :16],
                s,
                t,
                one,
                one,
            ),
            "per-head K scale": (q, k, v, out, s, t, torch.ones(4, device="cuda"), one),
            "uneven verify tokens": (q[:-1], k, v, out[:-1], s, t, one, one),
        }
        for name, args in rejected.items():
            with self.subTest(case=name):
                self.assertFalse(self.ops.can_verify_tp4(*args))


if __name__ == "__main__":
    unittest.main()
