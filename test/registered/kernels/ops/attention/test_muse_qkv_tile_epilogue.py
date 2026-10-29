"""TP4 Muse Spark fused QKV projection with its QK-norm / RoPE / FP8-cache
epilogue run by the last split-K program of each output tile (tile_epilogue),
against the projection followed by the separate epilogue kernel."""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

Q_HEADS, KV_HEADS, HEAD_DIM, HIDDEN = 32, 4, 64, 8192
OUT_FEATURES = 2 * (Q_HEADS + KV_HEADS) * HEAD_DIM  # Q, K, V and the output gate
EPS = 1e-5  # rms_norm_eps
Q_SCALE = 5.4730064863875  # Muse Spark query scale
SLOTS = 65536
WIDTHS = (8, 16, 24, 32, 40, 48)


def rope_cache(max_position=262144, theta=500000.0):
    inv = 1.0 / theta ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM)
    freqs = torch.outer(torch.arange(max_position).float(), inv)
    return torch.cat([freqs.cos(), freqs.sin()], dim=1).cuda()


def make_inputs(m, seed, sliding):
    """A verify step's attention input, positions and cache slots. One row is
    graph padding (slot -1); on sliding layers one slot has no SWA slot."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, HIDDEN, generator=g).to(torch.bfloat16)
    positions = torch.randint(0, 262144, (m,), generator=g)
    locations = torch.randperm(SLOTS, generator=g)[:m]
    locations[m // 3] = -1
    mapping = None
    if sliding:
        mapping = torch.randperm(SLOTS, generator=g)
        mapping[locations[0]] = -1
        mapping = mapping.cuda()
    return x.cuda(), positions.cuda(), locations.cuda(), mapping


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "gfx950 TP4 QKV epilogue")
class TestMuseQkvTileEpilogue(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.attention import qk_norm_rope_cache

        cls.ops = qk_norm_rope_cache
        g = torch.Generator().manual_seed(0)
        weight = torch.randn(OUT_FEATURES, HIDDEN, generator=g) * 0.02
        cls.weight = weight.to(torch.bfloat16).cuda()
        cls.cos_sin = rope_cache()
        cls.k_scale = torch.tensor([0.0371], device="cuda")
        cls.v_scale = torch.tensor([0.0213], device="cuda")

    def project(self, x, positions, locations, mapping, rope, caches, tile_epilogue):
        return self.ops.qkv_proj_qk_norm_rope_cache(
            x,
            self.weight,
            positions,
            self.cos_sin if rope else None,
            *caches,
            locations,
            mapping,
            EPS,
            Q_SCALE,
            self.k_scale,
            self.v_scale,
            num_q_heads=Q_HEADS,
            num_kv_heads=KV_HEADS,
            tile_epilogue=tile_epilogue,
        )

    @staticmethod
    def empty_caches():
        shape = (SLOTS, KV_HEADS, HEAD_DIM)
        return [
            torch.zeros(shape, dtype=torch.float8_e4m3fn, device="cuda")
            for _ in range(2)
        ]

    def assert_same(self, got, expected, got_caches, expected_caches, msg):
        for name, a, b in zip(("q", "k", "v", "gate"), got, expected):
            self.assertTrue(torch.equal(a, b), f"{msg}: {name}")
        for name, a, b in zip(("k cache", "v cache"), got_caches, expected_caches):
            self.assertTrue(
                torch.equal(a.view(torch.uint8), b.view(torch.uint8)), f"{msg}: {name}"
            )

    def test_matches_two_kernel_path(self):
        """Sliding layers (RoPE, full -> SWA slot map) and full layers (no RoPE)."""
        for sliding in (True, False):
            for m in WIDTHS:
                x, positions, locations, mapping = make_inputs(m, m, sliding)
                caches = {name: self.empty_caches() for name in ("two", "one")}
                expected = self.project(
                    x, positions, locations, mapping, sliding, caches["two"], False
                )
                got = self.project(
                    x, positions, locations, mapping, sliding, caches["one"], True
                )
                self.assert_same(
                    got,
                    expected,
                    caches["one"],
                    caches["two"],
                    f"sliding={sliding} m={m}",
                )

    def test_graph_replay(self):
        """Replays read new inputs, and every launch leaves its tickets ready for
        the next one."""
        for m in (16, 48):
            inputs = [make_inputs(m, seed, True) for seed in (1, 2)]
            x, positions, locations, mapping = (t.clone() for t in inputs[0])
            caches = self.empty_caches()
            self.project(x, positions, locations, mapping, True, caches, True)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = self.project(
                    x, positions, locations, mapping, True, caches, True
                )
            for replay in range(6):
                src = inputs[replay % 2]
                for dst, t in zip((x, positions, locations, mapping), src):
                    dst.copy_(t)
                for cache in caches:
                    cache.zero_()
                for out in outputs:
                    out.fill_(float("nan"))
                graph.replay()
                reference = self.empty_caches()
                expected = self.project(*src, True, reference, False)
                self.assert_same(
                    outputs, expected, caches, reference, f"m={m} replay {replay}"
                )


if __name__ == "__main__":
    unittest.main()
