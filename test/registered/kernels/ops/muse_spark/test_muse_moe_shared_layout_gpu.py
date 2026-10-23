"""Explicit, opt-in gfx950 component validation using official weights/routes.

No synthetic checkpoint is substituted. Run separately for TP4/E64 and TP8/E32:
  PYTHONPATH=python SGLANG_MUSE_MOE_SINGLE_LAYOUT=1 SGLANG_MUSE_A4W4_MOE=1 \
    MUSE_MOE_BACKEND=flydsl python <this-file> --fixture /path/to/rank.pt \
    --baseline-root /path/to/untouched/official-source

Trusted torch.save fixture schema (CPU tensors are accepted):
  metadata: {source_commit: 93406f598d109f1991c34ac9e938dad51c05c34b,
             checkpoint: <official checkpoint identity>, tp: 4|8, ep: 4|8,
             local_start: <global expert start>, layer: <layer index>}
  experts: {w13_weight, w2_weight, w13_weight_scale, w2_weight_scale}
  norm_weight: tensor[4096]; eps: float; gain: float
  cases: {"1": {x, ids, scores}, "8": ..., "16": ..., "128": ...,
          "255": ..., "256": ..., "1024": ..., "1025": ..., "16384": ...}
Use actual captured activations/routes (topk=8). Cases may slice or repeat real
captured rows to cover the required sizes; declare that derivation in
metadata.input_scope. These are component stress cases, not workload profiles.
16384 is optional with --skip-prefill; other listed M values are required.
Route stress cases derive all-local/no-local ids from those captured routes.

Tests cover native A16 fallback at both topologies, compact A4 verify only at
TP8, A4 prefill thresholds, graph replay, A4 alias storage, and partial reload.
Use production configs, including K256 for single-layout A16 at 513..1024 rows.
Set --bench for 30-iteration synchronized median
latency AFTER correctness; no tuning or performance claim is made by this file.
This is a component test, not the parent's full-model relL2 < .02 validation.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

COMMIT = "93406f598d109f1991c34ac9e938dad51c05c34b"
ROOT = Path(__file__).resolve().parents[5]
PREFIX = "python/sglang/srt/rocm_muse_spark/"
FIXTURE = os.environ.get("MUSE_SINGLE_LAYOUT_GPU_FIXTURE")
BASELINE = os.environ.get("MUSE_SINGLE_LAYOUT_BASELINE_ROOT")
SKIP_PREFILL = False
BENCH = False


def baseline_module(root, relative, name):
    path = root / relative
    original = subprocess.check_output(
        ["git", "show", f"{COMMIT}:{relative}"], cwd=ROOT
    )
    if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(original).digest():
        raise RuntimeError(f"Reference differs from untouched baseline: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(
    FIXTURE and BASELINE, "requires explicit official fixture and untouched baseline"
)
class TestOfficialSharedLayoutGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from sglang.srt.rocm_muse_spark import muse_moe_flydsl, mxfp4_moe, single_layout

        cls.torch, cls.ops, cls.fly, cls.layout = (
            torch,
            mxfp4_moe,
            muse_moe_flydsl,
            single_layout,
        )
        if not torch.cuda.is_available() or not torch.cuda.get_device_properties(
            0
        ).gcnArchName.startswith("gfx950"):
            raise unittest.SkipTest("requires gfx950")
        if (
            os.environ.get("SGLANG_MUSE_MOE_SINGLE_LAYOUT") != "1"
            or os.environ.get("MUSE_MOE_BACKEND") != "flydsl"
            or not mxfp4_moe.A4W4_ENABLED
        ):
            raise RuntimeError(
                "Set single-layout=1, A4W4=1 and backend=flydsl before import"
            )
        fixture = torch.load(FIXTURE, map_location="cpu", weights_only=True)
        meta = fixture["metadata"]
        assert (
            meta["source_commit"] == COMMIT
            and meta["checkpoint"]
            and meta["layer"] not in (0, 61)
        )
        assert (meta["tp"], meta["ep"]) in ((4, 4), (8, 8))
        print(json.dumps({"fixture": str(FIXTURE), "metadata": meta}), flush=True)
        cls.e = 256 // meta["ep"]
        cls.start = int(meta["local_start"])
        cls.norm = fixture["norm_weight"].cuda()
        cls.eps, cls.gain = float(fixture["eps"]), float(fixture["gain"])
        cls.native = SimpleNamespace(
            num_local_experts=cls.e, expert_hidden_size=4096, intermediate_size=4096
        )
        cls.shared = SimpleNamespace(
            num_local_experts=cls.e, expert_hidden_size=4096, intermediate_size=4096
        )
        for name in single_layout.NAMES:
            value = fixture["experts"][name].cuda().contiguous()
            setattr(cls.native, name, value)
            setattr(
                cls.shared, name, torch.nn.Parameter(value.clone(), requires_grad=False)
            )
        single_layout.initialize(cls.shared, True)
        single_layout.validate_scope(
            cls.shared, tp=meta["tp"], ep=meta["ep"], moe_tp=1, arch="gfx950"
        )
        cls.ptrs = [getattr(cls.shared, n).data_ptr() for n in single_layout.NAMES]
        cls.required = [1, 8, 16, 128, 255, 256, 1024, 1025] + (
            [] if SKIP_PREFILL else [16384]
        )
        cls.cases = fixture["cases"]  # Keep captured cases on CPU until needed.
        assert all(str(m) in cls.cases for m in cls.required), (
            "missing required captured M"
        )
        root = Path(BASELINE)
        cls.ref_fly = baseline_module(
            root, PREFIX + "muse_moe_flydsl.py", "official_native_a16"
        )
        cls.ref_ops = baseline_module(
            root, PREFIX + "mxfp4_moe.py", "official_mxfp4_moe"
        )
        cls.ref_compact = baseline_module(
            root,
            "python/sglang/kernels/ops/moe/mxfp4_compact_a4_moe.py",
            "official_compact_a4",
        )
        # Reference module-local bindings point at untouched source; installed
        # AITER is never modified or monkeypatched.
        cls.ref_ops.compact_a4_experts = cls.ref_compact.compact_a4_experts
        with torch.no_grad():
            cls.ref_ops.prepare_a4w4_weights(cls.native)
            mxfp4_moe.prepare_a4w4_weights(cls.shared)
        for actual, expected in zip(cls.shared.a4w4_weights, cls.native.a4w4_weights):
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), (
                "actual AITER shuffle mismatch"
            )
        aliases = [t.data_ptr() for t in cls.shared.a4w4_weights]
        assert aliases == cls.ptrs, "A4 tuple must alias Parameters"

    def case(self, m, routes="captured"):
        torch = self.torch
        case = self.cases[str(m)]
        x, ids, scores = [case[k].cuda().contiguous() for k in ("x", "ids", "scores")]
        self.assertEqual(tuple(x.shape), (m, 4096))
        self.assertEqual(tuple(ids.shape), (m, 8))
        self.assertEqual(tuple(scores.shape), (m, 8))
        if routes == "all-local":
            ids = ids.remainder(self.e) + self.start
        elif routes == "no-local":
            ids = torch.full_like(ids, (self.start + self.e) % 256)
        return x, ids, scores

    def invoke(self, role, case, shared):
        x, ids, scores = case
        obj = self.shared if shared else self.native
        if role == "native-a16":
            fn = self.fly.muse_moe_flydsl if shared else self.ref_fly.muse_moe_flydsl
            kwargs = {
                "eps": self.eps,
                "post_norm_gain": self.gain,
                "has_norm": True,
                "w13_scale": obj.w13_weight_scale,
                "w2_scale": obj.w2_weight_scale,
            }
            if shared:
                kwargs["single_layout"] = True
            return fn(
                x,
                scores,
                ids,
                obj.w13_weight,
                obj.w2_weight,
                self.start,
                self.norm,
                **kwargs,
            )
        ops = self.ops if shared else self.ref_ops
        fn = ops.run_compact_a4 if role == "compact-a4" else ops.run_a4w4
        return fn(x, scores, ids, obj, self.start, self.norm, self.eps, self.gain)

    def check(self, actual, expected, label):
        torch = self.torch
        self.assertTrue(bool(torch.isfinite(actual).all()), label)
        self.assertTrue(bool(torch.isfinite(expected).all()), label)
        diff = (actual.float() - expected.float()).double().norm().item()
        norm = expected.double().norm().item()
        rel = diff / max(norm, 1e-30)
        snr = math.inf if diff == 0 else 20 * math.log10(max(norm, 1e-30) / diff)
        exact = torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        print(
            json.dumps(
                {"case": label, "byte_exact": exact, "rel_l2": rel, "snr_db": snr}
            ),
            flush=True,
        )
        self.assertLess(rel, 0.005, label)
        self.assertGreaterEqual(snr, 30, label)
        self.assertTrue(exact, f"address-only change must be byte exact: {label}")

    def bench(self, fn):
        for _ in range(3):
            fn()
        self.torch.cuda.synchronize()
        times = []
        for _ in range(30):
            start = time.perf_counter()
            fn()
            self.torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        return statistics.median(times)

    def test_role_matrix(self):
        for m in self.required:
            # Compact eligibility is the original TP8/32, multiple-of-8 <=512.
            roles = ["native-a16"]
            if self.e == 32 and m % 8 == 0 and m <= 512:
                roles.append("compact-a4")
            if m >= 256:
                roles.append("a4-prefill")
            for routes in ("captured", "all-local", "no-local"):
                case = self.case(m, routes)
                for role in roles:
                    with self.subTest(m=m, routes=routes, role=role):
                        expected = self.invoke(role, case, False)
                        actual = self.invoke(role, case, True)
                        self.check(actual, expected, f"E{self.e}/M{m}/{routes}/{role}")
                        if BENCH and routes == "captured":
                            baseline = self.bench(
                                lambda role=role, case=case: self.invoke(
                                    role, case, False
                                )
                            )
                            candidate = self.bench(
                                lambda role=role, case=case: self.invoke(
                                    role, case, True
                                )
                            )
                            print(
                                json.dumps(
                                    {
                                        "E": self.e,
                                        "M": m,
                                        "role": role,
                                        "median_30_baseline_ms": baseline,
                                        "median_30_candidate_ms": candidate,
                                    }
                                ),
                                flush=True,
                            )

    def test_a16_schedule_boundaries(self):
        from sglang.srt.rocm_muse_spark import muse_moe_single_layout_gemm as gemm

        # Slice captured fixture rows to exercise both sides of the K-tile
        # change and the transition to the larger LDS schedule. Observe the
        # actual compiler arguments, not just the host config dictionary.
        source = self.case(1025)
        launcher = gemm._launcher
        for m, block_m, unroll in (
            (16, 16, 16),
            (512, 16, 16),
            (513, 32, 8),
            (1023, 32, 8),
            (1024, 32, 8),
            (1025, 64, 4),
        ):
            case = tuple(t[:m].contiguous() for t in source)
            calls = []

            def record(*args, _calls=calls, **kwargs):
                _calls.append((args[3], args[10]))
                return launcher(*args, **kwargs)

            expected = self.invoke("native-a16", case, False)
            with patch.object(gemm, "_launcher", side_effect=record):
                actual = self.invoke("native-a16", case, True)
            self.assertEqual(calls, [(block_m, unroll)] * 2)
            self.check(actual, expected, f"E{self.e}/M{m}/a16-schedule-boundary")

    def test_z_graph_replay_and_partial_reload(self):
        torch = self.torch
        for update_row, (role, m) in enumerate(
            [("native-a16", 8), ("a4-prefill", 256)]
            + ([("compact-a4", 8)] if self.e == 32 else []),
            start=1,
        ):
            case = self.case(m)
            # Warm compilation and allocations on a side stream before capture.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.invoke(role, case, True)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self.invoke(role, case, True)
            expected = self.invoke(role, case, False)
            for _ in range(3):
                graph.replay()
                self.check(output, expected, f"{role}/graph-replay")
            torch.cuda.synchronize()  # quiescent reload
            with torch.no_grad():
                self.layout.begin_load(self.shared)
                # Update one real checkpoint row, not all tensors; preserve all
                # untouched rows. Apply the identical update to the reference.
                self.shared.w2_weight[0, 0].copy_(self.native.w2_weight[0, update_row])
                self.native.w2_weight[0, 0].copy_(self.native.w2_weight[0, update_row])
                self.layout.finish_load(self.shared)
                self.ops.prepare_a4w4_weights(self.shared)
                self.ops.prepare_a4w4_weights(self.shared)  # idempotent
                self.ref_ops.prepare_a4w4_weights(self.native)
            self.assertEqual(
                self.ptrs,
                [getattr(self.shared, n).data_ptr() for n in self.layout.NAMES],
            )
            self.assertEqual(
                self.ptrs, [t.data_ptr() for t in self.shared.a4w4_weights]
            )
            expected = self.invoke(role, case, False)
            graph.replay()
            self.check(output, expected, f"{role}/graph-after-partial-reload")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fixture", default=FIXTURE)
    parser.add_argument("--baseline-root", default=BASELINE)
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--bench", action="store_true")
    args, remaining = parser.parse_known_args()
    FIXTURE, BASELINE, SKIP_PREFILL, BENCH = (
        args.fixture,
        args.baseline_root,
        args.skip_prefill,
        args.bench,
    )
    # unittest's decorator is evaluated before command-line parsing.
    TestOfficialSharedLayoutGPU.__unittest_skip__ = not bool(FIXTURE and BASELINE)
    unittest.main(argv=[__file__, *remaining])
