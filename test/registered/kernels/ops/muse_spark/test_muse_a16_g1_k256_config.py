# ruff: noqa: S102  # Execute only selected AST nodes from the checked-in host code.
"""CPU selector/launch contracts; no torch, FlyDSL, or GPU imports.

Run: python test/registered/kernels/ops/muse_spark/test_muse_a16_g1_k256_config.py
Numerical and performance qualification remains a separate GPU gate.
"""

import ast
import itertools
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "python/sglang/srt/rocm_muse_spark/muse_moe_flydsl.py"
TREE = ast.parse(SOURCE.read_text())
FUNCTIONS = {node.name: node for node in TREE.body if isinstance(node, ast.FunctionDef)}
CONFIG_NODES = [
    node
    for node in TREE.body
    if isinstance(node, ast.Assign)
    and any(
        isinstance(target, ast.Name)
        and target.id.startswith(("_SMALL_M", "_MID_M", "_LARGE_M"))
        for target in node.targets
    )
]
NAMESPACE = {"os": SimpleNamespace(environ={})}
exec(
    compile(
        ast.Module(
            CONFIG_NODES
            + [FUNCTIONS["_pick_flydsl_config"], FUNCTIONS["_launch_gemm"]],
            type_ignores=[],
        ),
        str(SOURCE),
        "exec",
    ),
    NAMESPACE,
)
PICK = NAMESPACE["_pick_flydsl_config"]


class TestA16G1K256Config(unittest.TestCase):
    def test_qualified_component_config_exact(self):
        # Full dictionary from the qualified G1-only component arm. G2 still
        # consumes the shared K512 setting; no other tile choice may change.
        expected = {
            "BLOCK_M": 16,
            "BLOCK_N": 128,
            "BLOCK_K": 128,
            "GROUP_M": 4,
            "num_warps": 2,
            "num_stages": 2,
            "FLYDSL_K_UNROLL": 16,
            "GEMM1_FLYDSL_K_UNROLL": 8,
        }
        for rows in (24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128):
            with self.subTest(rows=rows):
                self.assertEqual(
                    PICK(rows, 8, 64, True, True, shape=(4096, 4096, 4096)),
                    expected,
                )
                legacy = PICK(rows, 8, 64, True, True)
                self.assertEqual(
                    legacy,
                    {k: v for k, v in expected.items() if k != "GEMM1_FLYDSL_K_UNROLL"},
                )

    def test_unqualified_boundaries_and_shapes_keep_legacy_config(self):
        rows = (
            0,
            1,
            4,
            8,
            16,
            23,
            24,
            25,
            31,
            32,
            33,
            40,
            48,
            56,
            64,
            72,
            79,
            80,
            81,
            88,
            95,
            96,
            97,
            104,
            111,
            112,
            113,
            120,
            127,
            128,
            129,
            256,
            512,
            513,
            1024,
            2048,
        )
        shapes = (
            None,
            (2048, 4096, 4096),
            (4096, 2048, 4096),
            (4096, 4096, 2048),
            (8192, 8192, 8192),
            (4096, 4096, 4096),
        )
        for m, topk, experts, fp4, sl1, shape in itertools.product(
            rows, (4, 8), (32, 64, 128), (False, True), (False, True), shapes
        ):
            if (
                m in (24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128)
                and topk == 8
                and experts == 64
                and fp4
                and sl1
                and shape == (4096, 4096, 4096)
            ):
                continue
            with self.subTest(M=m, topk=topk, E=experts, fp4=fp4, sl1=sl1, shape=shape):
                self.assertEqual(
                    PICK(m, topk, experts, fp4, sl1, shape=shape),
                    PICK(m, topk, experts, fp4, sl1),
                )

    def test_production_passes_dimensions_only_for_automatic_config(self):
        # Execute the actual host selection statement: no tensor/device mocks
        # and no duplicate selection policy in this test.
        branch = next(
            node
            for node in FUNCTIONS["muse_moe_flydsl"].body
            if isinstance(node, ast.If) and ast.unparse(node.test) == "cfg is None"
        )
        code = compile(ast.Module([branch], type_ignores=[]), str(SOURCE), "exec")
        namespace = dict(
            NAMESPACE,
            M=24,
            topk=8,
            E=64,
            mxfp4=True,
            single_layout=True,
            K=4096,
            I=4096,
            H=4096,
            cfg=None,
        )
        exec(code, namespace)
        self.assertEqual(namespace["cfg"]["GEMM1_FLYDSL_K_UNROLL"], 8)
        for dim in ("K", "I", "H"):
            other = dict(namespace, cfg=None)
            other[dim] = 2048
            exec(code, other)
            self.assertNotIn("GEMM1_FLYDSL_K_UNROLL", other["cfg"])
        explicit = {"FLYDSL_K_UNROLL": 4, "GEMM1_FLYDSL_K_UNROLL": 2}
        namespace["cfg"] = explicit
        namespace["_pick_flydsl_config"] = lambda *a, **k: self.fail(
            "Explicit cfg selected"
        )
        exec(code, namespace)
        self.assertIs(namespace["cfg"], explicit)
        self.assertEqual(explicit, {"FLYDSL_K_UNROLL": 4, "GEMM1_FLYDSL_K_UNROLL": 2})

    def test_actual_stage_adapter_keeps_g2_and_input_config_unchanged(self):
        calls = []
        name = "sglang.srt.rocm_muse_spark.muse_moe_single_layout_gemm"
        module = ModuleType(name)
        module.launch_muse_moe_gemm = lambda *args: calls.append(args)
        cfg = PICK(24, 8, 64, True, True, shape=(4096, 4096, 4096))
        saved = dict(cfg)
        with patch.dict(sys.modules, {name: module}):
            for gate_up in (True, False):
                NAMESPACE["_launch_gemm"](
                    *([None] * 7),
                    4096,
                    4096,
                    192,
                    8,
                    gate_up,
                    gate_up,
                    True,
                    cfg,
                    single_layout=True,
                )
        self.assertEqual([call[-2]["FLYDSL_K_UNROLL"] for call in calls], [8, 16])
        self.assertEqual([call[-2]["BLOCK_N"] for call in calls], [128, 128])
        self.assertEqual([call[-1] for call in calls], [1, 1])
        self.assertEqual(cfg, saved)


if __name__ == "__main__":
    unittest.main()
