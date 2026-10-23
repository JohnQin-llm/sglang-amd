"""CPU address/lifecycle tests; no torch import or device use on the CPU path.

Run: python test/registered/kernels/ops/muse_spark/test_muse_moe_shared_layout.py -v
GPU component tests are in test_muse_moe_shared_layout_gpu.py (explicit fixture).
The optional pinned-vendor oracle uses MUSE_SINGLE_LAYOUT_VENDOR_ROOT.
"""

import ast
import hashlib
import importlib.util
import math
import os
import random
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "python/sglang/srt/rocm_muse_spark"
spec = importlib.util.spec_from_file_location(
    "single_layout", SOURCE / "single_layout.py"
)
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)
VENDOR = (
    Path(os.environ["MUSE_SINGLE_LAYOUT_VENDOR_ROOT"])
    if os.environ.get("MUSE_SINGLE_LAYOUT_VENDOR_ROOT")
    else None
)


def coordinates(index, shape):
    result = []
    for size in reversed(shape):
        result.append(index % size)
        index //= size
    assert index == 0
    return result[::-1]


def flatten(coords, shape):
    result = 0
    for coord, size in zip(coords, shape):
        result = result * size + coord
    return result


class SymbolicTensor:
    """Only shape/index algebra; each value is its original native byte index.

    This executes the real vendor reshape/permute functions without torch or
    storage allocation. It is not a GPU execution or numerical kernel test.
    """

    def __init__(self, shape, getter=None, dtype="torch.uint8"):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.get = getter or (lambda index: index)
        self.dtype = dtype
        self.device = SimpleNamespace(type="cuda")  # metadata for host guards only
        self.storage_identity = object()
        self.copies = 0

    def element_size(self):
        return 1

    def is_contiguous(self):
        return True

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], str):
            return SymbolicTensor(self.shape, self.get, shape[0])
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        shape = list(shape)
        if -1 in shape:
            shape[shape.index(-1)] = math.prod(self.shape) // -math.prod(shape)
        assert math.prod(shape) == math.prod(self.shape)
        return SymbolicTensor(shape, self.get, self.dtype)

    def permute(self, *axes):
        shape, getter = self.shape, self.get
        out_shape = tuple(shape[a] for a in axes)

        def get(index):
            out_coords = coordinates(index, out_shape)
            native = [0] * len(axes)
            for axis, coord in zip(axes, out_coords):
                native[axis] = coord
            return getter(flatten(native, shape))

        return SymbolicTensor(out_shape, get, self.dtype)

    def contiguous(self):
        return self

    def copy_(self, other):
        assert self.shape == other.shape
        self.get = other.get
        self.copies += 1
        return self

    def __setitem__(self, slices, other):
        # Supported dimensions require no padding: vendor's [:m,:n] fills all.
        assert tuple(s.stop for s in slices) == self.shape == other.shape
        self.copy_(other)


def official_offset(expert, n, k, e, rows, cols, kind):
    """Independent literal AITER reshape axes, NOT the collapsed builder map."""
    if kind == 0:
        shape = (e, rows // 16, 16, cols // 32, 2, 16)
        axes = (0, 1, 3, 4, 2, 5)
    elif kind == 1:
        shape = (e, rows // 16, 16, cols // 64, 4, 16)
        axes = (0, 1, 3, 4, 2, 5)
    elif kind == 2:
        shape = (e * rows // 32, 2, 16, cols // 8, 2, 4)
        axes = (0, 3, 5, 2, 4, 1)
    else:
        shape = (e, rows // 32, 2, 16, cols // 8, 2, 4)
        axes = (0, 1, 4, 6, 3, 5, 2)
    coords = coordinates((expert * rows + n) * cols + k, shape)
    return flatten([coords[a] for a in axes], [shape[a] for a in axes])


def sample_points(e, rows, cols):
    for expert in (0, 1, e - 1):
        for n in (0, 1, 15, 16, 31, 32, 255, 256, 4095, rows - 1):
            for k in sorted({0, 1, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, cols - 1}):
                yield expert, n, k
    rng = random.Random(1020)
    for _ in range(200):
        yield rng.randrange(e), rng.randrange(rows), rng.randrange(cols)


def experts(e=32):
    obj = SimpleNamespace(
        num_local_experts=e, expert_hidden_size=4096, intermediate_size=4096
    )
    for name, rows, cols in zip(
        layout.NAMES, (8192, 4096, 8192, 4096), (2048, 2048, 128, 128)
    ):
        setattr(obj, name, SymbolicTensor((e, rows, cols)))
    layout.initialize(obj, True)
    tp = 256 // e
    layout.validate_scope(obj, tp=tp, ep=tp, moe_tp=1, arch="gfx950")
    return obj


class TestMappings(unittest.TestCase):
    def test_four_actual_reshape_permutations_and_inverse(self):
        for e in (32, 64):
            for kind, (rows, cols) in enumerate(
                ((8192, 2048), (4096, 2048), (8192, 128), (4096, 128))
            ):
                tensor = SymbolicTensor((e, rows, cols))
                shuffled = layout._permuted(tensor, kind >= 2, False)
                restored = layout._permuted(shuffled, kind >= 2, True)
                offset = layout.scale_offset if kind >= 2 else layout.weight_offset
                for expert, n, k in sample_points(e, rows, cols):
                    native = (expert * rows + n) * cols + k
                    expected = official_offset(expert, n, k, e, rows, cols, kind)
                    self.assertEqual(
                        expert * rows * cols + offset(n, k, cols), expected
                    )
                    self.assertEqual(shuffled.get(expected), native)
                    self.assertEqual(restored.get(native), native)

    @unittest.skipUnless(
        VENDOR is not None and (VENDOR / "ops/shuffle.py").exists(),
        "set MUSE_SINGLE_LAYOUT_VENDOR_ROOT to the pinned AITER sources",
    )
    def test_execute_pinned_vendor_functions(self):
        source = (VENDOR / "ops/shuffle.py").read_bytes()
        self.assertEqual(
            hashlib.sha256(source).hexdigest(),
            "38f41cd233c529cba6d18f0654c267c3279df75d50295263ffa3f0df8f427b0b",
        )
        selected = {
            "shuffle_weight",
            "shuffle_weight_a16w4",
            "shuffle_scale",
            "shuffle_scale_a16w4",
        }
        nodes = [
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name in selected
        ]
        torch = SimpleNamespace(
            Tensor=SymbolicTensor,
            uint8="torch.uint8",
            float32="torch.float32",
            float4_e2m1fn_x2="fp4x2",
            empty=lambda *shape, dtype, device: SymbolicTensor(shape, dtype=dtype),
        )
        ns = {"torch": torch}
        exec(  # noqa: S102 - execute only local pinned/source AST for CPU tests
            compile(
                ast.Module(body=nodes, type_ignores=[]),
                str(VENDOR / "ops/shuffle.py"),
                "exec",
            ),
            ns,
        )
        # Execute the actual e8m0 wrapper too; its one import resolves to the
        # actual shuffle_scale extracted above, with no installed AITER import.
        fp4 = (VENDOR / "utility/fp4_utils.py").read_bytes()
        self.assertEqual(
            hashlib.sha256(fp4).hexdigest(),
            "6f4ecb4dd9eb22d30653264c851d000ce95b0ec8bc1fb40718b72c96eb3b70eb",
        )
        wrapper = next(
            n
            for n in ast.parse(fp4).body
            if isinstance(n, ast.FunctionDef) and n.name == "e8m0_shuffle"
        )
        wrapper.body = [n for n in wrapper.body if not isinstance(n, ast.ImportFrom)]
        exec(  # noqa: S102 - execute only local pinned/source AST for CPU tests
            compile(
                ast.Module(body=[wrapper], type_ignores=[]), "e8m0_shuffle", "exec"
            ),
            ns,
        )
        for e in (32, 64):
            tensors = [
                SymbolicTensor((e, 8192, 2048), dtype="fp4x2"),
                SymbolicTensor((e, 4096, 2048), dtype="fp4x2"),
                SymbolicTensor((e * 8192, 128)),
                SymbolicTensor((e * 4096, 128)),
            ]
            outputs = [
                ns["shuffle_weight"](tensors[0], (16, 16)),
                ns["shuffle_weight_a16w4"](tensors[1], 16, False),
                ns["e8m0_shuffle"](tensors[2]),
                ns["shuffle_scale_a16w4"](tensors[3], e, False),
            ]
            for kind, output in enumerate(outputs):
                rows, cols = (
                    (8192 if kind % 2 == 0 else 4096),
                    (2048 if kind < 2 else 128),
                )
                offset = layout.weight_offset if kind < 2 else layout.scale_offset
                for expert, n, k in sample_points(e, rows, cols):
                    self.assertEqual(
                        output.get(expert * rows * cols + offset(n, k, cols)),
                        (expert * rows + n) * cols + k,
                    )

    def test_native_vector_fragments_and_scale_dwords(self):
        for n in range(64):
            for k in range(0, 2048, 16):
                base = layout.weight_offset(n, k)
                self.assertEqual(base % 16, 0)
                self.assertEqual(
                    [layout.weight_offset(n, k + b) for b in range(16)],
                    list(range(base, base + 16)),
                )
            shuffled = layout._permuted(
                SymbolicTensor((1, 64, 128), lambda i: (i * 197 + i // 128) % 256),
                True,
                False,
            )
            for k in range(0, 128, 4):
                actual = sum(
                    shuffled.get(layout.scale_offset(n, k + b)) << (8 * b)
                    for b in range(4)
                )
                expected = sum(
                    ((n * 128 + k + b) * 197 + n) % 256 << (8 * b) for b in range(4)
                )
                self.assertEqual(actual, expected)


class TestLifecycle(unittest.TestCase):
    def test_partial_reload_stable_storage_and_idempotence(self):
        obj = experts()
        identities = [getattr(obj, n).storage_identity for n in layout.NAMES]
        layout.begin_load(obj)
        layout.finish_load(obj)
        layout.transform(obj)
        layout.transform(obj)
        self.assertEqual([getattr(obj, n).copies for n in layout.NAMES], [1] * 4)
        layout.begin_load(obj)
        for name in layout.NAMES:
            self.assertEqual(getattr(obj, name).get(13579), 13579)
        obj.w2_weight.get = lambda i: i + 7  # partial native checkpoint update
        layout.finish_load(obj)
        layout.transform(obj)
        self.assertEqual(
            identities, [getattr(obj, n).storage_identity for n in layout.NAMES]
        )
        self.assertEqual(
            obj.w2_weight.get(layout.weight_offset(31, 17)), 31 * 2048 + 17 + 7
        )
        self.assertEqual(
            obj.w13_weight.get(layout.weight_offset(31, 17)), 31 * 2048 + 17
        )
        layout.require_ready(obj)

    def test_scope_rejection_before_mutation(self):
        for tp, ep, e in ((4, 4, 64), (8, 8, 32)):
            obj = experts(e)
            layout.validate_scope(obj, tp=tp, ep=ep, moe_tp=1, arch="gfx950")
            for changes in ({"tp": 2}, {"ep": 1}, {"moe_tp": 2}, {"arch": "gfx942"}):
                args = {"tp": tp, "ep": ep, "moe_tp": 1, "arch": "gfx950"}
                args.update(changes)
                with self.assertRaises(ValueError):
                    layout.validate_scope(obj, **args)
            obj.w2_weight.dtype = "torch.int32"
            with self.assertRaises(ValueError):
                layout.validate_scope(obj, tp=tp, ep=ep, moe_tp=1, arch="gfx950")
            self.assertEqual([getattr(obj, n).copies for n in layout.NAMES], [0] * 4)

    def test_option_changes_and_unready_states_fail_closed(self):
        obj = experts()
        valid = {"enabled": True, "a4w4": True, "backend": "flydsl"}
        self.assertTrue(layout.check_options(obj, **valid))
        for changes in (
            {"enabled": False},
            {"a4w4": False},
            {"backend": "triton"},
            {"backend": "auto"},
            {"compact_unchanged": False},
        ):
            with self.assertRaises(RuntimeError):
                layout.check_options(obj, **(valid | changes))
        for state in ("native", "loading", "transforming", "failed"):
            obj._single_layout_state = state
            with self.assertRaises(RuntimeError):
                layout.require_ready(obj)
        with self.assertRaises(RuntimeError):
            layout.check_options(obj, **valid)
        with self.assertRaises(RuntimeError):
            layout.begin_load(obj)

    def test_interrupted_transform_is_terminal(self):
        obj = experts()

        def fail(_):
            raise RuntimeError("injected interrupted copy")

        obj.w2_weight.copy_ = fail
        with self.assertRaisesRegex(RuntimeError, "injected"):
            layout.transform(obj)
        self.assertEqual(obj._single_layout_state, "failed")
        self.assertIsNone(obj.a4w4_weights)
        with self.assertRaises(RuntimeError):
            layout.begin_load(obj)

    def test_real_load_wrapper_partial_and_interrupted_load(self):
        tree = ast.parse((SOURCE / "model.py").read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "MuseSparkTextForCausalLM"
        )
        node = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "load_weights"
        )
        node.decorator_list = []
        node.args.args[1].annotation = None
        ns = {"single_layout": layout}
        exec(  # noqa: S102 - execute only local pinned/source AST for CPU tests
            compile(
                ast.Module(body=[node], type_ignores=[]),
                "actual_model_load_wrapper",
                "exec",
            ),
            ns,
        )
        obj = experts()

        def load(weights):
            for name, getter in weights:
                getattr(obj, name).get = getter
            layout.finish_load(obj)
            layout.transform(obj)

        model = SimpleNamespace(
            _single_layout_modules=lambda: [obj], _load_weights_native=load
        )
        ns["load_weights"](model, [("w2_weight", lambda i: i + 9)])
        ns["load_weights"](model, [])
        self.assertEqual(
            obj.w2_weight.get(layout.weight_offset(17, 31)), 17 * 2048 + 31 + 9
        )

        def interrupted():
            yield "w13_weight", lambda i: 42
            raise RuntimeError("interrupted iterator")

        with self.assertRaisesRegex(RuntimeError, "interrupted iterator"):
            ns["load_weights"](model, interrupted())
        self.assertEqual(obj._single_layout_state, "failed")
        self.assertEqual(model._single_layout_state, "failed")
        self.assertFalse(model._single_layout_load_active)

    def test_unvalidated_transform_is_rejected(self):
        obj = experts()
        layout.initialize(obj, True)
        with self.assertRaisesRegex(RuntimeError, "scope must be validated"):
            layout.transform(obj)
        self.assertEqual([getattr(obj, n).copies for n in layout.NAMES], [0] * 4)
        layout.require_ready(SimpleNamespace())  # legacy default-off callers

    def test_actual_online_update_and_export_hooks(self):
        components = ROOT / "python/sglang/srt/model_executor/model_runner_components"
        functions = {}
        for filename, name in (
            ("weight_exporter.py", "_assert_weight_export_supported"),
            ("weight_updater.py", "_unsupported_derived_weight_cache_error"),
        ):
            tree = ast.parse((components / filename).read_text())
            node = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == name
            )
            node.returns = None
            for arg in node.args.args:
                arg.annotation = None
            exec(  # noqa: S102 - execute only local source host-hook AST
                compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"),
                functions,
            )
        obj = experts()
        model = SimpleNamespace(modules=lambda: [obj])
        self.assertEqual(
            functions["_unsupported_derived_weight_cache_error"](model), layout.ERROR
        )
        with self.assertRaisesRegex(
            RuntimeError, "cannot be updated online or exported"
        ):
            functions["_assert_weight_export_supported"](model)
        default = SimpleNamespace(modules=lambda: [SimpleNamespace()])
        functions["_assert_weight_export_supported"](default)

    def test_serialization_guards_and_default_off(self):
        class Base:
            def _save_to_state_dict(self, *args):
                return "native-save"

            def _load_from_state_dict(self, *args):
                return "native-load"

        class Model(layout.SingleLayoutStateDictGuard, Base):
            pass

        model = Model()
        layout.initialize(model, False)
        self.assertEqual(model._save_to_state_dict(), "native-save")
        self.assertEqual(model._load_from_state_dict(), "native-load")
        layout.initialize(model, True)
        for method in (model._save_to_state_dict, model._load_from_state_dict):
            with self.assertRaises(RuntimeError):
                method()
        self.assertEqual(model._derived_weight_cache_error, layout.ERROR)
        self.assertEqual(model._weight_export_error, layout.ERROR)


if __name__ == "__main__":
    unittest.main()
