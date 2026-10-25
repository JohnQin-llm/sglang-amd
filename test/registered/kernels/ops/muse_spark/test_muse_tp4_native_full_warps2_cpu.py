"""CPU-only execution of actual dispatch and launch adapter; no torch/GPU import."""

import ast
import json
from pathlib import Path
import subprocess
import types

ROOT = Path(__file__).resolve().parents[5]
BACKEND = "python/sglang/srt/layers/attention/aiter_backend.py"
KERNEL = "python/sglang/kernels/ops/attention/gqa_verify_splitkv.py"
BASE = "93406f598d109f1991c34ac9e938dad51c05c34b"


class Tensor:
    def __init__(self, shape, dtype, stride=None, device="cuda:0"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        if stride is None:
            stride = []
            n = 1
            for size in reversed(shape):
                stride.append(n)
                n *= size
            stride = tuple(reversed(stride))
        self._stride = tuple(stride)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def is_cuda(self):
        return self.device.startswith("cuda:")

    def stride(self, d=None):
        return self._stride if d is None else self._stride[d]

    def numel(self):
        n = 1
        for size in self.shape:
            n *= size
        return n

    def __len__(self):
        return self.shape[0]


class Jit:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def call(*a, **kw):
            self.calls.append((grid, a, kw))

        return call


def extract(text, names, ns):
    tree = ast.parse(text)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    for n in nodes:
        n.decorator_list = []
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "<actual source>", "exec"), ns
    )


def fixture(B=8, H=32):
    return dict(
        q=Tensor((B * 8, H, 64), "bf16"),
        k=Tensor((50000, 16, H // 8, 64), "fp8"),
        v=Tensor((50000, 16, H // 8, 64), "fp8"),
        out=Tensor((B * 8, H, 64), "bf16"),
        cu=Tensor((B + 1,), "i32"),
        seq=Tensor((B,), "i64"),
        table=Tensor((B, 2064), "i32"),
        kd=Tensor((1,), "f32"),
        vd=Tensor((1,), "f32"),
        scale=0.125,
    )


def main():
    text = (ROOT / BACKEND).read_text()
    kernel = (ROOT / KERNEL).read_text()
    torch = types.SimpleNamespace(
        bfloat16="bf16",
        float8_e4m3fn="fp8",
        int32="i32",
        int64="i64",
        float32="f32",
        empty=lambda shape, **kw: Tensor(shape, kw["dtype"], device=kw["device"]),
    )
    producer, reducer = Jit(), Jit()
    ns = {
        "torch": torch,
        "HD": 64,
        "GQA": 8,
        "QL": 8,
        "_verify_partials": producer,
        "_verify_reduce": reducer,
    }
    extract(kernel, ["pick_splits", "gqa_verify_splitkv"], ns)
    calls = []
    real_wrapper = ns["gqa_verify_splitkv"]

    def launch(*a, **kw):
        calls.append(("folded", kw))
        return real_wrapper(*a, **kw)

    def stub(name):
        def call(*a, **kw):
            calls.append((name, kw))

        return call

    ns.update(
        gqa_verify_splitkv=launch,
        swa_verify_splitkv=stub("swa"),
        gqa_verify_splitkv_2kv=stub("ep8"),
        gqa_verify_splitkv_4kv=stub("4kv"),
        can_gqa_verify_splitkv_4kv=lambda *a: False,
        fp8_dtype="fp8",
        _MUSE_VERIFY_FULL_GQA=(8, 64, 64, 4),
        _muse_verify_v2_logged=set(),
        pick_splits_v2=lambda b: 64,
        logger=types.SimpleNamespace(info=lambda *a: None),
    )
    extract(
        text,
        [
            "_muse_tp4_native_low_warps_context",
            "_muse_tp4_native_low_warps_abi",
            "_muse_verify_attention",
        ],
        ns,
    )
    checks = []

    def run(f, kernels, window=(-1, -1), qlen=8, softcap=0, sinks=None):
        calls.clear()
        producer.calls.clear()
        reducer.calls.clear()
        ret = ns["_muse_verify_attention"](
            f["q"],
            f["k"],
            f["v"],
            f["out"],
            f["cu"],
            qlen,
            f["seq"],
            f["table"],
            f["scale"],
            window,
            softcap,
            f["kd"],
            f["vd"],
            sinks,
            frozenset(kernels),
        )
        return ret, list(calls), list(producer.calls), list(reducer.calls)

    for B in (1, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17):
        f = fixture(B)
        ret, c, p, red = run(f, {"full_gqa", "full_gqa_tp4_low_warps"})
        assert ret == (B >= 8)
        if B >= 8:
            assert p[-1][0] == (4, B * 64) and len(red) == 1
            assert p[-1][2] == (
                {"num_warps": 1 if B == 8 else 2, "num_stages": 2}
                if B in (8, 10, 12, 14, 16)
                else {"num_warps": 4, "num_stages": 2}
            )
            assert red[-1][2] == {"num_warps": 2}
        checks.append(f"actual_caller_and_JIT_B{B}")
    f = fixture()
    ret, c, p, red = run(f, {"full_gqa"})
    assert p[-1][2] == {"num_warps": 4, "num_stages": 2}
    checks.append("no_context_token_keeps_default")
    mutations = {
        "Qdtype": lambda f: setattr(f["q"], "dtype", "f32"),
        "outdtype": lambda f: setattr(f["out"], "dtype", "f32"),
        "KVdtype": lambda f: (
            setattr(f["k"], "dtype", "fnuz"),
            setattr(f["v"], "dtype", "fnuz"),
        ),
        "Qstride": lambda f: setattr(f["q"], "_stride", (4096, 64, 1)),
        "outstride": lambda f: setattr(f["out"], "_stride", (4096, 64, 1)),
        "Kstride": lambda f: setattr(f["k"], "_stride", (8192, 256, 64, 1)),
        "Vstride": lambda f: setattr(f["v"], "_stride", (8192, 256, 64, 1)),
        "seqdtype": lambda f: setattr(f["seq"], "dtype", "i32"),
        "cudtype": lambda f: setattr(f["cu"], "dtype", "i64"),
        "tablestride": lambda f: setattr(f["table"], "_stride", (4128, 2)),
        "tabledtype": lambda f: setattr(f["table"], "dtype", "i64"),
        "seqstride": lambda f: setattr(f["seq"], "_stride", (2,)),
        "custride": lambda f: setattr(f["cu"], "_stride", (2,)),
        "scale": lambda f: f.update(scale=0.25),
        "noneKscale": lambda f: f.update(kd=None),
        "noneVscale": lambda f: f.update(vd=None),
        "scale_dtype": lambda f: setattr(f["kd"], "dtype", "bf16"),
        "scale_vector": lambda f: f.update(kd=Tensor((4,), "f32")),
        "foreign_device": lambda f: setattr(f["table"], "device", "cuda:1"),
        "cpu_device": lambda f: setattr(f["q"], "device", "cpu"),
    }
    for name, mut in mutations.items():
        f = fixture()
        mut(f)
        ret, c, p, red = run(f, {"full_gqa", "full_gqa_tp4_low_warps"})
        assert all(row[2] == {"num_warps": 4, "num_stages": 2} for row in p), name
        checks.append("negative_" + name)
    for B in (4, 8, 12, 16):
        f = fixture(B)
        ret, c, p, red = run(
            f, {"swa_gqa", "swa_splitkv", "full_gqa_tp4_low_warps"}, window=(2046, 0)
        )
        assert all(row[2] == {"num_warps": 4, "num_stages": 2} for row in p)
        checks.append(f"SWA_unchanged_B{B}")
        f = fixture(B, 16)
        ret, c, p, red = run(f, {"full_v2", "full_gqa", "full_gqa_tp4_low_warps"})
        assert c[0][0] == "ep8" and not p
        checks.append(f"EP8_unchanged_B{B}")
    for opts in ({"qlen": 1}, {"softcap": 1}, {"sinks": object()}, {"window": (0, 0)}):
        ret, c, p, red = run(fixture(), {"full_gqa", "full_gqa_tp4_low_warps"}, **opts)
        assert not p
        checks.append("unsupported_mode_" + str(list(opts)))
    parallel = types.SimpleNamespace(
        tp_size=4, attn_tp_size=4, moe_ep_size=4, attn_dcp_size=1
    )
    props = types.SimpleNamespace(gcnArchName="gfx950:sramecc+:xnack-")
    ns["get_parallel"] = lambda: parallel
    torch.cuda = types.SimpleNamespace(
        current_device=lambda: 0, get_device_properties=lambda _: props
    )
    runner = types.SimpleNamespace(
        is_draft_worker=False,
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(
                architectures=["MuseSparkForConditionalGeneration"]
            )
        ),
    )
    ctx = ns["_muse_tp4_native_low_warps_context"]
    assert ctx(runner, {"full_gqa"})
    for field, value in [
        ("tp_size", 8),
        ("attn_tp_size", 8),
        ("moe_ep_size", 8),
        ("attn_dcp_size", 2),
    ]:
        old = getattr(parallel, field)
        setattr(parallel, field, value)
        assert not ctx(runner, {"full_gqa"})
        setattr(parallel, field, old)
        checks.append("context_" + field)
    for arch in ("gfx942", "gfx951", "gfx1100"):
        props.gcnArchName = arch
        assert not ctx(runner, {"full_gqa"})
        checks.append("context_" + arch)
    props.gcnArchName = "gfx950"
    runner.is_draft_worker = True
    assert not ctx(runner, {"full_gqa"})
    runner.is_draft_worker = False
    runner.model_config.hf_config.architectures = ["Other"]
    assert not ctx(runner, {"full_gqa"})
    runner.model_config.hf_config.architectures = ["MuseSparkForConditionalGeneration"]
    assert not ctx(runner, set())
    checks.extend(["context_draft", "context_other_model", "context_full_gqa_disabled"])
    base = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", BASE + ":" + KERNEL], text=True
    )
    trees = [ast.parse(x) for x in (base, kernel)]
    for name in ("_verify_partials", "_verify_reduce", "pick_splits"):
        a, b = [
            next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == name)
            for t in trees
        ]
        assert ast.dump(a) == ast.dump(b), name
        checks.append("unchanged_AST_" + name)
    assert kernel == base
    checks.append("entire_native_kernel_module_byte_unchanged")
    print(
        json.dumps(
            {
                "status": "passed_CPU_contracts_only",
                "checks": checks,
                "count": len(checks),
                "GPU_executed": False,
                "runtime_qualified": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
