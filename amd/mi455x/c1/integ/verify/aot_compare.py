"""AOT-compile (gfx1250, no GPU) the in-tree kernels and their measured sources; compare the amdgcn.

- attn_v2_split: /data/muse455/attn2/kern/attn_v2.py vs sglang.kernels.ops.attention.muse_attn_v2 (worktree),
  served launch config T128 NSEG128 QSPLIT0 FOLD1 w4 st2.
- _qkv_epilogue_split: vagm/gemm_bench.py's copy vs the worktree's, M8 N4608 SK8 HQ32 HK4 (with / without map).
"""
import importlib.util
import os
import re
import sys

os.environ["TRITON_CACHE_DIR"] = "/tmp/integ_aot_cache"
import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402

OUT = "/data/muse455/integ/verify/asm"
os.makedirs(OUT, exist_ok=True)
TGT = GPUTarget("hip", "gfx1250", 32)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def compile_(fn, ptrs, const, other="i32", nw=4, ns=2, wpe=0):
    names = fn.arg_names
    sig = {}
    for n in names:
        sig[n] = "constexpr" if n in const else ptrs.get(n, "i64" if n.startswith(("sk", "sv")) else other)
    div = [names.index(n) for n in names if (n in ptrs and ptrs[n].startswith("*")) or n.startswith(("sk", "sv", "q_stride"))]
    src = triton.compiler.ASTSource(fn=fn, signature=sig, constexprs={(names.index(k),): v for k, v in const.items()},
                                    attrs={(i,): [["tt.divisibility", 16]] for i in div})
    k = triton.compile(src, target=TGT, options=dict(num_warps=nw, num_stages=ns, waves_per_eu=wpe))
    return k.asm["amdgcn"]


def body(a):
    # drop names / paths / metadata that carry the module or file name
    keep = []
    for line in a.splitlines():
        if re.search(r"\.file|\.loc|file_name|\.amdhsa_kernel|\.name:|\.symbol:|^\s*\.(globl|type|size|section)|"
                     r"^[A-Za-z_][\w.$]*:|Lfunc_end|\.debug|; -- End|DW_|\.cfi", line):
            continue
        keep.append(line)
    return "\n".join(keep)


def cmp(tag, a, b):
    open(f"{OUT}/{tag}_ref.amdgcn", "w").write(a)
    open(f"{OUT}/{tag}_integ.amdgcn", "w").write(b)
    st = {m: re.search(rf"; {m}: (\S+)", b).group(1) for m in ("NumVgprs", "Occupancy", "ScratchSize")}
    same = body(a) == body(b)
    n = sum(1 for l in body(b).splitlines() if l.strip() and not l.strip().startswith(";"))
    print(f"{tag}: ISA identical to the measured source: {same}  ({n} lines, {st})", flush=True)
    return same


sys.path.insert(0, "/data/muse455/integ/sglang/python")
ref = load("/data/muse455/attn2/kern/attn_v2.py", "attn2_attn_v2")
integ = load("/data/muse455/integ/sglang/python/sglang/kernels/ops/attention/muse_attn_v2.py", "integ_attn_v2")
ptrs = {"segm_out_ptr": "*fp32", "segm_max_ptr": "*fp32", "segm_sum_ptr": "*fp32", "q_ptr": "*bf16",
        "k_ptr": "*fp8e4nv", "v_ptr": "*fp8e4nv", "bt_ptr": "*i32", "seq_lens_ptr": "*i64", "cu_q_ptr": "*i32",
        "k_descale_ptr": "*fp32", "v_descale_ptr": "*fp32", "scale": "fp32"}
const = dict(NQ=32, G=8, D=64, PAGE=16, BLOCK_M=64, TILE=128, NSEG=128, QSPLIT=0, FOLD=1, DUAL=0, PFD=0, PF_SCOPE=0)
ok = cmp("attn_v2_T128_S128", compile_(ref.attn_v2_split, ptrs, const), compile_(integ.attn_v2_split, ptrs, const))

# the vagm copy calls the served module's helpers; load the served file for it (read only)
served = load("/data/muse455/sglang/python/sglang/kernels/ops/attention/qk_norm_rope_cache.py", "served_qk")
src = open("/data/muse455/k2/cand/vagm/vagm/gemm_bench.py").read()
start = src.index("@triton.jit\ndef _qkv_epilogue_split")
end = src.index("def graph_time")
ns = {"triton": triton, "tl": triton.language, "Q": served}
tmp = "/tmp/vagm_epi_split.py"
open(tmp, "w").write("import triton\nimport triton.language as tl\nimport served_qk as Q\n" + src[start:end])
vepi = load(tmp, "vagm_epi_split")
from sglang.kernels.ops.attention import qk_norm_rope_cache as IQ  # noqa: E402

eptrs = {n: "*fp32" for n in ("P", "COS", "KS", "VS")}
eptrs.update({"POS": "*i64", "LOC": "*i64", "MAP": "*i64", "KC": "*fp8e4nv", "VC": "*fp8e4nv", "QO": "*bf16",
              "KO": "*bf16", "VO": "*bf16", "GO": "*bf16"})
for has_map in (0, 1):
    ec = dict(M=8, N=4608, SK=8, HQ=32, HK=4, D=64, EPS=1e-6, SCALE=0.125, ROPE=1, HAS_MAP=has_map, COS_STRIDE=64)
    ok &= cmp(f"qkv_epilogue_split_map{has_map}", compile_(vepi._qkv_epilogue_split, eptrs, ec, nw=1, ns=2),
              compile_(IQ._qkv_epilogue_split, eptrs, ec, nw=1, ns=2))
print("ALL_IDENTICAL" if ok else "DIFFERENT")
