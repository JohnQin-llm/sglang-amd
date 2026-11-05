"""AOT-compile attn3_split for gfx1250 (no GPU) and dump amdgcn / ttgir + stats.
usage: aot.py TILE=64 NSEG=128 NBUF=4 [num_warps=4] --out name [--kern attn3_split] [--mod kern.attn3]"""
import importlib
import os
import re
import sys

sys.path.insert(0, "/data/muse455/attn3")
os.environ.setdefault("TRITON_CACHE_DIR", "/data/muse455/attn3/tcache_aot")
import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.experimental.gluon._runtime import GluonASTSource  # noqa: E402

args = sys.argv[1:]


def opt(name, default):
    return args[args.index(name) + 1] if name in args else default


mod, kname, out = opt("--mod", "kern.attn3"), opt("--kern", "attn3_split"), opt("--out", "a")
kv = dict(a.split("=") for a in args if "=" in a and not a.startswith("--"))
fn = getattr(importlib.import_module(mod), kname)
nw = int(kv.pop("num_warps", 4))
wpe = int(kv.pop("waves_per_eu", 0))
const = dict(NQ=32, NKV=4, G=8, D=64, PAGE=16, BLOCK_M=64)
const.update({k: int(v) for k, v in kv.items()})
names = fn.arg_names
ptrs = {"segm_out_ptr": "*fp32", "segm_max_ptr": "*fp32", "segm_sum_ptr": "*fp32", "q_ptr": "*bf16",
        "k_ptr": "*fp8e4nv", "v_ptr": "*fp8e4nv", "bt_ptr": "*i32", "seq_lens_ptr": "*i64", "cu_q_ptr": "*i32",
        "k_descale_ptr": "*fp32", "v_descale_ptr": "*fp32", "out_ptr": "*bf16", "cnt_ptr": "*i32"}
sig = {}
for n in names:
    if n in const:
        sig[n] = "constexpr"
    elif n in ptrs:
        sig[n] = ptrs[n]
    elif n == "scale":
        sig[n] = "fp32"
    else:
        sig[n] = "i32"
div = [names.index(n) for n in names if n in ptrs or n.endswith("stride") or n.startswith("q_stride")]
src = GluonASTSource(fn=fn, signature=sig, constexprs={(names.index(k),): v for k, v in const.items() if k in names},
                     attrs={(i,): [["tt.divisibility", 16]] for i in div})
k = triton.compile(src, target=GPUTarget("hip", "gfx1250", 32), options=dict(num_warps=nw, waves_per_eu=wpe))
a = k.asm["amdgcn"]
os.makedirs("/data/muse455/attn3/asm", exist_ok=True)
open(f"/data/muse455/attn3/asm/{out}.amdgcn", "w").write(a)
open(f"/data/muse455/attn3/asm/{out}.ttgir", "w").write(k.asm["ttgir"])
open(f"/data/muse455/attn3/asm/{out}.llir", "w").write(k.asm["llir"])
st = {m: (re.search(rf"; {m}: (\S+)", a) or re.search(rf"\.{m.lower()}[^,\n]*,? (\S+)", a)) for m in ("NumVgprs", "NumSgprs", "Occupancy", "ScratchSize")}
st = {k: (v.group(1) if v else None) for k, v in st.items()}
print(out, st, "shared", k.metadata.shared)
