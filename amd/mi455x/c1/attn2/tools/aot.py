"""AOT-compile a split kernel for gfx1250 (no GPU needed) and dump its amdgcn + stats.
usage: aot.py <module> <kernel> TILE=64 NSEG=128 num_warps=4 num_stages=1 [QSPLIT=0 FOLD=1 DUAL=0] --out name"""
import importlib, os, re, sys
sys.path.insert(0, "/data/muse455/attn2")
os.environ.setdefault("TRITON_CACHE_DIR", "/data/muse455/attn2/tcache_aot")
import triton
from triton.backends.compiler import GPUTarget
mod, kname = sys.argv[1], sys.argv[2]
kv = dict(a.split("=") for a in sys.argv[3:] if "=" in a and not a.startswith("--"))
out = sys.argv[sys.argv.index("--out") + 1]
fn = getattr(importlib.import_module(mod), kname)
nw, ns = int(kv.pop("num_warps", 4)), int(kv.pop("num_stages", 2))
wpe = int(kv.pop("waves_per_eu", 0))
const = dict(NQ=32, G=8, D=64, PAGE=16, BLOCK_M=64, **{k: int(v) for k, v in kv.items()})
names = fn.arg_names
sig = {}
ptrs = {"segm_out_ptr": "*fp32", "segm_max_ptr": "*fp32", "segm_sum_ptr": "*fp32", "q_ptr": "*bf16",
        "k_ptr": "*fp8e4nv", "v_ptr": "*fp8e4nv", "bt_ptr": "*i32", "seq_lens_ptr": "*i64", "cu_q_ptr": "*i32",
        "k_descale_ptr": "*fp32", "v_descale_ptr": "*fp32", "scale": "fp32"}
for n in names:
    if n in const:
        sig[n] = "constexpr"
    elif n in ptrs:
        sig[n] = ptrs[n]
    else:
        sig[n] = "i64" if n.startswith(("sk", "sv")) else "i32"
div = [names.index(n) for n in names if n in ptrs and n != "scale" or n.startswith(("sk", "sv", "q_stride"))]
src = triton.compiler.ASTSource(fn=fn, signature=sig, constexprs={(names.index(k),): v for k, v in const.items()},
                                attrs={(i,): [["tt.divisibility", 16]] for i in div})
k = triton.compile(src, target=GPUTarget("hip", "gfx1250", 32),
                   options=dict(num_warps=nw, num_stages=ns, waves_per_eu=wpe))
a = k.asm["amdgcn"]
open(f"/data/muse455/attn2/asm/{out}.amdgcn", "w").write(a)
open(f"/data/muse455/attn2/asm/{out}.ttgir", "w").write(k.asm["ttgir"])
st = {m: re.search(rf"; {m}: (\S+)", a).group(1) for m in ("NumVgprs", "Occupancy", "ScratchSize")}
print(out, st, "shared", k.metadata.shared)
