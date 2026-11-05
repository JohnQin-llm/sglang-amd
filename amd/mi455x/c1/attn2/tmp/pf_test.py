import os, sys
os.environ.setdefault("TRITON_CACHE_DIR", "/data/muse455/attn2/tcache_aot")
import triton, triton.language as tl
from triton.backends.compiler import GPUTarget

@triton.jit
def k(x_ptr, y_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    p = x_ptr + offs * 256
    a = p.to(tl.int64, bitcast=True)
    d = tl.inline_asm_elementwise("global_prefetch_b8 $1, off", "=v,v", [a], dtype=tl.int32, is_pure=False, pack=1)
    tl.store(y_ptr + offs, tl.load(x_ptr + offs))

src = triton.compiler.ASTSource(fn=k, signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "N": "constexpr"}, constexprs={(2,): 128})
kk = triton.compile(src, target=GPUTarget("hip", "gfx1250", 32), options=dict(num_warps=4))
print("\n".join(l for l in kk.asm["amdgcn"].split("\n") if "prefetch" in l or "global_load" in l))
