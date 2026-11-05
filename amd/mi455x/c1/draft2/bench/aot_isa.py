"""CPU-only: compile Triton kernels for gfx1250 ahead of time (no GPU) and summarize the ISA.

    python3 aot_isa.py OUT_DIR
Writes <kernel>.s and prints, per kernel: VGPRs, instruction count, global loads and where the first
reduction barrier / LDS op sits relative to the loads (load hoisting check).
"""
import sys

sys.path.insert(0, "/data/muse455/draft2/sglang/python")
import os
import re

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

TGT = GPUTarget("hip", "gfx1250", 32)


def compile_fn(fn, signature, constexprs, num_warps):
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton.compile(src, target=TGT, options={"num_warps": num_warps, "num_stages": 1})


def summarize(name, k, out_dir):
    asm = k.asm["amdgcn"]
    open(os.path.join(out_dir, name + ".s"), "w").write(asm)
    lines = [l.strip() for l in asm.splitlines() if l.strip() and not l.strip().startswith((";", ".", "//"))]
    ins = [l for l in lines if re.match(r"^[a-z_]", l) and not l.endswith(":")]
    loads = [i for i, l in enumerate(ins) if l.startswith(("global_load", "buffer_load"))]
    waits = [i for i, l in enumerate(ins) if l.startswith(("s_wait_loadcnt", "s_waitcnt"))]
    lds = [i for i, l in enumerate(ins) if l.startswith(("ds_", "s_barrier"))]
    vg = re.search(r"\.vgpr_count:\s+(\d+)", asm)
    first_lds = lds[0] if lds else -1
    before = sum(1 for i in loads if i < first_lds)
    print(f"{name}: {len(ins)} instr, vgpr {vg.group(1) if vg else '?'}, global loads {len(loads)} "
          f"({before} before the first LDS/barrier at instr {first_lds}), load waits {len(waits)}; "
          f"load positions {loads[:40]}")


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    from sglang.srt.rocm_muse_spark.fused_norm_residual import (
        _fused_norm_residual_hoist_kernel,
        _fused_norm_residual_triton_kernel,
    )

    sig = {"BRANCH": "*bf16", "STREAM": "*fp32", "OUTPUT": "*bf16", "NEW_STREAM_OUT": "*fp32", "ROUTER_OUT": "*fp32",
           "POST_WEIGHT": "*bf16", "PRE_WEIGHT": "*bf16", "ALPHA": "*fp32", "BETA": "*fp32",
           "post_norm_eps": "fp32", "pre_norm_eps": "fp32"}
    for c in ("H", "BLOCK_H", "POST_NORM_IS_INPUT_SCALE", "USE_RESIDUAL_GATE", "RESIDUAL_GATE_CHANNEL_WISE",
              "POST_NORM_HAS_WEIGHT", "PRE_NORM_HAS_WEIGHT", "HAS_ROUTER", "PRE_NORM_BF16", "M_BUCKET"):
        sig[c] = "constexpr"
    for inscale in (False, True):
        ce = {"H": 8192, "BLOCK_H": 8192, "POST_NORM_IS_INPUT_SCALE": inscale, "USE_RESIDUAL_GATE": True,
              "RESIDUAL_GATE_CHANNEL_WISE": True, "POST_NORM_HAS_WEIGHT": True, "PRE_NORM_HAS_WEIGHT": True,
              "HAS_ROUTER": False, "PRE_NORM_BF16": False, "M_BUCKET": 0}
        for nm, fn in (("served", _fused_norm_residual_triton_kernel), ("hoist", _fused_norm_residual_hoist_kernel)):
            for w in (8, 16):
                summarize(f"norm_{nm}_inscale{int(inscale)}_w{w}", compile_fn(fn, sig, ce, w), out_dir)


if __name__ == "__main__":
    main(sys.argv[1])
