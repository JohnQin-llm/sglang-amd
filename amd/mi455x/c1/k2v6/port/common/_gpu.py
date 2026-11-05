"""Test harness for the gfx1250 common primitives: one-CTA kernels over raw
pointers, ISA capture, and the GPU lock / server checks the tests run under.

A test builds ``body(tid, a, b, c, d)`` (four Int64 device addresses) into a
kernel, ``run`` launches it on torch tensors, ``isa`` returns the kernel's
gfx1250 assembly (FLYDSL_DEBUG_DUMP_ASM) so a test can check the instructions
a primitive lowers to.
"""

from __future__ import annotations

import glob
import itertools
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PORT = os.path.dirname(_HERE)
if _PORT not in sys.path:
    sys.path.insert(0, _PORT)

DUMP = os.environ.setdefault("FLYDSL_DUMP_DIR", f"/tmp/k2_common_asm_{os.getpid()}")
os.environ.setdefault("FLYDSL_DUMP_IR", "1")
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
import torch  # noqa: E402
from flydsl.expr.typing import Int64  # noqa: E402

_ids = itertools.count()


def kernel(body, threads=32, name=None, lds=None, grid=1):
    """Launcher of a ``grid``-CTA kernel running ``body``.

    ``lds``: an ``fx.struct`` class; then ``body(tid, a, b, c, d, smem)`` gets
    its allocated view."""
    name = name or re.sub(r"\W", "_", f"k2c_{body.__name__}_{next(_ids)}")

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def kern(a: Int64, b: Int64, c: Int64, d: Int64):
        tid = fx.thread_idx.x
        if lds is not None:
            alloc = fx.SharedAllocator()
            sm = alloc.allocate(lds).peek()
            body(tid, a, b, c, d, sm)
        else:
            body(tid, a, b, c, d)

    @flyc.jit
    def launch(a: Int64, b: Int64, c: Int64, d: Int64, stream: fx.Stream = fx.Stream(None)):
        kern(a, b, c, d).launch(grid=(grid, 1, 1), block=(threads, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch


def _p(t):
    return 0 if t is None else t.data_ptr()


def run(launch, a=None, b=None, c=None, d=None):
    launch(_p(a), _p(b), _p(c), _p(d), stream=fx.Stream(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()


def isa(launch):
    """The gfx1250 assembly of a launched kernel (FLYDSL_DUMP_IR's final ISA)."""
    hits = sorted(glob.glob(os.path.join(DUMP, launch.kernel_name, "*_final_isa.s")))
    if not hits:
        raise FileNotFoundError(f"no ISA dump for {launch.kernel_name} under {DUMP}")
    return open(hits[-1], errors="ignore").read()


def kernel_text(asm, name):
    """The instructions of kernel ``name`` in an assembly dump."""
    start = asm.find(f"{name}:")
    end = asm.find(".Lfunc_end", start)
    return asm[start:end] if start >= 0 else asm


class Report:
    """PASS / FAIL lines; ``done`` exits non-zero on any failure."""

    def __init__(self, title):
        self.title, self.fails, self.n = title, [], 0
        print(f"== {title}  (device {torch.cuda.get_device_name()}, "
              f"{torch.cuda.get_device_properties(0).gcnArchName})")

    def check(self, what, ok, detail=""):
        self.n += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"  [{detail}]" if detail else ""))
        if not ok:
            self.fails.append(what)
        return ok

    def done(self):
        print(f"== {self.title}: {self.n - len(self.fails)}/{self.n} passed")
        if self.fails:
            sys.exit(1)
