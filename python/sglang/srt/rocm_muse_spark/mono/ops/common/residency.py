# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The residency proof a persistent launch needs before it may wait on any
other CTA: every CTA of the grid resident at once.

A built FlyDSL launch keeps its compiled module's IR, the GPU code object
embedded in it (``gpu.binary``). The code object is loaded through the HIP
module API and the runtime is asked how many CTAs of ``block`` threads a CU
holds (``hipModuleOccupancyMaxActiveBlocksPerMultiprocessor``, which accounts
for VGPRs, AGPRs, SGPRs, LDS and wave slots), together with the kernel's
register, LDS and scratch (spill) use. A grid fits when
``occupancy * CUs >= grid`` and nothing spills.
"""

import ctypes
import re
from dataclasses import dataclass

_BIN = re.compile(r'bin\s*=\s*"((?:[^"\\]|\\.)*)"', re.S)
_ESC = {"n": 10, "t": 9, '"': 34, "\\": 92}

# hipFunction_attribute
_ATTR_SHARED = 1  # HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
_ATTR_LOCAL = 3  # HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES (scratch a lane: spills)
_ATTR_REGS = 4  # HIP_FUNC_ATTRIBUTE_NUM_REGS


@dataclass(frozen=True)
class Residency:
    name: str
    block: int
    blocks_per_cu: int  # the runtime's occupancy for ``block`` threads
    regs: int  # architected VGPRs a lane (the runtime's NUM_REGS)
    lds: int  # static LDS bytes a CTA
    scratch: int  # private (spill) bytes a lane

    def fits(self, grid: int, cus: int) -> bool:
        return self.blocks_per_cu >= 1 and self.blocks_per_cu * cus >= grid

    def problems(self, grid: int, cus: int) -> list[str]:
        out = []
        if not self.fits(grid, cus):
            out.append(
                f"{self.name}: {self.blocks_per_cu} CTA(s) of {self.block} threads a "
                f"CU x {cus} CUs < grid {grid}"
            )
        if self.scratch:
            out.append(f"{self.name}: spills {self.scratch} B a lane")
        return out


def _unescape(s: str) -> bytes:
    out = bytearray()
    i = 0
    while i < len(s):
        c = s[i]
        if c != "\\":
            out += c.encode("utf-8")
            i += 1
            continue
        n = s[i + 1]
        if n in _ESC:
            out.append(_ESC[n])
            i += 2
        else:
            out.append(int(s[i + 1 : i + 3], 16))
            i += 3
    return bytes(out)


def code_objects(launch) -> list[bytes]:
    """The GPU code objects of every compiled variant of a FlyDSL @jit launch."""
    out = []
    for artifact in getattr(launch, "_mem_cache", {}).values():
        text = getattr(artifact, "_ir_text", None) or ""
        out += [_unescape(m.group(1)) for m in _BIN.finditer(text)]
    return out


def _kernel_names(elf: bytes) -> list[str]:
    """Kernel symbols: the ``<name>.kd`` descriptors' names."""
    names = re.findall(rb"([A-Za-z_][A-Za-z0-9_.$]*)\.kd\x00", elf)
    return sorted({n.decode() for n in names})


class _Hip:
    _lib = None

    @classmethod
    def lib(cls):
        if cls._lib is None:
            lib = ctypes.CDLL("libamdhip64.so")
            lib.hipModuleLoadData.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,
            ]
            lib.hipModuleGetFunction.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,
                ctypes.c_char_p,
            ]
            lib.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_size_t,
            ]
            lib.hipFuncGetAttribute.argtypes = [
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            lib.hipModuleUnload.argtypes = [ctypes.c_void_p]
            cls._lib = lib
        return cls._lib


def _check(err: int, what: str) -> None:
    if err != 0:
        raise RuntimeError(f"{what} failed: hipError {err}")


def residency(launch, block: int) -> list[Residency]:
    """Each kernel of a built launch: its occupancy at ``block`` threads a CTA
    and its register / LDS / spill use (needs a current HIP device)."""
    hip = _Hip.lib()
    out = []
    for elf in code_objects(launch):
        buf = ctypes.create_string_buffer(elf, len(elf))
        mod = ctypes.c_void_p()
        _check(hip.hipModuleLoadData(ctypes.byref(mod), buf), "hipModuleLoadData")
        try:
            for name in _kernel_names(elf):
                fn = ctypes.c_void_p()
                _check(
                    hip.hipModuleGetFunction(ctypes.byref(fn), mod, name.encode()),
                    "hipModuleGetFunction",
                )
                occ = ctypes.c_int()
                _check(
                    hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
                        ctypes.byref(occ), fn, block, 0
                    ),
                    "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",
                )
                attrs = {}
                for key, attr in (
                    ("regs", _ATTR_REGS),
                    ("lds", _ATTR_SHARED),
                    ("scratch", _ATTR_LOCAL),
                ):
                    v = ctypes.c_int()
                    _check(hip.hipFuncGetAttribute(ctypes.byref(v), attr, fn), "attr")
                    attrs[key] = v.value
                out.append(Residency(name, block, occ.value, **attrs))
        finally:
            hip.hipModuleUnload(mod)
    return out
