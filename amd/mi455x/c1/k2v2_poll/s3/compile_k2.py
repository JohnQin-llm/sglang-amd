"""Compile-only check of the gfx1250 K2 (no launch): trace, lower, dump the ISA and report VGPRs / spills / LDS.

docker exec muse_mi455x bash -c 'cd /tmp && PYTHONSAFEPATH=1 HIP_VISIBLE_DEVICES=0 \
    PYTHONPATH=/data/muse455/k2/cand/k2v2_land1 python3 /data/muse455/k2/cand/k2v2_land1/s3/compile_k2.py [act]'
"""

import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = os.path.join(HERE, "asm")
os.environ.setdefault("FLYDSL_GPU_ARCH", "gfx1250")
os.environ["COMPILE_ONLY"] = "1"
os.environ["FLYDSL_DEBUG_DUMP_ASM"] = "1"
os.environ["FLYDSL_DUMP_IR"] = "1"
os.environ["FLYDSL_DUMP_DIR"] = DUMP

from port import k2  # noqa: E402


def isa_stats(path):
    txt = open(path).read()
    out = {"isa": path}
    for k in ("vgpr_count", "sgpr_count", "group_segment_fixed_size", "private_segment_fixed_size",
              "vgpr_spill_count", "sgpr_spill_count"):
        m = re.search(r"\.%s:\s*(\d+)" % k, txt)
        if m:
            out[k] = int(m.group(1))
    m = re.search(r"; NumVgprs:\s*(\d+)", txt)
    if m:
        out["NumVgprs"] = int(m.group(1))
    m = re.search(r"; ScratchSize:\s*(\d+)", txt)
    if m:
        out["ScratchSize"] = int(m.group(1))
    out["lines"] = txt.count("\n")
    return out


def main():
    act = sys.argv[1] if len(sys.argv) > 1 else "a8"
    key = k2.K2Build(act=act)
    t0 = time.time()
    launch = k2.build(key)
    launch(*([0] * 20), 0, 0, 0, 0, 0, 0)
    res = {"act": act, "compile_s": round(time.time() - t0, 1),
           "peer_bytes": k2.peer_bytes(), "scratch_bytes": k2.scratch_bytes()}
    isas = sorted(glob.glob(os.path.join(DUMP, "**", "*final_isa.s"), recursive=True), key=os.path.getmtime)
    isas = [p for p in isas if "muse_mono_k2g" in p]
    if isas:
        res.update(isa_stats(isas[-1]))
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(HERE, f"compile_k2_{act}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
