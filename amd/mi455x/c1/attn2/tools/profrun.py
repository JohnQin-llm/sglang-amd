"""Run prof.py under rocprofv3 once per counter group (one pass each)."""
import os, subprocess, sys
ROOT = "/data/muse455/attn2"
groups = {
    "p1": "SQ_WAVES SQ_BUSY_CYCLES SQ_WAVE_CYCLES SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS",
    "p2": "SQ_INSTS_VALU_TRANS SQ_INSTS_VEC32_VALU_WMMA SQ_INST_CYCLES_VALU_WMMA SQ_INST_CYCLES_LDS SQ_WAIT_INST_ANY SQ_WAIT_ANY",
    "p3": "SQ_INST_LEVEL_LDS SQ_LEVEL_WAVES SQ_ITEM_CYCLES_VALU SQ_INST_CYCLES_VMEM SQ_CYCLES GRBM_GUI_ACTIVE",
}
args = sys.argv[1:]
for g, pmc in groups.items():
    d = f"{ROOT}/results/prof/{g}"
    cmd = ["rocprofv3", "--pmc", *pmc.split(), "--output-format", "csv", "-d", d, "-o", g, "--",
           sys.executable, f"{ROOT}/prof.py", *args]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd="/tmp", env=dict(os.environ, PYTHONPATH=ROOT))
    print(g, "rc", p.returncode, flush=True)
    open(f"{ROOT}/logs/prof_{g}.log", "w").write(p.stdout + p.stderr)
