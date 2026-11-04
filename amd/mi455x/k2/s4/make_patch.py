"""Build the S4 integration patch against the served tree WITHOUT touching it.

    python3 make_patch.py [--out /data/muse455/k2/ep1/s4/k2_gfx1250.patch]

a/ = copies of the served files; b/ = the same + hooks (apply_hooks.py) + the new package
python/sglang/srt/rocm_muse_spark/mono_gfx1250/ (the merged port tree, imports rewritten). The patch is
``diff -ruN a b`` with git-style paths; apply with ``git apply`` (or ``patch -p1``) inside the sglang checkout.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

SERVED = "/data/muse455/sglang"
PORT = "/data/muse455/k2/ep1/port"
PKG_REL = "python/sglang/srt/rocm_muse_spark/mono_gfx1250"
PKG_MOD = "sglang.srt.rocm_muse_spark.mono_gfx1250"
MODEL_REL = "python/sglang/srt/rocm_muse_spark/model.py"
WORK = "/tmp/k2g_s4"

SKIP_FILE = re.compile(r"^(test_.*|_gpu|bench.*|harness|probe.*|bw_probe|weights|ref|run_.*)\.py$")
SKIP_DIR = {"__pycache__", "results", "probe", "tests"}


def rewrite(src, rel):
    src = re.sub(r"\bfrom port\.", f"from {PKG_MOD}.", src)
    src = re.sub(r"\bfrom port import\b", f"from {PKG_MOD} import", src)
    src = re.sub(r"\bimport port\.", f"import {PKG_MOD}.", src)
    # P3's test-time local import of mxq
    src = src.replace("sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n", "")
    src = src.replace("import mxq as Q  # noqa: E402", f"from {PKG_MOD}.moe import mxq as Q  # noqa: E402")
    if "sys.path.insert" in src:
        print(f"WARNING: {rel} still edits sys.path", file=sys.stderr)
    return src


def copy_package(dst):
    for root, dirs, files in os.walk(PORT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIR]
        rel = os.path.relpath(root, PORT)
        for f in files:
            if not f.endswith(".py") or SKIP_FILE.match(f) or f == "validation.py":
                continue
            out = os.path.join(dst, rel, f)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(os.path.join(root, f)) as fi:
                src = fi.read()
            with open(out, "w") as fo:
                fo.write(rewrite(src, os.path.join(rel, f)))
    shutil.copy(os.path.join(PORT, "INTERFACE.md"), os.path.join(dst, "INTERFACE.md"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/muse455/k2/ep1/s4/k2_gfx1250.patch")
    args = ap.parse_args()
    shutil.rmtree(WORK, ignore_errors=True)
    p = os.path.join(WORK, MODEL_REL)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    shutil.copy(os.path.join(SERVED, MODEL_REL), p)
    git = lambda *a: subprocess.run(["git", "-c", "user.name=k2g", "-c", "user.email=k2g@local", *a],  # noqa: E731
                                    cwd=WORK, check=True, capture_output=True, text=True).stdout
    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "served")
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "apply_hooks.py"), p], check=True)
    copy_package(os.path.join(WORK, PKG_REL))
    git("add", "-A")
    patch = git("diff", "--cached", "--no-color")
    with open(args.out, "w") as f:
        f.write(patch)
    files = re.findall(r"^\+\+\+ b/(\S+)", patch, flags=re.M)
    print(f"{args.out}: {len(files)} files, {patch.count(chr(10))} lines")


if __name__ == "__main__":
    main()
