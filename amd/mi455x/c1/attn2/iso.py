"""Run bench.py once per variant in a child process, so a faulting variant only kills its own process.

usage: iso.py --ctx 131072 --variants a,b,c [extra bench.py args]
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
i = args.index("--variants")
names = args[i + 1].split(",")
rest = args[:i] + args[i + 2:]
for n in names:
    p = subprocess.run([sys.executable, os.path.join(HERE, "bench.py"), "--variants", n, *rest],
                       capture_output=True, text=True)
    lines = [ln for ln in (p.stdout + p.stderr).splitlines() if ln.startswith("[") or "fault" in ln.lower()
             or "Error" in ln]
    print(f"== {n} rc={p.returncode}", flush=True)
    for ln in lines[-6:]:
        print("   " + ln, flush=True)
