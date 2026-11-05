"""Driver run 3: chain overhead with ALU kernels; graph topology variants with relaxed polling."""
import os
import subprocess

B = "/data/muse455/pdltest/bench"
SETTINGS = [("default", {}), ("classic", {"DEBUG_HIP_GRAPH_CLASSIC_PATH": "1"}),
            ("seg+q2", {"DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING": "1", "DEBUG_HIP_FORCE_GRAPH_QUEUES": "2"})]


def run(cmd, extra=None, t=200):
    env = dict(os.environ, **(extra or {}))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=t, env=env)
        out, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = (e.stdout or b"").decode() + (e.stderr or b"").decode(), "TIMEOUT"
    keep = [l for l in out.splitlines() if l.startswith(("RESULT", "CONFIG", "ERR", "DEADLOCK"))]
    print(f"== {' '.join(cmd[1:])} {extra or ''} rc={rc}\n" + "\n".join(keep), flush=True)


for name, env in SETTINGS:
    run([f"{B}/lat", "chain"], env)
    for k2v, tail in (("251", "256"), ("251", "16"), ("224", "256")):
        run([f"{B}/lat", "graph", tail, "40", "4", k2v], env)
