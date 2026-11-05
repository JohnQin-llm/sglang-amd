"""Driver run 2: release latency with the flag reset inside the graph, chain overhead, graph topology variants."""
import os
import subprocess

B = "/data/muse455/pdltest/bench"
SETTINGS = [("default", {}), ("classic", {"DEBUG_HIP_GRAPH_CLASSIC_PATH": "1"}),
            ("seg+q2", {"DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING": "1", "DEBUG_HIP_FORCE_GRAPH_QUEUES": "2"})]


def run(cmd, extra=None, t=150):
    env = dict(os.environ, **(extra or {}))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=t, env=env)
        out, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = (e.stdout or b"").decode() + (e.stderr or b"").decode(), "TIMEOUT"
    keep = [l for l in out.splitlines() if l.startswith(("RESULT", "CONFIG", "ERR", "DEADLOCK"))]
    print(f"== {' '.join(cmd[1:])} {extra or ''} rc={rc}\n" + "\n".join(keep), flush=True)


for name, env in SETTINGS:
    for kind in ("0", "2"):
        run([f"{B}/lat", "release", kind], env)
    run([f"{B}/lat", "chain"], env)
for name, env in SETTINGS:
    for k2v in ("251", "224"):
        for tail in ("256", "16"):
            run([f"{B}/lat", "graph", tail, "40", "4", k2v], env, t=200)
