"""Driver (one k2safe run): clock calibration, release latency, stream-wait cost, graph-setting cost x 3 settings,
then the VGPR-threshold co-residency sweep."""
import os
import subprocess

B = "/data/muse455/pdltest/bench"
SETTINGS = [("default", {}), ("classic", {"DEBUG_HIP_GRAPH_CLASSIC_PATH": "1"}),
            ("seg+q2", {"DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING": "1", "DEBUG_HIP_FORCE_GRAPH_QUEUES": "2"})]


def run(cmd, extra=None, t=120):
    env = dict(os.environ, **(extra or {}))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=t, env=env)
        out, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = (e.stdout or b"").decode() + (e.stderr or b"").decode(), "TIMEOUT"
    keep = [l for l in out.splitlines() if l.startswith(("RESULT", "CONFIG", "ERR", "DEADLOCK"))]
    print(f"== {' '.join(cmd[1:])} {extra or ''} rc={rc}\n" + "\n".join(keep), flush=True)


run([f"{B}/lat", "calib"])
for name, env in SETTINGS:
    for kind in ("0", "1", "2"):
        run([f"{B}/lat", "release", kind], env)
    run([f"{B}/lat", "waitcost"], env)
    for tail in ("256", "16"):
        run([f"{B}/lat", "graph", tail, "40", "4"], env, t=180)
# co-residency: producer VGPRs vs the real QKV partials footprint (104 VGPRs, 43 KB LDS, 128 threads, 576 CTAs)
for pv in ("251", "240", "232", "224", "216", "200"):
    run([f"{B}/cores", pv, "104", "43", "576", "145", "256", "5000", "128"])
run([f"{B}/cores", "128", "104", "43", "576", "170", "256", "5000", "128"])   # producer forced to 1 CTA/CU by LDS
for pv in ("224", "200"):  # consumer streams 75.5 MB of weights beside the producer
    run([f"{B}/cores", pv, "104", "43", "576", "145", "256", "5000", "128", "72"])
