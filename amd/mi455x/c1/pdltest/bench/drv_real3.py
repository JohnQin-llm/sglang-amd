"""Driver run 6: real Triton QKV behind a K2-shaped kernel, 3 graph settings."""
import os
import subprocess

B = "/data/muse455/pdltest/bench"
SETTINGS = [("default", {}), ("classic", {"DEBUG_HIP_GRAPH_CLASSIC_PATH": "1"}),
            ("seg+q2", {"DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING": "1", "DEBUG_HIP_FORCE_GRAPH_QUEUES": "2"})]
for name, env in SETTINGS:
    try:
        r = subprocess.run(["python3", f"{B}/real_qkv.py"], capture_output=True, text=True, timeout=240, env=dict(os.environ, HANDS="4,12", **env))
        out, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = (e.stdout or b"").decode() + (e.stderr or b"").decode(), "TIMEOUT"
    keep = [l for l in out.splitlines() if l.startswith(("RESULT", "ERR")) or "Error" in l]
    print(f"== {name} rc={rc}\n" + "\n".join(keep[-8:]), flush=True)
