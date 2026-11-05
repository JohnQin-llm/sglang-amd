import subprocess
combos = [{}, {"DEBUG_HIP_FORCE_GRAPH_QUEUES": "2"}, {"DEBUG_HIP_FORCE_GRAPH_QUEUES": "4"},
          {"DEBUG_HIP_GRAPH_CLASSIC_PATH": "1"}, {"DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING": "0"},
          {"DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING": "1", "DEBUG_HIP_FORCE_GRAPH_QUEUES": "2"}]
for c in combos:
    env = {"HIP_VISIBLE_DEVICES": "0", "PATH": "/usr/bin:/bin", **c}
    r = subprocess.run(["/data/muse455/pdltest/pdl_test", "graph"], capture_output=True, text=True, timeout=60, env=env)
    res = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith("RESULT") or "->" in l and "no error" not in l]
    print(f"== graph {c or '(default)'}: " + " | ".join(res), flush=True)
