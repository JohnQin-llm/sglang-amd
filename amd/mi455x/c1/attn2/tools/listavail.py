import subprocess, sys
p = subprocess.run(["rocprofv3", "--list-avail", "-d", "0"], capture_output=True, text=True)
open("/data/muse455/attn2/logs/list_avail.txt", "w").write(p.stdout + p.stderr)
print("rc", p.returncode, len(p.stdout))
