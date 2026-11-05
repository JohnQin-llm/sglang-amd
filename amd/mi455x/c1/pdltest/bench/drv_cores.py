"""Driver (one k2safe run): Triton QKV footprint, then the co-residency sweep."""
import subprocess

B = "/data/muse455/pdltest/bench"


def run(cmd, env=None, t=120):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=t, env=env)
    out = r.stdout + r.stderr
    print(f"== {' '.join(cmd[-8:])} rc={r.returncode}\n" + "\n".join(l for l in out.splitlines()
          if l.startswith(("RESULT", "CONFIG", "TRITON", "ERR", "Traceback")) or "Error" in l), flush=True)


run(["python3", f"{B}/tri_meta.py"], t=300)
it = "5000"
# prod_vgpr cons_vgpr cons_lds_kb cons_ctas prod_lds_kb nprod iters
for pv, cv, cl in [(251, 0, 0), (251, 0, 32), (251, 64, 0), (251, 128, 32), (128, 0, 0), (128, 64, 32), (128, 128, 32),
                   (64, 64, 32), (64, 128, 64), (64, 0, 176)]:
    run([f"{B}/cores", str(pv), str(cv), str(cl), "576", "145", "256", it])
# producer on fewer CTAs than CUs: free CUs exist
run([f"{B}/cores", "251", "64", "32", "576", "145", "192", it])
