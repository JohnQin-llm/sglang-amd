"""Kernel table of one VERIFY step + the VERIFY->DRAFT gap from a torch-profiler trace (CPU only).

    python3 trace_verify.py TRACE.json.gz OUT_PREFIX
writes OUT_PREFIX.verify.txt (median verify step, kernels in order), OUT_PREFIX.gap.txt (kernels between the
end of a VERIFY annotation and the next DRAFT annotation), OUT_PREFIX.summary.json (per-name totals per step).
"""
import bisect
import collections
import gzip
import json
import re
import statistics
import sys


def main(path, pre):
    ev = json.load(gzip.open(path))["traceEvents"]
    ann = sorted((e["ts"], e["ts"] + e["dur"], e["name"]) for e in ev
                 if e.get("cat") == "gpu_user_annotation" and e.get("ph") == "X" and e.get("name", "").startswith("step["))
    k = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"),
               key=lambda e: e["ts"])
    del ev
    kts = [e["ts"] for e in k]

    def kern(t0, t1):
        i = bisect.bisect_left(kts, t0)
        out = []
        while i < len(k) and k[i]["ts"] < t1:
            out.append(k[i])
            i += 1
        return out

    ver = [a for a in ann if a[2].startswith("step[VERIFY bs=1]")]
    print("verify steps", len(ver), [round(a[1] - a[0]) for a in ver])
    med = sorted(ver, key=lambda a: a[1] - a[0])[len(ver) // 2]
    ks = kern(med[0], med[1])
    with open(pre + ".verify.txt", "w") as f:
        f.write(f"{med[2]} ann {med[1]-med[0]:.1f} us, {len(ks)} kernels\n")
        t = ks[0]["ts"]
        for e in ks:
            f.write(f"{e['ts']-ks[0]['ts']:9.1f} {e['dur']:7.1f} gap {e['ts']-t:6.1f} {e['name'][:150]}\n")
            t = e["ts"] + e["dur"]
    # per-name totals over all verify steps
    agg = collections.defaultdict(lambda: [0, 0.0])
    for a in ver:
        for e in kern(a[0], a[1]):
            nm = e["name"][:90]
            agg[nm][0] += 1
            agg[nm][1] += e["dur"]
    n = len(ver)
    summ = sorted(((v[1] / n, v[0] / n, nm) for nm, v in agg.items()), reverse=True)
    # gaps VERIFY end -> next DRAFT start
    gaps = []
    for i, a in enumerate(ann):
        if a[2].startswith("step[VERIFY") and i + 1 < len(ann) and ann[i + 1][2].startswith("step[DRAFT"):
            gaps.append((a[1], ann[i + 1][0]))
    with open(pre + ".gap.txt", "w") as f:
        for g0, g1 in gaps[:6]:
            ks = kern(g0, g1)
            f.write(f"== gap {g1-g0:.1f} us, {len(ks)} kernels, busy {sum(e['dur'] for e in ks):.1f}\n")
            t = g0
            for e in ks:
                f.write(f"{e['ts']-g0:9.1f} {e['dur']:7.1f} gap {e['ts']-t:6.1f} {e['name'][:150]}\n")
                t = e["ts"] + e["dur"]
    json.dump({"n_verify": n, "per_step": [[round(a, 2), round(c, 2), nm] for a, c, nm in summ]},
              open(pre + ".summary.json", "w"), indent=0)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
