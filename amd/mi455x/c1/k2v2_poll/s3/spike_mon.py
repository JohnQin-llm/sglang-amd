"""Host monitor for spike.py (run on the host beside a k2safe job): every --period ms, CLOCK_MONOTONIC ns plus
/proc/vmstat NUMA / migration / compaction counters and the CPU ticks of the observability agents (gpuagent,
alloy). Only changes are written (one JSON line each). Read-only: it never signals any process.

  python3 spike_mon.py --out /data/muse455/k2/cand/k2v2_poll/s3/spike/mon_<tag>.jsonl --seconds 900 &
"""

import argparse
import json
import os
import time

KEYS = ("numa_hint_faults", "numa_hint_faults_local", "numa_pages_migrated", "pgmigrate_success", "pgmigrate_fail",
        "compact_stall", "compact_migrate_scanned", "thp_fault_alloc", "thp_collapse_alloc", "pgmajfault",
        "numa_pte_updates", "nr_tlb_remote_flush", "nr_tlb_remote_flush_received")
AGENTS = ("gpuagent", "alloy", "amd-smi", "rocm-smi", "amdsmi", "rdc", "rdcd")


def vm():
    d = {}
    with open("/proc/vmstat") as f:
        for line in f:
            k, v = line.split()
            if k in KEYS:
                d[k] = int(v)
    return d


def agents():
    out = {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            with open(f"/proc/{p}/stat") as f:
                s = f.read()
        except OSError:
            continue
        name = s[s.index("(") + 1:s.rindex(")")]
        if name in AGENTS:
            fl = s.rsplit(")", 1)[1].split()
            out[f"{name}/{p}"] = int(fl[11]) + int(fl[12])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=900)
    ap.add_argument("--period", type=float, default=5.0)
    ap.add_argument("--agent-every", type=int, default=20, help="agent scan every N periods (it walks /proc)")
    a = ap.parse_args()
    end = time.monotonic() + a.seconds
    last_v, last_a = {}, {}
    i = 0
    with open(a.out, "w") as f:
        while time.monotonic() < end:
            t = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            v = vm()
            dv = {k: x - last_v.get(k, x) for k, x in v.items() if x != last_v.get(k)}
            rec = {}
            if dv and last_v:
                rec["vm"] = dv
            last_v = v
            if i % a.agent_every == 0:
                ag = agents()
                if ag != last_a:
                    rec["agents"] = {k: x - last_a.get(k, 0) for k, x in ag.items() if x != last_a.get(k)}
                    gone = [k for k in last_a if k not in ag]
                    if gone:
                        rec["gone"] = gone
                last_a = ag
            if rec:
                rec["t"] = t
                f.write(json.dumps(rec) + "\n")
                f.flush()
            i += 1
            time.sleep(a.period / 1000)


if __name__ == "__main__":
    main()
