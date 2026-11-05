"""Env-switch sweep inside one k2safe hold: sweep.py [--script S] [--args "A B"] <layer> <name>:<K=V,K=V> ...
(one ``S`` run a config, default run_k2.py; ``--layers <layer>`` is passed first).

Logs and each config's run_k2 json go to logs/rf/<name>_L<layer>.{log,json}.
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "logs", "sweep")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="run_k2.py")
    ap.add_argument("--args", default="")
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("layer")
    ap.add_argument("configs", nargs="+")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for spec in a.configs:
        name, _, kvs = spec.partition(":")
        env = dict(os.environ)
        for kv in filter(None, kvs.split(",")):
            k, _, v = kv.partition("=")
            env[k] = v
        log = os.path.join(OUT, f"{name}_L{a.layer}.log")
        with open(log, "w") as f:
            rc = subprocess.run([sys.executable, os.path.join(HERE, a.script), "--layers", a.layer] + a.args.split(),
                                env=env, stdout=f, stderr=subprocess.STDOUT, timeout=a.timeout).returncode
        js = os.path.join(HERE, "run_k2_a8.json")
        if a.script == "run_k2.py" and os.path.isfile(js):
            shutil.copy(js, os.path.join(OUT, f"{name}_L{a.layer}.json"))
        print(f"[sweep] {name} ({kvs}) rc={rc}", flush=True)


if __name__ == "__main__":
    main()
