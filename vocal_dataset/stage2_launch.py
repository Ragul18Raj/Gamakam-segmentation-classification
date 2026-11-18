"""Run every stage-2 fold as its own single-threaded process (at most --jobs at a time), wait for all, then score.

usage: python stage2_launch.py [--jobs 14] [--runs main,block6,nosa]
  main   : results/stage2_main    leave-one-group-out, 11 folds (the run the pass marks apply to)
  block6 : results/stage2_block6  robustness split, 7 folds
  nosa   : results/stage2_nosa    Sa-free features (sin/cos zeroed), 11 folds
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", type=int, default=14)
ap.add_argument("--runs", default="main,block6,nosa")
a = ap.parse_args()

groups = sorted(pd.read_csv(ROOT / "metadata.csv", dtype=str).group.unique())
blocks = [f"B{k}" for k in range(1, 8)]
RUNS = {"main": ("results/stage2_main", groups, []),
        "block6": ("results/stage2_block6", blocks, ["--split", "block6"]),
        "nosa": ("results/stage2_nosa", groups, ["--nosa"])}
todo = []
for r in a.runs.split(","):
    out, folds, extra = RUNS[r]
    (ROOT / out).mkdir(parents=True, exist_ok=True)
    todo += [(out, k, extra) for k in folds]

t0, running, codes = time.time(), [], {}
while todo or running:
    while todo and len(running) < a.jobs:
        out, k, extra = todo.pop(0)
        fh = open(ROOT / out / f"fold_{k}.out", "w")
        p = subprocess.Popen([sys.executable, "stage2_vocal.py", "--fold", k, "--out", out] + extra,
                             cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
        running.append((p, out, k))
    for item in list(running):
        if item[0].poll() is not None:
            codes[(item[1], item[2])] = item[0].returncode
            running.remove(item)
            print(f"{item[1]} {item[2]} exit {item[0].returncode} ({time.time() - t0:.0f}s)", flush=True)
    time.sleep(5)

bad = {k: v for k, v in codes.items() if v}
print("all folds finished;", "failures: " + str(bad) if bad else "no failures", flush=True)
for r in a.runs.split(","):
    out, folds, _ = RUNS[r]
    if all((ROOT / out / f"fold_{k}.json").exists() for k in folds):
        with open(ROOT / out / "score.out", "w") as fh:
            subprocess.run([sys.executable, "stage2_score.py", out], cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
        print("scored", out, flush=True)
    else:
        print("NOT scored (missing fold files):", out, flush=True)
