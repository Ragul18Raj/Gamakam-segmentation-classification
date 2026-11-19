"""Run all folds of the 3 variants, at most 14 single-threaded processes at a time, then score each with stage2_score.py."""
import os, subprocess, sys, time
from pathlib import Path
VD = Path(__file__).resolve().parents[2]; env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
G = [f"P{k:02d}" for k in range(1, 12)]; V = ["both", "weights", "mask"]; todo = [(v, g) for v in V for g in G]; run = []; bad = []; t0 = time.time()
for v in V: (VD / "results/length_fix" / v).mkdir(parents=True, exist_ok=True)
while todo or run:
    while todo and len(run) < 14:
        v, g = todo.pop(0); o = f"results/length_fix/{v}"; fh = open(VD / o / f"fold_{g}.out", "w")
        run.append((subprocess.Popen([sys.executable, "results/length_fix/train_fold.py", "--variant", v, "--fold", g, "--out", o], cwd=VD, env=env, stdout=fh, stderr=subprocess.STDOUT), v, g))
    for r in list(run):
        if r[0].poll() is not None:
            run.remove(r); print(r[1], r[2], "exit", r[0].returncode, f"{time.time()-t0:.0f}s", flush=True)
            if r[0].returncode: bad.append(r[1:])
    time.sleep(3)
print("failures", bad, flush=True)
for v in V:
    with open(VD / f"results/length_fix/{v}/score.out", "w") as fh: print(v, "score exit", subprocess.run([sys.executable, "stage2_score.py", f"results/length_fix/{v}"], cwd=VD, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode, flush=True)
