"""Run every leave-one-song-out fold of the simulation as its own single-threaded process, then score.
usage: python launch_folds.py <gamaka_sim dir>"""
import os
import subprocess
import sys
import time

SIM = sys.argv[1]
env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
jobs = []
t0 = time.time()
for out, extra in (("rerun_clean", []), ("rerun_stress", ["--stress"])):
    os.makedirs(os.path.join(SIM, out), exist_ok=True)
    for k in range(1, 8):
        log = open(os.path.join(SIM, out, f"fold{k}.out"), "w")
        jobs.append(subprocess.Popen([sys.executable, "run_eval.py", "--folds", str(k), "--out", out] + extra,
                                     cwd=SIM, env=env, stdout=log, stderr=subprocess.STDOUT))
codes = [j.wait() for j in jobs]
print("fold exit codes:", codes, f"({time.time() - t0:.0f}s)", flush=True)
for out in ("rerun_clean", "rerun_stress"):
    with open(os.path.join(SIM, out + "_score.out"), "w") as fh:
        subprocess.run([sys.executable, "score.py", out], cwd=SIM, env=env, stdout=fh, stderr=subprocess.STDOUT)
print("scored", flush=True)
