"""EXPLORATORY: compare the retrained Stage 2 variants with the original run on the same held-out folds."""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
HERE = Path(__file__).resolve().parent; VD = HERE.parents[1]; sys.path.insert(0, str(VD))
from stage2_vocal import match_iou
from sklearn.metrics import f1_score
RUN = {"original": VD / "results/stage2_main", "both (primary)": HERE / "both", "weights only": HERE / "weights", "mask only": HERE / "mask"}; N = {1: "Kampita", 2: "Nokku", 3: "Etra", 4: "Irakka"}
F = {k: {Path(p).stem[5:]: json.load(open(p))["clips"] for p in sorted(glob.glob(str(d / "fold_*.json")))} for k, d in RUN.items()}; G = sorted(F["original"])
def counts(cl):
    t = {l: 0 for l in N}; r = dict(t); e = dict(t)
    for c in cl:
        a = [tuple(x) for x in c["ref"]]; b = [tuple(x) for x in c["cnn"]]; tp, _, _, _ = match_iou(a, b, .3)
        for l in N: t[l] += tp[l]; r[l] += sum(x[2] == l for x in a); e[l] += sum(x[2] == l for x in b)
    return t, r, e
def ev(cl):
    t, r, e = counts(cl); return float(np.mean([2 * t[l] / (r[l] + e[l]) if r[l] + e[l] else 0 for l in N]))
def fr(cl): return f1_score(np.concatenate([c["y"] for c in cl]), np.concatenate([c["cnn_frames"] for c in cl]), labels=range(5), average="macro", zero_division=0)
rows = []
for k, f in F.items():
    cl = [c for g in G for c in f[g]]; t, r, e = counts(cl); row = dict(run=k, event=ev(cl), frame=fr(cl), detections=sum(e.values()), correct=sum(t.values()))
    for l, n in N.items(): row[f"{n} F1"] = 2 * t[l] / (r[l] + e[l]); row[f"{n} P"] = t[l] / e[l] if e[l] else 0; row[f"{n} R"] = t[l] / r[l]; row[f"{n} n"] = e[l]
    rows.append(row)
T = pd.DataFrame(rows).set_index("run"); print(T[["event", "frame", "detections", "correct"]].round(3).to_string()); print(); print(T[[f"{n} {s}" for n in N.values() for s in ("F1", "P", "R", "n")]].T.round(2).to_string())
rng = np.random.default_rng(20261118); base = F["original"]
for k in list(F)[1:]:
    a = np.array([ev(F[k][g]) for g in G]); b = np.array([ev(base[g]) for g in G]); d = []; e = []
    for _ in range(2000):
        s = rng.choice(G, len(G)); x = [c for g in s for c in F[k][g]]; y = [c for g in s for c in base[g]]; d.append(ev(x) - ev(y)); e.append(fr(x) - fr(y))
    print(f"\n{k}: event change {T.event[k]-T.event['original']:+.3f} (group bootstrap 95% CI {np.percentile(d,2.5):+.3f} to {np.percentile(d,97.5):+.3f}); frame change {T.frame[k]-T.frame['original']:+.3f} (CI {np.percentile(e,2.5):+.3f} to {np.percentile(e,97.5):+.3f}); folds better {int((a>b).sum())}, worse {int((a<b).sum())} of 11; Wilcoxon two-sided p {wilcoxon(a,b).pvalue:.3f}")
