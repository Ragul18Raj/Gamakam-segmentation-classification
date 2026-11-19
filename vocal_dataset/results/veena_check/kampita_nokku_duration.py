"""EXPLORATORY: is Kampita vs Nokku in the vocal data decided by length? Leave-one-group-out, Kampita and Nokku events only (classifiers see 2 classes)."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score as BA
HERE = Path(__file__).resolve().parent; VD = HERE.parents[1]; sys.path.insert(0, str(VD)); warnings.filterwarnings("ignore")
from stage1_vocal import CL4, LR, NOLEN, centre, dist_matrix, resample, seg_feats, logo_feat, logo_dtw
from vocal_pitch import event_curve, load_cents, load_tables
meta, ev = load_tables(); m = meta.set_index("recording"); cur, lab, grp = [], [], []
for rec, e in ev.groupby("recording", sort=False):
    t, c, _, v, _ = load_cents(rec, m.loc[rec, "sa_hz"])
    for _, r in e.iterrows():
        x, nv = event_curve(t, c, v, r.start_s, r.end_s)
        if nv >= 5 and r.label in ("Kampita", "Nokku"): cur.append(x); lab.append(int(r.label == "Nokku")); grp.append(m.loc[rec, "group"])
y, g = np.array(lab), np.array(grp); d = np.array([len(c) / 100 for c in cur]); F = np.array([seg_feats(c) for c in cur])
print("events", dict(zip(["Kampita", "Nokku"], np.bincount(y))))
for n, s in (("Kampita", 0), ("Nokku", 1)): print(f"  {n} length s: min {d[y==s].min():.2f}, p5 {np.percentile(d[y==s],5):.2f}, median {np.median(d[y==s]):.2f}, p95 {np.percentile(d[y==s],95):.2f}, max {d[y==s].max():.2f}")
P = {"Duration only": logo_feat(F[:, [0]], y, g, LR), "All 9 features": logo_feat(F, y, g, LR), "No length features": logo_feat(F[:, NOLEN], y, g, LR),
     "DTW, length removed": logo_dtw(dist_matrix([centre(resample(c)) for c in cur]), y, g, 3, 2)}
lo, hi = np.percentile(d[y == 0], 5), np.percentile(d[y == 1], 95); Z = (d >= lo) & (d <= hi)
print(f"overlap zone {lo:.2f}-{hi:.2f} s: Kampita {int((Z&(y==0)).sum())}, Nokku {int((Z&(y==1)).sum())}")
for k, p in P.items():
    print(f"  {k:22s} balanced accuracy all {BA(y,p):.3f} | overlap zone {BA(y[Z],p[Z]):.3f}  (Kampita {np.mean(p[Z&(y==0)]==0):.2f}, Nokku {np.mean(p[Z&(y==1)]==1):.2f})")
# The p5-p95 zone above holds only 3 events, so also score length-matched pairs: each Nokku paired with the unused Kampita closest in length, within 30 ms.
order = np.argsort(-d[y == 1]); N = np.where(y == 1)[0][order]; K = list(np.where(y == 0)[0]); pairs = []
for n in N:
    if not K: break
    j = min(K, key=lambda k: abs(d[k] - d[n]))
    if abs(d[j] - d[n]) <= .03: pairs.append((j, n)); K.remove(j)
ka, no = map(np.array, zip(*pairs)); print(f"\nlength-matched pairs (within 30 ms): {len(pairs)}; lengths {d[no].min():.2f}-{d[no].max():.2f} s, median {np.median(d[no]):.2f} s; groups {len(set(g[np.r_[ka,no]]))}")
rng = np.random.default_rng(20261118); idx = np.r_[ka, no]
for k, p in P.items():
    a, b = np.mean(p[ka] == 0), np.mean(p[no] == 1); ga = sorted(set(g[idx])); bs = []
    for _ in range(2000):
        s = rng.choice(ga, len(ga)); ks = np.concatenate([ka[g[ka] == h] for h in s]); ns = np.concatenate([no[g[no] == h] for h in s])
        if len(ks) and len(ns): bs.append((np.mean(p[ks] == 0) + np.mean(p[ns] == 1)) / 2)
    print(f"  {k:22s} balanced accuracy {(a+b)/2:.3f} (group bootstrap 95% CI {np.percentile(bs,2.5):.2f}-{np.percentile(bs,97.5):.2f})  Kampita {a:.2f}, Nokku {b:.2f}")
