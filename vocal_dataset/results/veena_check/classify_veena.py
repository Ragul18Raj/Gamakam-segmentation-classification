"""EXPLORATORY (18 Nov 2026): train on the singer's 293 marked gamakas, test on veena; and test how much the vocal result needs duration.
Frozen before any classifier ran here. Quality rule (no labels): a clip passes if under 3% of its cut frames have the raw tracker more
than 600 cents from the cleaned pitch. Main clips: plavita #02.1 and bhairiga #03.1; Jaru_asc_1, _2 (Etra); Jaru_desc_1, _2 melgana (Irakka).
Run: /usr/bin/python3 classify_veena.py  (after extract_veena.py)"""
import pickle, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
HERE = Path(__file__).resolve().parent; VD = HERE.parents[1]; sys.path.insert(0, str(VD))
warnings.filterwarnings("ignore")
from stage1_vocal import CL4, FEAT, LR, NOLEN, centre, dtw, dist_matrix, resample, seg_feats, logo_feat, logo_dtw
from vocal_pitch import event_curve, load_cents, load_tables
MAIN = ["Kampita (plavita)#02.1", "Kampita (plavita) bhairiga#03.1", "Jaru_asc_1", "Jaru_asc_2", "Jaru_desc_1_melgana", "Jaru_desc_2_melgana"]
meta, ev = load_tables(); m = meta.set_index("recording"); cur, lab, grp = [], [], []
for rec, e in ev.groupby("recording", sort=False):
    t, c, _, v, _ = load_cents(rec, m.loc[rec, "sa_hz"])
    for _, r in e.iterrows():
        x, nv = event_curve(t, c, v, r.start_s, r.end_s)
        if nv >= 5: cur.append(x); lab.append(CL4.index(r.label)); grp.append(m.loc[rec, "group"])
y, g = np.array(lab), np.array(grp); F = np.array([seg_feats(c) for c in cur]); L0 = int(np.median([len(c) for c in cur]))
print(f"vocal training events {len(y)}, median length {L0/100:.2f} s"); durs = pd.Series([len(c)/100 for c in cur]).groupby(y).describe()[["25%","50%","75%"]]; durs.index = CL4; print(durs.round(2).to_string())
def band(a, b): return max(5, int(.2 * max(len(a), len(b))))
def knn(q, tr, yt, k=3):
    d = np.array([dtw(q, b, band(q, b)) for b in tr]); o = np.argsort(d, kind="stable")[:k]; v = np.bincount(yt[o], minlength=4); best = np.flatnonzero(v == v.max())
    return next(yt[i] for i in o if yt[i] in best)
D = pickle.load(open(HERE / "veena_curves.pkl", "rb"))
for o in D:
    r, c = o["raw"][o["a"]:o["b"]], o["masked"][o["a"]:o["b"]]; ok = ~np.isnan(r) & ~np.isnan(c); o["off8"] = float(np.mean(np.abs(r[ok] - c[ok]) > 600)); o["qc"] = o["off8"] < .03; o["main"] = o["name"] in MAIN
    o["take"] = ("plavita #02" if "#02" in o["name"] else "plavita bhairiga #03" if "bhairiga" in o["name"] else "plavitam Todi") if o["label"] == "Kampita" else ("asc [OG_1]" if "OG_1" in o["name"] else "asc") if "asc" in o["name"] else ("desc Melgana" if "melgana" in o["name"] else "desc Shankarabharanam")
assert sum(o["main"] for o in D) == 6 and all(o["qc"] for o in D if o["main"])
Vc = [o["curve"] for o in D]; yv = np.array([CL4.index(o["label"]) for o in D]); Fv = np.array([seg_feats(c) for c in Vc]); St = [resample(c, L0) for c in Vc]; Fs = np.array([seg_feats(c) for c in St])
Tc = [centre(c) for c in cur]; Tn = [centre(resample(c)) for c in cur]; P = {}
P["Full features"] = LR().fit(F, y).predict(Fv)
P["No length features"] = LR().fit(F[:, NOLEN], y).predict(Fv[:, NOLEN])
P["Duration only"] = LR().fit(F[:, [0]], y).predict(Fv[:, [0]])
P["DTW shape"] = np.array([knn(centre(c), Tc, y) for c in Vc])
P["DTW, length removed"] = np.array([knn(centre(resample(c)), Tn, y) for c in Vc])
P["Full features, clip stretched"] = LR().fit(F, y).predict(Fs)
P["DTW, clip stretched"] = np.array([knn(centre(c), Tc, y) for c in St])
T = pd.DataFrame({"clip": [o["name"] for o in D], "true": [o["label"] for o in D], "take": [o["take"] for o in D], "main": [o["main"] for o in D], "passes_quality": [o["qc"] for o in D], "off8": [round(o["off8"], 3) for o in D], "how_cut": [o["how"] for o in D], "dur_s": [len(c)/100 for c in Vc], "net_c": Fv[:, 4].round(0), **{k: [CL4[i] for i in v] for k, v in P.items()}})
T.to_csv(HERE / "veena_predictions.csv", index=False); M = list(P)
sh = lambda s: s.replace("Jaaru (Etra)", "Etra").replace("Jaaru (Irakka)", "Irakka")
print("\nMAIN CLIPS"); X = T[T.main].set_index("clip")[["true", "dur_s"] + M].applymap(lambda v: sh(v) if isinstance(v, str) else v); print(X.T.to_string())
def summ(S, name):
    print(f"\n{name}: {len(S)} clips; always-Kampita gets {(S.true == 'Kampita').mean():.2f}")
    r = {k: {"all": (S[k] == S.true).mean(), **{sh(c): (S[S.true == c][k] == c).mean() for c in S.true.unique()}} for k in M}; print(pd.DataFrame(r).T.round(2).to_string())
summ(T[T.passes_quality], "ALL CLIPS PASSING THE QUALITY RULE"); summ(T, "ALL 38 CLIPS")
print("\nfails quality rule:", T[~T.passes_quality][["clip", "off8"]].values.tolist())
print("\nby take, correct/total"); print(T[T.passes_quality].groupby(["true", "take"]).apply(lambda s: pd.Series({k: f"{(s[k] == s.true).sum()}/{len(s)}" for k in M})).to_string())
print("\nwhat each clip type is called (passing clips)")
for k in ("Full features", "No length features", "Duration only", "DTW, length removed"): print(k); print(pd.crosstab(T[T.passes_quality].true.map(sh), T[T.passes_quality][k].map(sh)).to_string())
print("\nveena cut length vs vocal"); print(T.groupby("true").dur_s.describe()[["min", "50%", "max"]].round(2).to_string())
# vocal: remove duration from the held-out events by stretching every one to the same length
def mf(p): return f1_score(y, p, average="macro", labels=range(4))
base = {"F-LR": logo_feat(F, y, g, LR), "DTW-3NN": logo_dtw(dist_matrix(Tc), y, g, 3, 4)}
pl, pd_ = np.empty_like(y), np.empty_like(y)
for G in np.unique(g):
    te, tr = np.where(g == G)[0], np.where(g != G)[0]; L = int(np.median([len(cur[i]) for i in tr])); st = [resample(cur[i], L) for i in te]
    pl[te] = LR().fit(F[tr], y[tr]).predict(np.array([seg_feats(c) for c in st])); trc = [Tc[i] for i in tr]
    pd_[te] = [knn(centre(c), trc, y[tr]) for c in st]
print("\nVOCAL held-out macro-F1, normal vs every test event stretched to one length")
print(f"  F-LR    {mf(base['F-LR']):.3f} -> {mf(pl):.3f}   per class {dict(zip(CL4, f1_score(y, pl, average=None, labels=range(4)).round(2)))}")
print(f"  DTW-3NN {mf(base['DTW-3NN']):.3f} -> {mf(pd_):.3f}   per class {dict(zip(CL4, f1_score(y, pd_, average=None, labels=range(4)).round(2)))}")
