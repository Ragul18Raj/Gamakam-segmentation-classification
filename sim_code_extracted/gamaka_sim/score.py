"""Aggregate leave-one-song-out results into the numbers used on the slide."""
import json, glob, sys
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from pipeline import match_events, f1

d = sys.argv[1] if len(sys.argv) > 1 else "results"
folds = [json.load(open(p)) for p in sorted(glob.glob(f"{d}/fold*.json"))]
print(f"{len(folds)} folds")
NAMES = {0: "none", 1: "G1", 2: "G2", 3: "G3"}
out = {}

# stage 1
y = sum((f["stage1"]["y"] for f in folds), [])
p = sum((f["stage1"]["p"] for f in folds), [])
acc_folds = [accuracy_score(f["stage1"]["y"], f["stage1"]["p"]) for f in folds]
out["stage1"] = dict(n=len(y), acc=accuracy_score(y, p), macro_f1=f1_score(y, p, average="macro"),
                     acc_fold_min=min(acc_folds), acc_fold_max=max(acc_folds),
                     chance=max(np.bincount(y)[1:]) / len(y),
                     cm=confusion_matrix(y, p, labels=[1, 2, 3]).tolist())

clips = [c for f in folds for c in f["clips"]]
Y = np.concatenate([c["y"] for c in clips])


def system(key, frames_key):
    r = {}
    if frames_key:
        P = np.concatenate([c[frames_key] for c in clips])
        r["frame_macro_f1"] = f1_score(Y, P, labels=[0, 1, 2, 3], average="macro", zero_division=0)
        r["frame_cm"] = confusion_matrix(Y, P, labels=[0, 1, 2, 3]).tolist()
    tp = {1: 0, 2: 0, 3: 0}; nr = dict(tp); ne = dict(tp)
    abs_err, exact = [], 0
    for c in clips:
        est = c[key] if key else []
        a, b, e = match_events([tuple(x) for x in c["ref"]], [tuple(x) for x in est])
        for l in (1, 2, 3):
            tp[l] += a[l]; nr[l] += b[l]; ne[l] += e[l]
        errs = [abs(e[l] - b[l]) for l in (1, 2, 3)]
        abs_err += errs
        exact += all(x == 0 for x in errs)
    per = {NAMES[l]: f1(tp[l], nr[l], ne[l]) for l in (1, 2, 3)}
    r["event_f1_per_class"] = per
    r["event_f1_macro"] = float(np.mean(list(per.values())))
    r["count_mae_per_type_per_clip"] = float(np.mean(abs_err))
    r["clips_all_counts_exact"] = exact / len(clips)
    # per fold event macro F1
    pf = []
    for f in folds:
        tp2 = {1: 0, 2: 0, 3: 0}; nr2 = dict(tp2); ne2 = dict(tp2)
        for c in f["clips"]:
            a, b, e = match_events([tuple(x) for x in c["ref"]], [tuple(x) for x in (c[key] if key else [])])
            for l in (1, 2, 3):
                tp2[l] += a[l]; nr2[l] += b[l]; ne2[l] += e[l]
        pf.append(float(np.mean([f1(tp2[l], nr2[l], ne2[l]) for l in (1, 2, 3)])))
    r["event_f1_fold_min"], r["event_f1_fold_max"] = min(pf), max(pf)
    # breakdowns
    for fac in ("tempo", "octave"):
        br = {}
        for v in sorted({c[fac] for c in clips}):
            tp3 = {1: 0, 2: 0, 3: 0}; nr3 = dict(tp3); ne3 = dict(tp3)
            for c in clips:
                if c[fac] != v:
                    continue
                a, b, e = match_events([tuple(x) for x in c["ref"]], [tuple(x) for x in (c[key] if key else [])])
                for l in (1, 2, 3):
                    tp3[l] += a[l]; nr3[l] += b[l]; ne3[l] += e[l]
            br[v] = float(np.mean([f1(tp3[l], nr3[l], ne3[l]) for l in (1, 2, 3)]))
        r[f"event_f1_by_{fac}"] = br
    return r


out["all_none"] = system(None, None)
out["all_none"]["frame_macro_f1"] = f1_score(Y, np.zeros_like(Y), labels=[0, 1, 2, 3], average="macro", zero_division=0)
out["rule_knn"] = system("rule", "rule_frames")
out["cnn"] = system("cnn", "cnn_frames")
P_raw = np.concatenate([c["cnn_raw"] for c in clips])
out["cnn"]["frame_macro_f1_before_postprocessing"] = f1_score(Y, P_raw, labels=[0, 1, 2, 3], average="macro")
out["n_clips"] = len(clips)
out["n_ref_events"] = sum(len(c["ref"]) for c in clips)
out["frame_share_none"] = float((Y == 0).mean())

json.dump(out, open(f"{d}/summary.json", "w"), indent=1)


def show(k, v, ind=""):
    if isinstance(v, dict):
        print(f"{ind}{k}:"); [show(a, b, ind + "  ") for a, b in v.items()]
    elif isinstance(v, float):
        print(f"{ind}{k}: {v:.3f}")
    else:
        print(f"{ind}{k}: {v}")


for k, v in out.items():
    show(k, v)
