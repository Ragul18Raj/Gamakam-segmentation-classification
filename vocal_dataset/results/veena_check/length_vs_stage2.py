"""EXPLORATORY: does gamaka length explain the Stage 2 failure? Reads pass-1 marks and the saved held-out CNN output; no model is trained."""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; VD = HERE.parents[1]; sys.path.insert(0, str(VD))
from stage2_vocal import match_iou
N = {1: "Kampita", 2: "Nokku", 3: "Etra", 4: "Irakka"}; ev = pd.read_csv(VD / "annotations/pass1_events.csv"); ev["dur"] = ev.end_s - ev.start_s
print("A. Marked lengths, all 293 events (s)"); print(ev.groupby("label").dur.describe(percentiles=[.05, .5, .95])[["count", "min", "5%", "50%", "95%", "max"]].round(2).to_string())
print("share of events at least 1 s:", ev.groupby("label").dur.apply(lambda s: round(float((s >= 1).mean()), 2)).to_dict(), "| events under 1 s by type:", ev[ev.dur < 1].label.value_counts().to_dict())
cl = [c for p in sorted(glob.glob(str(VD / "results/stage2_main/fold_*.json"))) for c in json.load(open(p))["clips"]]
def iou(a, b):
    o = max(0, min(a[1], b[1]) - max(a[0], b[0])); return o / (a[1] - a[0] + b[1] - b[0] - o) if o > 0 else 0
R, E = [], []
for c in cl:
    r = [tuple(x) for x in c["ref"]]; e = [tuple(x) for x in c["cnn"]]; _, _, _, pr = match_iou(r, e, .3); hr = {p[0] for p in pr}; he = {p[1] for p in pr}
    for a in r:
        best = max([iou(a, b) for b in e] or [0]); R.append(dict(cls=N[a[2]], dur=a[1] - a[0], hit=a in hr, found=best >= .3, touched=best > 0, tempo=c["tempo"]))
    for b in e:
        best = max([iou(a, b) for a in r] or [0]); E.append(dict(cls=N[b[2]], dur=b[1] - b[0], tp=b in he, partner=best >= .3, touches=best > 0))
R, E = pd.DataFrame(R), pd.DataFrame(E); B = [0, .2, .4, .8, 9]; L = ["<0.2", "0.2-0.4", "0.4-0.8", ">=0.8"]; R["bin"] = pd.cut(R.dur, B, labels=L, right=False); E["bin"] = pd.cut(E.dur, B, labels=L, right=False)
print(f"\nB. CNN result for marked events by length: correct = right type and IoU>=0.3 ({len(R)} events)")
print(R.groupby("bin").agg(n=("hit", "size"), correct=("hit", "mean"), located_any_type=("found", "mean"), touched=("touched", "mean")).round(2).to_string())
print("\nC. Same length, different type: correct/total"); print(R.pivot_table(index="cls", columns="bin", values="hit", aggfunc=lambda s: f"{int(s.sum())}/{len(s)}", fill_value="-").reindex(N.values()).to_string())
print("\nD. CNN events by length"); print(E.groupby("bin").agg(n=("tp", "size"), correct=("tp", "mean"), no_marked_partner=("partner", lambda s: int((~s).sum())), touch_nothing=("touches", lambda s: int((~s).sum()))).round(2).to_string())
print("\nE. CNN output length vs marked length, median s (n)"); print(pd.DataFrame({"marked": R.groupby("cls").dur.median().round(2), "n_marked": R.groupby("cls").size(), "cnn": E.groupby("cls").dur.median().round(2), "n_cnn": E.groupby("cls").size(), "cnn_correct": E.groupby("cls").tp.sum()}).reindex(N.values()).to_string())
Y = np.concatenate([c["y"] for c in cl]); f = {N[k]: int((Y == k).sum()) for k in N}; t = sum(f.values()); print("\nF. training-signal share: frames per type", f, {k: round(v / t, 2) for k, v in f.items()}, "| event share", (R.cls.value_counts(normalize=True).round(2)).to_dict(), "| none share of all frames", round(float((Y == 0).mean()), 3))
fa = E[~E.partner]; print("\nG. CNN events with no marked partner:", len(fa), fa.cls.value_counts().to_dict(), "median length", round(fa.dur.median(), 2)); print("if every marked event were found with the same false alarms, event F1 by type:", {k: round(2 * n / (2 * n + int((fa.cls == k).sum())), 2) for k, n in R.cls.value_counts().items()})
