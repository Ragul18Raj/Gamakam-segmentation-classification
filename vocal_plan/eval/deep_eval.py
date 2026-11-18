"""Deeper evaluation of the simulated leave-one-song-out runs.

Reads fold*.json written by gamaka_sim/run_eval.py and adds what the summary leaves out:
per-fold spread, bootstrap confidence intervals, a paired Wilcoxon test (CNN vs rule-based + 3-NN),
onset/offset error of matched events, per-class precision/recall, and confusion-matrix figures.
Everything here is computed on SIMULATED data and every figure says so.

usage: python deep_eval.py <gamaka_sim dir> <clean results dir> <stress results dir> <out dir>
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

SIM, CLEAN, STRESS, OUT = sys.argv[1:5]
sys.path.insert(0, SIM)
from pipeline import f1  # noqa: E402

os.makedirs(OUT, exist_ok=True)
DARK, TEAL, ORANGE, MUTED, LINE = "#1F2A30", "#2A9D8F", "#D9622B", "#5B6B73", "#D5DDE0"
NAMES = {1: "G1", 2: "G2", 3: "G3"}
TOL = 0.2
rng = np.random.default_rng(0)
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": DARK,
                     "xtick.color": DARK, "ytick.color": DARK, "axes.spines.top": False, "axes.spines.right": False})


def match(ref, est, tol=TOL):
    """Same rule as pipeline.match_events (same class, |onset diff| <= tol, greedy one-to-one), keeping the pairs."""
    pairs = []
    for l in (1, 2, 3):
        R = [e for e in ref if e[2] == l]
        E = [e for e in est if e[2] == l]
        cand = sorted((abs(r[0] - x[0]), i, j) for i, r in enumerate(R) for j, x in enumerate(E) if abs(r[0] - x[0]) <= tol)
        ur, ue = set(), set()
        for _, i, j in cand:
            if i not in ur and j not in ue:
                ur.add(i); ue.add(j); pairs.append((R[i], E[j], l))
    return pairs


def counts(clips, key):
    tp = {l: 0 for l in (1, 2, 3)}; nr = dict(tp); ne = dict(tp)
    for c in clips:
        ref = [tuple(x) for x in c["ref"]]; est = [tuple(x) for x in c[key]]
        for _, _, l in match(ref, est):
            tp[l] += 1
        for l in (1, 2, 3):
            nr[l] += sum(1 for e in ref if e[2] == l); ne[l] += sum(1 for e in est if e[2] == l)
    return tp, nr, ne


def macro_f1(clips, key):
    tp, nr, ne = counts(clips, key)
    return float(np.mean([f1(tp[l], nr[l], ne[l]) for l in (1, 2, 3)]))


def analyse(d):
    folds = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "fold*.json")), key=lambda p: int(p.split("fold")[-1][:-5]))]
    clips = [c for f in folds for c in f["clips"]]
    r = {"n_folds": len(folds), "n_clips": len(clips)}
    for key in ("cnn", "rule"):
        per_fold = [macro_f1(f["clips"], key) for f in folds]
        boot = [macro_f1([clips[i] for i in rng.integers(0, len(clips), len(clips))], key) for _ in range(1000)]
        tp, nr, ne = counts(clips, key)
        prec = {NAMES[l]: tp[l] / ne[l] if ne[l] else 0.0 for l in (1, 2, 3)}
        rec = {NAMES[l]: tp[l] / nr[l] if nr[l] else 0.0 for l in (1, 2, 3)}
        on, off = [], []
        for c in clips:
            for rv, ev, _ in match([tuple(x) for x in c["ref"]], [tuple(x) for x in c[key]]):
                on.append(1000 * abs(ev[0] - rv[0])); off.append(1000 * abs(ev[1] - rv[1]))
        r[key] = dict(event_f1_pooled=macro_f1(clips, key), event_f1_per_fold=per_fold,
                      fold_mean=float(np.mean(per_fold)), fold_std=float(np.std(per_fold, ddof=1)),
                      ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                      precision=prec, recall=rec,
                      false_events=int(sum(ne.values()) - sum(tp.values())), missed_events=int(sum(nr.values()) - sum(tp.values())),
                      onset_ms_median=float(np.median(on)), onset_ms_mean=float(np.mean(on)),
                      offset_ms_median=float(np.median(off)), offset_ms_mean=float(np.mean(off)),
                      onset_ms=on, offset_ms=off)
    diff = np.array(r["cnn"]["event_f1_per_fold"]) - np.array(r["rule"]["event_f1_per_fold"])
    w = wilcoxon(r["cnn"]["event_f1_per_fold"], r["rule"]["event_f1_per_fold"]) if np.any(diff != 0) else None
    r["paired"] = dict(cnn_minus_rule_per_fold=diff.tolist(), mean_diff=float(diff.mean()),
                       folds_cnn_better=int((diff > 0).sum()), folds_rule_better=int((diff < 0).sum()),
                       wilcoxon_stat=float(w.statistic) if w else None, wilcoxon_p=float(w.pvalue) if w else None)
    Y = np.concatenate([c["y"] for c in clips])
    for key, fk in (("cnn", "cnn_frames"), ("rule", "rule_frames")):
        P = np.concatenate([c[fk] for c in clips])
        cm = np.zeros((4, 4), int)
        np.add.at(cm, (Y, P), 1)
        r[key]["frame_cm"] = cm.tolist()
    return r


res = {"clean": analyse(CLEAN), "harder": analyse(STRESS)}
json.dump(res, open(os.path.join(OUT, "deep_eval.json"), "w"), indent=1)

# ------------------------------------------------------------------ figure 1: per-fold event F1, CNN vs rule
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
for ax, (cond, title) in zip(axs, (("clean", "Standard"), ("harder", "Stress test"))):
    r = res[cond]; x = np.arange(1, r["n_folds"] + 1)
    for key, col, lab in (("cnn", TEAL, "Dilated CNN"), ("rule", ORANGE, "Rule-based + 3-NN")):
        y = r[key]["event_f1_per_fold"]
        ax.plot(x, y, "-o", color=col, lw=1.8, ms=6, label=lab)
        ax.axhline(np.mean(y), color=col, lw=1, ls="--", alpha=0.7)
    p = r["paired"]["wilcoxon_p"]
    ax.set_title(f"{title}  (Wilcoxon p = {p:.2f})" if p is not None else title, fontsize=11.5, color=DARK, loc="left")
    ax.set_xticks(x); ax.set_xticklabels([f"song {i}" for i in x], fontsize=9.5)
    ax.grid(axis="y", color=LINE, lw=0.8)
axs[0].set_ylabel("Event macro-F1 on the test song")
axs[0].set_ylim(0.6, 1.02)
axs[1].legend(frameon=False, loc="lower right", fontsize=10)
fig.text(0.995, 0.005, "simulated data", ha="right", va="bottom", fontsize=9, style="italic", color=MUTED)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "per_fold_f1.png"), dpi=220)
plt.close(fig)

# ------------------------------------------------------------------ figure 2: CNN frame confusion matrices (row %)
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.9))
labs = ["none", "G1", "G2", "G3"]
for ax, (cond, title) in zip(axs, (("clean", "Standard"), ("harder", "Stress test"))):
    cm = np.array(res[cond]["cnn"]["frame_cm"], float)
    pct = 100 * cm / cm.sum(1, keepdims=True)
    ax.imshow(pct, cmap="Greens", vmin=0, vmax=100)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{pct[i, j]:.1f}", ha="center", va="center", fontsize=10.5, color="white" if pct[i, j] > 55 else DARK)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs); ax.set_yticks(range(4)); ax.set_yticklabels(labs)
    ax.set_xlabel("Predicted (per frame)"); ax.set_ylabel("Singer's label")
    ax.set_title(title, fontsize=11.5, color=DARK, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
fig.text(0.995, 0.005, "simulated data · row %", ha="right", va="bottom", fontsize=9, style="italic", color=MUTED)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "confusion_cnn.png"), dpi=220)
plt.close(fig)

# ------------------------------------------------------------------ console summary
for cond in ("clean", "harder"):
    r = res[cond]
    print(f"== {cond}: {r['n_clips']} clips, {r['n_folds']} folds")
    for key in ("cnn", "rule"):
        k = r[key]
        print(f"  {key:5s} pooled {k['event_f1_pooled']:.3f} | folds {k['fold_mean']:.3f} ± {k['fold_std']:.3f} "
              f"| CI95 {k['ci95'][0]:.3f}-{k['ci95'][1]:.3f} | onset med {k['onset_ms_median']:.0f} ms, offset med {k['offset_ms_median']:.0f} ms "
              f"| false {k['false_events']}, missed {k['missed_events']}")
        print("        precision", {a: round(b, 3) for a, b in k["precision"].items()}, "recall", {a: round(b, 3) for a, b in k["recall"].items()})
    p = r["paired"]
    print(f"  paired: CNN-rule mean {p['mean_diff']:+.3f}, CNN better in {p['folds_cnn_better']}/7, rule better in {p['folds_rule_better']}/7, Wilcoxon p={p['wilcoxon_p']}")
