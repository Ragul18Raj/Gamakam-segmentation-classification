"""Detection limit by gamaka length for the frozen Step 3 primary run, using saved held-out events only. EXPLORATORY. No training.
A mark is 'located' when a CNN event of any type is paired with it one-to-one at IoU >= 0.3 (the Stage 2 matcher, with types ignored).
Run from vocal_dataset:  python results/detection_limit/real_detection_limit.py"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import chi2
HERE = Path(__file__).resolve().parent; VD = HERE.parents[1]; sys.path.insert(0, str(VD))
from stage2_score import ev_scores
from stage2_vocal import match_iou
FR, SEED, NB = 100, 20261118, 2000; TYPES = {1: "Kampita", 2: "Nokku", 3: "Etra", 4: "Irakka"}; FULL = {"Kampita": 1, "Nokku": 2, "Jaaru (Etra)": 3, "Jaaru (Irakka)": 4}
EDGES = [0, .2, .3, .4, .6, .8, np.inf]; BINS = ["<0.2", "0.2-0.3", "0.3-0.4", "0.4-0.6", "0.6-0.8", ">=0.8"]; GROUPS = [f"P{k:02d}" for k in range(1, 12)]
OUT = {n: HERE / n for n in ("real_limit_report.txt", "real_limit_numbers.csv", "marks_located.csv", "unmatched_detections.csv", "marks_below_type_minimum.csv", "real_detection_curve.png")}
if any(p.exists() for p in OUT.values()): sys.exit("Output files already exist; not overwriting them.")
L, NUM = [], []
def say(s=""): print(s, flush=True); L.append(str(s))
def num(section, group, metric, value, low=np.nan, high=np.nan, n=np.nan): NUM.append(dict(section=section, group=group, metric=metric, value=value, ci_low=low, ci_high=high, n=n))

# ---------------------------------------- load the frozen run and match it to the singer's marks
E = pd.read_csv(VD / "results/step3/events_all_B.csv"); P = pd.read_csv(VD / "results/step3/chosen_params.csv"); P = P[(P.variant == "all") & (P.cleanup == "B")].set_index("held_out_group")
clips, marks, dets, mins = [], [], [], {}
for g in GROUPS:
    for c in json.load(open(VD / f"results/stage2_main/fold_{g}.json"))["clips"]:
        x = E[E.track == c["id"]]; est = [(int(round((a - c["offset"]) * FR)) / FR, int(round((b - c["offset"]) * FR)) / FR, FULL[t]) for a, b, t in zip(x.start_s, x.end_s, x.type)]; ref = [tuple(r) for r in c["ref"]]
        clips.append(dict(ref=ref, B=est)); mins[g] = mins.get(g, 0) + len(c["y"]) / FR / 60
        rl = {r[:2]: r[2] for r in ref}; el = {e[:2]: e[2] for e in est}; _, _, _, pairs = match_iou([(s, e, 1) for s, e, _ in ref], [(s, e, 1) for s, e, _ in est]); hit = {a[:2]: b[:2] for a, b in pairs}; used = set(hit.values())
        for s, e, l in ref:
            d = hit.get((s, e)); marks.append(dict(track=c["id"], group=g, type=TYPES[l], start_s=round(s + c["offset"], 3), end_s=round(e + c["offset"], 3), length_s=round(e - s, 3), located=d is not None, named_correct=d is not None and el[d] == l, detected_type=TYPES[el[d]] if d else "", detected_length_s=round(d[1] - d[0], 3) if d else np.nan))
        for s, e, l in est:
            if (s, e) not in used: dets.append(dict(track=c["id"], group=g, detected_type=TYPES[l], start_s=round(s + c["offset"], 3), end_s=round(e + c["offset"], 3), length_s=round(e - s, 3)))
M, U = pd.DataFrame(marks), pd.DataFrame(dets); macro, pc, _ = ev_scores(clips, "B"); nd = sum(len(c["B"]) for c in clips); assert abs(macro - .462) < .0006 and nd == 321 and int(M.located.sum()) == 182 and len(M) == 293 and len(U) == nd - 182
M["bin"] = pd.cut(M.length_s, EDGES, right=False, labels=BINS); U["bin"] = pd.cut(U.length_s, EDGES, right=False, labels=BINS); tm = sum(mins.values()); TY = list(TYPES.values())

# ---------------------------------------- logistic models (unpenalised apart from a tiny ridge for numerical stability)
def fit(X, y, ridge=1e-6):
    b = np.zeros(X.shape[1])
    for _ in range(100):
        p = np.clip(1 / (1 + np.exp(-np.clip(X @ b, -30, 30))), 1e-9, 1 - 1e-9); w = p * (1 - p); step = np.linalg.solve((X.T * w) @ X + ridge * np.eye(X.shape[1]), X.T @ (y - p) - ridge * b); b += np.clip(step, -3, 3)
        if np.abs(step).max() < 1e-8: break
    return b
def dev(X, y, b): p = np.clip(1 / (1 + np.exp(-np.clip(X @ b, -30, 30))), 1e-12, 1 - 1e-12); return -2 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
def xp(a, s, p): return float(np.exp((np.log(p / (1 - p)) - a) / s)) if s > 0 else np.inf
def design(D):
    base = [np.ones(len(D)), np.log(D.length_s.values)]
    return np.column_stack(base), np.column_stack(base + [(D.type == t).values.astype(float) for t in TY[1:]])
X1, X2 = design(M); y = M.located.values.astype(float); b1, b2 = fit(X1, y), fit(X2, y); rng = np.random.default_rng(SEED); picks = rng.integers(0, 11, (NB, 11)); IDX = {g: np.flatnonzero(M.group == g) for g in GROUPS}; UI = {g: np.flatnonzero(U.group == g) for g in GROUPS}; grid = np.geomspace(.1, 3, 200)
rows = {k: [] for k in ("b1", "b2", "curve")}; bin_all = {(t, b): [] for t in ["All"] + TY for b in BINS}; named_all = {(t, b): [] for t in ["All"] + TY for b in BINS}; ua = {b: [] for b in ["All"] + BINS}
for s in picks:
    gs = [GROUPS[i] for i in s]; I = np.concatenate([IDX[g] for g in gs]); D = M.iloc[I]; A, B = X1[I], X2[I]; yy = y[I]; c1, c2 = fit(A, yy), fit(B, yy); rows["b1"].append(c1); rows["b2"].append(c2); rows["curve"].append(1 / (1 + np.exp(-np.clip(c1[0] + c1[1] * np.log(grid), -30, 30))))
    for t in ["All"] + TY:
        Q = D if t == "All" else D[D.type == t]
        for b in BINS:
            q = Q[Q.bin == b]
            if len(q): bin_all[(t, b)].append(q.located.mean()); named_all[(t, b)].append(q.named_correct.mean())
    minutes = sum(mins[g] for g in gs); V = U.iloc[np.concatenate([UI[g] for g in gs])]; ua["All"].append(len(V) / minutes)
    for b in BINS: ua[b].append(int((V.bin == b).sum()) / minutes)
def ci(v): v = np.asarray(v, float); return (np.nan, np.nan) if len(v) == 0 else (float(np.percentile(v, 2.5, method="lower")), float(np.percentile(v, 97.5, method="higher")))
R1, R2, C = np.array(rows["b1"]), np.array(rows["b2"]), np.array(rows["curve"]); rangeA = (M.length_s.min(), M.length_s.max())

say("DETECTION LIMIT, REAL MARKS. EXPLORATORY. Frozen Step 3 primary run; recorded vocal, one singer; 293 marks; held-out predictions from 11 groups.")
say(f"Located = paired one-to-one with a CNN event of any type at IoU >= 0.3. Located {int(M.located.sum())}/293 ({M.located.mean():.3f}); located and right type {int(M.named_correct.sum())}/293 ({M.named_correct.mean():.3f}). Event macro-F1 check {macro:.3f}, detections {nd}.")
say("\n1a. located ~ log(length). Length at which located probability reaches 50% and 80%; 95% CI from 2000 resamples of the 11 groups")
tab = []
for p in (.5, .8):
    v = xp(b1[0], b1[1], p); lo, hi = ci([xp(r[0], r[1], p) for r in R1]); num("real_model1", "All", f"X{int(p*100)}_ms", v * 1000, lo * 1000, hi * 1000, 293); tab.append(dict(point=f"X{int(p*100)}", length_ms=round(v * 1000), ci95=f"{lo*1000:.0f} to {hi*1000:.0f}", within_observed_lengths=bool(rangeA[0] <= v <= rangeA[1])))
say(pd.DataFrame(tab).to_string(index=False)); l, h = ci(R1[:, 1]); say(f"Slope per log(length): {b1[1]:.2f} (95% CI {l:.2f} to {h:.2f}); observed mark lengths {rangeA[0]*1000:.0f} to {rangeA[1]*1000:.0f} ms"); num("real_model1", "All", "slope_log_length", b1[1], l, h, 293)
say("\n1b. located ~ log(length) + type (Kampita is the reference; one common length slope)")
ch = dev(X1, y, b1) - dev(X2, y, b2); say(f"Likelihood-ratio test for adding type: chi2 = {ch:.2f}, df 3, p = {chi2.sf(ch, 3):.4f} (this p-value ignores grouping; use the bootstrap intervals below as the safer guide)"); l, h = ci(R2[:, 1]); say(f"Slope per log(length): {b2[1]:.2f} (95% CI {l:.2f} to {h:.2f})"); num("real_model2", "All", "slope_log_length", b2[1], l, h, 293); tab = []
for i, t in enumerate(TY):
    a = b2[0] + (b2[1 + i] if i else 0); ra = R2[:, 0] + (R2[:, 1 + i] if i else 0); q = M[M.type == t]; row = dict(type=t, n=len(q), observed_ms=f"{q.length_s.min()*1000:.0f}-{q.length_s.max()*1000:.0f}")
    for p in (.5, .8):
        v = xp(a, b2[1], p); lo, hi = ci([xp(u, r[1], p) for u, r in zip(ra, R2)]); inside = q.length_s.min() <= v <= q.length_s.max(); row[f"X{int(p*100)}_ms"] = round(v * 1000); row[f"X{int(p*100)}_ci95"] = f"{lo*1000:.0f} to {hi*1000:.0f}"; row[f"X{int(p*100)}_in_range"] = bool(inside); num("real_model2", t, f"X{int(p*100)}_ms", v * 1000, lo * 1000, hi * 1000, len(q))
    if i: l, h = ci(np.exp(R2[:, 1 + i])); row["odds_vs_Kampita"] = f"{np.exp(b2[1 + i]):.2f} ({l:.2f} to {h:.2f})"; num("real_model2", t, "odds_ratio_vs_Kampita", float(np.exp(b2[1 + i])), l, h, len(q))
    else: row["odds_vs_Kampita"] = "reference"
    tab.append(row)
say(pd.DataFrame(tab).to_string(index=False))
say("\n1c. Located rate by mark length (s): located/n, rate; group-bootstrap 95% CI for each rate"); B_ROWS = []
for t in ["All"] + TY:
    Q = M if t == "All" else M[M.type == t]
    for b in BINS:
        q = Q[Q.bin == b]; n = len(q)
        lo, hi = ci(bin_all[(t, b)]); nl, nh = ci(named_all[(t, b)]); r = q.located.mean() if n else np.nan; rn = q.named_correct.mean() if n else np.nan; B_ROWS.append(dict(type=t, length_s=b, n=n, located=int(q.located.sum()), located_rate=r, ci_low=lo, ci_high=hi, located_right_type=int(q.named_correct.sum()), right_type_rate=rn)); num("real_bins", f"{t} | {b}", "located_rate", r, lo, hi, n); num("real_bins", f"{t} | {b}", "located_and_right_type_rate", rn, nl, nh, n)
BT = pd.DataFrame(B_ROWS); say(BT.assign(located=BT.located.astype(str) + "/" + BT.n.astype(str), right_type=BT.located_right_type.astype(str) + "/" + BT.n.astype(str), located_rate=BT.located_rate.round(2), ci=[f"{a:.2f}-{b:.2f}" if b == b else "" for a, b in zip(BT.ci_low, BT.ci_high)])[["type", "length_s", "located", "located_rate", "ci", "right_type"]].to_string(index=False))

say(f"\n2. False alarms: detections not paired with any mark. Evaluated audio {tm:.2f} min (after the opening Sa is cut)"); l, h = ci(ua["All"]); say(f"Unmatched detections: {len(U)}/{nd} ({len(U)/nd:.1%}), {len(U)/tm:.2f} per minute (95% CI {l:.2f} to {h:.2f})"); num("false_alarms", "All", "per_minute", len(U) / tm, l, h, len(U)); ALL = pd.concat([U.length_s, M.detected_length_s.dropna()]); tb = pd.cut(ALL, EDGES, right=False, labels=BINS).value_counts(); tab = []
for b in BINS:
    n = int((U.bin == b).sum()); l, h = ci(ua[b]); tab.append(dict(detection_length_s=b, unmatched=n, all_detections=int(tb[b]), unmatched_share=round(n / max(int(tb[b]), 1), 2), per_minute=round(n / tm, 2), ci95=f"{l:.2f} to {h:.2f}")); num("false_alarms", b, "per_minute", n / tm, l, h, n); num("false_alarms", b, "share_of_detections_unmatched", n / max(int(tb[b]), 1), n=int(tb[b]))
say(pd.DataFrame(tab).to_string(index=False)); say("These counts treat every unmarked region as false, although the singer says orikai and possibly other gamakas are unmarked.")

say("\n3. Minimum lengths in the frozen decoder. Viterbi region minimum, then the per-type length filter after naming; effective minimum = the larger of the two")
tab = []; R = []
for g in GROUPS:
    q = P.loc[g]; row = dict(group=g, region_min_ms=round(q.D_region_s * 1000)); eff = {}
    for t in TY: eff[t] = max(q.D_region_s, q[f"L_{t}"]); row[f"{t}_ms"] = round(eff[t] * 1000)
    tab.append(row); anymin = min(eff.values())
    for i in M.index[M.group == g]:
        t = M.at[i, "type"]; ln = M.at[i, "length_s"]; R.append(dict(idx=i, type_min_s=eff[t], lowest_any_type_min_s=anymin, below_type_min=ln < eff[t] - 1e-9, max_iou_right_type=min(1, ln / eff[t]), max_iou_any_type=min(1, ln / anymin)))
say(pd.DataFrame(tab).to_string(index=False)); Z = pd.DataFrame(R).set_index("idx"); M = M.join(Z); F = M[M.below_type_min]; F[["track", "group", "type", "start_s", "end_s", "length_s", "type_min_s", "located", "named_correct", "detected_type", "detected_length_s", "max_iou_right_type"]].to_csv(OUT["marks_below_type_minimum.csv"], index=False)
tab = []
for t in ["All"] + TY:
    q = F if t == "All" else F[F.type == t]; n = int(len(M) if t == "All" else (M.type == t).sum()); tab.append(dict(type=t, marks=n, shorter_than_type_minimum=len(q), share=round(len(q) / n, 3), still_located=int(q.located.sum()), still_right_type=int(q.named_correct.sum()), impossible_right_type=int((q.max_iou_right_type < .3).sum()))); num("minimums", t, "marks_below_type_minimum", len(q), n=n); num("minimums", t, "below_minimum_but_right_type", int(q.named_correct.sum()), n=len(q)); num("minimums", t, "below_minimum_cannot_reach_iou_0.3", int((q.max_iou_right_type < .3).sum()), n=len(q))
say(pd.DataFrame(tab).to_string(index=False)); say("A detection must meet the minimum, but a shorter mark can still match it when IoU >= 0.3. A mark is impossible for its own type only when mark length / type minimum < 0.3.")
say("\n4. Controlled insertion test: NOT RUN. The Step 3 networks were not saved, so the frozen detector cannot process new pitch tracks without retraining.")

M.drop(columns=["bin"]).to_csv(OUT["marks_located.csv"], index=False); U.drop(columns=["bin"]).to_csv(OUT["unmatched_detections.csv"], index=False); pd.DataFrame(NUM).to_csv(OUT["real_limit_numbers.csv"], index=False)
# ---------------------------------------- figure
INK, MUTED, GRID, BLUE = "#26313f", "#667085", "#e6e8ec", "#1f5fa8"; plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(9, 5.4)); ms = grid * 1000; ax.fill_between(ms, np.percentile(C, 2.5, axis=0), np.percentile(C, 97.5, axis=0), color=BLUE, alpha=.15, lw=0); ax.plot(ms, 1 / (1 + np.exp(-(b1[0] + b1[1] * np.log(grid)))), color=BLUE, lw=2, label="Logistic fit with 95% CI")
A_ = BT[BT.type == "All"]; mid = [M[M.bin == b].length_s.median() * 1000 for b in BINS]; ax.errorbar(mid, A_.located_rate, yerr=[A_.located_rate - A_.ci_low, A_.ci_high - A_.located_rate], fmt="o", ms=7, color=BLUE, mec="white", mew=1.5, elinewidth=1.3, capsize=0, label="Located rate by length band")
for x, r, h, n in zip(mid, A_.located_rate, A_.ci_high, A_.n): ax.text(x, min(h + .025, 1.02), f"n={n}", ha="center", fontsize=8.5, color=MUTED)
for p in (.5, .8):
    v = xp(b1[0], b1[1], p) * 1000
    if 100 <= v <= 3000: ax.plot([v, v], [0, p], color=MUTED, lw=1, ls=(0, (3, 3))); ax.plot([100, v], [p, p], color=MUTED, lw=1, ls=(0, (3, 3))); ax.text(v * 1.03, .025, f"X{int(p*100)} {v:.0f} ms", fontsize=9.5, color=INK)
ax.set_xscale("log"); ax.set_xlim(100, 3000); ax.set_ylim(0, 1.06); T = [100, 200, 300, 500, 800, 1000, 2000, 3000]; ax.set_xticks(T); ax.set_xticklabels(T); ax.minorticks_off(); ax.set_xlabel("Marked gamaka length (ms, log scale)"); ax.set_ylabel("Probability located (any type, IoU ≥ 0.3)"); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}")); ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True); ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(length=0)
ax.set_title("Real marks: longer gamakas are located more reliably", loc="left", fontsize=13, color=INK, pad=12); ax.legend(frameon=False, loc="lower right", fontsize=9.5, bbox_to_anchor=(1, .08)); fig.tight_layout(); fig.savefig(OUT["real_detection_curve.png"], dpi=160); OUT["real_limit_report.txt"].write_text("\n".join(L) + "\n")
