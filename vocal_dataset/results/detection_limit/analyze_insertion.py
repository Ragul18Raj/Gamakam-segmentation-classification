"""Analyse the controlled insertion test and compare it with real marks, per gamaka type. EXPLORATORY.
X50/X80 are reported only when the fitted curve rises and crosses the level inside the tested 80-800 ms range.
Run from vocal_dataset:  python results/detection_limit/analyze_insertion.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; VD = HERE.parents[1]; sys.path[:0] = [str(VD), str(HERE)]
import run_insertion as RI
I = HERE / "insertion"; OUT = {n: I / n for n in ("insertion_report.txt", "insertion_rates.csv", "insertion_fits.csv", "pitch_shift_diagnostic.csv", "real_vs_insertion_by_type.png")}
if any(p.exists() for p in OUT.values()): sys.exit("Output files already exist; not overwriting them.")
T = pd.read_csv(I / "insertion_trials.csv"); M = pd.read_csv(HERE / "marks_located.csv"); TY = ["Kampita", "Nokku", "Etra", "Irakka"]; LEN = list(RI.LENGTHS); G = RI.GROUPS; NB = 2000; rng = np.random.default_rng(20261119); PICK = rng.integers(0, 11, (NB, 11)); L = []
def say(s=""): print(s, flush=True); L.append(str(s))
def fit(x, y, ridge=1e-6):
    X = np.c_[np.ones(len(x)), np.log(x)]; b = np.zeros(2)
    for _ in range(100):
        p = np.clip(1 / (1 + np.exp(-np.clip(X @ b, -30, 30))), 1e-9, 1 - 1e-9); w = p * (1 - p); s = np.linalg.solve((X.T * w) @ X + ridge * np.eye(2), X.T @ (y - p) - ridge * b); b += np.clip(s, -3, 3)
        if np.abs(s).max() < 1e-8: break
    return b
def xp(b, p): return float(np.exp((np.log(p / (1 - p)) - b[0]) / b[1])) if b[1] > 0 else np.inf
def pct(v): v = np.asarray(v, float); v = v[~np.isnan(v)]; return (np.nan, np.nan) if not len(v) else (float(np.percentile(v, 2.5, method="lower")), float(np.percentile(v, 97.5, method="higher")))
def samples(D):
    idx = {g: np.flatnonzero(D.group.values == g) for g in G}
    for s in PICK: yield np.concatenate([idx[G[i]] for i in s])
def run_fit(D, col, source, typ):
    x = D.ms.values.astype(float); y = D[col].values.astype(float); b = fit(x, y); B = [fit(x[i], y[i]) for i in samples(D) if y[i].min() != y[i].max()]; B = np.array(B); out = dict(source=source, type=typ, outcome=col, n=len(D), successes=int(y.sum()), slope_per_log_length=b[1], slope_ci_low=pct(B[:, 1])[0], slope_ci_high=pct(B[:, 1])[1], bootstrap_fits=len(B)); lo, hi = x.min(), x.max()
    for p in (.5, .8):
        k = f"X{int(p*100)}"; v = xp(b, p) if y.min() != y.max() else np.inf
        if b[1] <= 0: st, val, a, c = "no upward length trend", np.nan, np.nan, np.nan
        elif v < 80: st, val, a, c = "below 80 ms", np.nan, np.nan, np.nan
        elif v > 800: st, val, a, c = "not reached by 800 ms", np.nan, np.nan, np.nan
        else: a, c = pct([xp(r, p) for r in B]); st, val = ("inside tested range" if lo <= v <= hi else "inside tested range, outside observed marks"), v
        out.update({f"{k}_status": st, f"{k}_ms": val, f"{k}_ci_low": a, f"{k}_ci_high": c})
    return out, b, B
T["group"] = T.held_out_group; T["ms"] = T.length_ms; M["ms"] = M.length_s * 1000; M = M.rename(columns={"named_correct": "right_type"}); X = T[T.type != "Flat control"]; rates = []
for t in TY:
    D = X[X.type == t]
    for ms in LEN:
        q = D[D.ms == ms]; r = dict(type=t, length_ms=ms, insertions=len(q), region_found=int(q.region_found.sum()))
        for col in ("located", "right_type"):
            by = q.groupby("group")[col].agg(["sum", "size"]).reindex(G).fillna(0); a, n = by["sum"].values, by["size"].values; v = [a[s].sum() / max(n[s].sum(), 1) for s in PICK]; lo, hi = pct(v); r.update({col: int(q[col].sum()), f"{col}_rate": q[col].mean(), f"{col}_ci_low": lo, f"{col}_ci_high": hi})
        rates.append(r)
R = pd.DataFrame(rates); fits = []; FB = {}
for t in TY:
    for src, D in (("real marks", M[M.type == t]), ("insertions", X[X.type == t])):
        for col in ("located", "right_type"):
            o, b, B = run_fit(D, col, src, t); fits.append(o); FB[(src, t, col)] = (b, B, D.ms.min(), D.ms.max())
F = pd.DataFrame(fits)
say("CONTROLLED INSERTION TEST. EXPLORATORY. Frozen Step 3 detector and Viterbi settings; recorded vocal, one singer; contours from each fold's training groups; held-out tracks.")
say("Reproduction check passed in all 11 folds (correlation 1.0, event F1 difference 0). Located = paired at IoU >= 0.3 with a final detection of any type.\n")
say(f"Trials: {len(X)} gamaka insertions ({X.groupby(['type', 'ms']).size().min()} per type x length) and {int((T.type == 'Flat control').sum())} flat-contour controls. Pitch mismatch with the surrounding melody before smoothing: median {X.pitch_mismatch_cents.abs().median():.0f} c, 90th percentile {X.pitch_mismatch_cents.abs().quantile(.9):.0f} c.")
C = T[T.type == "Flat control"]; say(f"Flat control: located {int(C.located.sum())}/{len(C)}; candidate regions before naming and length filtering {int(C.region_found.sum())}/{len(C)}. Upper 97.5% bound on located rate {1 - .025 ** (1 / len(C)):.2%}.\n")
for col, name in (("located", "Located, any type"), ("right_type", "Located and right type")):
    say(name + ": count / insertions"); say(R.assign(v=R[col].astype(str) + "/" + R.insertions.astype(str)).pivot(index="type", columns="length_ms", values="v").reindex(TY).to_string()); say()
say("Group-bootstrap 95% CIs for every rate are in insertion_rates.csv.\n\nPer-type logistic fits: outcome ~ log(length). X values are shown only inside 80-800 ms")
def show(r, k): return f"{r[k+'_ms']:.0f} ({r[k+'_ci_low']:.0f}-{r[k+'_ci_high']:.0f})" + ("*" if "outside" in r[k + "_status"] else "") if r[k + "_ms"] == r[k + "_ms"] else r[k + "_status"]
say(pd.DataFrame([dict(source=r.source, type=r.type, outcome=r.outcome, n=r.n, successes=r.successes, slope=f"{r.slope_per_log_length:.2f} ({r.slope_ci_low:.2f} to {r.slope_ci_high:.2f})", X50=show(r, "X50"), X80=show(r, "X80")) for _, r in F.iterrows()]).to_string(index=False)); say("* inside 80-800 ms but outside the lengths observed for that real-mark type.")
SRC = RI.sources(); key = {(s["rec"], s["start_s"]): s for s in SRC}; sh = []
for _, r in X.iterrows():
    s = key[(r.source_recording, r.source_start_s)]; n = int(r.ms // 10); a = int(r.start_frame); c = RI.BY[r.track]; f = RI.fill(c["cents"].astype(float)); pad = RI.J + RI.CTX; left, right = np.median(f[a - pad:a - RI.J]), np.median(f[a + n + RI.J:a + n + pad]); cur, _ = RI.scaled(s, n); o = ((left - cur[0]) + (right - cur[-1])) / 2; sh.append(abs(((o + 600) % 1200) - 600))
X = X.assign(pitch_class_shift_c=sh, shift_band=pd.cut(sh, [0, 100, 300, 601], right=False, labels=["<100", "100-300", ">=300"]), length_band=np.where(X.ms >= 400, "400-800 ms", "80-300 ms")); P = X.groupby(["length_band", "shift_band"], observed=True).agg(insertions=("located", "size"), located=("located", "sum"), located_rate=("located", "mean"), right_type=("right_type", "sum")).reset_index(); P.to_csv(OUT["pitch_shift_diagnostic.csv"], index=False); say("\nDiagnostic: located rate by how far insertion moved the contour from its original pitch class (cents, octaves ignored)"); say(P.round(3).to_string(index=False)); Pt = X[X.ms >= 400].groupby(["type", "shift_band"], observed=True).located.agg(["sum", "size"]); say("\n400-800 ms insertions by type: located/insertions"); say(Pt.assign(v=Pt["sum"].astype(str) + "/" + Pt["size"].astype(str)).v.unstack().reindex(TY).to_string())
R.to_csv(OUT["insertion_rates.csv"], index=False); F.to_csv(OUT["insertion_fits.csv"], index=False)
INK, MUTED, GRID, BLUE, ORANGE = "#26313f", "#667085", "#e6e8ec", "#1f5fa8", "#c8641e"; plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": GRID, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED}); fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True); EDGE = [0, 200, 300, 400, 600, 800, 1e9]
for a, t in zip(ax.flat, TY):
    for src, col, mk in (("real marks", BLUE, "s"), ("insertions", ORANGE, "o")):
        b, B, lo, hi = FB[(src, t, "located")]; x = np.geomspace(max(lo, 80), min(hi, 3000), 120); Y = 1 / (1 + np.exp(-np.clip(B[:, [0]] + B[:, [1]] * np.log(x), -30, 30))); a.fill_between(x, np.percentile(Y, 2.5, axis=0), np.percentile(Y, 97.5, axis=0), color=col, alpha=.14, lw=0); a.plot(x, 1 / (1 + np.exp(-(b[0] + b[1] * np.log(x)))), color=col, lw=2)
    q = R[R.type == t]; a.errorbar(q.length_ms, q.located_rate, yerr=[q.located_rate - q.located_ci_low, q.located_ci_high - q.located_rate], fmt="o", ms=5.5, color=ORANGE, mec="white", mew=1, elinewidth=1.1)
    D = M[M.type == t]; bin_ = pd.cut(D.ms, EDGE, right=False); g = D.groupby(bin_, observed=True).agg(x=("ms", "median"), y=("located", "mean"), n=("located", "size")); g = g[g.n >= 5]; a.plot(g.x, g.y, "s", ms=6, color=BLUE, mec="white", mew=1)
    a.set_xscale("log"); a.set_xlim(70, 3000); a.set_ylim(0, 1.03); a.set_title(t, loc="left", fontsize=12, color=INK); a.grid(axis="y", color=GRID, lw=.8); a.set_axisbelow(True); a.spines[["top", "right", "left"]].set_visible(False); a.tick_params(length=0); a.set_xticks([100, 200, 400, 800, 1600, 3000]); a.set_xticklabels([100, 200, 400, 800, 1600, 3000]); a.minorticks_off(); a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
for a in ax[1]: a.set_xlabel("Gamaka length (ms, log scale)")
for a in ax[:, 0]: a.set_ylabel("Located (any type, IoU ≥ 0.3)")
H = [plt.Line2D([], [], color=BLUE, marker="s", lw=2, ms=6, label="Real marks: fit, 95% CI, length bands with n ≥ 5"), plt.Line2D([], [], color=ORANGE, marker="o", lw=2, ms=6, label="Controlled insertions: fit, 95% CI, tested lengths")]; fig.suptitle("Exploratory: located rate by gamaka length, real marks and controlled insertions", x=.01, ha="left", fontsize=13.5, color=INK); fig.legend(handles=H, loc="upper left", bbox_to_anchor=(.008, .955), ncol=2, frameon=False, fontsize=10); fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(OUT["real_vs_insertion_by_type.png"], dpi=150); OUT["insertion_report.txt"].write_text("\n".join(L) + "\n")
