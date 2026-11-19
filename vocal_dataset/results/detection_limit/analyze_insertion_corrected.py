"""CORRECTED analysis of the controlled insertion test, with a line-search logistic solver checked against scikit-learn. EXPLORATORY.
The first analysis script used a solver that failed to converge on some fits; its files are left in place, unchanged.
X50/X80 are reported only when the fitted curve rises and crosses the level inside the tested 80-800 ms range.
Run from vocal_dataset:  python results/detection_limit/analyze_insertion_corrected.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; VD = HERE.parents[1]; sys.path[:0] = [str(VD), str(HERE)]
import run_insertion as RI
I = HERE / "insertion"; C_DIR = I / "corrected"; OUT = {n: C_DIR / n for n in ("insertion_report.txt", "insertion_rates.csv", "insertion_fits.csv", "pitch_shift_diagnostic.csv", "real_vs_insertion_by_type.png")}
if any(p.exists() for p in OUT.values()): sys.exit("Output files already exist; not overwriting them.")
T = pd.read_csv(I / "insertion_trials.csv"); M = pd.read_csv(HERE / "marks_located.csv"); TY = ["Kampita", "Nokku", "Etra", "Irakka"]; LEN = list(RI.LENGTHS); G = RI.GROUPS; NB = 2000; rng = np.random.default_rng(20261119); PICK = rng.integers(0, 11, (NB, 11)); L = []
def say(s=""): print(s, flush=True); L.append(str(s))
def newton(X, y, ridge=1e-6):
    """Maximise the logistic log-likelihood with step halving, so every accepted step improves it."""
    b = np.zeros(X.shape[1])
    def obj(v):
        e = np.clip(X @ v, -30, 30); return float(np.sum(y * e - np.logaddexp(0, e)) - ridge * (v @ v) / 2)
    done = False
    for _ in range(300):
        e = np.clip(X @ b, -30, 30); p = 1 / (1 + np.exp(-e)); w = np.maximum(p * (1 - p), 1e-10); s = np.linalg.solve((X.T * w) @ X + ridge * np.eye(X.shape[1]), X.T @ (y - p) - ridge * b); t = 1.0; base = obj(b)
        while obj(b + t * s) < base - 1e-12 and t > 1e-8: t /= 2
        b = b + t * s
        if np.abs(t * s).max() < 1e-8: done = True; break
    return b, done
def fit(x, y):
    """Coefficients on ln(ms), estimated with length centred at 300 ms for numerical stability."""
    z = np.log(np.asarray(x, float) / 300); b, _ = newton(np.c_[np.ones(len(z)), z], np.asarray(y, float)); return np.array([b[0] - b[1] * np.log(300), b[1]])
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
from sklearn.linear_model import LogisticRegression
CHECK = []
for (src, t, col), (b, _, _, _) in FB.items():
    D = (M if src == "real marks" else X); D = D[D.type == t]; z = np.log(D.ms.values / 300)[:, None]; m = LogisticRegression(penalty=None, max_iter=100000, tol=1e-10).fit(z, D[col].values.astype(int)); CHECK.append(abs(float(m.coef_[0, 0]) - b[1]))
assert max(CHECK) < .01, f"solver disagrees with scikit-learn: {max(CHECK)}"
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
say(f"\nSolver check: all {len(CHECK)} per-type point fits agree with scikit-learn (largest slope difference {max(CHECK):.5f}).")
# Recheck of the pooled and type-adjusted real-mark models reported earlier, with the same bootstrap resamples.
TN = ["Kampita", "Nokku", "Etra", "Irakka"]; yy = M.located.values.astype(float); ls = np.log(M.length_s.values); c0 = np.log(.4); X1 = np.c_[np.ones(len(M)), ls - c0]; X2 = np.column_stack([np.ones(len(M)), ls - c0] + [(M.type == t).values.astype(float) for t in TN[1:]]); Pk = np.random.default_rng(20261118).integers(0, 11, (NB, 11)); ID = {g: np.flatnonzero(M.group.values == g) for g in G}; b1, ok1 = newton(X1, yy); b2, ok2 = newton(X2, yy); R1, R2, bad = [], [], 0
for s_ in Pk:
    i = np.concatenate([ID[G[j]] for j in s_]); a1, o1 = newton(X1[i], yy[i]); a2, o2 = newton(X2[i], yy[i]); R1.append(a1); R2.append(a2); bad += (not o1) + (not o2)
R1, R2 = np.array(R1), np.array(R2)
def xs(a, slope, p): return np.where(slope > 0, np.exp(np.clip((np.log(p / (1 - p)) - a) / np.where(slope > 0, slope, 1) + c0, -50, 50)) * 1000, np.inf)
RC = [dict(model="located ~ log(length)", term=f"X{int(p*100)}_ms", value=float(xs(b1[0], b1[1], p)), ci_low=pct(xs(R1[:, 0], R1[:, 1], p))[0], ci_high=pct(xs(R1[:, 0], R1[:, 1], p))[1]) for p in (.5, .8)] + [dict(model="located ~ log(length)", term="slope", value=b1[1], ci_low=pct(R1[:, 1])[0], ci_high=pct(R1[:, 1])[1]), dict(model="located ~ log(length) + type", term="slope", value=b2[1], ci_low=pct(R2[:, 1])[0], ci_high=pct(R2[:, 1])[1])]
for q, t in enumerate(TN):
    a = b2[0] + (b2[1 + q] if q else 0); ra = R2[:, 0] + (R2[:, 1 + q] if q else 0)
    for p in (.5, .8): v = xs(ra, R2[:, 1], p); RC.append(dict(model="located ~ log(length) + type", term=f"{t} X{int(p*100)}_ms", value=float(xs(a, b2[1], p)), ci_low=pct(v)[0], ci_high=pct(v)[1]))
    if q: RC.append(dict(model="located ~ log(length) + type", term=f"{t} odds vs Kampita", value=float(np.exp(b2[1 + q])), ci_low=pct(np.exp(R2[:, 1 + q]))[0], ci_high=pct(np.exp(R2[:, 1 + q]))[1]))
RC = pd.DataFrame(RC); RC.to_csv(C_DIR / "real_model_recheck.csv", index=False); say(f"\nRecheck of the earlier real-mark models with the corrected solver (point fits converged: {ok1 and ok2}; unconverged bootstrap fits: {bad} of {2 * NB})"); say(RC.round(3).to_string(index=False))
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
