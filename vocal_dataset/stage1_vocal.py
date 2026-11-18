"""Stage 1 (PLAN.md section 5): given the singer's start and end times, can the 4 gamaka types be told apart from pitch?

Run from vocal_dataset:  python stage1_vocal.py
Needs crepe/*.npz and sa_hz in metadata.csv (tonic_from_opening_sa.py).
Writes results/stage1/: results.json, predictions.csv, report.txt, confusion.png, curves_by_class.png

Splits (leave-one-group-out, predictions pooled over held-out groups):
  main    : metadata.csv group (PROVISIONAL until the singer confirms)  -> 11 folds
  block6  : TRACK 1-6, 7-12, ..., 25-30, VARNAM                          -> 7 folds (robustness)
Label sets: 4 classes (main) and 3 classes with Etra + Irakka merged into Jaaru (secondary).
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from vocal_pitch import event_curve, load_cents, load_tables

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "stage1"
CL4 = ["Kampita", "Nokku", "Jaaru (Etra)", "Jaaru (Irakka)"]
CL3 = ["Kampita", "Nokku", "Jaaru"]
FR, MIN_VOICED, N_PERM, N_BOOT, SEED = 100, 5, 1000, 2000, 20261117
NOLEN = [1, 3, 4, 5, 6, 7, 8]  # features without dur and turns
FEAT = ["dur", "range", "turns", "turn_rate", "net", "abs_net", "exc_up", "exc_down", "mean_abs_vel"]


def uf3(x):
    p = np.concatenate([x[:1], x, x[-1:]])
    return (p[:-2] + p[1:-1] + p[2:]) / 3


def seg_feats(c):
    """The 9 stage-1 features of veena_stage1 (same code as methods_explore/classify.py)."""
    sm = uf3(c)
    v = np.diff(sm)
    sgn = np.sign(v[np.abs(v) > 1.0])
    turns = int(np.sum(sgn[1:] != sgn[:-1])) if len(sgn) > 1 else 0
    dur = len(c) / FR
    base = c[0]
    return [dur, np.percentile(c, 95) - np.percentile(c, 5), turns, turns / dur, c[-3:].mean() - c[:3].mean(),
            abs(c[-3:].mean() - c[:3].mean()), c.max() - base, base - c.min(), np.abs(v).mean() * FR]


@njit(cache=True)
def dtw(a, b, band):
    """Absolute-difference DTW inside a Sakoe-Chiba band, normalised by n + m."""
    n, m = len(a), len(b)
    w = max(band, abs(n - m))
    inf = 1e18
    prev = np.full(m + 1, inf)
    cur = np.full(m + 1, inf)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur[:] = inf
        for j in range(max(1, i - w), min(m, i + w) + 1):
            cur[j] = abs(a[i - 1] - b[j - 1]) + min(prev[j], prev[j - 1], cur[j - 1])
        prev, cur = cur, prev
    return prev[m] / (n + m)


def centre(c):
    return np.asarray(c - np.median(c), dtype=np.float64)


def resample(c, n=100):
    return np.interp(np.linspace(0, len(c) - 1, n), np.arange(len(c)), c)


def dist_matrix(curves, band_frac=0.2):
    N = len(curves)
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            M[i, j] = M[j, i] = dtw(curves[i], curves[j], max(5, int(band_frac * max(len(curves[i]), len(curves[j])))))
    return M


LR = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))  # noqa: E731
KN = lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(3))  # noqa: E731


def logo_feat(X, y, g, mk):
    p = np.empty_like(y)
    for grp in np.unique(g):
        te = g == grp
        p[te] = mk().fit(X[~te], y[~te]).predict(X[te])
    return p


def logo_dtw(M, y, g, k, ncls):
    p = np.empty_like(y)
    for grp in np.unique(g):
        te, tr = np.where(g == grp)[0], np.where(g != grp)[0]
        for i in te:
            o = tr[np.argsort(M[i, tr], kind="stable")[:k]]
            v = np.bincount(y[o], minlength=ncls)
            best = np.flatnonzero(v == v.max())
            p[i] = next(y[q] for q in o if y[q] in best)  # tie: the nearest neighbour among the tied classes
    return p


def run(cfg, y, g, ncls):
    return logo_feat(cfg[1], y, g, cfg[2]) if cfg[0] == "f" else logo_dtw(cfg[1], y, g, cfg[2], ncls)


def mf1(y, p, ncls):
    return float(f1_score(y, p, average="macro", labels=range(ncls)))


def evaluate(y, g, CFG, CTRL, ncls, names_cls, perm=True):
    """All configurations on one split and one label set, plus the within-group permutation test (family = CFG)."""
    res, pred = {}, {}
    for name, cfg in {**CFG, **CTRL}.items():
        p = run(cfg, y, g, ncls)
        pred[name] = p
        res[name] = dict(macro_f1=mf1(y, p, ncls), acc=float((p == y).mean()),
                         recall=dict(zip(names_cls, recall_score(y, p, average=None, labels=range(ncls)).tolist())),
                         confusion=confusion_matrix(y, p, labels=range(ncls)).tolist())
    maj = np.full_like(y, np.bincount(y).argmax())
    res["majority"] = dict(macro_f1=mf1(y, maj, ncls), acc=float((maj == y).mean()))
    if perm:
        rng = np.random.default_rng(SEED)
        names = list(CFG)
        N = np.zeros((N_PERM, len(names)))
        groups = [np.where(g == grp)[0] for grp in np.unique(g)]
        for b in range(N_PERM):
            yp = y.copy()
            for ix in groups:
                yp[ix] = y[ix][rng.permutation(len(ix))]
            for j, n in enumerate(names):
                N[b, j] = mf1(yp, run(CFG[n], yp, g, ncls), ncls)
            if (b + 1) % 200 == 0:
                print("   perm", b + 1, flush=True)
        mx = N.max(axis=1)
        for j, n in enumerate(names):
            o = res[n]["macro_f1"]
            res[n].update(p_single=float((1 + (N[:, j] >= o).sum()) / (N_PERM + 1)),
                          p_family=float((1 + (mx >= o).sum()) / (N_PERM + 1)),
                          null_mean=float(N[:, j].mean()), null_95=float(np.percentile(N[:, j], 95)))
        res["family_null_95"] = float(np.percentile(mx, 95))
        res["family_null_mean"] = float(mx.mean())
    return res, pred


def boot_groups(y, p, g, ncls):
    r = np.random.default_rng(SEED)
    gs = np.unique(g)
    ix = {k: np.where(g == k)[0] for k in gs}
    out = [mf1(y[i], p[i], ncls) for i in (np.concatenate([ix[k] for k in r.choice(gs, len(gs))]) for _ in range(N_BOOT))]
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def block6(rec):
    return "B7_VARNAM" if rec == "VARNAM" else f"B{(int(rec.split()[-1]) - 1) // 6 + 1}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta, ev = load_tables()
    m = meta.set_index("recording")
    curves, rows, excluded, folds = [], [], [], {}
    for rec, e in ev.groupby("recording", sort=False):
        t, cents, conf, voiced, n_fold = load_cents(rec, m.loc[rec, "sa_hz"])
        folds[rec] = n_fold
        for _, r in e.iterrows():
            c, nv = event_curve(t, cents, voiced, r.start_s, r.end_s)
            if nv < MIN_VOICED:
                excluded.append(dict(recording=rec, start_s=r.start_s, end_s=r.end_s, label=r.label, voiced_frames=nv))
                continue
            curves.append(c)
            rows.append(dict(recording=rec, start_s=r.start_s, end_s=r.end_s, label=r.label, group=m.loc[rec, "group"],
                             tempo=m.loc[rec, "tempo"] or "(varnam)", block6=block6(rec), voiced_frames=nv,
                             median_cents=float(np.median(c))))
    d = pd.DataFrame(rows)
    print(f"events used {len(d)}, excluded (<{MIN_VOICED} voiced frames) {len(excluded)}; octave slips folded {sum(folds.values())}")
    y4 = d.label.map({c: i for i, c in enumerate(CL4)}).values
    y3 = d.label.map(lambda s: 2 if s.startswith("Jaaru") else CL3.index(s)).values

    F = np.array([seg_feats(c) for c in curves])
    rs = np.random.default_rng(SEED)
    scr = [rs.permutation(c) for c in curves]
    Fs = np.array([seg_feats(c) for c in scr])
    print("DTW matrices ...", flush=True)
    DM = dist_matrix([centre(c) for c in curves])
    DMnl = dist_matrix([centre(resample(c)) for c in curves])
    DMs = dist_matrix([centre(c) for c in scr])

    CFG = {"F-LR": ("f", F, LR), "F-3NN": ("f", F, KN), "DTW-3NN": ("d", DM, 3), "DTW-1NN": ("d", DM, 1)}
    CTRL = {"control: duration only LR": ("f", F[:, [0]], LR),
            "control: median pitch probe LR": ("f", d[["median_cents"]].values, LR),
            "control: length removed F-LR": ("f", F[:, NOLEN], LR),
            "control: length removed DTW-3NN": ("d", DMnl, 3),
            "control: scramble F-LR": ("f", Fs, LR),
            "control: scramble DTW-3NN": ("d", DMs, 3)}

    R = {"n_events": int(len(d)), "excluded": excluded, "octave_slips_folded": folds,
         "n_per_class": d.label.value_counts().to_dict(), "groups": sorted(d.group.unique().tolist())}
    preds = {}
    for split, gcol in (("main", "group"), ("block6", "block6")):
        g = d[gcol].values
        for lab, y, ncls, names in (("4class", y4, 4, CL4), ("3class", y3, 3, CL3)):
            key = f"{split}_{lab}"
            print(f"== {key}", flush=True)
            res, pred = evaluate(y, g, CFG, CTRL, ncls, names)
            for n in CFG:
                print(f"   {n:34s} F1 {res[n]['macro_f1']:.3f}  p_fam {res[n]['p_family']:.3f}", flush=True)
            R[key] = res
            preds[key] = pred

    # main analysis extras
    main = R["main_4class"]
    g = d.group.values
    best = max(CFG, key=lambda n: main[n]["macro_f1"])
    main["best"] = best
    for n in {best, "DTW-3NN"}:
        main[n]["ci95_groups"] = boot_groups(y4, preds["main_4class"][n], g, 4)
    tracks = (d.recording != "VARNAM").values
    for n in list(CFG) + list(CTRL):
        p = preds["main_4class"][n]
        main[n]["tracks_only_macro_f1"] = mf1(y4[tracks], p[tracks], 4)
        main[n]["by_tempo"] = {tp: dict(n=int((d.tempo == tp).sum()), macro_f1=mf1(y4[(d.tempo == tp).values], p[(d.tempo == tp).values], 4),
                                        acc=float((p == y4)[(d.tempo == tp).values].mean()))
                               for tp in ("slow", "medium", "fast", "(varnam)")}
        main[n]["per_group_acc"] = {k: float((p == y4)[g == k].mean()) for k in np.unique(g)}
    probe = main["control: median pitch probe LR"]["macro_f1"]
    bm = main[best]
    R["PASS_stage1"] = bool(bm["macro_f1"] >= 0.50 and bm["p_family"] < 0.05 and bm["macro_f1"] > probe)
    R["DTW_confirmed"] = bool(main["DTW-3NN"]["macro_f1"] >= 0.50 and main["DTW-3NN"]["p_family"] < 0.05)
    json.dump(R, open(OUT / "results.json", "w"), indent=1, default=float)

    P = d.copy()
    for key, pr in preds.items():
        for n, p in pr.items():
            names = CL4 if key.endswith("4class") else CL3
            P[f"{key} | {n}"] = [names[i] for i in p]
    P.to_csv(OUT / "predictions.csv", index=False)
    d.assign(**{f: F[:, i] for i, f in enumerate(FEAT)}).to_csv(OUT / "event_features.csv", index=False)
    report(R, d, F, best)
    figures(R, d, curves, best)


def report(R, d, F, best):
    L = []
    main = R["main_4class"]
    L.append("STAGE 1 (recorded vocal, 1 singer; grouping PROVISIONAL until the singer confirms it)")
    L.append(f"events used {R['n_events']} (excluded {len(R['excluded'])} with < {MIN_VOICED} voiced frames); per class {R['n_per_class']}")
    L.append(f"octave slips folded: {sum(R['octave_slips_folded'].values())} runs over all recordings")
    for key in ("main_4class", "main_3class", "block6_4class", "block6_3class"):
        r = R[key]
        L.append(f"\n--- {key}  (majority macro-F1 {r['majority']['macro_f1']:.3f}; family null mean {r['family_null_mean']:.3f}, "
                 f"95th {r['family_null_95']:.3f})")
        L.append(f"{'configuration':36s} {'macroF1':>7s} {'acc':>6s} {'p_single':>8s} {'p_family':>8s} {'tracksonly':>10s}")
        for n, v in r.items():
            if not isinstance(v, dict) or "macro_f1" not in v or n == "majority":
                continue
            ps = f"{v['p_single']:8.3f} {v['p_family']:8.3f}" if "p_single" in v else f"{'':8s} {'':8s}"
            to = f"{v['tracks_only_macro_f1']:10.3f}" if "tracks_only_macro_f1" in v else ""
            L.append(f"{n:36s} {v['macro_f1']:7.3f} {v['acc']:6.3f} {ps} {to}")
    L.append(f"\nbest trained configuration (main, 4 classes): {best}, 95% CI over groups {main[best]['ci95_groups']}")
    L.append(f"DTW-3NN 95% CI over groups {main['DTW-3NN']['ci95_groups']}")
    for n in sorted({best, "DTW-3NN"}):
        L.append(f"\n{n}: recall {', '.join(f'{k} {v:.2f}' for k, v in main[n]['recall'].items())}")
        L.append("confusion (rows = singer's label, cols = predicted; order " + ", ".join(CL4) + ")")
        L.append(str(np.array(main[n]["confusion"])))
        L.append("by tempo: " + ", ".join(f"{k} n={v['n']} F1={v['macro_f1']:.2f}" for k, v in main[n]["by_tempo"].items()))
        L.append("per group acc: " + ", ".join(f"{k} {v:.2f}" for k, v in main[n]["per_group_acc"].items()))
    L.append("\nclass means of the 9 features:")
    L.append(d.assign(**{f: F[:, i] for i, f in enumerate(FEAT)}).groupby("label")[FEAT].mean().round(2).to_string())
    probe = main["control: median pitch probe LR"]["macro_f1"]
    L.append(f"\nPASS criterion: best >= 0.50 ({main[best]['macro_f1']:.3f}), p_family < 0.05 ({main[best]['p_family']:.3f}), "
             f"beats median-pitch probe ({probe:.3f})  ->  {'PASS' if R['PASS_stage1'] else 'FAIL'}")
    L.append(f"veena DTW result confirmed on vocal (DTW-3NN >= 0.50 and p_family < 0.05): {R['DTW_confirmed']}")
    txt = "\n".join(L)
    print(txt)
    (OUT / "report.txt").write_text(txt, encoding="utf-8")


def figures(R, d, curves, best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main = R["main_4class"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, n in zip(axs, [best, "DTW-3NN"] if best != "DTW-3NN" else ["DTW-3NN", "F-LR"]):
        C = np.array(main[n]["confusion"])
        Cn = C / C.sum(axis=1, keepdims=True)
        ax.imshow(Cn, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{C[i, j]}\n{Cn[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if Cn[i, j] > 0.6 else "black")
        ax.set_xticks(range(4), [c.replace("Jaaru ", "") for c in CL4], fontsize=8)
        ax.set_yticks(range(4), [c.replace("Jaaru ", "") for c in CL4], fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("singer's label")
        ax.set_title(f"{n}: macro-F1 {main[n]['macro_f1']:.2f}, p_fam {main[n]['p_family']:.3f}", fontsize=9)
    fig.suptitle("Stage 1, recorded vocal, leave-one-group-out (grouping provisional)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "confusion.png", dpi=110)
    plt.close(fig)

    col = {"Kampita": "#4daf4a", "Nokku": "#e41a1c", "Jaaru (Etra)": "#377eb8", "Jaaru (Irakka)": "#984ea3"}
    fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, c in zip(axs, CL4):
        ix = np.where(d.label.values == c)[0]
        for i in ix:
            x = curves[i]
            ax.plot(np.arange(len(x)) / FR, x - x[0], color=col[c], alpha=0.25, lw=0.8)
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_title(f"{c} (n={len(ix)})", fontsize=9)
        ax.set_xlabel("time from event start (s)")
    axs[0].set_ylabel("cents relative to the first frame")
    axs[0].set_ylim(-700, 700)
    fig.suptitle("Every annotated event's pitch curve (recorded vocal), shifted to start at 0", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "curves_by_class.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
