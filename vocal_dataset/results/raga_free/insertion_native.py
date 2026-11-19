"""Native-speed insertion test for the raga-free detector. EXPLORATORY.

Each fold uses its saved raga-free network and its tuned Viterbi, naming and length settings.
Contours come only from that fold's training groups and are inserted into sung, unmarked parts of held-out tracks,
at least 0.5 s from real marks and 0.2 s from the raga-free run's detections.

Nokku, Etra and Irakka: the source length must be 0.8 to 1.25 times the target length.
Kampita: whole pitch cycles, measured between neighbouring pitch minima, are cropped or repeated until the built contour is
0.8 to 1.25 times the target length. Every contour is then resampled to the exact target length.
Flat contours are the control. A cell can be empty when no eligible training contour exists.
Run from vocal_dataset:  python results/raga_free/insertion_native.py
"""
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
VD = HERE.parents[1]
sys.path[:0] = [str(VD), str(VD / "results/two_stage"), str(VD / "results/detection_limit"), str(HERE)]
import rf_common as R  # noqa: E402
import run_insertion as RI  # noqa: E402

TS, TV, torch = RI.TS, R.TV, R.TV["torch"]
PP = TS.PP
GROUPS, TYPES, SHORT = TS.GROUPS, TS.TYPES, TS.SHORT
LENGTHS, PER_FOLD, SEED = RI.LENGTHS, RI.PER_FOLD, 20261121
OUT_DIR = HERE / "insertion_native"
OUT = {n: OUT_DIR / n for n in ("native_insertion_report.txt", "native_insertion_trials.csv", "native_insertion_rates.csv", "native_insertion_fits.csv", "empty_cells.csv")}
PARAM = pd.read_csv(HERE / "chosen_params.csv").set_index("held_out_group")
RI.E = pd.read_csv(HERE / "events_raga_free.csv")     # sites avoid this detector's own detections
NET = {}


def get_net(g):
    if g not in NET:
        w = int(PARAM.loc[g, "window_frames"])
        n = R.new_net()
        n.load_state_dict(torch.load(HERE / f"weights/W{w}/fold_{g}.pt", map_location="cpu")["state_dict"])
        n.eval()
        NET[g] = n
    return NET[g]


def identity(g):
    """Unmodified held-out tracks must reproduce the saved raga-free probabilities."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    w = int(PARAM.loc[g, "window_frames"])
    z = np.load(HERE / f"probs/W{w}/fold_{g}.npz")
    worst = 0.0
    for c in RI.CL:
        if c["group"] == g:
            p = TV["predict"](get_net(g), R.features_for(c["cents"], c["conf"], w), False)
            worst = max(worst, float(np.abs(p - z[f"test|{c['id']}"]).max()))
    return g, worst


def resample(x, n):
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def cycles(curve):
    """Whole cycles as (start, end) index pairs between neighbouring pitch minima."""
    smooth = uniform_filter1d(curve, 5, mode="nearest")
    lows, _ = find_peaks(-smooth, prominence=20, distance=5)
    return list(zip(lows[:-1], lows[1:]))


def kampita_contour(src, n, rng):
    """Crop or repeat whole cycles, keeping pitch continuous at each join."""
    cy = cycles(src["curve"])
    if not cy:
        return None
    period = np.mean([b - a for a, b in cy])
    count = int(round(n / period))
    if count < 1:
        return None
    first = int(rng.integers(len(cy)))
    cur, conf, level = [], [], None
    for i in range(count):
        a, b = cy[(first + i) % len(cy)]
        piece = src["curve"][a:b + 1].astype(float)
        if level is not None:
            piece = piece + level - piece[0]
            piece = piece[1:]
            q = src["conf"][a + 1:b + 1]
        else:
            q = src["conf"][a:b + 1]
        level = piece[-1]
        cur.append(piece)
        conf.append(q)
    cur, conf = np.concatenate(cur), np.concatenate(conf)
    if not .8 <= len(cur) / n <= 1.25:
        return None
    return resample(cur, n), resample(conf, n), count, count > len(cy)


def build(src, t, n, rng):
    if t == 1:
        return kampita_contour(src, n, rng)
    if not .8 <= len(src["curve"]) / n <= 1.25:
        return None
    return resample(src["curve"], n), resample(src["conf"], n), 0, False


def plan(sung):
    rng = np.random.default_rng(SEED)
    trials, empty = [], []
    for k, g in enumerate(GROUPS):
        val = GROUPS[(k + 1) % 11]
        tracks = [c["id"] for c in RI.CL if c["group"] == g]
        sites = {ms: [s for t in tracks for s in RI.sites(t, ms // 10, sung)] for ms in LENGTHS}
        for t in (*TYPES, 0):
            pool = [i for i, s in enumerate(SRC) if s["type"] == t and s["group"] not in (g, val)]
            for ms in LENGTHS:
                n = ms // 10
                order = rng.permutation(pool)
                made = []
                for i in order:
                    got = (np.zeros(n), np.full(n, .9), 0, False) if t == 0 else build(SRC[int(i)], t, n, rng)
                    if got is not None:
                        made.append((int(i), got))
                    if len(made) == PER_FOLD:
                        break
                if not made:
                    empty += [dict(held_out_group=g, type=SHORT.get(t, "Flat control"), length_ms=ms, reason="no eligible training contour")] * PER_FOLD
                    continue
                used = set()
                for r in range(PER_FOLD):
                    i, (cur, conf, count, repeated) = made[r % len(made)]
                    net = cur[-1] - cur[0]
                    cand = [s for s in sites[ms] if (s[0], s[1]) not in used]
                    if not cand:
                        empty.append(dict(held_out_group=g, type=SHORT.get(t, "Flat control"), length_ms=ms, reason="no insertion site"))
                        continue
                    mis = np.array([(s[3] - s[2]) - net for s in cand])
                    ok = np.flatnonzero(np.abs(mis) <= RI.TOL)
                    j = int(rng.choice(ok)) if len(ok) else int(np.abs(mis).argmin())
                    s = cand[j]
                    used.add((s[0], s[1]))
                    trials.append(dict(group=g, type=t, length_ms=ms, source=i, track=s[0], a=s[1], L=s[2], R=s[3], mismatch=float(mis[j]), cur=cur, conf=conf, cycles=count, repeated=repeated))
    return trials, empty


def run_trial(tr):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    g, c, src = tr["group"], RI.BY[tr["track"]], SRC[tr["source"]]
    q = PARAM.loc[g]
    n, a = tr["length_ms"] // 10, tr["a"]
    x, conf = RI.insert(c, a, n, tr["cur"], tr["conf"], tr["L"], tr["R"])
    p = TV["predict"](get_net(g), R.features_for(x, conf, int(q.window_frames)), False).astype(float)
    regions = TS.regions_b(dict(y=np.zeros(p.shape[1]), logp2=np.log(p.T + 1e-8)), (q.D_region_s, q.switch, q.alpha), PRIOR[g])
    overlap = [(s, e) for s, e in regions if s < a + n and e > a]
    truth = (a / 100, (a + n) / 100, 1)

    def pair(ev):
        got = TS.match_iou([truth], [(s / 100, e / 100, 1) for s, e, *_ in ev])[3]
        return got[0][1] if got else None

    t, cents, _, voiced, _ = TS.PITCH[c["id"]]
    rec, v = np.asarray(cents, float).copy(), voiced.copy()
    good = ~np.isnan(x) & (conf >= .5)
    rec[c["i0"]:][good] = x[good]
    v[c["i0"]:] = good
    named = []
    for s, e in overlap:
        curve, nv = TS.event_curve(t, rec, v, s / 100 + c["offset"], e / 100 + c["offset"])
        if nv >= 5:
            cur = TS.centre(curve)
            row = np.array([TS.dtw(cur, b, max(5, int(.2 * max(len(cur), len(b))))) for b in TS.LIB_CURVE])
            label = TS.name_from_row(row, {g, src["group"]})
            if e - s >= int(round(q[f"L_{SHORT[label]}"] * 100)):
                named.append((s, e, label))
    hit = pair(named)
    det = next((z for z in named if hit and z[0] / 100 == hit[0] and z[1] / 100 == hit[1]), None)
    return dict(held_out_group=g, track=tr["track"], type=SHORT.get(tr["type"], "Flat control"), length_ms=tr["length_ms"], start_frame=a,
                source_recording=src["rec"], source_start_s=src["start_s"], source_group=src["group"], source_length_ms=src["native_ms"], cycles_used=tr["cycles"], cycles_repeated=tr["repeated"],
                pitch_mismatch_cents=round(tr["mismatch"], 1), region_found=pair(overlap) is not None, located=det is not None, right_type=det is not None and det[2] == tr["type"],
                detected_type=SHORT[det[2]] if det else "", detected_length_ms=(det[1] - det[0]) * 10 if det else np.nan)


def newton(X, y, ridge=1e-6):
    b = np.zeros(X.shape[1])

    def obj(v):
        e = np.clip(X @ v, -30, 30)
        return float(np.sum(y * e - np.logaddexp(0, e)) - ridge * (v @ v) / 2)

    for _ in range(300):
        e = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-e))
        w = np.maximum(p * (1 - p), 1e-10)
        s = np.linalg.solve((X.T * w) @ X + ridge * np.eye(X.shape[1]), X.T @ (y - p) - ridge * b)
        t = 1.0
        base = obj(b)
        while obj(b + t * s) < base - 1e-12 and t > 1e-8:
            t /= 2
        b = b + t * s
        if np.abs(t * s).max() < 1e-8:
            break
    return b


def fit(x, y):
    z = np.log(np.asarray(x, float) / 300)
    b = newton(np.c_[np.ones(len(z)), z], np.asarray(y, float))
    return np.array([b[0] - b[1] * np.log(300), b[1]])


def xp(b, p):
    return float(np.exp((np.log(p / (1 - p)) - b[0]) / b[1])) if b[1] > 0 else np.inf


def pct(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return (np.nan, np.nan) if not len(v) else (float(np.percentile(v, 2.5, method="lower")), float(np.percentile(v, 97.5, method="higher")))


def analyse(T, empty):
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(str(s))

    picks = np.random.default_rng(SEED).integers(0, 11, (2000, 11))
    X = T[T.type != "Flat control"]
    F = T[T.type == "Flat control"]
    rates, fits = [], []
    for t in SHORT.values():
        D = X[X.type == t]
        for ms in LENGTHS:
            q = D[D.length_ms == ms]
            row = dict(type=t, length_ms=ms, insertions=len(q), region_found=int(q.region_found.sum()))
            for col in ("located", "right_type"):
                by = q.groupby("held_out_group")[col].agg(["sum", "size"]).reindex(GROUPS).fillna(0)
                a, n = by["sum"].values, by["size"].values
                v = [a[s].sum() / n[s].sum() if n[s].sum() else np.nan for s in picks]
                lo, hi = pct(v)
                row.update({col: int(q[col].sum()), f"{col}_rate": q[col].mean() if len(q) else np.nan, f"{col}_ci_low": lo, f"{col}_ci_high": hi})
            rates.append(row)
        idx = {g: np.flatnonzero(D.held_out_group.values == g) for g in GROUPS}
        x = D.length_ms.values.astype(float)
        tested = sorted(D.length_ms.unique())
        for col in ("located", "right_type"):
            y = D[col].values.astype(float)
            row = dict(type=t, outcome=col, insertions=len(D), successes=int(y.sum()), tested_lengths_ms=" ".join(map(str, tested)))
            if y.min() == y.max():
                row["slope"] = np.nan
                row["X50_status"] = row["X80_status"] = "no variation"
                fits.append(row)
                continue
            b = fit(x, y)
            B = []
            for s in picks:
                i = np.concatenate([idx[GROUPS[j]] for j in s])
                if y[i].min() != y[i].max():
                    B.append(fit(x[i], y[i]))
            B = np.array(B)
            row.update(slope=b[1], slope_ci_low=pct(B[:, 1])[0], slope_ci_high=pct(B[:, 1])[1])
            for p in (.5, .8):
                k = f"X{int(p * 100)}"
                v = xp(b, p)
                if b[1] <= 0:
                    row[f"{k}_status"] = "no upward length trend"
                elif v < tested[0]:
                    row[f"{k}_status"] = f"below {tested[0]} ms"
                elif v > tested[-1]:
                    row[f"{k}_status"] = f"not reached by {tested[-1]} ms"
                else:
                    lo, hi = pct([xp(r, p) for r in B])
                    row.update({f"{k}_status": "inside tested range", f"{k}_ms": v, f"{k}_ci_low": lo, f"{k}_ci_high": hi})
            fits.append(row)
    R_, Fit = pd.DataFrame(rates), pd.DataFrame(fits)
    R_.to_csv(OUT["native_insertion_rates.csv"], index=False)
    Fit.to_csv(OUT["native_insertion_fits.csv"], index=False)
    say("NATIVE-SPEED INSERTION TEST, RAGA-FREE DETECTOR. EXPLORATORY. Recorded vocal, one singer; contours from training groups, held-out tracks.")
    say(f"Gamaka insertions: {len(X)}. Flat controls: located {int(F.located.sum())}/{len(F)}, candidate regions {int(F.region_found.sum())}/{len(F)}, upper 97.5% bound {1 - .025 ** (1 / max(len(F), 1)):.2%}.")
    say(f"Pitch mismatch before smoothing: median {X.pitch_mismatch_cents.abs().median():.0f} c, 90th percentile {X.pitch_mismatch_cents.abs().quantile(.9):.0f} c.")
    say(f"Kampita insertions that repeated cycles: {int(X[X.type == 'Kampita'].cycles_repeated.sum())}/{int((X.type == 'Kampita').sum())}.\n")
    for col, title in (("located", "Located, any type"), ("right_type", "Located and right type")):
        say(title + ": count/insertions, blank = no eligible insertion")
        say(R_.assign(v=[f"{a}/{n}" if n else "" for a, n in zip(R_[col], R_.insertions)]).pivot(index="type", columns="length_ms", values="v").reindex(SHORT.values()).to_string())
        say()
    say("Logistic fits, outcome ~ log(length); X values appear only inside each type's tested lengths")
    say(Fit.round(3).to_string(index=False))
    if len(empty):
        E = pd.DataFrame(empty).groupby(["type", "length_ms", "reason"]).size().rename("missing_insertions").reset_index()
        E.to_csv(OUT["empty_cells.csv"], index=False)
        say("\nMissing planned insertions")
        say(E.to_string(index=False))
    else:
        pd.DataFrame(columns=["type", "length_ms", "reason", "missing_insertions"]).to_csv(OUT["empty_cells.csv"], index=False)
    OUT["native_insertion_report.txt"].write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    if any(p.exists() for p in OUT.values()):
        sys.exit("Output files already exist; not overwriting them.")
    OUT_DIR.mkdir(exist_ok=True)
    PP.viterbi(np.zeros((30, 5)), np.array([0, 2, 2, 2, 2], np.int64), 1.0)
    TS.dtw(np.zeros(5), np.zeros(5), 5)
    with mp.Pool(16) as pool:
        SUNG = dict(pool.map(RI.sung_mask, list(RI.BY)))
    SRC = RI.sources()
    SRC.append(dict(rec="flat control", start_s=0.0, group="none", type=0, native_ms=0, curve=np.zeros(100), conf=np.full(100, .9)))
    PRIOR = {g: TS.priors2(k) for k, g in enumerate(GROUPS)}
    with mp.Pool(11) as pool:
        same = dict(pool.map(identity, GROUPS))
    print("largest probability difference on untouched tracks:", max(same.values()), flush=True)
    assert max(same.values()) < 1e-5
    trials, empty = plan(SUNG)
    print(f"planned {len(trials)} insertions; missing planned insertions {len(empty)}", flush=True)
    with mp.Pool(26) as pool:
        result = pool.map(run_trial, trials, chunksize=4)
    T = pd.DataFrame(result)
    T.to_csv(OUT["native_insertion_trials.csv"], index=False)
    analyse(T, empty)
