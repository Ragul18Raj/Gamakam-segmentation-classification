"""Step 1: clean up the saved Stage 2 CNN probabilities, with no new training. EXPLORATORY.

Method A: smooth probabilities -> top class per frame -> events -> merge small gaps -> extend -> drop short events.
Method B: duration-aware Viterbi decoding, with a minimum length per gamaka type.

Tuning rule: for each held-out group, settings are chosen only from the OTHER 10 groups, using their held-out
predictions, then applied once to the held-out group. Those other models did train on the held-out group's tracks,
so this is nested cross-validation over held-out predictions, not a fully separate inner training.

Singer's marks, event matching and scoring come from stage2_vocal.py / stage2_score.py unchanged.
Run from vocal_dataset:  python results/postproc/postprocess.py
"""
import itertools
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.ndimage import uniform_filter1d
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix, f1_score

HERE = Path(__file__).resolve().parent
VD = HERE.parents[1]
sys.path.insert(0, str(VD))
from stage2_score import ev_scores  # noqa: E402
from stage2_vocal import match_iou, runs  # noqa: E402

FR = 100                                  # frames per second
TYPES = (1, 2, 3, 4)
SHORT = {1: "Kampita", 2: "Nokku", 3: "Etra", 4: "Irakka"}
FULL = {1: "Kampita", 2: "Nokku", 3: "Jaaru (Etra)", 4: "Jaaru (Irakka)"}
GROUPS = [f"P{k:02d}" for k in range(1, 12)]
PROCS, SEED, N_BOOT = 28, 20261118, 2000

# Search grids (fixed before any result from this script was seen).
A_W_MS = (50, 70, 90, 110)                # smoothing window
A_G_MS = (0, 50, 100, 150)                # largest same-type gap to merge
A_E_MS = (0, 20, 40, 60)                  # extension at each end
A_L_S = tuple(round(0.10 + 0.05 * i, 2) for i in range(11))    # minimum length per type, 0.10 to 0.60 s
B_D_KAMPITA = (0.10, 0.20, 0.30, 0.40)    # minimum Kampita length, s
B_D_OTHER = (0.10, 0.14, 0.20)            # minimum Nokku / Etra / Irakka length, s
B_SWITCH = (1.0, 3.0, 6.0, 10.0, 20.0)    # log-probability cost of each switch into or out of "none"
B_ALPHA = (0.0, 0.5, 1.0)                 # how strongly to divide emissions by class priors


# ------------------------------------------------------------ data
def load_data():
    """All held-out tracks, grouped by held-out group. Probabilities come from the verified rerun."""
    data = {}
    for g in GROUPS:
        fold = json.load(open(VD / f"results/stage2_main/fold_{g}.json"))
        z = np.load(HERE / f"probs/fold_{g}.npz")
        data[g] = []
        for c in fold["clips"]:
            p = z[f"test|{c['id']}"].astype(np.float64)
            data[g].append(dict(id=c["id"], offset=c["offset"], ref=[tuple(x) for x in c["ref"]],
                                cnn=[tuple(x) for x in c["cnn"]], y=np.array(c["y"]),
                                cnn_frames=np.array(c["cnn_frames"]), p=p, logp=np.log(p.T + 1e-8)))
    return data


def seconds(ev):
    return [(s / FR, e / FR, l) for s, e, l in ev]


def to_frames(ev, T):
    lab = np.zeros(T, np.int64)
    for s, e, l in ev:
        lab[s:e] = l
    return lab


def label_events(lab):
    """Join equal neighbouring frame labels into (start, end, type); end is exclusive; 'none' is skipped."""
    return [(s, e, l) for s, e, l in runs(lab) if l]


# ------------------------------------------------------------ Method A
def merge_gaps(ev, gap_fr):
    """Join neighbouring events of the same type when the gap between them is at most gap_fr frames."""
    out = []
    for s, e, l in ev:
        if out and out[-1][2] == l and s - out[-1][1] <= gap_fr:
            out[-1] = (out[-1][0], e, l)
        else:
            out.append((s, e, l))
    return out


def extend_events(ev, ext_fr, T):
    """Extend both ends, stopping at the track edges and at half the gap to a neighbour, so events never overlap."""
    out = []
    for i, (s, e, l) in enumerate(ev):
        left = s if i == 0 else (s - ev[i - 1][1]) // 2
        right = T - e if i == len(ev) - 1 else (ev[i + 1][0] - e) // 2
        out.append((s - min(ext_fr, left), e + min(ext_fr, right), l))
    return out


def method_a_events(p, w_ms, g_ms, e_ms):
    """Steps 1-4 of Method A, before the per-type length filter. Returns events in frames."""
    smooth = uniform_filter1d(p, w_ms // 10, axis=1, mode="nearest")
    ev = label_events(smooth.argmax(0))
    return extend_events(merge_gaps(ev, g_ms // 10), e_ms // 10, p.shape[1])


def drop_short(ev, min_fr):
    return [x for x in ev if x[1] - x[0] >= min_fr[x[2]]]


def method_a(p, cfg):
    return drop_short(method_a_events(p, cfg["W_ms"], cfg["G_ms"], cfg["E_ms"]),
                      {l: int(round(cfg[f"L_{SHORT[l]}"] * FR)) for l in TYPES})


def a_counts(combo):
    """For one (W, G, E): true positives and detections per group, type and minimum length."""
    w, g, e = combo
    tp = np.zeros((len(GROUPS), 5, len(A_L_S)), int)
    ne = tp.copy()
    for gi, grp in enumerate(GROUPS):
        for c in DATA[grp]:
            ev = method_a_events(c["p"], w, g, e)
            for li, length in enumerate(A_L_S):
                kept = [x for x in ev if x[1] - x[0] >= int(round(length * FR))]
                t, _, n, _ = match_iou(c["ref"], seconds(kept))
                for l in TYPES:
                    tp[gi, l, li] += t[l]
                    ne[gi, l, li] += n[l]
    return tp, ne


def tune_a():
    """Length limits affect only their own type's F1, so each type's best limit can be chosen exactly and separately."""
    combos = list(itertools.product(A_W_MS, A_G_MS, A_E_MS))
    with mp.Pool(PROCS) as pool:
        res = pool.map(a_counts, combos, chunksize=1)
    chosen = {}
    for k, grp in enumerate(GROUPS):
        use = [i for i in range(len(GROUPS)) if i != k]
        nr = NREF[use].sum(0)
        best = None
        for combo, (tp, ne) in zip(combos, res):
            t, n = tp[use].sum(0), ne[use].sum(0)
            f = 2 * t / np.maximum(nr[:, None] + n, 1)     # [type, length]
            pick = {l: int(f[l].argmax()) for l in TYPES}  # argmax keeps the smallest limit on ties
            score = float(np.mean([f[l, pick[l]] for l in TYPES]))
            if best is None or score > best[0] + 1e-12:    # first (simplest) setting wins ties
                best = (score, combo, pick)
        score, (w, g, e), pick = best
        chosen[grp] = dict(W_ms=w, G_ms=g, E_ms=e, **{f"L_{SHORT[l]}": A_L_S[pick[l]] for l in TYPES}, tuning_event_f1=score)
    return chosen


# ------------------------------------------------------------ Method B
@njit(cache=True)
def viterbi(em, dur, switch):
    """em [T, 5] = log emission per class; dur[l] = minimum frames of type l.
    States: 0 = none; each type is a chain of dur[l] states, and only the last one may repeat.
    A gamaka is entered from and left to "none", each at cost `switch`. Returns one class label per frame."""
    T = em.shape[0]
    first, last = np.zeros(5, np.int64), np.zeros(5, np.int64)
    n = 1
    for l in range(1, 5):
        first[l] = n
        n += dur[l]
        last[l] = n - 1
    cls = np.zeros(n, np.int64)
    for l in range(1, 5):
        cls[first[l]:last[l] + 1] = l
    v = np.full(n, -np.inf)
    v[0] = em[0, 0]
    new = np.empty(n)
    back = np.zeros((T, n), np.int16)
    for t in range(1, T):
        new[:] = -np.inf
        best, arg = v[0], 0                    # stay in none, or finish a gamaka
        for l in range(1, 5):
            if v[last[l]] - switch > best:
                best, arg = v[last[l]] - switch, last[l]
        new[0] = best + em[t, 0]
        back[t, 0] = arg
        for l in range(1, 5):
            new[first[l]] = v[0] - switch + em[t, l]      # a gamaka can start only from none
            back[t, first[l]] = 0
            for s in range(first[l] + 1, last[l] + 1):    # walk the minimum-duration chain
                new[s] = v[s - 1] + em[t, l]
                back[t, s] = s - 1
            s = last[l]                                   # after the minimum, the gamaka may continue
            if v[s] + em[t, l] > new[s]:
                new[s] = v[s] + em[t, l]
                back[t, s] = s
        v, new = new, v
    s, best = 0, v[0]                                     # a track cannot end inside an unfinished chain
    for l in range(1, 5):
        if v[last[l]] > best:
            s, best = last[l], v[last[l]]
    out = np.zeros(T, np.int64)
    for t in range(T - 1, -1, -1):
        out[t] = cls[s]
        s = back[t, s]
    return out


def method_b(c, cfg, prior):
    dur = np.array([0] + [int(round(cfg[f"D_{SHORT[l]}"] * FR)) for l in TYPES], np.int64)
    em = c["logp"] - cfg["alpha"] * np.log(prior)
    return label_events(viterbi(em, dur, float(cfg["switch"])))


def priors(groups):
    y = np.concatenate([c["y"] for g in groups for c in DATA[g]])
    return np.bincount(y, minlength=5) / len(y)


def b_cfg(combo):
    dk, dn, de, di, sw, al = combo
    return dict(D_Kampita=dk, D_Nokku=dn, D_Etra=de, D_Irakka=di, switch=sw, alpha=al)


def b_score(job):
    """Event macro-F1 of one Method B setting, pooled over the 10 tuning groups of one held-out group."""
    k, ci = job
    cfg = b_cfg(B_COMBOS[ci])
    use = [g for i, g in enumerate(GROUPS) if i != k]
    prior = priors(use)
    tp, nr, ne = np.zeros(5), np.zeros(5), np.zeros(5)
    for g in use:
        for c in DATA[g]:
            t, r, n, _ = match_iou(c["ref"], seconds(method_b(c, cfg, prior)))
            for l in TYPES:
                tp[l] += t[l]
                nr[l] += r[l]
                ne[l] += n[l]
    return k, ci, float(np.mean([2 * tp[l] / max(nr[l] + ne[l], 1) for l in TYPES]))


def tune_b():
    jobs = [(k, ci) for k in range(len(GROUPS)) for ci in range(len(B_COMBOS))]
    best = {}
    with mp.Pool(PROCS) as pool:
        for k, ci, score in pool.imap(b_score, jobs, chunksize=20):   # imap keeps grid order, so the first setting wins ties
            if k not in best or score > best[k][0] + 1e-12:
                best[k] = (score, ci)
    return {g: dict(b_cfg(B_COMBOS[best[k][1]]), tuning_event_f1=best[k][0]) for k, g in enumerate(GROUPS)}


# ------------------------------------------------------------ scoring and reporting
def frame_macro(cm):
    """Same value as sklearn macro-F1 over the 5 labels with zero_division=0."""
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.divide(2 * tp, den, out=np.zeros(5), where=den > 0)))


def event_macro(tp, nr, ne):
    """Same rule as stage2_score.ev_scores: average over types present in the marks or detections."""
    den = nr + ne
    return float(np.mean((2 * tp / np.maximum(den, 1))[den > 0])) if (den > 0).any() else 0.0


def group_table(clips_by_group, key):
    tp, nr, ne, cm = [], [], [], []
    for g in GROUPS:
        t, r, n = np.zeros(4), np.zeros(4), np.zeros(4)
        for c in clips_by_group[g]:
            a, b, d, _ = match_iou(c["ref"], c[key])
            for i, l in enumerate(TYPES):
                t[i] += a[l]
                r[i] += b[l]
                n[i] += d[l]
        y = np.concatenate([c["y"] for c in clips_by_group[g]])
        f = np.concatenate([c[key + "_frames"] for c in clips_by_group[g]])
        tp.append(t), nr.append(r), ne.append(n), cm.append(confusion_matrix(y, f, labels=range(5)))
    return np.array(tp), np.array(nr), np.array(ne), np.array(cm)


def main():
    outputs = [HERE / n for n in ("step1_report.txt", "events_method_A.csv", "events_method_B.csv", "chosen_params.csv")]
    if any(p.exists() for p in outputs):
        sys.exit("Output files already exist; not overwriting them: " + ", ".join(p.name for p in outputs if p.exists()))
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(str(s))

    print("tuning Method A ...", flush=True)
    a_cfg = tune_a()
    print("tuning Method B ...", flush=True)
    b_chosen = tune_b()

    # Apply each group's settings once to that held-out group.
    for k, g in enumerate(GROUPS):
        prior = priors([x for x in GROUPS if x != g])
        for c in DATA[g]:
            T = len(c["y"])
            ea = method_a(c["p"], a_cfg[g])
            eb = method_b(c, b_chosen[g], prior)
            for ev in (ea, eb):
                assert all(ev[i][1] <= ev[i + 1][0] for i in range(len(ev) - 1)), "overlapping events"
            assert all(e - s >= int(round(a_cfg[g][f"L_{SHORT[l]}"] * FR)) for s, e, l in ea)
            assert all(e - s >= int(round(b_chosen[g][f"D_{SHORT[l]}"] * FR)) for s, e, l in eb)
            c["A"], c["B"] = seconds(ea), seconds(eb)
            c["A_frames"], c["B_frames"] = to_frames(ea, T), to_frames(eb, T)
            c["original"], c["original_frames"] = c["cnn"], c["cnn_frames"]

    runs_ = {"original": "original", "Method A": "A", "Method B": "B"}
    clips = [c for g in GROUPS for c in DATA[g]]
    Y = np.concatenate([c["y"] for c in clips])
    rows, per, tab = [], {}, {}
    for name, key in runs_.items():
        macro, pc, _ = ev_scores(clips, key)
        fr = f1_score(Y, np.concatenate([c[key + "_frames"] for c in clips]), labels=range(5), average="macro", zero_division=0)
        tab[name] = group_table(DATA, key)
        tp, nr, ne, cm = [x.sum(0) for x in tab[name]]
        assert abs(event_macro(tp, nr, ne) - macro) < 1e-12 and abs(frame_macro(cm) - fr) < 1e-12
        rows.append(dict(run=name, event=macro, frame=fr, detections=int(ne.sum()), correct=int(tp.sum())))
        per[name] = pc
    assert abs(rows[0]["event"] - 0.395) < 0.0006 and abs(rows[0]["frame"] - 0.497) < 0.0006, "original run not reproduced"

    say("STEP 1: CNN output clean-up, no retraining. EXPLORATORY. Recorded vocal, one singer; 11 held-out groups pooled.")
    say("Settings for each held-out group were tuned on the held-out predictions of the other 10 groups.\n")
    say(pd.DataFrame(rows).round(3).to_string(index=False))
    say("\nPer type: F1 / precision / recall / detections")
    t = pd.DataFrame({name: {FULL[l]: "{f1:.2f} / {precision:.2f} / {recall:.2f} / {n_est}".format(**per[name][FULL[l]]) for l in TYPES} for name in runs_})
    say(t.to_string())

    say("\nEvents shorter than 0.2 s")
    for name, key in runs_.items():
        say(f"  {name:9s} {sum(int(round((e - s) * FR)) < 20 for c in clips for s, e, _ in c[key])}")

    # Per-fold scores use the same functions as the pooled numbers.
    fold = pd.DataFrame(index=GROUPS)
    for name, key in runs_.items():
        fold[f"{name} event"] = [ev_scores(DATA[g], key)[0] for g in GROUPS]
        fold[f"{name} frame"] = [frame_macro(c) for c in tab[name][3]]
    for name in ("Method A", "Method B"):
        for m in ("event", "frame"):
            fold[f"{name} {m} change"] = fold[f"{name} {m}"] - fold[f"original {m}"]
    say("\nPer-fold scores and change from the original")
    say(fold[[c for c in fold if "change" in c or c.startswith("original")]].round(3).to_string())

    say(f"\nChange from the original, pooled score; group bootstrap {N_BOOT} resamples of the 11 groups; two-sided Wilcoxon over 11 folds")
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(GROUPS), (N_BOOT, len(GROUPS)))
    o = tab["original"]
    for name in ("Method A", "Method B"):
        m = tab[name]
        for label in ("event", "frame"):
            boot = []
            for s in picks:
                if label == "event":
                    boot.append(event_macro(*[x[s].sum(0) for x in m[:3]]) - event_macro(*[x[s].sum(0) for x in o[:3]]))
                else:
                    boot.append(frame_macro(m[3][s].sum(0)) - frame_macro(o[3][s].sum(0)))
            delta = next(r[label] for r in rows if r["run"] == name) - rows[0][label]
            diff = fold[f"{name} {label} change"].values
            p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            say(f"  {name} {label}: {delta:+.3f} (95% CI {np.percentile(boot, 2.5):+.3f} to {np.percentile(boot, 97.5):+.3f}); "
                f"folds better {int((diff > 0).sum())}, worse {int((diff < 0).sum())}; Wilcoxon p = {p:.3f}")

    pa, pb = pd.DataFrame(a_cfg).T, pd.DataFrame(b_chosen).T
    say("\nMethod A settings chosen per held-out group (ms, length limits in s)")
    say(pa.round(3).to_string())
    say("\nMethod B settings chosen per held-out group (minimum lengths in s)")
    say(pb.round(3).to_string())

    # Save results only after everything above has succeeded.
    for key, path in (("A", outputs[1]), ("B", outputs[2])):
        pd.DataFrame([dict(track=c["id"], start_s=round(s + c["offset"], 3), end_s=round(e + c["offset"], 3), type=FULL[l])
                      for c in clips for s, e, l in c[key]], columns=["track", "start_s", "end_s", "type"]).to_csv(path, index=False)
    pd.concat([pa.assign(method="A"), pb.assign(method="B")]).rename_axis("held_out_group").to_csv(outputs[3])
    outputs[0].write_text("\n".join(lines) + "\n")


B_COMBOS = list(itertools.product(B_D_KAMPITA, B_D_OTHER, B_D_OTHER, B_D_OTHER, B_SWITCH, B_ALPHA))
DATA = load_data()                         # loaded once; worker processes share it through fork
NREF = np.zeros((len(GROUPS), 5), int)
for _i, _g in enumerate(GROUPS):
    for _c in DATA[_g]:
        for _s, _e, _l in _c["ref"]:
            NREF[_i, _l] += 1

if __name__ == "__main__":
    viterbi(np.zeros((30, 5)), np.array([0, 2, 2, 2, 2], np.int64), 1.0)   # compile once before the workers start
    main()
