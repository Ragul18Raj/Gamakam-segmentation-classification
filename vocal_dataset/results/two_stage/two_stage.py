"""Step 2: find gamaka regions with a gamaka-vs-none CNN, clean them up, then name each region with Stage 1 DTW-3NN. EXPLORATORY.

Two versions of the region clean-up, using the Step 1 operations:
  A: smooth -> top class -> merge small gaps -> extend -> name -> drop events shorter than a per-type minimum
  B: minimum-duration Viterbi (none / gamaka) -> name -> drop events shorter than a per-type minimum

Tuning for a held-out group k uses only the other 10 groups. When scoring tuning group j, the DTW library excludes both
j and k, so neither the group being scored nor the held-out group supplies names. The final held-out run uses a library
of the 10 groups other than k. Region CNNs for j did train on k's tracks, the same small leak as in Step 1.
Run from vocal_dataset:  python results/two_stage/two_stage.py
"""
import itertools
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
VD = HERE.parents[1]
sys.path[:0] = [str(VD), str(VD / "results/postproc")]
import postprocess as PP  # noqa: E402  (Step 1 helpers; its grids and marks are reused)
from stage1_vocal import CL4, centre, dist_matrix, dtw  # noqa: E402
from stage2_score import ev_scores  # noqa: E402
from stage2_vocal import match_iou  # noqa: E402
from vocal_pitch import event_curve, load_cents, load_tables  # noqa: E402

FR, TYPES, GROUPS, SHORT, FULL, DATA = PP.FR, PP.TYPES, PP.GROUPS, PP.SHORT, PP.FULL, PP.DATA
L_GRID = PP.A_L_S
A_GRID = list(itertools.product(PP.A_W_MS, PP.A_G_MS, PP.A_E_MS))
B_GRID = list(itertools.product((0.10, 0.14, 0.20, 0.30, 0.40), PP.B_SWITCH, PP.B_ALPHA))   # minimum region length, switch cost, prior strength
CONFIGS = [("A", c) for c in A_GRID] + [("B", c) for c in B_GRID]
LABEL_ID = {n: i + 1 for i, n in enumerate(CL4)}       # same type numbers as Stage 2
MIN_VOICED = 5                                         # Stage 1 rule: fewer voiced frames cannot be named
CLIPS = {c["id"]: c for g in GROUPS for c in DATA[g]}
GROUP_OF = {c["id"]: g for g in GROUPS for c in DATA[g]}
REGIONS, DIST = {}, {}                                 # filled in main, then shared with workers through fork


# ------------------------------------------------------------ data
for g in GROUPS:                                       # binary probabilities [none, gamaka] from the new CNNs
    z = np.load(HERE / f"probs/fold_{g}.npz")
    for c in DATA[g]:
        c["p2"] = z[f"test|{c['id']}"].astype(np.float64)
        assert c["p2"].shape == (2, len(c["y"]))
        c["logp2"] = np.log(c["p2"].T + 1e-8)

meta, marks = load_tables()
SA = meta.set_index("recording").sa_hz
GROUP_NAME = meta.set_index("recording").group
PITCH = {r: load_cents(r, SA[r]) for r in meta.recording}
LIB_KEY, LIB_CURVE, LIB_Y, LIB_G = [], [], [], []
for rec, e in marks.groupby("recording", sort=False):    # same event order as stage1_vocal.main
    t, cents, _, voiced, _ = PITCH[rec]
    for _, r in e.iterrows():
        x, nv = event_curve(t, cents, voiced, r.start_s, r.end_s)
        assert nv >= MIN_VOICED
        LIB_KEY.append((rec, r.start_s)), LIB_CURVE.append(centre(x)), LIB_Y.append(LABEL_ID[r.label]), LIB_G.append(GROUP_NAME[rec])
LIB_Y, LIB_G = np.array(LIB_Y), np.array(LIB_G)


def name_from_row(row, exclude):
    """Stage 1 DTW-3NN rule: three nearest allowed library events; a tie goes to the nearest tied class."""
    allowed = np.flatnonzero(~np.isin(LIB_G, list(exclude)))
    near = allowed[np.argsort(row[allowed], kind="stable")[:3]]
    votes = np.bincount(LIB_Y[near], minlength=5)
    best = np.flatnonzero(votes == votes.max())
    return int(next(LIB_Y[i] for i in near if LIB_Y[i] in best))


def check_library():
    """The naming code must give exactly the saved Stage 1 held-out DTW-3NN predictions."""
    M = dist_matrix(LIB_CURVE)
    saved = pd.read_csv(VD / "results/stage1/predictions.csv").set_index(["recording", "start_s"])["main_4class | DTW-3NN"]
    bad = sum(name_from_row(M[i], {LIB_G[i]}) != LABEL_ID[saved[LIB_KEY[i]]] for i in range(len(LIB_Y)))
    assert bad == 0, f"{bad} naming differences from Stage 1"
    return len(LIB_Y)


# ------------------------------------------------------------ region finding
def priors2(k):
    y = np.concatenate([c["y"] for i, g in enumerate(GROUPS) if i != k for c in DATA[g]])
    p = float((y > 0).mean())
    return np.array([1 - p, p])


def regions_a(c, cfg):
    w, g, e = cfg
    smooth = uniform_filter1d(c["p2"], w // 10, axis=1, mode="nearest")
    ev = PP.extend_events(PP.merge_gaps(PP.label_events(smooth.argmax(0)), g // 10), e // 10, len(c["y"]))
    return [(s, x) for s, x, _ in ev]


def regions_b(c, cfg, prior):
    d, switch, alpha = cfg
    em = np.full((len(c["y"]), 5), -1e9)              # Step 1 decoder with only the "none" and "gamaka" classes usable
    em[:, :2] = c["logp2"] - alpha * np.log(prior)
    return [(s, x) for s, x, _ in PP.label_events(PP.viterbi(em, np.array([0, int(round(d * FR)), 1, 1, 1], np.int64), float(switch)))]


def region_job(job):
    ci, k = job
    kind, cfg = CONFIGS[ci]
    if kind == "A":
        return job, {i: regions_a(c, cfg) for i, c in CLIPS.items()}
    prior = priors2(k)
    return job, {i: regions_b(c, cfg, prior) for i, c in CLIPS.items()}


def region_key(ci, k):
    return (ci, None) if CONFIGS[ci][0] == "A" else (ci, k)   # A does not use class priors, so it is the same for every k


def dist_job(key):
    cid, s, e = key
    c = CLIPS[cid]
    t, cents, _, voiced, _ = PITCH[cid]
    x, nv = event_curve(t, cents, voiced, s / FR + c["offset"], e / FR + c["offset"])
    if nv < MIN_VOICED:
        return key, None
    q = centre(x)
    return key, np.array([dtw(q, b, max(5, int(.2 * max(len(q), len(b))))) for b in LIB_CURVE], np.float32)


def named_events(cid, regions, exclude):
    out = []
    for s, e in regions:
        row = DIST[(cid, s, e)]
        if row is not None:
            out.append((s, e, name_from_row(row, exclude)))
    return out


def tune_job(job):
    """Best per-type length limits for one configuration, scored only on the 10 groups other than k."""
    ci, k = job
    test = GROUPS[k]
    reg = REGIONS[region_key(ci, k)]
    tp, ne = np.zeros((5, len(L_GRID)), int), np.zeros((5, len(L_GRID)), int)
    for g in GROUPS:
        if g == test:
            continue
        for c in DATA[g]:
            ev = named_events(c["id"], reg[c["id"]], {g, test})
            for li, length in enumerate(L_GRID):
                kept = [x for x in ev if x[1] - x[0] >= int(round(length * FR))]
                t, _, n, _ = match_iou(c["ref"], PP.seconds(kept))
                for l in TYPES:
                    tp[l, li] += t[l]
                    ne[l, li] += n[l]
    nr = np.delete(PP.NREF, k, axis=0).sum(0)
    f = 2 * tp / np.maximum(nr[:, None] + ne, 1)
    pick = {l: int(f[l].argmax()) for l in TYPES}
    return job, float(np.mean([f[l, pick[l]] for l in TYPES])), pick


# ------------------------------------------------------------ reporting helpers
def detection_stats(clips, key):
    """Ignore type to measure region finding; then ask how many matched regions received the marked type."""
    tp = nr = ne = same = 0
    for c in clips:
        ref = {(s, e): l for s, e, l in c["ref"]}
        est = {(s, e): l for s, e, l in c[key]}
        a, b, d, pairs = match_iou([(s, e, 1) for s, e in ref], [(s, e, 1) for s, e in est])
        tp += a[1]
        nr += b[1]
        ne += d[1]
        same += sum(ref[r[:2]] == est[x[:2]] for r, x in pairs)
    p, r = tp / max(ne, 1), tp / nr
    return dict(precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-12), located=tp, right_type_of_located=same / max(tp, 1))


def main():
    outputs = [HERE / n for n in ("step2_report.txt", "events_two_stage_A.csv", "events_two_stage_B.csv", "chosen_params.csv")]
    if any(p.exists() for p in outputs):
        sys.exit("Output files already exist; not overwriting them.")
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(str(s))

    n_lib = check_library()
    print(f"naming check passed: all {n_lib} Stage 1 DTW-3NN held-out predictions reproduced", flush=True)
    PP.viterbi(np.zeros((30, 5)), np.array([0, 2, 2, 2, 2], np.int64), 1.0)

    jobs = sorted({region_key(ci, k) for ci in range(len(CONFIGS)) for k in range(len(GROUPS))}, key=str)
    with mp.Pool(PP.PROCS) as pool:
        for job, reg in pool.imap_unordered(region_job, jobs, chunksize=4):
            REGIONS[job] = reg
    keys = sorted({(cid, s, e) for reg in REGIONS.values() for cid, rs in reg.items() for s, e in rs})
    print(f"{len(keys)} distinct candidate regions to compare with the library", flush=True)
    with mp.Pool(PP.PROCS) as pool:
        DIST.update(pool.imap_unordered(dist_job, keys, chunksize=50))

    jobs = [(ci, k) for k in range(len(GROUPS)) for ci in range(len(CONFIGS))]
    with mp.Pool(PP.PROCS) as pool:
        tuned = pool.map(tune_job, jobs, chunksize=4)                 # map keeps grid order, so the first setting wins ties
    chosen = {}
    for (ci, k), score, pick in tuned:
        kind = CONFIGS[ci][0]
        if (kind, k) not in chosen or score > chosen[(kind, k)][0] + 1e-12:
            chosen[(kind, k)] = (score, ci, pick)

    # Apply the chosen settings once to each held-out group.
    params, unnamed = [], {"A": 0, "B": 0}
    for k, g in enumerate(GROUPS):
        for kind, key in (("A", "TA"), ("B", "TB")):
            score, ci, pick = chosen[(kind, k)]
            cfg = CONFIGS[ci][1]
            mins = {l: int(round(L_GRID[pick[l]] * FR)) for l in TYPES}
            names = ("W_ms", "G_ms", "E_ms") if kind == "A" else ("D_region_s", "switch", "alpha")
            params.append(dict(held_out_group=g, method=kind, **dict(zip(names, cfg)), **{f"L_{SHORT[l]}": L_GRID[pick[l]] for l in TYPES}, tuning_event_f1=score))
            for c in DATA[g]:
                reg = REGIONS[region_key(ci, k)][c["id"]]
                ev = named_events(c["id"], reg, {g})
                unnamed[kind] += len(reg) - len(ev)
                ev = PP.drop_short(ev, mins)
                assert all(ev[i][1] <= ev[i + 1][0] for i in range(len(ev) - 1))
                c[key], c[key + "_frames"] = PP.seconds(ev), PP.to_frames(ev, len(c["y"]))

    # Step 1 Method B is included only as a comparison row.
    s1 = pd.read_csv(VD / "results/postproc/events_method_B.csv")
    for c in CLIPS.values():
        x = s1[s1.track == c["id"]]
        ev = [(int(round((a - c["offset"]) * FR)), int(round((b - c["offset"]) * FR)), LABEL_ID[t]) for a, b, t in zip(x.start_s, x.end_s, x.type)]
        c["S1B"], c["S1B_frames"] = PP.seconds(ev), PP.to_frames(ev, len(c["y"]))
        c["original"], c["original_frames"] = c["cnn"], c["cnn_frames"]

    runs_ = {"original": "original", "Step 1 Method B": "S1B", "Two-stage A": "TA", "Two-stage B": "TB"}
    clips = list(CLIPS.values())
    Y = np.concatenate([c["y"] for c in clips])
    rows, per, tab = [], {}, {}
    for name, key in runs_.items():
        macro, pc, _ = ev_scores(clips, key)
        fr = f1_score(Y, np.concatenate([c[key + "_frames"] for c in clips]), labels=range(5), average="macro", zero_division=0)
        tab[name] = PP.group_table(DATA, key)
        tp, nr, ne, cm = [x.sum(0) for x in tab[name]]
        assert abs(PP.event_macro(tp, nr, ne) - macro) < 1e-12 and abs(PP.frame_macro(cm) - fr) < 1e-12
        rows.append(dict(run=name, event=macro, frame=fr, detections=int(ne.sum()), correct=int(tp.sum())))
        per[name] = pc
    assert abs(rows[0]["event"] - .395) < .0006 and abs(rows[1]["event"] - .417) < .0006, "reference runs not reproduced"

    say("STEP 2: gamaka-vs-none CNN regions named by Stage 1 DTW-3NN. EXPLORATORY. Recorded vocal, one singer; 11 held-out groups pooled.")
    say("Settings for each held-out group were tuned only on the other 10 groups.\n")
    say(pd.DataFrame(rows).round(3).to_string(index=False))
    say("\nPer type: F1 / precision / recall / detections")
    say(pd.DataFrame({n: {FULL[l]: "{f1:.2f} / {precision:.2f} / {recall:.2f} / {n_est}".format(**per[n][FULL[l]]) for l in TYPES} for n in runs_}).to_string())
    say("\nRegion finding with type ignored (IoU >= 0.3), and how often a located region got the marked type")
    say(pd.DataFrame({n: detection_stats(clips, k) for n, k in runs_.items()}).T.round(3).to_string())
    say("\nEvents shorter than 0.2 s: " + ", ".join(f"{n} {sum(int(round((e - s) * FR)) < 20 for c in clips for s, e, _ in c[k])}" for n, k in runs_.items()))
    say(f"Held-out regions dropped before length filtering because they had fewer than {MIN_VOICED} voiced frames: A {unnamed['A']}, B {unnamed['B']}")

    fold = pd.DataFrame(index=GROUPS)
    for name, key in runs_.items():
        fold[f"{name} event"] = [ev_scores(DATA[g], key)[0] for g in GROUPS]
        fold[f"{name} frame"] = [PP.frame_macro(c) for c in tab[name][3]]
    compare = ("Two-stage A", "Two-stage B")
    for name in compare:
        for m in ("event", "frame"):
            fold[f"{name} {m} change"] = fold[f"{name} {m}"] - fold[f"original {m}"]
    say("\nPer-fold scores and change from the original")
    say(fold[[c for c in fold if c.startswith("original") or "change" in c]].round(3).to_string())
    say(f"\nChange from the original, pooled score; group bootstrap {PP.N_BOOT} resamples; two-sided Wilcoxon over 11 folds")
    picks = np.random.default_rng(PP.SEED).integers(0, len(GROUPS), (PP.N_BOOT, len(GROUPS)))
    o = tab["original"]
    for name in compare:
        m = tab[name]
        for label in ("event", "frame"):
            if label == "event":
                boot = [PP.event_macro(*[x[s].sum(0) for x in m[:3]]) - PP.event_macro(*[x[s].sum(0) for x in o[:3]]) for s in picks]
            else:
                boot = [PP.frame_macro(m[3][s].sum(0)) - PP.frame_macro(o[3][s].sum(0)) for s in picks]
            delta = next(r[label] for r in rows if r["run"] == name) - rows[0][label]
            diff = fold[f"{name} {label} change"].values
            p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            say(f"  {name} {label}: {delta:+.3f} (95% CI {np.percentile(boot, 2.5):+.3f} to {np.percentile(boot, 97.5):+.3f}); "
                f"folds better {int((diff > 0).sum())}, worse {int((diff < 0).sum())}; Wilcoxon p = {p:.3f}")
    P = pd.DataFrame(params)
    for kind in ("A", "B"):
        say(f"\nTwo-stage {kind} settings chosen per held-out group")
        say(P[P.method == kind].dropna(axis=1, how="all").drop(columns="method").round(3).to_string(index=False))

    for key, path in (("TA", outputs[1]), ("TB", outputs[2])):
        pd.DataFrame([dict(track=c["id"], start_s=round(s + c["offset"], 3), end_s=round(e + c["offset"], 3), type=FULL[l])
                      for c in clips for s, e, l in c[key]], columns=["track", "start_s", "end_s", "type"]).to_csv(path, index=False)
    P.to_csv(outputs[3], index=False)
    outputs[0].write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
