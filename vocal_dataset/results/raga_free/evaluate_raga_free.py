"""Evaluate the raga-free Stage 2 detector with the frozen Step 3 route. EXPLORATORY.

Route: gamaka-vs-none probabilities -> Viterbi clean-up -> Stage 1 DTW-3NN naming -> per-type minimum length.
For each held-out group, the median window and every clean-up setting are chosen only from the other 10 groups.
DTW naming uses median-subtracted contours only. While tuning on group j for held-out group k, its library excludes j and k.
Run from vocal_dataset:  python results/raga_free/evaluate_raga_free.py
"""
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
VD = HERE.parents[1]
sys.path[:0] = [str(VD), str(VD / "results/two_stage"), str(HERE)]
import two_stage as TS  # noqa: E402

PP = TS.PP
DATA, GROUPS, FR, TYPES, FULL, SHORT = TS.DATA, TS.GROUPS, TS.FR, TS.TYPES, TS.FULL, TS.SHORT
CLIPS = list(TS.CLIPS.values())
WINDOWS = (75, 51, 101)                 # 75 frames is listed first, so it wins exact tuning ties
B_CONFIGS = [i for i, (kind, _) in enumerate(TS.CONFIGS) if kind == "B"]
EDGES, BANDS = [0, .2, .3, .4, .6, .8, np.inf], ["<0.2", "0.2-0.3", "0.3-0.4", "0.4-0.6", "0.6-0.8", ">=0.8"]
OUT = {n: HERE / n for n in ("raga_free_report.txt", "chosen_params.csv", "events_raga_free.csv", "per_fold.csv", "located_by_length.csv")}


def from_csv(path):
    table = pd.read_csv(path)
    out = {}
    for c in CLIPS:
        x = table[table.track == c["id"]]
        out[c["id"]] = [(int(round((a - c["offset"]) * FR)), int(round((b - c["offset"]) * FR)), TS.LABEL_ID[t])
                        for a, b, t in zip(x.start_s, x.end_s, x.type)]
    return out


def store(c, key, ev):
    c[key], c[key + "_frames"] = PP.seconds(ev), PP.to_frames(ev, len(c["y"]))


def tune_window(w):
    """Return, for each held-out group, the best Viterbi setting for this median window and its held-out events."""
    for g in GROUPS:
        z = np.load(HERE / f"probs/W{w}/fold_{g}.npz")
        for c in DATA[g]:
            c["p2"] = z[f"test|{c['id']}"].astype(np.float64)
            assert c["p2"].shape == (2, len(c["y"]))
            c["logp2"] = np.log(c["p2"].T + 1e-8)
    TS.REGIONS.clear()
    jobs = [(ci, k) for ci in B_CONFIGS for k in range(len(GROUPS))]
    with mp.Pool(PP.PROCS) as pool:
        for job, reg in pool.imap_unordered(TS.region_job, jobs, chunksize=4):
            TS.REGIONS[job] = reg
    new = sorted({(cid, s, e) for reg in TS.REGIONS.values() for cid, rs in reg.items() for s, e in rs} - set(TS.DIST))
    with mp.Pool(PP.PROCS) as pool:
        TS.DIST.update(pool.imap_unordered(TS.dist_job, new, chunksize=50))
    jobs = [(ci, k) for k in range(len(GROUPS)) for ci in B_CONFIGS]
    with mp.Pool(PP.PROCS) as pool:
        tuned = pool.map(TS.tune_job, jobs, chunksize=4)
    best = {}
    for (ci, k), score, pick in tuned:
        if k not in best or score > best[k][0] + 1e-12:
            best[k] = (score, ci, pick)
    result = {}
    for k, g in enumerate(GROUPS):
        score, ci, pick = best[k]
        d, switch, alpha = TS.CONFIGS[ci][1]
        mins = {l: int(round(TS.L_GRID[pick[l]] * FR)) for l in TYPES}
        events = {}
        for c in DATA[g]:
            ev = PP.drop_short(TS.named_events(c["id"], TS.REGIONS[(ci, k)][c["id"]], {g}), mins)
            assert all(ev[i][1] <= ev[i + 1][0] for i in range(len(ev) - 1))
            events[c["id"]] = ev
        result[g] = dict(score=score, events=events, params=dict(
            held_out_group=g, window_frames=w, D_region_s=d, switch=switch, alpha=alpha,
            **{f"L_{SHORT[l]}": TS.L_GRID[pick[l]] for l in TYPES}, tuning_event_f1=score))
    print("finished window", w, flush=True)
    return result


def count_errors(clips, key):
    per_type, per_track = [], []
    for c in clips:
        diff = [sum(x[2] == l for x in c[key]) - sum(x[2] == l for x in c["ref"]) for l in TYPES]
        per_type += [abs(v) for v in diff]
        per_track.append(abs(sum(diff)))
    return float(np.mean(per_type)), float(np.mean(per_track))


def located_table(key):
    rows = []
    for c in CLIPS:
        ref = {(s, e): l for s, e, l in c["ref"]}
        est = {(s, e): l for s, e, l in c[key]}
        _, _, _, pairs = TS.match_iou([(s, e, 1) for s, e in ref], [(s, e, 1) for s, e in est])
        hit = {r[:2]: x[:2] for r, x in pairs}
        for (s, e), l in ref.items():
            d = hit.get((s, e))
            rows.append(dict(type=SHORT[l], length_s=round(e - s, 3), located=d is not None, right_type=d is not None and est[d] == l))
    t = pd.DataFrame(rows)
    t["band"] = pd.cut(t.length_s, EDGES, right=False, labels=BANDS)
    return t


def main():
    if any(p.exists() for p in OUT.values()):
        sys.exit("Output files already exist; not overwriting them.")
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(str(s))

    TS.check_library()
    PP.viterbi(np.zeros((30, 5)), np.array([0, 2, 2, 2, 2], np.int64), 1.0)
    results = {w: tune_window(w) for w in WINDOWS}
    frozen = from_csv(VD / "results/step3/events_all_B.csv")
    for c in CLIPS:
        c["original"], c["original_frames"] = c["cnn"], c["cnn_frames"]
        store(c, "frozen", frozen[c["id"]])
    params = []
    for g in GROUPS:
        chosen = max(WINDOWS, key=lambda w: results[w][g]["score"])     # max keeps the first listed window on exact ties
        params.append(results[chosen][g]["params"])
        for c in DATA[g]:
            store(c, "RF", results[chosen][g]["events"][c["id"]])
            for w in WINDOWS:
                store(c, f"W{w}", results[w][g]["events"][c["id"]])

    runs = {"original": "original", "frozen raga-aware (Step 3)": "frozen", "RAGA-FREE, tuned window": "RF"}
    runs.update({f"raga-free, {w * 10} ms window": f"W{w}" for w in sorted(WINDOWS)})
    Y = np.concatenate([c["y"] for c in CLIPS])
    rows, per, tab = [], {}, {}
    for name, key in runs.items():
        macro, pc, _ = TS.ev_scores(CLIPS, key)
        frame = f1_score(Y, np.concatenate([c[key + "_frames"] for c in CLIPS]), labels=range(5), average="macro", zero_division=0)
        tab[key] = PP.group_table(DATA, key)
        tp, nr, ne, cm = [x.sum(0) for x in tab[key]]
        assert abs(PP.event_macro(tp, nr, ne) - macro) < 1e-12 and abs(PP.frame_macro(cm) - frame) < 1e-12
        by_type, by_track = count_errors(CLIPS, key)
        rows.append(dict(run=name, event=macro, frame=frame, detections=int(ne.sum()), correct=int(tp.sum()),
                         count_error_per_type_track=by_type, total_count_error_per_track=by_track))
        per[name] = pc
    assert abs(rows[0]["event"] - .395) < .0006 and abs(rows[1]["event"] - .462) < .0006

    say("RAGA-FREE STAGE 2. EXPLORATORY. Recorded vocal, one singer; 11 held-out groups pooled; settings tuned only on the other 10 groups.")
    say("Inputs: cents minus running median, velocity, voiced flag, CREPE confidence. Stage 1 DTW naming uses median-subtracted contours.\n")
    say(pd.DataFrame(rows).round(3).to_string(index=False))
    say("\nPer type: F1 / precision / recall / detections")
    say(pd.DataFrame({n: {FULL[l]: "{f1:.2f} / {precision:.2f} / {recall:.2f} / {n_est}".format(**per[n][FULL[l]]) for l in TYPES} for n in runs}).T.to_string())
    say("\nRegion finding with type ignored (IoU >= 0.3)")
    say(pd.DataFrame({n: TS.detection_stats(CLIPS, k) for n, k in runs.items()}).T.round(3).to_string())

    fold = pd.DataFrame(index=GROUPS)
    for name in ("frozen", "RF"):
        fold[f"{name}_event"] = [TS.ev_scores(DATA[g], name)[0] for g in GROUPS]
        fold[f"{name}_frame"] = [PP.frame_macro(c) for c in tab[name][3]]
    fold["event_change"], fold["frame_change"] = fold.RF_event - fold.frozen_event, fold.RF_frame - fold.frozen_frame
    fold["window_frames"] = [p["window_frames"] for p in params]
    say("\nPer fold, raga-free against the frozen raga-aware run")
    say(fold.round(3).to_string())
    picks = np.random.default_rng(PP.SEED).integers(0, len(GROUPS), (PP.N_BOOT, len(GROUPS)))
    say(f"\nChange from the frozen raga-aware run; group bootstrap {PP.N_BOOT} resamples; two-sided Wilcoxon over 11 folds")
    for label in ("event", "frame"):
        if label == "event":
            boot = [PP.event_macro(*[x[s].sum(0) for x in tab["RF"][:3]]) - PP.event_macro(*[x[s].sum(0) for x in tab["frozen"][:3]]) for s in picks]
            delta = PP.event_macro(*[x.sum(0) for x in tab["RF"][:3]]) - PP.event_macro(*[x.sum(0) for x in tab["frozen"][:3]])
        else:
            boot = [PP.frame_macro(tab["RF"][3][s].sum(0)) - PP.frame_macro(tab["frozen"][3][s].sum(0)) for s in picks]
            delta = PP.frame_macro(tab["RF"][3].sum(0)) - PP.frame_macro(tab["frozen"][3].sum(0))
        diff = fold[f"{label}_change"].values
        p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
        say(f"  {label}: {delta:+.3f} (95% CI {np.percentile(boot, 2.5):+.3f} to {np.percentile(boot, 97.5):+.3f}); folds better {int((diff > 0).sum())}, worse {int((diff < 0).sum())}; Wilcoxon p = {p:.3f}")

    tables = []
    for name, key in (("frozen raga-aware", "frozen"), ("raga-free", "RF")):
        t = located_table(key)
        for typ in ["All"] + list(SHORT.values()):
            q = t if typ == "All" else t[t.type == typ]
            for band in BANDS:
                b = q[q.band == band]
                tables.append(dict(run=name, type=typ, length_s=band, marks=len(b), located=int(b.located.sum()), right_type=int(b.right_type.sum())))
    L = pd.DataFrame(tables)
    L.to_csv(OUT["located_by_length.csv"], index=False)
    say("\nReal marks located, by length band and type: frozen raga-aware -> raga-free (located/marks)")
    view = L.assign(v=L.located.astype(str) + "/" + L.marks.astype(str)).pivot_table(index=["type", "length_s"], columns="run", values="v", aggfunc="first", sort=False)
    say(view[["frozen raga-aware", "raga-free"]].to_string())
    P = pd.DataFrame(params)
    say("\nSettings chosen for each held-out group")
    say(P.round(3).to_string(index=False))
    P.to_csv(OUT["chosen_params.csv"], index=False)
    fold.to_csv(OUT["per_fold.csv"], index_label="held_out_group")
    pd.DataFrame([dict(track=c["id"], start_s=round(s + c["offset"], 3), end_s=round(e + c["offset"], 3), type=FULL[l])
                  for c in CLIPS for s, e, l in c["RF"]], columns=["track", "start_s", "end_s", "type"]).to_csv(OUT["events_raga_free.csv"], index=False)
    OUT["raga_free_report.txt"].write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
