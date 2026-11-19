"""Step 3 evaluation: apply the Step 2 clean-up, tuning and DTW-3NN naming to each trained detector. EXPLORATORY.
Primary run, fixed before training: variant 'all' with Viterbi clean-up (B). Everything else is a diagnostic.
Run from vocal_dataset after all 55 models have been trained:  python results/step3/evaluate_step3.py"""
import multiprocessing as mp, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score
HERE = Path(__file__).resolve().parent; VD = HERE.parents[1]; sys.path[:0] = [str(VD), str(VD / "results/two_stage")]
import two_stage as TS
PP = TS.PP; DATA, GROUPS, FR, TYPES, FULL, SHORT = TS.DATA, TS.GROUPS, TS.FR, TS.TYPES, TS.FULL, TS.SHORT
VARIANTS = ["base", "swara", "bilstm", "heads", "all"]; TITLE = dict(base="PyTorch base", swara="+ swara distance", bilstm="+ BiLSTM", heads="+ onset/offset heads", all="all three (primary)")
CLIPS = list(TS.CLIPS.values())

def run_variant(v):
    """Same steps as two_stage.main: find regions, name them, tune on the other 10 groups, apply once to the held-out group."""
    for g in GROUPS:
        z = np.load(HERE / f"probs/{v}/fold_{g}.npz")
        for c in DATA[g]:
            c["p2"] = z[f"test|{c['id']}"].astype(np.float64); assert c["p2"].shape == (2, len(c["y"])); c["logp2"] = np.log(c["p2"].T + 1e-8)
    TS.REGIONS.clear(); jobs = sorted({TS.region_key(ci, k) for ci in range(len(TS.CONFIGS)) for k in range(len(GROUPS))}, key=str)
    with mp.Pool(PP.PROCS) as p:
        for job, reg in p.imap_unordered(TS.region_job, jobs, chunksize=4): TS.REGIONS[job] = reg
    new = sorted({(cid, s, e) for reg in TS.REGIONS.values() for cid, rs in reg.items() for s, e in rs} - set(TS.DIST))
    with mp.Pool(PP.PROCS) as p: TS.DIST.update(p.imap_unordered(TS.dist_job, new, chunksize=50))
    with mp.Pool(PP.PROCS) as p: tuned = p.map(TS.tune_job, [(ci, k) for k in range(len(GROUPS)) for ci in range(len(TS.CONFIGS))], chunksize=4)
    best = {}
    for (ci, k), score, pick in tuned:
        kd = TS.CONFIGS[ci][0]
        if (kd, k) not in best or score > best[(kd, k)][0] + 1e-12: best[(kd, k)] = (score, ci, pick)
    rows = []
    for k, g in enumerate(GROUPS):
        for kd in "AB":
            score, ci, pick = best[(kd, k)]; cfg = TS.CONFIGS[ci][1]; names = ("W_ms", "G_ms", "E_ms") if kd == "A" else ("D_region_s", "switch", "alpha"); mins = {l: int(round(TS.L_GRID[pick[l]] * FR)) for l in TYPES}
            rows.append(dict(variant=v, cleanup=kd, held_out_group=g, **dict(zip(names, cfg)), **{f"L_{SHORT[l]}": TS.L_GRID[pick[l]] for l in TYPES}, tuning_event_f1=score))
            for c in DATA[g]:
                ev = PP.drop_short(TS.named_events(c["id"], TS.REGIONS[TS.region_key(ci, k)][c["id"]], {g}), mins); key = f"{v}_{kd}"
                assert all(ev[i][1] <= ev[i + 1][0] for i in range(len(ev) - 1)); c[key], c[key + "_frames"] = PP.seconds(ev), PP.to_frames(ev, len(c["y"]))
    print("finished", v, flush=True); return rows

def main():
    files = [HERE / "step3_report.txt", HERE / "chosen_params.csv"] + [HERE / f"events_{v}_{k}.csv" for v in VARIANTS for k in "AB"]
    if any(f.exists() for f in files): sys.exit("Output files already exist; not overwriting them.")
    lines = []
    def say(s=""): print(s, flush=True); lines.append(str(s))
    TS.check_library(); PP.viterbi(np.zeros((30, 5)), np.array([0, 2, 2, 2, 2], np.int64), 1.0); params = [r for v in VARIANTS for r in run_variant(v)]
    s2 = pd.read_csv(VD / "results/two_stage/events_two_stage_B.csv")
    for c in CLIPS:
        x = s2[s2.track == c["id"]]; ev = [(int(round((a - c["offset"]) * FR)), int(round((b - c["offset"]) * FR)), TS.LABEL_ID[t]) for a, b, t in zip(x.start_s, x.end_s, x.type)]
        c["S2B"], c["S2B_frames"] = PP.seconds(ev), PP.to_frames(ev, len(c["y"])); c["original"], c["original_frames"] = c["cnn"], c["cnn_frames"]
    Y = np.concatenate([c["y"] for c in CLIPS]); R = {"original": "original", "Step 2 Two-stage B": "S2B", **{TITLE[v]: f"{v}_B" for v in VARIANTS}}; RA = {TITLE[v]: f"{v}_A" for v in VARIANTS}; tab, per = {}, {}
    def row(n, k):
        m, pc, _ = TS.ev_scores(CLIPS, k); fr = f1_score(Y, np.concatenate([c[k + "_frames"] for c in CLIPS]), labels=range(5), average="macro", zero_division=0); tab[k] = PP.group_table(DATA, k); t, nr, ne, cm = [x.sum(0) for x in tab[k]]
        assert abs(PP.event_macro(t, nr, ne) - m) < 1e-12 and abs(PP.frame_macro(cm) - fr) < 1e-12; per[n] = pc; return dict(run=n, event=m, frame=fr, detections=int(ne.sum()), correct=int(t.sum()))
    B = [row(n, k) for n, k in R.items()]; A = [row(n + " (A)", k) for n, k in RA.items()]; assert abs(B[0]["event"] - .395) < .0006 and abs(B[1]["event"] - .437) < .0006
    say("STEP 3: detector additions. EXPLORATORY. Recorded vocal, one singer; 11 held-out groups pooled. Settings tuned only on the other 10 groups for each held-out group.")
    say("Primary run: all three additions, Viterbi clean-up, Stage 1 DTW-3NN naming. Existing inputs already include velocity and confidence; context is already about 2.5 s.\n")
    say("Viterbi clean-up (B)"); say(pd.DataFrame(B).round(3).to_string(index=False)); say("\nSmooth/merge/extend clean-up (A)"); say(pd.DataFrame(A).round(3).to_string(index=False))
    say("\nPer type, Viterbi clean-up: F1 / precision / recall / detections"); say(pd.DataFrame({n: {FULL[l]: "{f1:.2f} / {precision:.2f} / {recall:.2f} / {n_est}".format(**per[n][FULL[l]]) for l in TYPES} for n in R}).T.to_string())
    say("\nRegion finding with type ignored (IoU >= 0.3), Viterbi clean-up"); say(pd.DataFrame({n: TS.detection_stats(CLIPS, k) for n, k in R.items()}).T.round(3).to_string())
    say("\nEvents shorter than 0.2 s: " + ", ".join(f"{n} {sum(int(round((e - s) * FR)) < 20 for c in CLIPS for s, e, _ in c[k])}" for n, k in R.items()))
    fold = {k: dict(event=np.array([TS.ev_scores(DATA[g], k)[0] for g in GROUPS]), frame=np.array([PP.frame_macro(c) for c in tab[k][3]])) for k in R.values()}; picks = np.random.default_rng(PP.SEED).integers(0, len(GROUPS), (PP.N_BOOT, len(GROUPS)))
    pooled = {k: (PP.event_macro(*[x.sum(0) for x in tab[k][:3]]), PP.frame_macro(tab[k][3].sum(0))) for k in R.values()}
    def compare(a, b, title):
        for j, lab in enumerate(("event", "frame")):
            if lab == "event": bs = [PP.event_macro(*[x[s].sum(0) for x in tab[a][:3]]) - PP.event_macro(*[x[s].sum(0) for x in tab[b][:3]]) for s in picks]
            else: bs = [PP.frame_macro(tab[a][3][s].sum(0)) - PP.frame_macro(tab[b][3][s].sum(0)) for s in picks]
            d = fold[a][lab] - fold[b][lab]; p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            say(f"  {title} {lab}: {pooled[a][j] - pooled[b][j]:+.3f} (95% CI {np.percentile(bs, 2.5):+.3f} to {np.percentile(bs, 97.5):+.3f}); folds better {int((d > 0).sum())}, worse {int((d < 0).sum())}; Wilcoxon p = {p:.3f}")
    say(f"\nChange from the original; group bootstrap {PP.N_BOOT} resamples; two-sided Wilcoxon over 11 folds")
    for v in VARIANTS: compare(f"{v}_B", "original", TITLE[v])
    say("\nChange from the PyTorch base with the same clean-up")
    for v in VARIANTS[1:]: compare(f"{v}_B", "base_B", TITLE[v])
    say("\nPer-fold event F1, Viterbi clean-up"); say(pd.DataFrame({n: fold[k]["event"] for n, k in R.items()}, index=GROUPS).round(3).to_string())
    P = pd.DataFrame(params); say("\nPrimary run settings by held-out group"); say(P[(P.variant == "all") & (P.cleanup == "B")].dropna(axis=1, how="all").drop(columns=["variant", "cleanup"]).round(3).to_string(index=False))
    P.to_csv(files[1], index=False)
    for v in VARIANTS:
        for kd in "AB":
            pd.DataFrame([dict(track=c["id"], start_s=round(s + c["offset"], 3), end_s=round(e + c["offset"], 3), type=FULL[l]) for c in CLIPS for s, e, l in c[f"{v}_{kd}"]], columns=["track", "start_s", "end_s", "type"]).to_csv(HERE / f"events_{v}_{kd}.csv", index=False)
    files[0].write_text("\n".join(lines) + "\n")
if __name__ == "__main__": main()
