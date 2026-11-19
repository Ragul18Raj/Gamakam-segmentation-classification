"""Controls for the raga-free Stage 2 detector. EXPLORATORY.

1. Transposition: shift every held-out track by +300 and -300 cents and rerun the whole frozen route.
2. Median-pitch floor: name events from Sa-relative median pitch alone.
3. Information probes. All numbered tracks are Kalyani and the VARNAM's raga is unlabelled, so a raga probe cannot be trained.
   Instead, a swara probe uses group cross-validation to ask whether Sa-relative note position can be read out.
   A track probe uses separate time blocks inside each track, because held-out groups contain different track identities.
Run from vocal_dataset:  python results/raga_free/controls.py
"""
import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
VD = HERE.parents[1]
sys.path[:0] = [str(VD), str(VD / "results/two_stage"), str(HERE)]
warnings.filterwarnings("ignore")
import rf_common as R  # noqa: E402
import two_stage as TS  # noqa: E402

PP, S, TV = TS.PP, R.S, R.TV
torch = TV["torch"]
GROUPS, FR, TYPES, FULL, SHORT = TS.GROUPS, TS.FR, TS.TYPES, TS.FULL, TS.SHORT
CL = S.load_real_dataset("group")
BY = {c["id"]: c for c in CL}
RF = pd.read_csv(HERE / "chosen_params.csv").set_index("held_out_group")
FZ = pd.read_csv(VD / "results/step3/chosen_params.csv")
FZ = FZ[(FZ.variant == "all") & (FZ.cleanup == "B")].set_index("held_out_group")
OUT = {n: HERE / n for n in ("controls_report.txt", "transposition_test.csv", "median_pitch_floor.csv", "information_probes.csv")}
SWARA = np.array([0, 200, 400, 600, 700, 900, 1100, 1200.0])
HALF = 15
NETS = {}


def net_for(kind, g):
    if (kind, g) not in NETS:
        if kind == "raga-free":
            n = R.new_net()
            w = int(RF.loc[g, "window_frames"])
            state = torch.load(HERE / f"weights/W{w}/fold_{g}.pt", map_location="cpu")["state_dict"]
        else:
            n = TV["Net"](7, True, True)
            state = torch.load(VD / f"results/detection_limit/retrained/weights/fold_{g}.pt", map_location="cpu")["state_dict"]
        n.load_state_dict(state)
        n.eval()
        NETS[(kind, g)] = n
    return NETS[(kind, g)]


def inputs(kind, c, g, shift=0.0):
    if kind == "raga-free":
        return R.features_for(c["cents"], c["conf"], int(RF.loc[g, "window_frames"]), shift)
    x = (np.asarray(c["cents"], np.float64) + shift).astype(np.float32)
    return TV["with_inputs"](S.feats(dict(cents=x, conf=c["conf"]), False), True)


def regions_and_events(kind, g, c, shift):
    k = GROUPS.index(g)
    q = RF.loc[g] if kind == "raga-free" else FZ.loc[g]
    p = TV["predict"](net_for(kind, g), inputs(kind, c, g, shift), False).astype(float)
    d = dict(y=np.zeros(p.shape[1]), logp2=np.log(p.T + 1e-8))
    regions = TS.regions_b(d, (q.D_region_s, q.switch, q.alpha), TS.priors2(k))
    t, cents, _, voiced, _ = TS.PITCH[c["id"]]              # same recording-time curve cut as the main naming route
    x = np.asarray(cents, np.float64) + shift
    named = []
    for s, e in regions:
        curve, nv = TS.event_curve(t, x, voiced, s / FR + c["offset"], e / FR + c["offset"])
        if nv >= 5:
            cur = TS.centre(curve)
            row = np.array([TS.dtw(cur, b, max(5, int(.2 * max(len(cur), len(b))))) for b in TS.LIB_CURVE])
            named.append((s, e, TS.name_from_row(row, {g})))
    mins = {l: int(round(q[f"L_{SHORT[l]}"] * FR)) for l in TYPES}
    return p[1], regions, PP.drop_short(named, mins)


def transpose_job(job):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    kind, g, shift = job
    return job, {c["id"]: regions_and_events(kind, g, c, shift) for c in CL if c["group"] == g}


def scored(results, kind, shift):
    clips = []
    for g in GROUPS:
        for c in TS.DATA[g]:
            _, _, ev = results[(kind, g, shift)][c["id"]]
            clips.append(dict(ref=c["ref"], B=PP.seconds(ev), y=c["y"], B_frames=PP.to_frames(ev, len(c["y"]))))
    Y = np.concatenate([c["y"] for c in clips])
    frame = f1_score(Y, np.concatenate([c["B_frames"] for c in clips]), labels=range(5), average="macro", zero_division=0)
    macro, pc, _ = TS.ev_scores(clips, "B")
    return macro, frame, sum(len(c["B"]) for c in clips), sum(TS.match_iou(c["ref"], c["B"])[0][l] for c in clips for l in TYPES)


def median_cents(c, s, e):
    """Median Sa-relative pitch of an event; s and e are seconds from the start of the evaluated clip."""
    t, cents, _, voiced, _ = TS.PITCH[c["id"]]
    x, nv = TS.event_curve(t, cents, voiced, s + c["offset"], e + c["offset"])
    return float(np.median(x)) if nv >= 5 else np.nan


def median_floor(results):
    rows = []
    for c in CL:
        for s, e, l in c["ann"]:
            m = median_cents(c, s, e)
            rows.append(dict(group=c["group"], type=l, median=m))
    M = pd.DataFrame(rows).dropna()
    pred = np.zeros(len(M), int)
    clips = []
    for g in GROUPS:
        tr, te = M.group != g, M.group == g
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(M.loc[tr, ["median"]], M.loc[tr, "type"])
        pred[te.values] = clf.predict(M.loc[te, ["median"]])
        q = RF.loc[g]
        mins = {l: int(round(q[f"L_{SHORT[l]}"] * FR)) for l in TYPES}
        for c in TS.DATA[g]:
            _, regions, _ = results[("raga-free", g, 0.0)][c["id"]]
            full = BY[c["id"]]
            ev = []
            for a, b in regions:
                m = median_cents(full, a / FR, b / FR)
                if m == m:
                    ev.append((a, b, int(clf.predict(pd.DataFrame({"median": [m]}))[0])))
            clips.append(dict(ref=c["ref"], B=PP.seconds(PP.drop_short(ev, mins))))
    return f1_score(M.type, pred, average="macro", labels=list(TYPES)), TS.ev_scores(clips, "B")


def sample_frames(c, seed):
    x = np.asarray(c["cents"], np.float64)
    v = ~np.isnan(x) & (np.asarray(c["conf"]) >= .5)
    idx = np.flatnonzero(v)
    idx = idx[(idx >= HALF) & (idx < len(x) - HALF)]
    return np.sort(np.random.default_rng(seed).choice(idx, min(300, len(idx)), replace=False))


def swara_labels(c, idx):
    pc = np.mod(np.asarray(c["cents"], np.float64)[idx], 1200)
    return np.abs(pc[:, None] - SWARA).argmin(1) % 7


@torch.no_grad()
def representation(kind, rep, g, c, idx):
    f = inputs(kind, c, g)
    if rep == "input":
        return np.stack([f[:, i - HALF:i + HALF + 1].ravel() for i in idx])
    n = net_for(kind, g)
    h = n.conv(torch.from_numpy(f)[None])
    h = n.lstm(h.transpose(1, 2))[0][0].numpy()
    return h[idx]


def probe_job(job):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    kind, rep, probe, g = job
    X, y, grp, pos = [], [], [], []
    for n, c in enumerate(CL):
        idx = sample_frames(c, 20261120 + n)
        X.append(representation(kind, rep, g, c, idx))
        y.append(swara_labels(c, idx) if probe == "swara" else np.full(len(idx), n))
        grp += [c["group"]] * len(idx)
        pos.append(idx / max(len(c["cents"]) - 1, 1))
    X, y, grp, pos = np.vstack(X), np.concatenate(y), np.array(grp), np.concatenate(pos)
    if probe == "swara":
        tr, te = grp != g, grp == g
    else:
        block = np.minimum((pos * 5).astype(int), 4)
        edge = np.abs(pos * 5 - np.round(pos * 5)) < .05
        tr, te = (block % 2 == 0) & ~edge, (block % 2 == 1) & ~edge
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=1.0)).fit(X[tr], y[tr])
    p = clf.predict(X[te])
    labels, counts = np.unique(y[tr], return_counts=True)
    chance = np.mean(y[te] == labels[counts.argmax()])
    return dict(model=kind, representation=rep, probe=probe, fold=g, n_test=int(te.sum()), correct=int((p == y[te]).sum()),
                accuracy=float((p == y[te]).mean()), balanced_accuracy=float(balanced_accuracy_score(y[te], p)), majority_chance=float(chance), classes=len(labels))


def main():
    if any(p.exists() for p in OUT.values()):
        sys.exit("Output files already exist; not overwriting them.")
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(str(s))

    PP.viterbi(np.zeros((30, 5)), np.array([0, 2, 2, 2, 2], np.int64), 1.0)
    TS.dtw(np.zeros(5), np.zeros(5), 5)
    shifts = (0.0, 300.0, -300.0)
    jobs = [(kind, g, s) for kind in ("raga-free", "frozen raga-aware") for g in GROUPS for s in shifts]
    with mp.Pool(26) as pool:
        results = dict(pool.imap_unordered(transpose_job, jobs, chunksize=1))
    rows = []
    for kind in ("raga-free", "frozen raga-aware"):
        base = scored(results, kind, 0.0)
        p0 = np.concatenate([results[(kind, g, 0.0)][c["id"]][0] for g in GROUPS for c in TS.DATA[g]])
        for s in shifts:
            m = scored(results, kind, s)
            p = np.concatenate([results[(kind, g, s)][c["id"]][0] for g in GROUPS for c in TS.DATA[g]])
            same = sum(results[(kind, g, s)][c["id"]][2] == results[(kind, g, 0.0)][c["id"]][2] for g in GROUPS for c in TS.DATA[g])
            rows.append(dict(model=kind, shift_cents=int(s), event=m[0], frame=m[1], detections=m[2], correct=m[3], event_change=m[0] - base[0], frame_change=m[1] - base[1],
                             largest_gamaka_probability_change=float(np.abs(p - p0).max()), mean_gamaka_probability_change=float(np.abs(p - p0).mean()), tracks_with_identical_events=f"{same}/31"))
    T = pd.DataFrame(rows)
    assert abs(T[(T.model == "raga-free") & (T.shift_cents == 0)].event.iloc[0] - .387) < .0006
    assert abs(T[(T.model == "frozen raga-aware") & (T.shift_cents == 0)].event.iloc[0] - .462) < .0006
    T.to_csv(OUT["transposition_test.csv"], index=False)
    say("RAGA-FREE CONTROLS. EXPLORATORY. Recorded vocal, one singer.\n")
    say("Transposition test: every held-out track shifted before the full route")
    say(T.round(6).to_string(index=False))

    mark_f1, (ev_f1, pc, _) = median_floor(results)
    pd.DataFrame([dict(measure="marked events, type from median pitch, macro-F1", value=mark_f1), dict(measure="raga-free regions named from median pitch, event macro-F1", value=ev_f1),
                  *[dict(measure=f"raga-free regions named from median pitch, {k} event F1", value=v["f1"]) for k, v in pc.items()]]).to_csv(OUT["median_pitch_floor.csv"], index=False)
    say(f"\nMedian-pitch probe: macro-F1 on the singer's marked boundaries {mark_f1:.3f}. Event macro-F1 when raga-free regions are named from median pitch alone: {ev_f1:.3f}.")
    say("Per type event F1 for that floor: " + ", ".join(f"{k} {v['f1']:.2f}" for k, v in pc.items()))

    jobs = [(kind, rep, probe, g) for kind in ("raga-free", "frozen raga-aware") for rep in ("input", "hidden") for probe in ("swara", "track") for g in GROUPS]
    with mp.Pool(26) as pool:
        P = pd.DataFrame(pool.map(probe_job, jobs, chunksize=1))
    P.to_csv(OUT["information_probes.csv"], index=False)
    summary = []
    for key, q in P.groupby(["probe", "model", "representation"], sort=False):
        acc = q.correct.sum() / q.n_test.sum() if key[0] == "swara" else q.accuracy.mean()
        summary.append(dict(probe=key[0], model=key[1], representation=key[2], accuracy=acc, balanced_accuracy=q.balanced_accuracy.mean(), majority_chance=(q.majority_chance * q.n_test).sum() / q.n_test.sum(), classes=int(q.classes.max())))
    say("\nInformation probes. Swara: leave-one-group-out, 7 Kalyani note classes. Track: separate within-track time blocks, 31 classes")
    say(pd.DataFrame(summary).round(3).to_string(index=False))
    say("\nA raga probe and leave-one-raga-out are not possible: TRACK 1-30 are all labelled Kalyani and the VARNAM has no raga label.")
    OUT["controls_report.txt"].write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
