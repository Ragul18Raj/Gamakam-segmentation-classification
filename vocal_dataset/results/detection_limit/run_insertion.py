"""Controlled insertion test for the frozen Step 3 detector. EXPLORATORY. No training, no change to marks or Viterbi settings.
For each held-out group, the fold's frozen network, Viterbi settings, per-type length limits and class priors are used.
Each trial time-scales one marked contour of a given type from that fold's TRAINING groups to a target length, inserts it into one held-out track,
and runs the whole frozen pipeline on that modified track. The original pitch of the track is restored for the next trial.

Sites: sung, unmarked stretches at least 0.5 s from every real mark, and 0.2 s from every saved detection of the frozen run, so an existing detection cannot earn credit.
Pitch: the contour keeps its shape and net movement; it is shifted so the mismatch with the surrounding melody is shared equally at both ends; joins are raised-cosine ramps over 50 ms just outside the event.
Naming: the frozen DTW-3NN rule. The library excludes the held-out group and also the source contour's group, so a contour cannot match itself or its other-tempo versions.
Run from vocal_dataset:  python results/detection_limit/run_insertion.py   (add --test 12 to run a few trials and write nothing)"""
import argparse, multiprocessing as mp, sys
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf
HERE = Path(__file__).resolve().parent; VD = HERE.parents[1]; sys.path[:0] = [str(VD), str(VD / "results/two_stage")]
import two_stage as TS
import stage2_vocal as S
PP = TS.PP; OUT = HERE / "insertion/insertion_trials.csv"; LENGTHS = (80, 120, 160, 200, 250, 300, 400, 600, 800); PER_FOLD = 4; J, CTX, TOL, SEED = 5, 10, 60.0, 20261119; TYPES = TS.TYPES; GROUPS = TS.GROUPS
TV = {"__file__": str(VD / "results/step3/train_variant.py"), "__name__": "frozen_trainer"}
exec((VD / "results/step3/train_variant.py").read_text().split("\na = argparse.ArgumentParser()")[0], TV)   # the exact Net, input and predict code
CL = S.load_real_dataset("group"); BY = {c["id"]: c for c in CL}; P = pd.read_csv(VD / "results/step3/chosen_params.csv"); P = P[(P.variant == "all") & (P.cleanup == "B")].set_index("held_out_group"); E = pd.read_csv(VD / "results/step3/events_all_B.csv"); NET = {}

def otsu(x):
    h, e = np.histogram(x, 80); c = (e[:-1] + e[1:]) / 2; w = h.cumsum() / h.sum(); m = (h * c).cumsum() / h.sum()
    with np.errstate(all="ignore"): s = (m[-1] * w - m) ** 2 / (w * (1 - w))
    return c[np.nanargmax(s)]
def sung_mask(cid):
    """Voiced and loud enough to be singing, because CREPE also reports confident pitch on the drone during rests."""
    c = BY[cid]; y, sr = sf.read(VD / f"audio/{cid}.mp3", dtype="float64", always_2d=True); m = y.mean(1); hop = int(sr / 100); T = len(c["cents"]); i0 = c["i0"]; q = np.r_[0, np.cumsum(m ** 2)]; a = np.maximum(0, (i0 + np.arange(T) - 1) * hop).astype(int); b = np.minimum(len(m), (i0 + np.arange(T) + 1) * hop).astype(int)
    db = 10 * np.log10((q[b] - q[a]) / np.maximum(b - a, 1) + 1e-12); return cid, (db >= otsu(db)) & ~np.isnan(c["cents"])
def sources():
    out = []; meta, ev = TS.meta, TS.marks
    for rec, e in ev.groupby("recording", sort=False):
        t, cents, conf, v, _ = TS.PITCH[rec]
        for _, r in e.iterrows():
            idx = np.where((t >= r.start_s) & (t < r.end_s))[0]; vi = idx[v[idx]]; span = np.arange(vi[0], vi[-1] + 1); out.append(dict(rec=rec, start_s=r.start_s, group=TS.GROUP_NAME[rec], type=TS.LABEL_ID[r.label], native_ms=round((r.end_s - r.start_s) * 1000), curve=np.interp(span, vi, cents[vi]).astype(float), conf=np.clip(np.asarray(conf[span], float), .5, 1)))
    return out
def fill(x): i = np.flatnonzero(~np.isnan(x)); return np.interp(np.arange(len(x)), i, x[i]).astype(float)
def dilate(m, r): return np.convolve(m.astype(int), np.ones(2 * r + 1, int), mode="same") > 0
def sites(cid, n, sung):
    c = BY[cid]; T = len(c["cents"]); mark = np.zeros(T, bool); det = np.zeros(T, bool); x = E[E.track == cid]
    for s, e, _ in c["ann"]: mark[max(0, int(round(s * 100))):int(round(e * 100))] = True
    for s, e in zip(x.start_s, x.end_s): det[max(0, int(round((s - c["offset"]) * 100))):int(round((e - c["offset"]) * 100))] = True
    bad = np.r_[0, np.cumsum(dilate(mark, 50) | dilate(det, 20))]; good = np.r_[0, np.cumsum(sung[cid])]; f = fill(c["cents"].astype(float)); out = []; pad = J + CTX
    for a in range(pad, T - n - pad, 3):
        if bad[a + n] - bad[a] == 0 and (good[a + n + pad] - good[a - pad]) / (n + 2 * pad) >= .9: out.append((cid, a, float(np.median(f[a - pad:a - J])), float(np.median(f[a + n + J:a + n + pad]))))
    return out
def scaled(sr, n):
    u = np.linspace(0, len(sr["curve"]) - 1, n); i = np.arange(len(sr["curve"])); return np.interp(u, i, sr["curve"]), np.interp(u, i, sr["conf"])
def insert(c, a, n, cur, cf, L, R):
    x = c["cents"].astype(float).copy(); q = np.asarray(c["conf"], float).copy(); f = fill(x); o = ((L - cur[0]) + (R - cur[-1])) / 2; new = cur + o; x[a:a + n] = new; q[a:a + n] = cf; dl, dr = new[0] - f[a - 1], new[-1] - f[a + n]
    for i in range(J):
        w = .5 - .5 * np.cos(np.pi * (i + 1) / (J + 1)); x[a - J + i] = f[a - J + i] + w * dl; x[a + n + i] = f[a + n + i] + (1 - w) * dr
    q[a - J:a] = np.maximum(q[a - J:a], .6); q[a + n:a + n + J] = np.maximum(q[a + n:a + n + J], .6); return x.astype(np.float32), q.astype(np.float32)
def get_net(g):
    if g not in NET:
        n = TV["Net"](7, True, True); n.load_state_dict(TV["torch"].load(HERE / f"retrained/weights/fold_{g}.pt", map_location="cpu")["state_dict"]); n.eval(); NET[g] = n
    return NET[g]
def run_trial(tr):
    TV["torch"].set_num_threads(1); TV["torch"].use_deterministic_algorithms(True); g = tr["group"]; k = GROUPS.index(g); c = BY[tr["track"]]; sr = SRC[tr["source"]]; n = tr["length_ms"] // 10; a = tr["a"]; cur, cf = scaled(sr, n); x, q = insert(c, a, n, cur, cf, tr["L"], tr["R"])
    p = TV["predict"](get_net(g), S.feats(dict(cents=x, conf=q), False), True).astype(float); cfgq = P.loc[g]; d = dict(y=np.zeros(p.shape[1]), logp2=np.log(p.T + 1e-8)); reg = TS.regions_b(d, (cfgq.D_region_s, cfgq.switch, cfgq.alpha), PRIOR[g]); truth = (a / 100, (a + n) / 100, 1); ov = [(s, e) for s, e in reg if s < a + n and e > a]
    def pair(ev):
        _, _, _, pr = TS.match_iou([truth], [(s / 100, e / 100, 1) for s, e, *_ in ev]); return pr[0][1] if pr else None
    found = pair(ov) is not None; ok = ~np.isnan(x) & (q >= .5); named = []
    for s, e in ov:
        i = s + np.flatnonzero(ok[s:e])
        if len(i) >= 5:
            cv = TS.centre(np.interp(np.arange(i[0], i[-1] + 1), i, x[i]).astype(float)); row = np.array([TS.dtw(cv, b, max(5, int(.2 * max(len(cv), len(b))))) for b in TS.LIB_CURVE]); l = TS.name_from_row(row, {g, sr["group"]})
            if e - s >= int(round(cfgq[f"L_{TS.SHORT[l]}"] * 100)): named.append((s, e, l))
    hit = pair(named); det = next((z for z in named if hit and z[0] / 100 == hit[0] and z[1] / 100 == hit[1]), None)
    return dict(held_out_group=g, track=tr["track"], type=TS.SHORT.get(tr["type"], "Flat control"), length_ms=tr["length_ms"], start_frame=a, source_recording=sr["rec"], source_start_s=sr["start_s"], source_group=sr["group"], source_length_ms=sr["native_ms"], pitch_mismatch_cents=round(tr["mismatch"], 1), region_found=found, located=det is not None, right_type=det is not None and det[2] == tr["type"], detected_type=TS.SHORT[det[2]] if det else "", detected_length_ms=(det[1] - det[0]) * 10 if det else np.nan)
def identity(g):
    """The insertion code path, given an untouched track, must reproduce the retrained network's saved probabilities."""
    TV["torch"].set_num_threads(1); TV["torch"].use_deterministic_algorithms(True); z = np.load(HERE / f"retrained/probs/all/fold_{g}.npz"); worst = 0.0
    for c in CL:
        if c["group"] == g: worst = max(worst, float(np.abs(TV["predict"](get_net(g), S.feats(dict(cents=c["cents"], conf=c["conf"]), False), True) - z[f"test|{c['id']}"]).max()))
    return g, worst
def plan(sung):
    rng = np.random.default_rng(SEED); by_type = {t: [i for i, s in enumerate(SRC) if s["type"] == t] for t in (*TYPES, 0)}; trials = []; short = []
    for k, g in enumerate(GROUPS):
        val = GROUPS[(k + 1) % 11]; tracks = [c["id"] for c in CL if c["group"] == g]; cand = {ms: [s for t in tracks for s in sites(t, ms // 10, sung)] for ms in LENGTHS}
        for t in (*TYPES, 0):                   # type 0 is the flat-contour control
            pool = rng.permutation([i for i in by_type[t] if SRC[i]["group"] not in (g, val)]); z = 0
            for ms in LENGTHS:
                used = set()
                for _ in range(PER_FOLD):
                    src = int(pool[z % len(pool)]); z += 1; cur, _c = scaled(SRC[src], ms // 10); net = cur[-1] - cur[0]; C = [s for s in cand[ms] if (s[0], s[1]) not in used]
                    if not C: short.append((g, t, ms)); continue
                    mis = np.array([(s[3] - s[2]) - net for s in C]); ok = np.flatnonzero(np.abs(mis) <= TOL); i = int(rng.choice(ok)) if len(ok) else int(np.abs(mis).argmin()); s = C[i]; used.add((s[0], s[1])); trials.append(dict(group=g, type=t, length_ms=ms, source=src, track=s[0], a=s[1], L=s[2], R=s[3], mismatch=float(mis[i])))
    return trials, short
if __name__ == "__main__":
    A = argparse.ArgumentParser(); A.add_argument("--test", type=int, default=0); A = A.parse_args()
    if not A.test and OUT.exists(): sys.exit("Output already exists; not overwriting it.")
    PP.viterbi(np.zeros((30, 5)), np.array([0, 2, 2, 2, 2], np.int64), 1.0); TS.dtw(np.zeros(5), np.zeros(5), 5)
    with mp.Pool(16) as p: SUNG = dict(p.map(sung_mask, list(BY)))
    SRC = sources(); SRC.append(dict(rec="flat control", start_s=0.0, group="none", type=0, native_ms=0, curve=np.zeros(100), conf=np.full(100, .9))); PRIOR = {g: TS.priors2(k) for k, g in enumerate(GROUPS)}; T, short = plan(SUNG); print(f"planned {len(T)} trials; slots with no site: {len(short)}; sung share median {np.median([m.mean() for m in SUNG.values()]):.2f}; pitch mismatch abs median {np.median([abs(t['mismatch']) for t in T]):.0f} c, p90 {np.percentile([abs(t['mismatch']) for t in T], 90):.0f} c", flush=True)
    with mp.Pool(11) as p: same = dict(p.map(identity, GROUPS))
    print("largest probability difference on untouched tracks:", max(same.values()), flush=True); assert max(same.values()) < 1e-5
    if A.test: T = [T[i] for i in np.linspace(0, len(T) - 1, A.test).astype(int)]
    with mp.Pool(26) as p: R = p.map(run_trial, T, chunksize=4)
    D = pd.DataFrame(R); print(D.groupby(["type"]).agg(n=("located", "size"), region=("region_found", "mean"), located=("located", "mean"), right=("right_type", "mean")).round(2).to_string() if not A.test else D.iloc[:, [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14]].to_string())
    if not A.test: D.to_csv(OUT, index=False); pd.DataFrame(short, columns=["group", "type", "length_ms"]).to_csv(HERE / "insertion/slots_without_site.csv", index=False)
