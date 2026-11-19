"""Run the current raga-free gamaka detector on one recording and write one JSON result. EXPLORATORY.

A fresh process runs this file for every analysis, so it always imports the model code currently on disk.
Dataset recordings use the fold model that did not train on them. Other recordings use the average of the 11 fold models
and the median of the 11 tuned settings; that combination has not been evaluated on new singers or ragas.
"""
import argparse, json, os, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, soundfile as sf

VD = Path(__file__).resolve().parents[1]; RF_DIR = VD / "results/raga_free"
sys.path[:0] = [str(VD), str(VD / "results/two_stage"), str(RF_DIR)]
A = argparse.ArgumentParser()
for name in ("--audio", "--output", "--progress", "--fingerprint", "--manifest"): A.add_argument(name, required=True)
A.add_argument("--known", default=""); A.add_argument("--pitch", default=""); A.add_argument("--name", default="recording"); A = A.parse_args()

def progress(stage, message, fraction):
    tmp = Path(A.progress + ".tmp"); tmp.write_text(json.dumps(dict(stage=stage, message=message, fraction=fraction, time=time.time()))); tmp.replace(A.progress)

progress("loading", "Loading the current model code", .08)
import rf_common as R
import two_stage as TS
from vocal_pitch import fold_octave_slips
PP, TV, S = TS.PP, R.TV, R.S; torch = TV["torch"]; torch.set_num_threads(min(8, os.cpu_count() or 1))
PARAM = pd.read_csv(RF_DIR / "chosen_params.csv").set_index("held_out_group"); GROUPS, SHORT, FULL, FR = TS.GROUPS, TS.SHORT, TS.FULL, TS.FR

def load_net(g):
    w = int(PARAM.loc[g, "window_frames"]); net = R.new_net(); net.load_state_dict(torch.load(RF_DIR / f"weights/W{w}/fold_{g}.pt", map_location="cpu")["state_dict"]); net.eval(); return net, w

def rms_envelope(path, n):
    y, sr = sf.read(path, dtype="float64", always_2d=True); m = y.mean(1); hop = sr / FR; q = np.r_[0, np.cumsum(m ** 2)]; i = (np.arange(n) * hop).astype(int); a = np.clip(i - int(hop), 0, len(m)); b = np.clip(i + int(hop), 0, len(m)); e = np.sqrt((q[b] - q[a]) / np.maximum(b - a, 1)); top = max(float(np.percentile(e, 99.5)), 1e-9); return np.clip(e / top, 0, 1), len(m) / sr

def rounded(v, digits=2): return None if v is None or not np.isfinite(v) else round(float(v), digits)

if A.known:
    t, rec_cents, rec_conf, rec_voiced, _ = TS.PITCH[A.known]; clips = {c["id"]: c for c in S.load_real_dataset("group")}; c = clips[A.known]; g = c["group"]; k = GROUPS.index(g); q = PARAM.loc[g]
    x, conf, offset, i0, folds, exclude = c["cents"], c["conf"], float(c["offset"]), int(c["i0"]), [g], {g}; prior = TS.priors2(k); ref = c["ann"]; mode = f"Dataset recording: fold model {g}, which did not train on this passage"
else:
    z = np.load(A.pitch); t, f, rec_conf = z["time_s"], z["f0_hz"], z["confidence"]; rec_voiced = rec_conf >= .5
    if rec_voiced.sum() < 20: raise SystemExit("Too little confident pitch was found in this recording.")
    base = float(np.median(f[rec_voiced])); rec_cents = 1200 * np.log2(np.maximum(f, 1e-3) / base); rec_cents, _ = fold_octave_slips(rec_cents, rec_voiced)
    x = np.where(rec_voiced, rec_cents, np.nan).astype(np.float32); conf = rec_conf.astype(np.float32); offset, i0, folds, exclude = 0.0, 0, GROUPS, set(); prior = TS.priors2(-1); ref = []; mode = "New recording: average of the 11 fold models"
    numeric = PARAM.drop(columns=["window_frames", "tuning_event_f1"]).median(); q = numeric
T = len(x); probs = []
for n, g in enumerate(folds):
    progress("detecting", f"Running detector {n + 1} of {len(folds)}", .15 + .5 * n / len(folds)); net, w = load_net(g); probs.append(TV["predict"](net, R.features_for(x, conf, w), False).astype(float))
p = np.mean(probs, axis=0); progress("decoding", "Finding gamaka regions", .68)
regions = TS.regions_b(dict(y=np.zeros(T), logp2=np.log(p.T + 1e-8)), (float(q.D_region_s), float(q.switch), float(q.alpha)), prior); mins = {l: int(round(float(q[f"L_{SHORT[l]}"]) * FR)) for l in TS.TYPES}
allowed = np.flatnonzero(~np.isin(TS.LIB_G, list(exclude))); events, too_short, unvoiced = [], 0, 0
for n, (s, e) in enumerate(regions):
    progress("naming", f"Naming region {n + 1} of {len(regions)}", .7 + .22 * n / max(len(regions), 1)); curve, nv = TS.event_curve(t, rec_cents, rec_voiced, s / FR + offset, e / FR + offset)
    if nv < 5: unvoiced += 1; continue
    cur = TS.centre(curve); row = np.array([TS.dtw(cur, b, max(5, int(.2 * max(len(cur), len(b))))) for b in TS.LIB_CURVE]); label = TS.name_from_row(row, exclude)
    if e - s < mins[label]: too_short += 1; continue
    near = allowed[np.argsort(row[allowed], kind="stable")[:3]]; votes = [SHORT[int(TS.LIB_Y[i])] for i in near]
    events.append(dict(start=round(s / FR + offset, 3), end=round(e / FR + offset, 3), type=SHORT[label], full_type=FULL[label], label=label, frame_start=s, frame_end=e, confidence=round(float(p[1, s:e].mean()), 3), neighbours=votes, nearest_distance_cents=round(float(row[near[0]]), 1), pitch_range_cents=round(float(np.percentile(curve, 95) - np.percentile(curve, 5)), 1)))

marks, summary = [], None
if ref:
    pairs = TS.match_iou([(s, e, 1) for s, e, _ in ref], [(ev["frame_start"] / FR, ev["frame_end"] / FR, 1) for ev in events])[3]; rm = {(r[0], r[1]): d[:2] for r, d in pairs}; byspan = {(ev["frame_start"] / FR, ev["frame_end"] / FR): ev for ev in events}
    for s, e, l in ref:
        d = rm.get((s, e)); ev = byspan.get(d) if d else None; right = bool(ev and ev["label"] == l); marks.append(dict(start=round(s + offset, 3), end=round(e + offset, 3), type=SHORT[l], full_type=FULL[l], located=ev is not None, right_type=right))
        if ev: ev["status"] = "right type" if right else f"overlaps a marked {SHORT[l]}"
    for ev in events: ev.setdefault("status", "no matching mark")
    saved = pd.read_csv(RF_DIR / "events_raga_free.csv"); saved = saved[saved.track == A.known]; same = [(round(a, 3), round(b, 3), n) for a, b, n in zip(saved.start_s, saved.end_s, saved.type)] == [(ev["start"], ev["end"], ev["full_type"]) for ev in events]
    summary = dict(marks=len(marks), located=sum(m["located"] for m in marks), right_type=sum(m["right_type"] for m in marks), unmatched_detections=sum(ev["status"] == "no matching mark" for ev in events), matches_saved_evaluation=same)
progress("plotting", "Preparing the display", .95); env, duration = rms_envelope(A.audio, len(t)); voiced_cents = rec_cents[rec_voiced]; middle = float(np.median(voiced_cents)) if len(voiced_cents) else 0; step = max(1, int(np.ceil(len(t) / 60000))); prob = np.full(len(t), np.nan); use = min(T, len(t) - i0); prob[i0:i0 + use] = p[1, :use]; idx = np.arange(0, len(t), step)
for ev in events: ev.pop("frame_start"); ev.pop("frame_end"); ev.pop("label")
params = {k: round(float(q[k]), 3) for k in ("D_region_s", "switch", "alpha", "L_Kampita", "L_Nokku", "L_Etra", "L_Irakka")}
result = dict(name=A.name, duration=round(duration, 3), hop_s=step / FR, known_track=A.known or None, generated_at=time.time(), events=events, marks=marks, summary=summary, rejected=dict(too_short_after_naming=too_short, too_little_voiced_pitch=unvoiced), plot=dict(pitch=[rounded(rec_cents[i] - middle, 1) if rec_voiced[i] else None for i in idx], gamaka_probability=[rounded(prob[i], 3) for i in idx], envelope=[round(float(env[i]), 3) for i in idx]), model=dict(fingerprint=A.fingerprint, files=json.loads(Path(A.manifest).read_text()), mode=mode, folds=list(folds), window_frames=[int(PARAM.loc[g, "window_frames"]) for g in folds], settings=params, uses_sa_or_raga=False))
tmp = Path(A.output + ".tmp"); tmp.write_text(json.dumps(result, separators=(",", ":"))); tmp.replace(A.output); progress("done", "Analysis complete", 1)
