"""Name the gamaka in one selected stretch of a recording. EXPLORATORY dashboard tool; it never writes annotation files.

Naming reuses the Stage 1 code unchanged:
  vocal_pitch.event_curve   cut the selection and interpolate across unvoiced frames
  stage1_vocal.centre       subtract the median (no Sa or raga needed)
  stage1_vocal.dtw          DTW inside a Sakoe-Chiba band, distance divided by n + m (mean absolute cents per step)
  two_stage.name_from_row   three nearest allowed library events; a tie goes to the nearest tied type
The library is two_stage.LIB_CURVE: every event in annotations/pass1_events.csv, built once when this module loads.
For a dataset recording, events from its own passage group (all tempo versions) are excluded, as in the Stage 1
held-out evaluation. New code here: the reject threshold, the subsequence search, and the two-gamaka check, which runs
the dashboard's raga-free detector and Viterbi on the selection plus 2.5 s of context on each side.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from numba import njit
from scipy.ndimage import median_filter

HERE = Path(__file__).resolve().parent; VD = HERE.parent; RF_DIR = VD / "results/raga_free"; CONFIG = HERE / "classify_config.json"
sys.path[:0] = [str(VD), str(VD / "results/two_stage"), str(RF_DIR)]
import two_stage as TS  # noqa: E402  (builds the Stage 1 library at import)
from vocal_pitch import CONF, event_curve, fold_octave_slips  # noqa: E402

FR, MIN_VOICED, MIN_LEN_S, MAX_LEN_S, CONTEXT_S = TS.FR, TS.MIN_VOICED, 0.08, 60.0, 2.5
SHORT, FULL, GROUPS = TS.SHORT, TS.FULL, TS.GROUPS


class Reject(Exception):
    """A selection that cannot be named; the message is shown to the user."""


# ------------------------------------------------------------ library (from two_stage, same order)
LIB_TRACK, LIB_START, LIB_END = [], [], []
for rec, e in TS.marks.groupby("recording", sort=False):
    for r in e.itertuples():
        LIB_TRACK.append(rec), LIB_START.append(float(r.start_s)), LIB_END.append(float(r.end_s))
assert [(a, b) for a, b in zip(LIB_TRACK, LIB_START)] == [tuple(k) for k in TS.LIB_KEY], "library order changed"
LIB_TRACK = np.array(LIB_TRACK)
GROUP_TRACKS = {g: sorted(TS.GROUP_NAME.index[TS.GROUP_NAME == g]) for g in GROUPS}


def lib_row(q):
    """Stage 1 distance from one centred contour to every library event (same band rule as stage1_vocal.dist_matrix)."""
    return np.array([TS.dtw(q, b, max(5, int(.2 * max(len(q), len(b))))) for b in TS.LIB_CURVE])


def name3(row, exclude):
    label = TS.name_from_row(row, exclude)
    allowed = np.flatnonzero(~np.isin(TS.LIB_G, list(exclude)))
    near = allowed[np.argsort(row[allowed], kind="stable")[:3]]
    return label, near, int(sum(TS.LIB_Y[i] == label for i in near))


def config():
    c = json.loads(CONFIG.read_text())
    return float(c["reject_distance"]), bool(c.get("tuned", False))


# ------------------------------------------------------------ recordings
_TRACKS = {}


def load_track(known="", pitch="", audio=""):
    """Pitch for one recording. Dataset recordings use the same cents as Stage 1 (two_stage.PITCH); new recordings use
    the dashboard's cached CREPE file, in cents from the recording's median with octave slips folded, as analyze.py does."""
    key = known or pitch
    if key in _TRACKS:
        return _TRACKS[key]
    if known:
        t, cents, conf, voiced, _ = TS.PITCH[known]
        g = TS.GROUP_NAME[known]
        tr = dict(name=known, t=t, cents=cents, conf=conf, voiced=voiced, group=g, exclude={g}, audio=audio)
    else:
        z = np.load(pitch)
        t, f, conf = z["time_s"], z["f0_hz"], z["confidence"]
        voiced = conf >= CONF
        if voiced.sum() < MIN_VOICED:
            raise Reject("Too little confident pitch was found in this recording.")
        cents = 1200 * np.log2(np.maximum(f, 1e-3) / float(np.median(f[voiced])))
        cents, _ = fold_octave_slips(cents, voiced)
        tr = dict(name="", t=t, cents=cents, conf=conf, voiced=voiced, group=None, exclude=set(), audio=audio)
    _TRACKS[key] = tr
    return tr


def rms_envelope(path, n):
    """Loudness envelope at the pitch frame rate (same as dashboard/analyze.py)."""
    y, sr = sf.read(path, dtype="float64", always_2d=True); m = y.mean(1); hop = sr / FR; q = np.r_[0, np.cumsum(m ** 2)]
    i = (np.arange(n) * hop).astype(int); a = np.clip(i - int(hop), 0, len(m)); b = np.clip(i + int(hop), 0, len(m))
    e = np.sqrt((q[b] - q[a]) / np.maximum(b - a, 1)); top = max(float(np.percentile(e, 99.5)), 1e-9)
    return np.clip(e / top, 0, 1), len(m) / sr


def view(req):
    tr = load_track(req.get("known", ""), req.get("pitch", ""), req.get("audio", ""))
    if "view" not in tr:
        t, c, v = tr["t"], tr["cents"], tr["voiced"]
        mid = float(np.median(c[v])) if v.any() else 0.0
        env, duration = rms_envelope(tr["audio"], len(t)) if tr["audio"] else (np.zeros(len(t)), float(t[-1]))
        step = max(1, int(np.ceil(len(t) / 60000))); idx = np.arange(0, len(t), step)
        tr["view"] = dict(duration=round(duration, 3), hop_s=step / FR, t0=float(t[0]),
                          pitch=[round(float(c[i] - mid), 1) if v[i] else None for i in idx],
                          envelope=[round(float(env[i]), 3) for i in idx])
    g = tr["group"]
    return dict(tr["view"], known_track=tr["name"] or None, group=g, group_tracks=GROUP_TRACKS.get(g, []), library_size=len(TS.LIB_CURVE),
                library_used=int((~np.isin(TS.LIB_G, list(tr["exclude"]))).sum()))


# ------------------------------------------------------------ selection
def cut(tr, s, e):
    """Selection contour with Stage 1's event_curve; also the frame index where the contour starts."""
    curve, nv = event_curve(tr["t"], tr["cents"], tr["voiced"], s, e)
    idx = np.where((tr["t"] >= s) & (tr["t"] < e))[0]
    v = idx[tr["voiced"][idx]]
    return curve, nv, (int(v[0]) if len(v) else -1)


def check(tr, s, e):
    if not (np.isfinite(s) and np.isfinite(e)) or e <= s:
        raise Reject("The end of the selection must be after its start.")
    if e - s < MIN_LEN_S - 1e-9:
        raise Reject(f"The selection is {round((e - s) * 1000)} ms long. At least {round(MIN_LEN_S * 1000)} ms is needed.")
    if e - s > MAX_LEN_S:
        raise Reject(f"The selection is {e - s:.1f} s long. Please select at most {MAX_LEN_S:.0f} s.")
    if s >= tr["t"][-1] + 1 / FR:
        raise Reject("The selection starts after the end of the pitch track.")


def name_span(tr, s, e):
    curve, nv, first = cut(tr, s, e)
    if nv < MIN_VOICED:
        raise Reject(f"Only {nv} voiced pitch frame{'s' if nv != 1 else ''} in the selection. At least {MIN_VOICED} are needed "
                     "(10 ms frames with CREPE confidence of at least 0.5).")
    q = TS.centre(curve); row = lib_row(q); label, near, votes = name3(row, tr["exclude"])
    return dict(curve=q, nv=nv, first=first, row=row, label=label, near=near, votes=votes, dist=float(row[near[0]]))


# ------------------------------------------------------------ subsequence search (new code)
@njit(cache=True)
def subseq_dtw(x, b, lo, hi):
    """Open-begin, open-end DTW of template b inside x. Returns (cost / (len(b) + span), start, end exclusive),
    keeping only spans of lo..hi frames. Same step costs and normalisation style as stage1_vocal.dtw."""
    n, m = len(x), len(b)
    inf = 1e18
    prev = np.zeros(n + 1); prev_s = np.arange(n + 1)
    cur = np.full(n + 1, inf); cur_s = np.zeros(n + 1, np.int64)
    for i in range(1, m + 1):
        cur[0] = inf
        for j in range(1, n + 1):
            c = abs(b[i - 1] - x[j - 1])
            best, src = prev[j - 1], prev_s[j - 1]
            if prev[j] < best:
                best, src = prev[j], prev_s[j]
            if cur[j - 1] < best:
                best, src = cur[j - 1], cur_s[j - 1]
            cur[j] = c + best; cur_s[j] = src
        prev, cur = cur, prev
        prev_s, cur_s = cur_s, prev_s
    out, bs, be = inf, -1, -1
    for j in range(1, n + 1):
        span = j - prev_s[j]
        if span < lo or span > hi:
            continue
        v = prev[j] / (m + span)
        if v < out:
            out, bs, be = v, prev_s[j], j
    return out, bs, be


def subsequence(tr, s, e, top=12):
    """Find the stretch inside the selection that best matches a library event, then name that stretch with the
    normal Stage 1 rule. Each template is compared with the selection minus a running median as long as the template,
    so a gamaka sitting above or below the selection's median can still be found."""
    curve, nv, first = cut(tr, s, e)
    if nv < MIN_VOICED:
        raise Reject(f"Only {nv} voiced pitch frames in the selection. At least {MIN_VOICED} are needed.")
    min_fr = int(round(MIN_LEN_S * FR)); allowed = np.flatnonzero(~np.isin(TS.LIB_G, list(tr["exclude"])))
    local, cand = {}, []
    for i in allowed:
        b = TS.LIB_CURVE[i]; m = len(b); w = min(len(curve), max(5, 2 * (m // 10) * 5 + 1))
        if w not in local:
            local[w] = (curve - median_filter(curve, size=w, mode="nearest")).astype(np.float64)
        lo, hi = max(min_fr, (m + 1) // 2), 2 * m
        if lo > len(curve):
            continue
        d, a, z = subseq_dtw(local[w], b, lo, hi)
        if a >= 0:
            cand.append((d, a, z))
    if not cand:
        raise Reject("No stretch inside the selection matched any library gamaka. Try a longer selection or turn the search off.")
    cand.sort()
    t = tr["t"]; best, seen = None, set()
    for d, a, z in cand:
        if (a, z) in seen:
            continue
        seen.add((a, z))
        ss, ee = float(t[first + a]), float(t[first + z - 1]) + 1 / FR
        try:
            r = name_span(tr, ss, ee)
        except Reject:
            continue
        r.update(span=(round(ss, 3), round(ee, 3)), search_cost=float(d))
        if best is None or r["dist"] < best["dist"]:
            best = r
        if len(seen) >= top:
            break
    if best is None:
        raise Reject("The best-matching stretches had too little voiced pitch to name.")
    return best


# ------------------------------------------------------------ two-gamaka check (dashboard detector + Viterbi)
_DET = {}


def detector():
    if not _DET:
        import rf_common as R
        torch = R.TV["torch"]; torch.set_num_threads(min(4, os.cpu_count() or 1))
        param = pd.read_csv(RF_DIR / "chosen_params.csv").set_index("held_out_group"); nets = {}
        for g in GROUPS:
            w = int(param.loc[g, "window_frames"]); net = R.new_net()
            net.load_state_dict(torch.load(RF_DIR / f"weights/W{w}/fold_{g}.pt", map_location="cpu")["state_dict"]); net.eval(); nets[g] = (net, w)
        _DET.update(R=R, param=param, nets=nets, median=param.drop(columns=["window_frames", "tuning_event_f1"]).median())
    return _DET


def detector_events(tr, s, e):
    """Raga-free detector on the selection plus CONTEXT_S each side (the network was trained with about 2.5 s of
    context). Dataset recordings use the fold model that did not train on them; others average the 11 fold models,
    exactly as dashboard/analyze.py does. Only events with at least half their length inside the selection are kept."""
    D = detector(); t = tr["t"]
    i0 = int(np.searchsorted(t, s - CONTEXT_S)); i1 = int(np.searchsorted(t, e + CONTEXT_S))
    x = np.where(tr["voiced"], tr["cents"], np.nan)[i0:i1].astype(np.float32); conf = tr["conf"][i0:i1].astype(np.float32)
    if tr["group"]:
        folds, q, prior = [tr["group"]], D["param"].loc[tr["group"]], TS.priors2(GROUPS.index(tr["group"]))
    else:
        folds, q, prior = GROUPS, D["median"], TS.priors2(-1)
    p = np.mean([D["R"].TV["predict"](D["nets"][g][0], D["R"].features_for(x, conf, D["nets"][g][1]), False).astype(float) for g in folds], axis=0)
    regions = TS.regions_b(dict(y=np.zeros(len(x)), logp2=np.log(p.T + 1e-8)), (float(q.D_region_s), float(q.switch), float(q.alpha)), prior)
    mins = {l: int(round(float(q[f"L_{SHORT[l]}"]) * FR)) for l in TS.TYPES}; out = []
    for a, z in regions:
        es, ee = float(t[i0]) + a / FR, float(t[i0]) + z / FR
        inside = max(0.0, min(ee, e) - max(es, s))
        if inside < .5 * (ee - es):
            continue
        curve, nv = event_curve(t, tr["cents"], tr["voiced"], es, ee)
        if nv < MIN_VOICED:
            continue
        label, _, votes = name3(lib_row(TS.centre(curve)), tr["exclude"])
        if z - a < mins[label]:
            continue
        out.append(dict(start=round(es, 3), end=round(ee, 3), type=SHORT[label], votes=f"{votes}/3", detector_score=round(float(p[1, a:z].mean()), 3)))
    return out


# ------------------------------------------------------------ request
def pts(c, step=1):
    return [round(float(v), 1) for v in c[::step]]


def classify(req):
    t0 = time.perf_counter(); tm = {}
    tr = load_track(req.get("known", ""), req.get("pitch", ""), req.get("audio", ""))
    s, e = round(float(req["start"]), 3), round(float(req["end"]), 3)
    check(tr, s, e)
    whole = name_span(tr, s, e); tm["name_ms"] = (time.perf_counter() - t0) * 1000
    r, best_span = whole, None
    if req.get("subsequence"):
        t1 = time.perf_counter(); r = subsequence(tr, s, e); best_span = dict(start=r["span"][0], end=r["span"][1]); tm["search_ms"] = (time.perf_counter() - t1) * 1000
    t1 = time.perf_counter(); det = detector_events(tr, s, e); tm["detector_ms"] = (time.perf_counter() - t1) * 1000
    thr, tuned = config(); dist = r["dist"]; unknown = dist > thr
    label = "unknown" if unknown else ("confident" if r["votes"] == 3 else "unsure")
    span = best_span or dict(start=s, end=e)
    nb = [dict(id=int(i), type=SHORT[int(TS.LIB_Y[i])], full_type=FULL[int(TS.LIB_Y[i])], track=str(LIB_TRACK[i]), group=str(TS.LIB_G[i]),
               start=LIB_START[i], end=LIB_END[i], distance=round(float(r["row"][i]), 2), contour=pts(TS.LIB_CURVE[i])) for i in r["near"]]
    tm["total_ms"] = (time.perf_counter() - t0) * 1000
    return dict(type=None if unknown else SHORT[r["label"]], closest_type=SHORT[r["label"]], full_type=FULL[r["label"]],
                votes=f"{r['votes']}/3", votes_n=r["votes"], distance=round(dist, 2), threshold=thr, threshold_tuned=tuned, label=label,
                selection=dict(start=s, end=e, voiced_frames=whole["nv"]), classified_span=dict(span, voiced_frames=r["nv"]),
                contour=pts(r["curve"]), contour_t0=float(tr["t"][r["first"]]), whole_selection=dict(type=SHORT[whole["label"]], votes=f"{whole['votes']}/3", distance=round(whole["dist"], 2)),
                best_match=best_span, nearest=nb, detector_events=det, multiple_gamakas=len(det) >= 2,
                excluded_group=tr["group"], excluded_tracks=GROUP_TRACKS.get(tr["group"], []), library_used=int((~np.isin(TS.LIB_G, list(tr["exclude"]))).sum()),
                known_track=tr["name"] or None, timing_ms={k: round(v, 1) for k, v in tm.items()})


def warm():
    """Compile the numba functions and load the detector so the first request is fast."""
    q = TS.LIB_CURVE[0]; TS.dtw(q, q, 5); subseq_dtw(np.r_[q, q], q, 5, 2 * len(q)); detector()


def handle(req):
    cmd = req.get("cmd")
    if cmd == "classify":
        return classify(req)
    if cmd == "view":
        return view(req)
    if cmd == "ping":
        return dict(pong=True, library_size=len(TS.LIB_CURVE))
    raise ValueError(f"unknown command {cmd!r}")
