"""
anchor_feasibility_check.py  --  GO / NO-GO test for steps 1-3 of the raga-first plan.

Question it answers: on YOUR recordings, can we reliably find steady notes
("anchors") and do they land on the raga's swaras? If not, the per-swara
fingerprint plan is dead and we switch to the anchor/turning-point version.

Standalone file -- does not touch your existing pipeline.

Inputs (either):
  - CREPE activation matrix saved as .npy, shape [T, 360]
  - CREPE CSV (time,frequency,confidence), e.g. from the crepe CLI

Usage:
  Single file:
    python anchor_feasibility_check.py --input song.npy --tonic 146.8
  Batch (CSV with columns: path,tonic_hz):
    python anchor_feasibility_check.py --manifest recordings.csv
  Optional raga notes (cents above Sa, default = all 12 semitones):
    --swaras 0,200,400,500,700,900,1000     # Harikambhoji notes

NOTE: the steady-note logic is MY implementation of the hysteresis idea
(settle within 20 cents, leave beyond 50 cents). Check it against the
Rao et al. paper before treating results as that method's results.
"""

import argparse
import csv
import os

import numpy as np
from scipy.signal import medfilt

# ---------------- parameters (all judgment calls -- tune them) ----------------
SETTLE_CENTS = 20       # all frames within +/- this of their mean -> note starts
LEAVE_CENTS = 50        # note ends when pitch moves further than this from its center
MIN_NOTE_MS = 80        # a steady note must last at least this long (guess)
MAX_DRIFT_CENTS = 30    # v2: an anchor's own start-to-end drift. Stops slow glides being
                        # absorbed into chains of fake "steady notes".
CONF_THRESH = 0.5       # CREPE confidence below this = unvoiced
MEDIAN_MS = 50          # smoothing window to kill CREPE jitter
LONG_MOVE_S = 1.0       # movements longer than this likely span several svaras
GESTURE_MIN_S = 0.15    # v2: movements shorter than this are micro-slivers, not gestures
TRIVIAL_RANGE_CENTS = 200   # v2: a clip whose whole pitch range is under this tests nothing
# v3: oscillation grouping. A kampita shows up as anchors alternating between two
# pitch levels; v2 called each peak and trough a separate "steady note" and shredded
# the gesture. These group such runs back into one oscillation unit.
OSC_MIN_STEP_CENTS = 40    # smallest swing that counts as an oscillation step
OSC_MAX_STEP_CENTS = 500   # larger than this is a phrase, not an oscillation
OSC_MAX_GAP_S = 0.35       # max time between consecutive turning points
OSC_MIN_TURNS = 3          # need at least 3 turning points (two swings)
# v3: octave-error flag is now relative to the clip's own range, so concert tracks
# spanning three octaves don't flag every upper-octave note.
FAR_BEYOND_BAND_CENTS = 600   # this far outside the clip's 5th-95th percentile band
TURN_MIN_CENTS = 25     # min excursion for a turning point to count
SNAP_TOL_CENTS = 40     # anchor counts as "on a swara" within this distance

# ---------------- go / no-go thresholds (judgment calls) ----------------
GO_ANCHOR_ON_SWARA = 0.75   # >= 75% of anchors should sit on a raga note
GO_LONG_MOVE_SHARE = 0.30   # <= 30% of voiced time in long unanchored movements


# ---------------------------- loading ----------------------------
def crepe_activation_to_pitch(act):
    """Same local weighted average CREPE uses internally."""
    cents_map = np.linspace(0, 7180, 360) + 1997.3794084376191
    T = act.shape[0]
    cents = np.zeros(T)
    conf = act.max(axis=1)
    for t in range(T):
        c = int(np.argmax(act[t]))
        lo, hi = max(0, c - 4), min(360, c + 5)
        w = act[t, lo:hi]
        cents[t] = np.sum(w * cents_map[lo:hi]) / max(np.sum(w), 1e-9)
    freq = 10.0 * 2 ** (cents / 1200.0)
    return freq, conf


def load_pitch(path, hop_ms):
    if path.endswith(".npy"):
        act = np.load(path)
        if act.ndim != 2 or act.shape[1] != 360:
            raise ValueError(f"{path}: expected [T,360] activation, got {act.shape}")
        freq, conf = crepe_activation_to_pitch(act)
        times = np.arange(len(freq)) * hop_ms / 1000.0
    else:
        data = np.genfromtxt(path, delimiter=",", names=True)
        times, freq, conf = data["time"], data["frequency"], data["confidence"]
        if len(times) > 1:
            hop_ms = float(np.median(np.diff(times)) * 1000)
    return times, freq, conf, hop_ms


# ---------------------------- core ----------------------------
def to_cents(freq, conf, tonic_hz, hop_ms):
    voiced = (conf >= CONF_THRESH) & (freq > 0)
    cents = np.full(len(freq), np.nan)
    cents[voiced] = 1200 * np.log2(freq[voiced] / tonic_hz)
    k = max(3, int(round(MEDIAN_MS / hop_ms)) | 1)  # odd kernel
    filled = np.where(voiced, cents, 0.0)
    smooth = medfilt(filled, k)
    smooth[~voiced] = np.nan
    return smooth, voiced


def find_steady_notes(cents, voiced, hop_ms):
    """Returns list of (start, end, center_cents). end is exclusive."""
    n_min = max(2, int(round(MIN_NOTE_MS / hop_ms)))
    T = len(cents)
    notes, i = [], 0
    while i + n_min <= T:
        win = cents[i:i + n_min]
        if voiced[i:i + n_min].all():
            m = win.mean()
            if np.max(np.abs(win - m)) <= SETTLE_CENTS:
                j = i + n_min
                while j < T and voiced[j] and abs(cents[j] - m) <= LEAVE_CENTS:
                    j += 1
                seg = cents[i:j]
                drift = abs(np.mean(seg[:3]) - np.mean(seg[-3:])) if len(seg) >= 6 else 0.0
                if drift <= MAX_DRIFT_CENTS:      # v2: reject moving plateaus
                    notes.append((i, j, float(np.mean(seg))))
                    i = j
                    continue
                i += 1
                continue
        i += 1
    return notes


def find_movements(voiced, notes, T):
    """Voiced, non-note runs. Returns list of (start, end, bounded_both_sides)."""
    in_note = np.zeros(T, bool)
    for s, e, _ in notes:
        in_note[s:e] = True
    moves, t = [], 0
    while t < T:
        if voiced[t] and not in_note[t]:
            s = t
            while t < T and voiced[t] and not in_note[t]:
                t += 1
            left = s > 0 and in_note[s - 1]
            right = t < T and in_note[t]
            moves.append((s, t, left and right))
        else:
            t += 1
    return moves


def count_turning_points(seg):
    """Direction changes whose excursion exceeds TURN_MIN_CENTS."""
    if len(seg) < 3:
        return 0
    turns, anchor, direction = 0, seg[0], 0
    for v in seg[1:]:
        d = v - anchor
        if direction >= 0 and d <= -TURN_MIN_CENTS:
            turns += direction == 1
            direction, anchor = -1, v
        elif direction <= 0 and d >= TURN_MIN_CENTS:
            turns += direction == -1
            direction, anchor = 1, v
        elif (direction == 1 and v > anchor) or (direction == -1 and v < anchor):
            anchor = v
    return turns


def find_oscillations(notes, hop_ms):
    """v3: group runs of anchors that alternate up/down into single oscillation units.

    Returns (runs, member_idx) where each run is a dict describing one oscillation
    and member_idx is the set of anchor indices swallowed by some run.
    """
    sec = hop_ms / 1000.0
    runs, member = [], set()
    i = 0
    while i < len(notes) - 2:
        run = [i]
        direction = 0
        j = i
        while j + 1 < len(notes):
            (s0, e0, c0), (s1, e1, c1) = notes[j], notes[j + 1]
            step = c1 - c0
            gap = (s1 - e0) * sec
            if not (OSC_MIN_STEP_CENTS <= abs(step) <= OSC_MAX_STEP_CENTS) or gap > OSC_MAX_GAP_S:
                break
            d = 1 if step > 0 else -1
            if direction != 0 and d == direction:   # must alternate
                break
            direction = d
            run.append(j + 1)
            j += 1
        if len(run) >= OSC_MIN_TURNS:
            cs = [notes[k][2] for k in run]
            start, end = notes[run[0]][0], notes[run[-1]][1]
            dur = (end - start) * sec
            runs.append({
                "start": start, "end": end, "duration": dur,
                "n_turns": len(run),
                "extent": float(max(cs) - min(cs)),
                "center": float(np.mean(cs)),
                "rate": (len(run) - 1) / dur if dur > 0 else 0.0,
            })
            member.update(run)
            i = run[-1] + 1
        else:
            i += 1
    return runs, member


def swara_distance(center, swaras):
    folded = center % 1200
    d = np.abs(folded - np.array(swaras))
    return float(np.min(np.minimum(d, 1200 - d)))


def analyse(path, tonic_hz, swaras, hop_ms, plot_dir, plot_seconds):
    times, freq, conf, hop_ms = load_pitch(path, hop_ms)
    cents, voiced = to_cents(freq, conf, tonic_hz, hop_ms)
    T = len(cents)
    notes = find_steady_notes(cents, voiced, hop_ms)
    osc_runs, osc_members = find_oscillations(notes, hop_ms)
    moves = find_movements(voiced, notes, T)
    sec = hop_ms / 1000.0

    voiced_s = voiced.sum() * sec
    note_s = sum(e - s for s, e, _ in notes) * sec
    move_durs = np.array([(e - s) * sec for s, e, _ in moves]) if moves else np.array([])
    long_s = move_durs[move_durs > LONG_MOVE_S].sum() if len(move_durs) else 0.0
    dists = np.array([swara_distance(c, swaras) for _, _, c in notes]) if notes else np.array([])
    vcents = cents[voiced]
    pitch_range = float(np.nanmax(vcents) - np.nanmin(vcents)) if len(vcents) else 0.0
    median_pitch = float(np.nanmedian(vcents)) if len(vcents) else 0.0
    centers = [c for _, _, c in notes]
    lo_b, hi_b = (np.percentile(vcents, [5, 95]) if len(vcents) else (0.0, 0.0))
    far = [c for c in centers
           if c < lo_b - FAR_BEYOND_BAND_CENTS or c > hi_b + FAR_BEYOND_BAND_CENTS]
    held = [c for k, c in enumerate(centers) if k not in osc_members]
    held_d = np.array([swara_distance(c, swaras) for c in held]) if held else np.array([])
    osc_s = sum(r["duration"] for r in osc_runs)
    gesture_s = move_durs[move_durs >= GESTURE_MIN_S].sum() if len(move_durs) else 0.0
    turns = [count_turning_points(cents[s:e]) for s, e, _ in moves]

    r = {
        "file": os.path.basename(path),
        "voiced_s": voiced_s,
        "steady_share": note_s / voiced_s if voiced_s else 0.0,
        "n_notes": len(notes),
        "anchor_on_swara": float(np.mean(dists <= SNAP_TOL_CENTS)) if len(dists) else 0.0,
        "anchor_median_dist": float(np.median(dists)) if len(dists) else float("nan"),
        "n_moves": len(moves),
        "move_median_s": float(np.median(move_durs)) if len(move_durs) else 0.0,
        "long_move_share": long_s / voiced_s if voiced_s else 0.0,
        "bounded_share": float(np.mean([b for *_, b in moves])) if moves else 0.0,
        "turns_median": float(np.median(turns)) if turns else 0.0,
        "anchor_dists": dists,
        "pitch_range": pitch_range,
        "trivial": pitch_range < TRIVIAL_RANGE_CENTS,
        "n_far": len(far),
        "far_centers": [round(c) for c in far],
        "gesture_share": gesture_s / voiced_s if voiced_s else 0.0,
        "n_osc": len(osc_runs),
        "osc_share": osc_s / voiced_s if voiced_s else 0.0,
        "osc_dur": float(np.median([r["duration"] for r in osc_runs])) if osc_runs else 0.0,
        "osc_extent": float(np.median([r["extent"] for r in osc_runs])) if osc_runs else 0.0,
        "osc_rate": float(np.median([r["rate"] for r in osc_runs])) if osc_runs else 0.0,
        "osc_turns": float(np.median([r["n_turns"] for r in osc_runs])) if osc_runs else 0.0,
        "held_dists": held_d,
        "n_held": len(held),
        "anchor_centers": [c for _, _, c in notes],
    }
    if plot_dir:
        make_plot(r, times, cents, notes, swaras, plot_dir, plot_seconds)
    return r


def make_plot(r, times, cents, notes, swaras, plot_dir, plot_seconds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    m = times <= times[0] + plot_seconds
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(times[m], cents[m], color="0.55", lw=0.9, label="pitch (movement)")
    first = True
    for s, e, c in notes:
        if times[s] > times[0] + plot_seconds:
            break
        ax.plot(times[s:e], cents[s:e], color="tab:blue", lw=2.2,
                label="steady note (anchor)" if first else None)
        first = False
    lo, hi = np.nanmin(cents[m]) if m.any() else 0, np.nanmax(cents[m]) if m.any() else 1200
    for octave in range(int(np.floor(lo / 1200)) - 1, int(np.ceil(hi / 1200)) + 1):
        for sw in swaras:
            y = sw + 1200 * octave
            if lo - 100 <= y <= hi + 100:
                ax.axhline(y, color="tab:red", lw=0.5, alpha=0.4)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cents above Sa")
    ax.set_title(f"{r['file']}  |  red lines = raga notes, blue = anchors found")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = os.path.join(plot_dir, os.path.splitext(r["file"])[0] + "_anchors.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------- report ----------------------------
def verdict(ok):
    return "GO     " if ok else "NO-GO  "


def nearest_swara(center, swaras):
    f = center % 1200
    return int(min(swaras, key=lambda sw: min(abs(f - sw), 1200 - abs(f - sw))))


def report(results, swaras):
    print("\n" + "=" * 78)
    for r in results:
        print(f"\n{r['file']}  ({r['voiced_s']:.1f}s voiced)")
        print(f"  steady-note share of voiced time : {r['steady_share']:.0%}")
        print(f"  anchors found                    : {r['n_notes']}")
        print(f"  anchors on a raga note (<= {SNAP_TOL_CENTS}c) : {r['anchor_on_swara']:.0%}"
              f"   (median distance {r['anchor_median_dist']:.0f}c)")
        counts = {}
        for c in r["anchor_centers"]:
            k = nearest_swara(c, swaras)
            counts[k] = counts.get(k, 0) + 1
        print("  anchors per swara (cents: count)  : " +
              ", ".join(f"{k}:{counts[k]}" for k in sorted(counts)))
        print(f"  movements                        : {r['n_moves']}  "
              f"(median {r['move_median_s']:.2f}s, median turning points {r['turns_median']:.0f})")
        print(f"  movements with anchors both sides: {r['bounded_share']:.0%}")
        print(f"  voiced time in gestures >= {GESTURE_MIN_S}s  : {r['gesture_share']:.0%}"
              "   <- real gestures, not micro-slivers")
        if r["n_osc"]:
            print(f"  OSCILLATIONS (kampita-like)      : {r['n_osc']}  "
                  f"median {r['osc_dur']:.2f}s, {r['osc_turns']:.0f} turning points, "
                  f"{r['osc_extent']:.0f}c wide, {r['osc_rate']:.1f} swings/s")
            print(f"  voiced time in oscillations      : {r['osc_share']:.0%}")
        else:
            print("  OSCILLATIONS (kampita-like)      : none found")
        on_held = (float(np.mean(r["held_dists"] <= SNAP_TOL_CENTS))
                   if len(r["held_dists"]) else float("nan"))
        print(f"  held notes only, on a raga note  : "
              + (f"{on_held:.0%} of {r['n_held']}" if r["n_held"] else "n/a")
              + "   <- excludes oscillation turning points")
        print(f"  pitch range of clip              : {r['pitch_range']:.0f}c"
              + ("   *** TRIVIAL: clip never leaves one note, its score proves nothing ***"
                 if r["trivial"] else ""))
        if r["n_far"]:
            print(f"  anchors outside clip's pitch band: {r['n_far']} "
                  f"at {r['far_centers'][:12]}{' ...' if r['n_far'] > 12 else ''}"
                  "   <- outside 5-95 percentile band + 600c; check by ear")
        print(f"  voiced time in movements > {LONG_MOVE_S:.0f}s   : {r['long_move_share']:.0%}")

    usable = [r for r in results if not r["trivial"]]
    n_trivial = len(results) - len(usable)
    all_d = np.concatenate([r["anchor_dists"] for r in usable]) if usable else np.array([])
    on = float(np.mean(all_d <= SNAP_TOL_CENTS)) if len(all_d) else 0.0
    tot_v = sum(r["voiced_s"] for r in usable)
    long_share = sum(r["long_move_share"] * r["voiced_s"] for r in usable) / tot_v if tot_v else 0.0
    gest = sum(r["gesture_share"] * r["voiced_s"] for r in usable) / tot_v if tot_v else 0.0
    n_far = sum(r["n_far"] for r in usable)

    print("\n" + "=" * 78)
    print("OVERALL  (pooled over non-trivial clips only)")
    if n_trivial:
        print(f"  EXCLUDED {n_trivial} trivial clip(s): whole pitch range under "
              f"{TRIVIAL_RANGE_CENTS}c, so their scores mean nothing")
    if n_far:
        print(f"  NOTE: {n_far} anchor(s) sit outside their clip's own pitch band. "
              "Octave folding makes these look on-swara; check by ear.")
    n_osc = sum(r["n_osc"] for r in usable)
    osc_share = sum(r["osc_share"] * r["voiced_s"] for r in usable) / tot_v if tot_v else 0.0
    held_all = np.concatenate([r["held_dists"] for r in usable if len(r["held_dists"])]) \
        if any(len(r["held_dists"]) for r in usable) else np.array([])
    on_held = float(np.mean(held_all <= SNAP_TOL_CENTS)) if len(held_all) else float("nan")
    print(f"  INFO    oscillations found: {n_osc}, covering {osc_share:.0%} of voiced time")
    print("         -> 0 on kampita material means oscillation grouping is still wrong")
    if len(held_all):
        print(f"  INFO    held notes only, on a raga note: {on_held:.0%} of {len(held_all)}"
              "  (oscillation turning points excluded)")
    print(f"  {verdict(on >= GO_ANCHOR_ON_SWARA)} anchors land on raga notes: {on:.0%} "
          f"(need >= {GO_ANCHOR_ON_SWARA:.0%})")
    print("         -> if NO-GO: check tonic and CREPE tracking first")
    print("         -> WARNING: a GO here does NOT prove the tonic is right. A tonic off by a")
    print("            fifth (Sa<->Pa) can still land most anchors on raga notes. Check tonics")
    print("            by ear or against annotations, and look at 'anchors per swara' above.")
    print(f"  {verdict(long_share <= GO_LONG_MOVE_SHARE)} long unanchored movements: {long_share:.0%} "
          f"of voiced time (need <= {GO_LONG_MOVE_SHARE:.0%})")
    print("         -> if NO-GO: per-swara labels are off; switch to the anchor + turning-point")
    print("            version (the plan survives, the per-swara claim does not)")
    print(f"  INFO    real gestures (>= {GESTURE_MIN_S}s): {gest:.0%} of voiced time")
    print("         -> near 0% means the segmenter is slicing micro-slivers, not finding")
    print("            gamaka gestures, and the long-movement check above passed for free")
    print("=" * 78 + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input")
    ap.add_argument("--tonic", type=float, help="tonic (Sa) in Hz")
    ap.add_argument("--manifest", help="CSV with columns path,tonic_hz")
    ap.add_argument("--swaras", default="0,100,200,300,400,500,600,700,800,900,1000,1100",
                    help="raga notes in cents above Sa, comma-separated")
    ap.add_argument("--hop_ms", type=float, default=10.0, help="CREPE step size for .npy input")
    ap.add_argument("--plot_dir", default="anchor_plots")
    ap.add_argument("--plot_seconds", type=float, default=30.0)
    a = ap.parse_args()

    swaras = [float(x) for x in a.swaras.split(",")]
    jobs = []
    if a.manifest:
        with open(a.manifest, newline="") as f:
            for row in csv.DictReader(f):
                jobs.append((row["path"], float(row["tonic_hz"])))
    elif a.input and a.tonic:
        jobs.append((a.input, a.tonic))
    else:
        ap.error("give --input and --tonic, or --manifest")

    results = [analyse(p, t, swaras, a.hop_ms, a.plot_dir, a.plot_seconds) for p, t in jobs]
    report(results, swaras)


if __name__ == "__main__":
    main()
