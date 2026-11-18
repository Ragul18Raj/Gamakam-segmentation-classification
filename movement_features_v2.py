"""
movement_features.py -- build per-track movement tables and compare raga fingerprints.

This is the step after the feasibility checks. It stops asking "does segmentation work"
and starts producing the actual result: for each track, a table of every movement
(where it starts, where it ends, how it curves), then a fingerprint per track, then
pairwise distances between tracks.

The question it answers: does the fingerprint track the RAGA or the SINGER?

Needs anchor_feasibility_check_v3.py in the same folder (it reuses its segmentation).

Usage:
  python movement_features.py --manifest tracks.csv --out_dir features/

  tracks.csv columns: path,tonic_hz,raga,artist,swaras
    swaras = the raga's notes in cents, semicolon-separated, e.g. "0;200;400;500;700;900;1000"

Outputs:
  <out_dir>/<track>_movements.csv   one row per movement and per oscillation
  printed fingerprint distance matrices + the raga-vs-artist verdict
"""

import argparse
import csv
import os

import numpy as np

import anchor_feasibility_check_v3 as af

SEMITONES = np.arange(12) * 100.0          # fingerprint bins, raga-independent
TURN_BINS = [0, 1, 2, 3]                   # capped at 3+
EXTENT_EDGES = [0, 100, 250, 500, 1e9]     # cents
MIN_MOVE_S = 0.05        # v2: movements shorter than this are slivers; kept in the CSV,
                         # excluded from fingerprints
# v2: oscillation detection reads the PITCH CURVE, not the anchor chain. A veena
# kampita dwells at its peaks (so they become anchors); a vocal one doesn't (so they
# don't). Working from contour turning points covers both.
OSC_TURN_CENTS = 40      # smallest reversal that counts as a turning point
OSC_MAX_STEP = 500       # bigger than this is a phrase, not an oscillation
OSC_MAX_GAP_S = 0.35     # max time between consecutive turning points
OSC_MIN_TURNS = 3        # at least 3 turning points (two swings)


def turning_points(cents, voiced):
    """Positions and values of direction reversals bigger than OSC_TURN_CENTS."""
    pts = []
    anchor_i = None
    direction = 0
    for i, v in enumerate(cents):
        if not voiced[i] or np.isnan(v):
            anchor_i, direction = None, 0
            continue
        if anchor_i is None:
            anchor_i = i
            continue
        d = v - cents[anchor_i]
        if direction >= 0 and d <= -OSC_TURN_CENTS:
            if direction == 1:
                pts.append((anchor_i, float(cents[anchor_i])))
            direction, anchor_i = -1, i
        elif direction <= 0 and d >= OSC_TURN_CENTS:
            if direction == -1:
                pts.append((anchor_i, float(cents[anchor_i])))
            direction, anchor_i = 1, i
        elif (direction == 1 and v > cents[anchor_i]) or (direction == -1 and v < cents[anchor_i]):
            anchor_i = i
    return pts


def find_oscillations_contour(cents, voiced, hop_ms):
    """Runs of alternating turning points on the pitch curve = one oscillation."""
    sec = hop_ms / 1000.0
    pts = turning_points(cents, voiced)
    runs, i = [], 0
    while i < len(pts) - (OSC_MIN_TURNS - 1):
        run = [i]
        while run[-1] + 1 < len(pts):
            (i0, v0), (i1, v1) = pts[run[-1]], pts[run[-1] + 1]
            if abs(v1 - v0) > OSC_MAX_STEP or (i1 - i0) * sec > OSC_MAX_GAP_S:
                break
            run.append(run[-1] + 1)
        if len(run) >= OSC_MIN_TURNS:
            vals = [pts[k][1] for k in run]
            start, end = pts[run[0]][0], pts[run[-1]][0]
            dur = (end - start) * sec
            runs.append({"start": start, "end": end, "duration": dur,
                         "n_turns": len(run), "extent": float(max(vals) - min(vals)),
                         "center": float(np.mean(vals)),
                         "rate": (len(run) - 1) / dur if dur > 0 else 0.0})
            i = run[-1] + 1
        else:
            i += 1
    return runs


# ---------------------------------------------------------------- extraction
def extract(path, tonic_hz, swaras, hop_ms):
    times, freq, conf, hop_ms = af.load_pitch(path, hop_ms)
    cents, voiced = af.to_cents(freq, conf, tonic_hz, hop_ms)
    notes = af.find_steady_notes(cents, voiced, hop_ms)
    osc_runs, osc_members = af.find_oscillations(notes, hop_ms)       # veena-style chains
    contour_runs = find_oscillations_contour(cents, voiced, hop_ms)   # works for both
    moves = af.find_movements(voiced, notes, len(cents))
    sec = hop_ms / 1000.0

    ends = {e: k for k, (s, e, c) in enumerate(notes)}      # anchor ending at frame e
    starts = {s: k for k, (s, e, c) in enumerate(notes)}    # anchor starting at frame s
    osc_frames = set()
    for r in osc_runs:
        osc_frames.update(range(r["start"], r["end"]))
    contour_frames = set()
    for r in contour_runs:
        contour_frames.update(range(r["start"], r["end"] + 1))

    rows = []
    for s, e, bounded in moves:
        seg = cents[s:e]
        if len(seg) == 0:
            continue
        prev_k, next_k = ends.get(s), starts.get(e)
        # v2: fall back to the movement's own endpoints instead of discarding it
        if prev_k is not None:
            frm, frm_src = notes[prev_k][2], "anchor"
        else:
            frm, frm_src = float(seg[0]), "edge"
        if next_k is not None:
            to, to_src = notes[next_k][2], "anchor"
        else:
            to, to_src = float(seg[-1]), "edge"
        dur = (e - s) * sec
        n_turns = af.count_turning_points(seg)
        osc_like = len(contour_frames & set(range(s, e))) > (e - s) / 2
        rows.append({
            "type": "oscillation_in_movement" if osc_like else "movement",
            "start_s": round(s * sec, 3),
            "duration_s": round(dur, 3),
            "from_cents": round(frm), "to_cents": round(to),
            "from_swara": af.nearest_swara(frm, SEMITONES),
            "to_swara": af.nearest_swara(to, SEMITONES),
            "from_source": frm_src, "to_source": to_src,
            "extent_cents": round(float(np.nanmax(seg) - np.nanmin(seg))),
            "n_turns": n_turns,
            "direction": int(np.sign(seg[-1] - seg[0])),
            "in_oscillation": int(s in osc_frames),
            "bounded": int(bounded),
            "source": "",
            "used_in_fingerprint": int(dur >= MIN_MOVE_S),
        })

    # Emit BOTH detectors' oscillations. The contour detector finds smooth (vocal)
    # ones; v3's anchor chaining finds dwelling (veena) ones that break the contour
    # into pieces. Anchors inside either are excluded from held notes below, so if
    # only one kind were emitted the other's time would vanish from the output.
    # Drop a chain run only when a contour run CONTAINS it. Partial overlap used to
    # discard the chain run while its anchors stayed excluded from held notes, so the
    # uncovered part vanished. Overlapping rows are marked by "source" instead.
    def contained(r):
        return any(c["start"] <= r["start"] and c["end"] >= r["end"] - 1
                   for c in contour_runs)
    chain_keep = [r for r in osc_runs if not contained(r)]
    for r, src in ([(c, "contour") for c in contour_runs]
                   + [(c, "chain") for c in chain_keep]):
        rows.append({
            "type": "oscillation",
            "start_s": round(r["start"] * sec, 3),
            "duration_s": round(r["duration"], 3),
            "from_cents": round(r["center"]), "to_cents": round(r["center"]),
            "from_swara": af.nearest_swara(r["center"], SEMITONES),
            "to_swara": af.nearest_swara(r["center"], SEMITONES),
            "from_source": "anchor", "to_source": "anchor", "source": src,
            "extent_cents": round(r["extent"]),
            "n_turns": r["n_turns"],
            "direction": 0,
            "in_oscillation": 1,
            "bounded": "",
            "used_in_fingerprint": 1,
        })

    # A held note that clips the edge of an oscillation used to be discarded whole,
    # which cost some clips most of their pitch data over a few frames of overlap.
    # Trim instead: keep the parts outside the oscillation, recentred on what's left.
    held = []
    for k, (s_, e_, c_) in enumerate(notes):
        if k in osc_members:            # a turning point OF an oscillation, not a note
            continue
        free = [i for i in range(s_, e_) if i not in contour_frames]
        if len(free) == (e_ - s_):
            held.append((s_, e_, c_))
            continue
        run = []
        for i in free + [None]:         # split what's left into contiguous pieces
            if run and (i is None or i != run[-1] + 1):
                if (run[-1] + 1 - run[0]) * sec >= MIN_MOVE_S:
                    held.append((run[0], run[-1] + 1,
                                 float(np.nanmean(cents[run[0]:run[-1] + 1]))))
                run = []
            if i is not None:
                run.append(i)
    note_time = np.zeros(12)
    held_rows = []
    for s, e, c in held:
        note_time[int(round((c % 1200) / 100)) % 12] += (e - s) * sec
        held_rows.append({"start_s": s * sec, "duration_s": (e - s) * sec, "cents": float(c)})
    rows.sort(key=lambda r: r["start_s"])
    return rows, note_time, held_rows


# ---------------------------------------------------------------- fingerprints
def norm(v):
    t = v.sum()
    return v / t if t > 0 else v


def transition_vec(rows):
    """v2: every movement contributes, using its own endpoints when no anchor bounds it."""
    m = np.zeros((12, 12))
    for r in rows:
        if r["type"] == "oscillation" or not r["used_in_fingerprint"]:
            continue
        m[int(r["from_swara"]) // 100, int(r["to_swara"]) // 100] += 1
    return norm(m.ravel())


def shape_vec(rows):
    h = np.zeros((len(TURN_BINS), len(EXTENT_EDGES) - 1))
    for r in rows:
        if r["type"] == "oscillation" or not r["used_in_fingerprint"]:
            continue
        t = min(int(r["n_turns"]), TURN_BINS[-1])
        e = int(np.searchsorted(EXTENT_EDGES, r["extent_cents"], side="right") - 1)
        h[t, min(e, h.shape[1] - 1)] += 1
    return norm(h.ravel())


def jsd(p, q):
    """Jensen-Shannon distance: 0 = identical, 1 = nothing in common."""
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.where(b[mask] > 0, b[mask], 1e-12))))
    return float(np.sqrt(max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m))))


def matrix(names, vecs, title):
    print(f"\n{title}  (0 = identical, higher = more different)")
    w = max(len(n) for n in names) + 2
    print(" " * w + "".join(f"{n[:10]:>12}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:<{w}}" + "".join(f"{jsd(vecs[i], vecs[j]):>12.3f}" for j in range(len(names))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", default="features")
    ap.add_argument("--hop_ms", type=float, default=10.0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    names, notes_v, trans_v, shape_v, meta = [], [], [], [], []
    with open(a.manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            swaras = [float(x) for x in row["swaras"].split(";")]
            rows, note_time, _ = extract(row["path"], float(row["tonic_hz"]), swaras, a.hop_ms)
            stem = os.path.splitext(os.path.basename(row["path"]))[0]
            out = os.path.join(a.out_dir, f"{stem}_movements.csv")
            with open(out, "w", newline="", encoding="utf-8") as g:
                w = csv.DictWriter(g, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            label = f"{row['raga'][:8]}/{row['artist'][:8]}"
            names.append(label)
            meta.append((row["raga"], row["artist"]))
            notes_v.append(norm(note_time))
            trans_v.append(transition_vec(rows))
            shape_v.append(shape_vec(rows))
            n_mv = sum(r["type"] == "movement" for r in rows)
            n_oscm = sum(r["type"] == "oscillation_in_movement" for r in rows)
            n_osc = sum(r["type"] == "oscillation" for r in rows)
            n_used = sum(r["used_in_fingerprint"] and r["type"] != "oscillation" for r in rows)
            n_edge = sum(r["from_source"] == "edge" or r["to_source"] == "edge" for r in rows)
            print(f"{label:<22} {n_mv + n_oscm:>5} movements "
                  f"({n_oscm} inside oscillations, {n_used} used in fingerprints, "
                  f"{n_edge} using own endpoints), {n_osc:>3} oscillations -> {out}")

    matrix(names, notes_v, "WHERE THE NOTES ARE (should track raga)")
    matrix(names, trans_v, "NOTE-TO-NOTE TRANSITIONS (should track raga)")
    matrix(names, shape_v, "MOVEMENT SHAPES (may track singer)")

    print("\nRAGA vs SINGER")
    for label, vecs in (("notes", notes_v), ("transitions", trans_v), ("shapes", shape_v)):
        same_raga = [jsd(vecs[i], vecs[j]) for i in range(len(meta)) for j in range(i + 1, len(meta))
                     if meta[i][0] == meta[j][0] and meta[i][1] != meta[j][1]]
        same_art = [jsd(vecs[i], vecs[j]) for i in range(len(meta)) for j in range(i + 1, len(meta))
                    if meta[i][1] == meta[j][1] and meta[i][0] != meta[j][0]]
        if not same_raga or not same_art:
            print(f"  {label:<12} need at least one same-raga and one same-singer pair")
            continue
        sr, sa = float(np.mean(same_raga)), float(np.mean(same_art))
        verdict = "tracks RAGA" if sr < sa else "tracks SINGER"
        print(f"  {label:<12} same raga/diff singer {sr:.3f} | same singer/diff raga {sa:.3f}"
              f"  -> {verdict}")
    print("\nThese are averages over very few pairs. Treat as a direction, not a result.\n")


if __name__ == "__main__":
    main()