"""Part 1: verify CREPE outputs, classify filenames, group clips into sessions, and ESTIMATE tonics.

Tonic: one fixed value for the whole dataset, Sa = E4 = 329.3 Hz. This is an INFERENCE from
audio statistics, NOT a confirmed ground-truth tonic (see tonic_estimates_NOTE.md).

Per-session pooled histogram modes are still computed as supporting evidence: pooling the
high-confidence, sustained CREPE frames of every clip in a session and taking the mode of the
pooled pitch histogram. That mode is the most common sustained pitch in the session, which is
often a demonstrated swara rather than Sa, so it is NOT used as the tonic.

Sessions: every WAV is a Logic Pro bounce whose bext TimeReference gives the clip's position
on its project timeline. Clips sharing a filename label whose regions tile one timeline
without overlapping are treated as cut from one continuous take (one session). A label group
whose regions overlap is split by its "_OG_1" export series.

Outputs:
  tonic_estimates.csv    path,tonic_hz,method        (fixed dataset tonic for every file)
  session_tonics.csv     one row per session, with the top pooled histogram peaks (evidence only)
  dataset_summary.csv    full per-file verification table
"""
import csv
import re
import struct
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np
from crepe.core import to_local_average_cents

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Gamaka_veena_dataset"
CREPE_DIR = ROOT / "crepe_output"

FRAMES_PER_SEC = 100          # step_size = 10 ms
LENGTH_TOLERANCE = 2          # frames; CREPE (center=True) yields duration*100 + 1
CONF_THRESHOLDS = (0.8, 0.5)  # try strict first, relax if too few frames survive
MIN_FRAMES = 10
STABLE_CENTS = 20             # max |pitch jump| to both neighbours for a "sustained" frame
BIN_CENTS = 10
PEAK_WINDOW_CENTS = 30
# Sa = E for the whole dataset; octave choice does not affect anchor_feasibility_check.py,
# which folds pitches mod 1200 cents. E4 is the most populated E octave in the data.
DATASET_TONIC_HZ = 329.3
TONIC_METHOD = "fixed_dataset_tonic_E4_corroborated"
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SWARAS = ["S", "R1", "R2/G1", "G2/R3", "G3", "M1", "M2", "P", "D1", "D2/N1", "N2/D3", "N3"]

# Raga spellings as they appear in filenames -> standard name
RAGAS = {
    "shankrabarnam": "Shankarabharanam",
    "shankravarnam": "Shankarabharanam",
    "begada": "Begada",
    "kambhoji": "Kambhoji",
    "sudhsaveri": "Shuddha Saveri",
    "hindola": "Hindolam",
    "amritvarshini": "Amritavarshini",
    "mayamolagolav": "Mayamalavagowla",
    "kanahalpriya": "Kharaharapriya",
    "todi": "Todi",
}
# Tokens that look like raga labels but could not be matched confidently
UNCLEAR_RAGA_TOKENS = ("kalapriya", "bhairiga", "melgana")
GAMAKAS = ("ahatam", "andolita", "jaru", "kampita", "kandippu", "namitam", "nokku", "odukkul",
           "orikai", "pratyagata", "pratayagata", "ravai", "sphuritam", "sphutitam", "tribhinna", "vali")


def classify(name):
    low = name.lower()
    gamaka = next((g for g in GAMAKAS if low.startswith(g)), "")
    for token, raga in RAGAS.items():
        if token in low:
            return "raga", raga, gamaka
    for token in UNCLEAR_RAGA_TOKENS:
        if token in low:
            return "unclear", f"'{token}'?", gamaka
    return ("gamaka-type" if gamaka else "unclear"), "", gamaka


def group_label(filename):
    """Filename with the clip index / 'continuous' marker / export suffix removed."""
    s = re.sub(r"\s*Guitar_samples_3D_OG(_1)?\.wav$", "", filename)
    s = re.sub(r"_\d+_melgana$", "_melgana", s)   # Jaru_desc_N_melgana
    s = re.sub(r"[\s_]*continuous.*$", "", s)
    s = re.sub(r"(#\d+)?(\.\d+)?$", "", s)         # #take.index
    s = re.sub(r"[\s_]*!?\d*$", "", s)             # _N, _!N, _!
    return s.strip()


def wav_info(path):
    """(duration_s, bext timeline position in seconds or None)."""
    with wave.open(str(path)) as w:
        sr, dur = w.getframerate(), w.getnframes() / w.getframerate()
    b = path.read_bytes()
    i, ref = 12, None
    while i + 8 <= len(b):
        cid, n = b[i:i + 4], struct.unpack("<I", b[i + 4:i + 8])[0]
        if cid == b"bext" and n >= 346:
            ref = struct.unpack("<Q", b[i + 8 + 338:i + 8 + 346])[0] / sr
        i += 8 + n + (n & 1)
    return dur, ref


def overlaps(clips):
    clips = sorted(clips, key=lambda r: r["timeline_s"])
    return any(a["timeline_s"] + a["duration_s"] > b["timeline_s"] + 0.01 for a, b in zip(clips, clips[1:]))


def assign_sessions(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[group_label(r["filename"])].append(r)
    for label, clips in groups.items():
        if any(c["timeline_s"] is None for c in clips) or not overlaps(clips):
            for c in clips:
                c["session"] = label
            continue
        for c in clips:
            series = "take B" if c["filename"].endswith("_OG_1.wav") else "take A"
            c["session"] = f"{label} [{series}]"
        for series in ("take A", "take B"):
            sub = [c for c in clips if c["session"].endswith(f"[{series}]")]
            if overlaps(sub):
                raise RuntimeError(f"session {label} [{series}] still overlaps on the timeline")


def stable_frames(activation):
    """Cents (rel. 10 Hz) of confident, sustained frames and the confidence threshold used."""
    cents = to_local_average_cents(activation)
    conf = activation.max(axis=1)
    stable = np.zeros(len(cents), dtype=bool)
    if len(cents) >= 3:
        d = np.abs(np.diff(cents))
        stable[1:-1] = (d[:-1] < STABLE_CENTS) & (d[1:] < STABLE_CENTS)
    for thr in CONF_THRESHOLDS:
        sel = cents[(conf >= thr) & stable]
        if len(sel) >= MIN_FRAMES:
            return sel, thr
    return sel, None


def histogram_peaks(sel, n_peaks=3):
    """Top peaks of the smoothed pitch histogram as [(cents, share_of_frames), ...]."""
    # pad by the kernel half-width so the smoothed histogram keeps the same length as hist
    edges = np.arange(sel.min() - 5 * BIN_CENTS, sel.max() + 6 * BIN_CENTS, BIN_CENTS)
    hist, _ = np.histogram(sel, bins=edges)
    kernel = np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2)
    smooth = np.convolve(hist, kernel / kernel.sum(), mode="same")
    peaks = []
    for idx in np.argsort(smooth)[::-1]:
        center = edges[idx] + BIN_CENTS / 2
        if smooth[idx] <= 0 or any(abs(center - c) < 2 * PEAK_WINDOW_CENTS for c, _ in peaks):
            continue
        near = sel[np.abs(sel - center) <= PEAK_WINDOW_CENTS]
        if len(near) == 0 or (1 <= idx < len(smooth) - 1 and smooth[idx] < max(smooth[idx - 1], smooth[idx + 1])):
            continue
        peaks.append((float(np.median(near)), len(near) / len(sel)))
        if len(peaks) == n_peaks:
            break
    return peaks


def cents_to_hz(c):
    return 10 * 2 ** (c / 1200)


def note_name(hz):
    midi = 69 + 12 * np.log2(hz / 440)
    n = int(round(midi))
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}{(midi - n) * 100:+.0f}c"


def main():
    rows = []
    for wav in sorted(DATA_DIR.glob("*.wav"), key=lambda p: p.name.lower()):
        dur, ref = wav_info(wav)
        cat, raga, gamaka = classify(wav.stem)
        row = {"filename": wav.name, "duration_s": round(dur, 2), "timeline_s": ref, "has_crepe": "N",
               "frames": "", "expected_frames": round(dur * FRAMES_PER_SEC), "length_ok": "",
               "category": cat, "raga": raga, "gamaka_type": gamaka, "_sel": None}
        npy = CREPE_DIR / f"{wav.stem}.npy"
        if npy.exists():
            act = np.load(npy)
            row["has_crepe"] = "Y"
            row["frames"] = act.shape[0]
            row["length_ok"] = "Y" if (act.ndim == 2 and act.shape[1] == 360 and act.shape[0] > 0 and
                                       abs(act.shape[0] - dur * FRAMES_PER_SEC) <= LENGTH_TOLERANCE) else "N"
            if row["length_ok"] == "Y":
                row["_sel"], row["_thr"] = stable_frames(act)
        rows.append(row)
    assign_sessions(rows)

    sessions = defaultdict(list)
    for r in rows:
        sessions[r["session"]].append(r)
    session_rows = []
    for name, clips in sorted(sessions.items(), key=lambda x: x[0].lower()):
        pooled = [c["_sel"] for c in clips if c["_sel"] is not None and c["_thr"] is not None]
        srow = {"session": name, "n_clips": len(clips), "n_clips_used": len(pooled), "n_frames": 0,
                "pooled_mode_hz": "", "pooled_mode_note": "", "mode_vs_Sa_cents": "", "mode_swara": "",
                "peak1_share": "", "peak2": "", "peak3": ""}
        if pooled:
            sel = np.concatenate(pooled)
            peaks = histogram_peaks(sel)
            mode_c = peaks[0][0]
            vs_sa = (1200 * np.log2(cents_to_hz(mode_c) / DATASET_TONIC_HZ)) % 1200
            srow.update(n_frames=len(sel), pooled_mode_hz=round(cents_to_hz(mode_c), 2),
                        pooled_mode_note=note_name(cents_to_hz(mode_c)), mode_vs_Sa_cents=round(vs_sa),
                        mode_swara=SWARAS[int(round(vs_sa / 100)) % 12], peak1_share=round(peaks[0][1], 2))
            for k, (c, share) in enumerate(peaks[1:], 2):
                srow[f"peak{k}"] = f"{cents_to_hz(c):.1f}Hz ({c - mode_c:+.0f}c, {share:.2f})"
            for c in clips:
                c["session_pooled_mode_hz"] = srow["pooled_mode_hz"]
                if c["_sel"] is not None and c["_thr"] is not None:
                    own = histogram_peaks(c["_sel"], 1)[0][0]
                    c["clip_mode_hz"] = round(cents_to_hz(own), 2)
                    c["clip_mode_vs_session_cents"] = round(own - mode_c)
        session_rows.append(srow)

    fields = ["filename", "session", "duration_s", "timeline_s", "has_crepe", "frames", "expected_frames",
              "length_ok", "category", "raga", "gamaka_type", "tonic_hz", "session_pooled_mode_hz",
              "clip_mode_hz", "clip_mode_vs_session_cents"]
    with open(ROOT / "dataset_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "tonic_hz": DATASET_TONIC_HZ,
                        "timeline_s": round(r["timeline_s"], 3) if r["timeline_s"] is not None else ""})
    with open(ROOT / "session_tonics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(session_rows[0]))
        w.writeheader()
        w.writerows(session_rows)
    with open(ROOT / "tonic_estimates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "tonic_hz", "method"])
        for r in rows:
            w.writerow([f"Gamaka_veena_dataset/{r['filename']}", DATASET_TONIC_HZ, TONIC_METHOD])

    print(f"files={len(rows)} has_crepe={sum(r['has_crepe'] == 'Y' for r in rows)} "
          f"length_ok={sum(r['length_ok'] == 'Y' for r in rows)} sessions={len(session_rows)} "
          f"tonic={DATASET_TONIC_HZ} Hz for all files ({TONIC_METHOD})")
    for r in rows:
        if r["has_crepe"] == "N":
            print("MISSING CREPE:", r["filename"])
        elif r["length_ok"] == "N":
            print(f"LENGTH MISMATCH: {r['filename']} frames={r['frames']} expected~{r['expected_frames']}")
        elif r["_thr"] is None:
            print(f"NOT POOLED (too few confident sustained frames): {r['filename']}")
        elif r["_thr"] == 0.5:
            print(f"POOLED AT RELAXED CONF 0.5: {r['filename']}")


if __name__ == "__main__":
    main()
