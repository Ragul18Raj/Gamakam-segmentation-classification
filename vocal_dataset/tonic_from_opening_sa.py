"""Step 4: tonic (Sa) of every recording, measured from the Sa the singer sings at the start of each clip.

The singer told us (17 Nov 2026) that every clip opens with Sa, and that this opening Sa is not annotated.
So the tonic is MEASURED from that opening note, not guessed from a whole-track histogram.

Method, per recording (crepe/<rec>.npz, 10 ms frames):
 1. Look only at the opening window: 0 s up to 0.5 s before the first annotated gamaka.
 2. Keep "steady" frames: confidence >= 0.8 and |pitch change to the next frame| < 20 cents.
 3. Find the longest run of steady frames that stays within +-35 cents of its own median (the held Sa).
    Sa_hz = median pitch of that run; also report its length and spread (IQR, cents).
 4. Cross-check against the rest of the track: fold all steady frames after the opening into one octave
    around Sa and report where the most common held pitch sits (expected: Sa or Pa, i.e. ~0 or ~702 cents).
 5. Report the nearest equal-tempered note (A4 = 440 Hz) and the offset in cents.
Writes results/tonic.csv and fills sa_note / sa_hz in metadata.csv (source noted as 'measured from opening Sa').
Also writes results/tonic_opening.png (pitch of the opening window for every recording, with the Sa run marked).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CONF, STEP_C, HOLD_C = 0.8, 20.0, 35.0


def note_name(hz):
    m = 69 + 12 * np.log2(hz / 440.0)
    k = int(round(m))
    return f"{NAMES[k % 12]}{k // 12 - 1}", float(100 * (m - k))


def longest_hold(cents, ok):
    """Longest run of consecutive ok frames whose values stay within +-HOLD_C of the run median."""
    best = (0, 0)
    i, n = 0, len(cents)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < n and ok[j]:
            j += 1
        # inside this ok-run, grow windows that stay within the band
        a = i
        for b in range(i + 1, j + 1):
            seg = cents[a:b]
            while b - a > 1 and (seg.max() - seg.min()) > 2 * HOLD_C:
                a += 1
                seg = cents[a:b]
            if b - a > best[1] - best[0]:
                best = (a, b)
        i = j
    return best


def main():
    ev = pd.read_csv(ROOT / "annotations" / "pass1_events.csv")
    first = ev.groupby("recording").start_s.min()
    meta = pd.read_csv(ROOT / "metadata.csv", dtype=str).fillna("")
    rows, panels = [], []
    for rec in meta.recording:
        d = np.load(ROOT / "crepe" / f"{rec}.npz")
        t, f, c = d["time_s"], d["f0_hz"], d["confidence"]
        cents = 1200 * np.log2(np.maximum(f, 1e-3) / 440.0)
        step = np.abs(np.diff(cents, append=cents[-1]))
        steady = (c >= CONF) & (step < STEP_C)
        end = max(0.0, first[rec] - 0.5)
        w = t < end
        a, b = longest_hold(cents[w], steady[w])
        run = cents[a:b]
        sa_c = float(np.median(run)) if len(run) else np.nan
        sa_hz = 440.0 * 2 ** (sa_c / 1200)
        iqr = float(np.subtract(*np.percentile(run, [75, 25]))) if len(run) else np.nan
        # cross-check on the rest of the track
        rest = steady & (t >= end)
        fold = np.mod(cents[rest] - sa_c + 50, 1200) - 50  # -50..1150 cents above Sa
        h, edges = np.histogram(fold, bins=np.arange(-50, 1151, 10))
        hs = np.convolve(h, np.ones(3) / 3, mode="same")
        mode_c = float(edges[hs.argmax()] + 5)
        sa_frac = float(np.mean(np.abs(fold) <= 30)) if len(fold) else np.nan
        pa_frac = float(np.mean(np.abs(fold - 702) <= 30)) if len(fold) else np.nan
        nn, off = note_name(sa_hz)
        rows.append(dict(recording=rec, opening_window_s=round(end, 2), sa_start_s=round(t[a], 2) if len(run) else np.nan,
                         sa_len_s=round((b - a) / 100, 2), sa_hz=round(sa_hz, 2), sa_iqr_cents=round(iqr, 1),
                         nearest_note=nn, offset_cents=round(off, 1), rest_mode_cents_above_sa=round(mode_c, 0),
                         rest_frac_on_sa=round(sa_frac, 3), rest_frac_on_pa=round(pa_frac, 3),
                         confidence_mean_in_run=round(float(c[a:b].mean()), 3) if len(run) else np.nan))
        panels.append((rec, t[w], np.where(c[w] >= 0.5, f[w], np.nan), t[a] if len(run) else 0, t[b - 1] if len(run) else 0, sa_hz))
    tab = pd.DataFrame(rows)
    # consistency across recordings
    ref = np.median(tab.sa_hz)
    tab["cents_from_median_sa"] = (1200 * np.log2(tab.sa_hz / ref)).round(1)
    tab["check"] = np.where((tab.sa_len_s >= 1.0) & (tab.sa_iqr_cents <= 20) &
                            ((tab.rest_mode_cents_above_sa.abs() <= 30) | ((tab.rest_mode_cents_above_sa - 702).abs() <= 30) |
                             (tab.rest_frac_on_sa + tab.rest_frac_on_pa >= 0.25)), "ok", "LOOK")
    (ROOT / "results").mkdir(exist_ok=True)
    tab.to_csv(ROOT / "results" / "tonic.csv", index=False)
    print(tab.to_string(index=False))
    print(f"\nmedian Sa across recordings {ref:.2f} Hz = {note_name(ref)[0]} {note_name(ref)[1]:+.0f} c; "
          f"spread {tab.cents_from_median_sa.min():+.0f} to {tab.cents_from_median_sa.max():+.0f} cents")
    m = meta.set_index("recording")
    m["sa_hz"] = tab.set_index("recording").sa_hz.round(2).astype(str)
    m["sa_note"] = tab.set_index("recording").nearest_note
    m["notes"] = m["notes"].str.replace("; Sa measured from opening Sa (tonic_from_opening_sa.py)", "", regex=False) + \
        "; Sa measured from opening Sa (tonic_from_opening_sa.py)"
    m.reset_index().to_csv(ROOT / "metadata.csv", index=False)

    n = len(panels)
    fig, axs = plt.subplots((n + 3) // 4, 4, figsize=(16, 2.2 * ((n + 3) // 4)), squeeze=False)
    for ax, (rec, tt, ff, s0, s1, sa) in zip(axs.flat, panels):
        ax.plot(tt, ff, lw=0.8)
        ax.axvspan(s0, s1, color="orange", alpha=0.3)
        ax.axhline(sa, color="k", lw=0.6, ls="--")
        ax.set_yscale("log")
        ax.set_ylim(sa / 2 ** (7 / 12), sa * 2 ** (19 / 12))
        ax.set_title(f"{rec}: Sa {sa:.1f} Hz", fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axs.flat[n:]:
        ax.axis("off")
    fig.suptitle("Opening window of each recording (before the first annotated gamaka). Orange = the held run taken as Sa", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "tonic_opening.png", dpi=110)


if __name__ == "__main__":
    main()
