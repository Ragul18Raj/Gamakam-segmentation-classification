"""Step 3 spot checks: is the CREPE pitch line good enough on these recordings?

For ALL recordings (cheap), counts per minute of voiced audio:
  octave_jumps : adjacent voiced frames (conf >= 0.5) that differ by 1000-1400 cents
  big_jumps    : adjacent voiced frames that differ by > 400 cents (includes octave jumps)
  spikes       : a single voiced frame > 300 cents away from both neighbours, which agree with each other (< 100 c)
  and the same counts *inside annotated events*, since that is where it matters.
For 4 recordings (VARNAM + one slow, one medium, one fast from different passages):
  results/spotcheck/<rec>_pNN.png : spectrogram + pitch line, singer's events shaded by type, in 30 s pages
  results/spotcheck/<rec>_sonified.wav and <rec>_orig_plus_sonified.wav for listening (sonify_crepe.synth)
Writes results/pitch_quality.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from sonify_crepe import synth  # noqa: E402

SPOT = ["VARNAM", "TRACK 1", "TRACK 14", "TRACK 27"]  # VARNAM, slow (P01), medium (P05), fast (P09) - provisional tempos
COL = {"Kampita": "#4daf4a", "Nokku": "#e41a1c", "Jaaru (Etra)": "#377eb8", "Jaaru (Irakka)": "#984ea3"}
PAGE = 30.0


def quality(rec, ev):
    d = np.load(ROOT / "crepe" / f"{rec}.npz")
    t, f, c = d["time_s"], d["f0_hz"], d["confidence"]
    cents = 1200 * np.log2(np.maximum(f, 1e-3) / 440.0)
    v = c >= 0.5
    pair = v[1:] & v[:-1]
    dc = np.abs(np.diff(cents))
    octj = pair & (dc >= 1000) & (dc <= 1400)
    bigj = pair & (dc > 400)
    tri = v[2:] & v[1:-1] & v[:-2]
    spk = tri & (np.abs(cents[1:-1] - cents[:-2]) > 300) & (np.abs(cents[1:-1] - cents[2:]) > 300) & \
        (np.abs(cents[2:] - cents[:-2]) < 100)
    inev = np.zeros(len(t), bool)
    for _, e in ev[ev.recording == rec].iterrows():
        inev[(t >= e.start_s) & (t < e.end_s)] = True
    vmin = v.sum() / 6000
    return dict(recording=rec, minutes=round(len(t) / 6000, 2), voiced_frac=round(v.mean(), 3),
                voiced_frac_in_events=round(v[inev].mean(), 3) if inev.any() else np.nan,
                octave_jumps=int(octj.sum()), big_jumps=int(bigj.sum()), spikes=int(spk.sum()),
                big_jumps_per_voiced_min=round(bigj.sum() / max(vmin, 1e-9), 1),
                octave_jumps_in_events=int((octj & inev[1:]).sum()), big_jumps_in_events=int((bigj & inev[1:]).sum()),
                spikes_in_events=int((spk & inev[1:-1]).sum()),
                event_frames_low_conf=int((inev & ~v).sum()), event_frames=int(inev.sum()))


def plot_pages(rec, ev, sa_hz):
    d = np.load(ROOT / "crepe" / f"{rec}.npz")
    t, f, c = d["time_s"], d["f0_hz"], d["confidence"]
    y, sr = sf.read(str(ROOT / "audio" / f"{rec}.mp3"), dtype="float32", always_2d=True)
    mono = resample_poly(y.mean(axis=1), 1, 3)  # 16 kHz for the picture
    sr16 = sr // 3
    fv = np.where(c >= 0.5, f, np.nan)
    e = ev[ev.recording == rec]
    out = ROOT / "results" / "spotcheck"
    n_pages = int(np.ceil(t[-1] / PAGE))
    for p in range(n_pages):
        a, b = p * PAGE, min((p + 1) * PAGE, t[-1])
        fig, ax = plt.subplots(2, 1, figsize=(18, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
        seg = mono[int(a * sr16):int(b * sr16)]
        ax[0].specgram(seg, NFFT=2048, Fs=sr16, noverlap=2048 - 160, cmap="magma", vmin=-110, xextent=(a, b))
        m = (t >= a) & (t <= b)
        ax[0].plot(t[m], fv[m], color="cyan", lw=1.0)
        ax[0].set_yscale("log")
        ax[0].set_ylim(sa_hz / 2, sa_hz * 4)
        ax[0].set_ylabel("Hz")
        ax[1].plot(t[m], 1200 * np.log2(fv[m] / sa_hz), color="k", lw=0.9)
        ax[1].scatter(t[m][c[m] < 0.5], np.full((c[m] < 0.5).sum(), -1150), s=1, color="0.7")
        for _, r in e[(e.end_s > a) & (e.start_s < b)].iterrows():
            for x in ax:
                x.axvspan(r.start_s, r.end_s, color=COL[r.label], alpha=0.25, lw=0)
        for k in range(-12, 25, 12):
            ax[1].axhline(k * 100, color="0.6", lw=0.5)
        ax[1].axhline(700, color="0.8", lw=0.5, ls="--")
        ax[1].set_ylim(-1200, 2000)
        ax[1].set_ylabel("cents above Sa (Pa dashed)")
        ax[1].set_xlabel("time (s)")
        ax[1].legend(handles=[Patch(color=v, alpha=0.4, label=k) for k, v in COL.items()], loc="upper right", fontsize=8, ncol=4)
        ax[0].set_title(f"{rec}  {a:.0f}-{b:.0f} s: spectrogram + CREPE pitch (conf>=0.5); singer's events shaded "
                        f"(Sa = {sa_hz:.1f} Hz measured from opening Sa)")
        fig.tight_layout()
        fig.savefig(out / f"{rec.replace(' ', '_')}_p{p + 1:02d}.png", dpi=90)
        plt.close(fig)
    # listening files
    son = synth(t, f, c, 16000, 0.5, "sine")
    sf.write(out / f"{rec.replace(' ', '_')}_sonified.wav", son, 16000)
    o = 0.9 * mono / (np.abs(mono).max() + 1e-9)
    n = min(len(o), len(son))
    mix = 0.6 * o[:n] + 0.4 * son[:n]
    sf.write(out / f"{rec.replace(' ', '_')}_orig_plus_sonified.wav", 0.9 * mix / np.abs(mix).max(), 16000)


def main():
    ev = pd.read_csv(ROOT / "annotations" / "pass1_events.csv")
    meta = pd.read_csv(ROOT / "metadata.csv", dtype=str).fillna("")
    q = pd.DataFrame([quality(r, ev) for r in meta.recording])
    q.to_csv(ROOT / "results" / "pitch_quality.csv", index=False)
    print(q.to_string(index=False))
    tot = q.sum(numeric_only=True)
    print(f"\nTOTAL: octave jumps {int(tot.octave_jumps)} ({int(tot.octave_jumps_in_events)} inside events), "
          f"big jumps {int(tot.big_jumps)} ({int(tot.big_jumps_in_events)} inside events), spikes {int(tot.spikes)} "
          f"({int(tot.spikes_in_events)} inside events); event frames with conf < 0.5: "
          f"{int(tot.event_frames_low_conf)}/{int(tot.event_frames)} ({tot.event_frames_low_conf / tot.event_frames:.1%})")
    (ROOT / "results" / "spotcheck").mkdir(exist_ok=True)
    sa = dict(zip(meta.recording, meta.sa_hz.astype(float)))
    for r in SPOT:
        plot_pages(r, ev, sa[r])
        print("plotted + sonified", r)


if __name__ == "__main__":
    main()
