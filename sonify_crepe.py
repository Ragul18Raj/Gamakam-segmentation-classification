"""Turn CREPE output back into sound so you can hear the pitch contour.

Works with either:
  - CREPE CSV  (time, frequency, confidence)            e.g. clip.f0.csv
  - CREPE activations .npy  (frames x 360)              e.g. clip.activation.npy

Usage:
  python sonify_crepe.py clip.f0.csv
  python sonify_crepe.py clip.activation.npy --hop 10
  python sonify_crepe.py folder_of_outputs/            # does every file
Options:
  --conf 0.5      frames below this confidence are silent
  --tone veena    'sine' (pure, best for checking pitch) or 'veena' (harmonics, bright)
  --sr 16000

Writes <name>_sonified.wav next to each input.
Needs: pip install numpy soundfile
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def load_csv(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    names = d.dtype.names
    t = d[names[0]]
    f = d[names[1]]
    c = d[names[2]] if len(names) > 2 else np.ones_like(f)
    return t, f, c


def load_activation(path, hop_ms):
    act = np.load(path)  # (frames, 360), CREPE bins are 20 cents apart
    cents_map = 20 * np.arange(360) + 1997.3794084376191
    idx = act.argmax(axis=1)
    f, c = np.zeros(len(act)), act.max(axis=1)
    for i, k in enumerate(idx):  # weighted average around the peak, as CREPE does
        lo, hi = max(0, k - 4), min(360, k + 5)
        w = act[i, lo:hi]
        cents = (w * cents_map[lo:hi]).sum() / max(w.sum(), 1e-9)
        f[i] = 10 * 2 ** (cents / 1200)
    t = np.arange(len(act)) * hop_ms / 1000
    return t, f, c


def synth(t, f, c, sr, conf_thr, tone):
    n = int(t[-1] * sr) + 1
    ts = np.arange(n) / sr
    voiced = (c >= conf_thr).astype(float)
    # hold the last good pitch through gaps so the tone doesn't jump to 0 Hz
    f = f.copy()
    good = np.where(voiced > 0)[0]
    if len(good) == 0:
        return np.zeros(n)
    f = np.interp(np.arange(len(f)), good, f[good])
    freq = np.interp(ts, t, f)
    amp = np.interp(ts, t, voiced)
    k = int(0.01 * sr)  # 10 ms fade to avoid clicks
    amp = np.convolve(amp, np.ones(k) / k, mode="same")
    phase = 2 * np.pi * np.cumsum(freq) / sr
    if tone == "sine":
        y = np.sin(phase)
    else:
        y = sum(np.sin(h * phase) / h ** 1.2 for h in range(1, 9) if h * freq.max() < sr / 2)
    y = y * amp
    return 0.9 * y / (np.abs(y).max() + 1e-9)


def run(path, a):
    if path.suffix == ".npy":
        t, f, c = load_activation(path, a.hop)
    else:
        t, f, c = load_csv(path)
    y = synth(t, f, c, a.sr, a.conf, a.tone)
    out = path.with_name(path.stem + "_sonified.wav")
    sf.write(out, y, a.sr)
    print("wrote", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--hop", type=float, default=10, help="ms per frame (activations only)")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--tone", choices=["sine", "veena"], default="sine")
    p.add_argument("--sr", type=int, default=16000)
    a = p.parse_args()
    src = Path(a.input)
    files = sorted(src.glob("*.csv")) + sorted(src.glob("*.npy")) if src.is_dir() else [src]
    for fp in files:
        run(fp, a)
