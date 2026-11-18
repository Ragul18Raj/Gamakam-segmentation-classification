"""Step 3: CREPE pitch for every recording in audio/ (MP3, 48 kHz, stereo).

Settings copied from ../run_crepe.py: model_capacity='full', step_size=10 ms, viterbi=False, center=True.
Audio is read with soundfile (libsndfile 1.2.2 reads MP3) and the two channels are averaged to mono.
The left/right correlation is recorded so the averaging can be justified (or not).

Output per recording:
  crepe/<rec>.npy   [T, 360] float32 activation
  crepe/<rec>.npz   time_s, f0_hz, confidence  (decoded exactly as sonify_crepe.load_activation)
  results/channel_check.csv   rec, corr_LR, rms_L, rms_R, max_abs_diff
Existing outputs are skipped unless --force.
"""
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
AUD, OUT, RES = ROOT / "audio", ROOT / "crepe", ROOT / "results"
HOP_MS = 10


def decode(act, hop_ms=HOP_MS):
    """Same as sonify_crepe.load_activation (weighted average of 9 bins around the peak)."""
    cents_map = 20 * np.arange(360) + 1997.3794084376191
    idx = act.argmax(axis=1)
    c = act.max(axis=1)
    f = np.zeros(len(act))
    for i, k in enumerate(idx):
        lo, hi = max(0, k - 4), min(360, k + 5)
        w = act[i, lo:hi]
        cents = (w * cents_map[lo:hi]).sum() / max(w.sum(), 1e-9)
        f[i] = 10 * 2 ** (cents / 1200)
    t = np.arange(len(act)) * hop_ms / 1000
    return t, f, c


def main():
    import crepe

    OUT.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    files = sorted(AUD.glob("*.mp3"), key=lambda p: (p.stem != "VARNAM", int(p.stem.split()[-1]) if p.stem != "VARNAM" else 0))
    rows, t_all = [], time.time()
    for i, p in enumerate(files, 1):
        rec = p.stem
        y, sr = sf.read(str(p), dtype="float32", always_2d=True)
        L, R = y[:, 0], y[:, -1]
        corr = float(np.corrcoef(L, R)[0, 1]) if y.shape[1] > 1 else 1.0
        rows.append((rec, corr, float(np.sqrt((L ** 2).mean())), float(np.sqrt((R ** 2).mean())),
                     float(np.abs(L - R).max())))
        npy, npz = OUT / f"{rec}.npy", OUT / f"{rec}.npz"
        if npy.exists() and npz.exists() and "--force" not in sys.argv:
            print(f"[{i}/{len(files)}] skip {rec}", flush=True)
            continue
        t0 = time.time()
        mono = y.mean(axis=1)
        _, _, _, act = crepe.predict(mono, sr, model_capacity="full", viterbi=False, step_size=HOP_MS, verbose=0)
        assert act.ndim == 2 and act.shape[1] == 360 and len(act) > 0, act.shape
        act = act.astype(np.float32)
        tmp = OUT / f"{rec}.tmp.npy"
        np.save(tmp, act)
        tmp.replace(npy)
        t, f, c = decode(act)
        np.savez(npz, time_s=t, f0_hz=f, confidence=c)
        print(f"[{i}/{len(files)}] OK {rec}  {act.shape}  audio {len(mono) / sr:.1f}s  corrLR={corr:.4f}  "
              f"{time.time() - t0:.0f}s", flush=True)
    import csv
    with open(RES / "channel_check.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["recording", "corr_LR", "rms_L", "rms_R", "max_abs_diff"])
        w.writerows(rows)
    print(f"DONE in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
