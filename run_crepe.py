"""Run CREPE on every WAV in Gamaka_veena_dataset/ and save the [T, 360] activation matrix.

Settings: model_capacity='full', step_size=10 ms, viterbi=False (CREPE defaults otherwise, center=True).
Output: crepe_output/<wav stem>.npy  (existing outputs are skipped; pass --force to redo all)
"""
import sys
import time
import wave
from pathlib import Path

import numpy as np
import crepe

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Gamaka_veena_dataset"
OUT_DIR = ROOT / "crepe_output"


def load_wav(path):
    with wave.open(str(path)) as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"unsupported sample width {width * 8}-bit")
    audio = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    return audio, sr


def main():
    OUT_DIR.mkdir(exist_ok=True)
    wavs = sorted(DATA_DIR.glob("*.wav"))
    ok, failed = 0, []
    t_start = time.time()
    skipped = 0
    for i, wav in enumerate(wavs, 1):
        out = OUT_DIR / f"{wav.stem}.npy"
        if out.exists() and "--force" not in sys.argv:
            skipped += 1
            continue
        t0 = time.time()
        try:
            audio, sr = load_wav(wav)
            _, _, _, activation = crepe.predict(
                audio, sr, model_capacity="full", viterbi=False, step_size=10, verbose=0
            )
            if activation.ndim != 2 or activation.shape[1] != 360 or activation.shape[0] == 0:
                raise ValueError(f"unexpected activation shape {activation.shape}")
            # write to a temp file then rename, so an interrupted run never leaves a truncated .npy
            tmp = out.with_suffix(".tmp.npy")
            np.save(tmp, activation.astype(np.float32))
            tmp.replace(out)
            ok += 1
            print(f"[{i}/{len(wavs)}] OK   {wav.name}  {activation.shape}  {time.time() - t0:.1f}s", flush=True)
        except Exception as e:
            failed.append((wav.name, f"{type(e).__name__}: {e}"))
            print(f"[{i}/{len(wavs)}] FAIL {wav.name}  {type(e).__name__}: {e}", flush=True)
    total = time.time() - t_start
    print(f"\nDONE processed_ok={ok} skipped_existing={skipped} failed={len(failed)} total_time={total:.1f}s", flush=True)
    for name, err in failed:
        print(f"  FAILED: {name} -> {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
