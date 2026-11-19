"""Measure CREPE pitch for one uploaded recording with the Stage 2 settings. The decode step is imported from run_crepe_vocal.py."""
import os, sys
from pathlib import Path
sys.dont_write_bytecode = True
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, soundfile as sf
VD = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(VD))
from run_crepe_vocal import HOP_MS, decode

def main(audio, out):
    import crepe
    y, sr = sf.read(audio, dtype="float32", always_2d=True)
    _, _, _, act = crepe.predict(y.mean(axis=1), sr, model_capacity="full", viterbi=False, step_size=HOP_MS, verbose=0)
    t, f, c = decode(act.astype(np.float32))
    tmp = Path(str(out) + ".tmp.npz"); np.savez(tmp, time_s=t, f0_hz=f, confidence=c); tmp.replace(out)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
