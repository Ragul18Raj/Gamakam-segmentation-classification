"""Raga-free Stage 2 inputs. EXPLORATORY.
Four input channels: local movement (cents minus a running median), pitch velocity, voiced flag, CREPE confidence.
No Sa-relative sine/cosine and no swara distance. The running median is computed after any transposition."""
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
HERE = Path(__file__).resolve().parent; VD = HERE.parents[1]; sys.path.insert(0, str(VD))
TV = {"__file__": str(VD / "results/step3/train_variant.py"), "__name__": "frozen_trainer"}
exec((VD / "results/step3/train_variant.py").read_text().split("\na = argparse.ArgumentParser()")[0], TV)   # Net, predict and validation code are reused unchanged
S = TV["S"]; PL = S.P; WINDOWS = (51, 75, 101)                  # about 0.5, 0.75 and 1.0 s, odd so the median is centred

def track_base(cents, conf):
    """Gap filling and voicing, exactly as in pipeline.features but in float64 so transposition stays exact."""
    c = np.asarray(cents, np.float64); q = np.asarray(conf, np.float64); filled, voiced = PL.fill_short_gaps(c, (~np.isnan(c)) & (q >= .5)); return filled, voiced, np.nan_to_num(q)
def movement_features(filled, voiced, conf, w):
    local = np.clip(filled - median_filter(filled, size=w, mode="nearest"), -600, 600) / 100; vel = np.clip(uniform_filter1d(np.gradient(filled), 5) / 10, -10, 10); f = np.stack([local, vel, voiced.astype(float), conf]); f[:2, ~voiced] = 0; return f.astype(np.float32)
def features_for(cents, conf, w, shift=0.0):
    filled, voiced, q = track_base(cents, conf); return movement_features(filled + shift, voiced, q, w)
def new_net(): return TV["Net"](4, True, True)
