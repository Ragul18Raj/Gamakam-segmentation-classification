"""EXPLORATORY retraining of Stage 2 (18 Nov 2026, after results and the length analysis had been seen). stage2_vocal.py is imported, not edited.
Decided before training. Primary variant = both. weights and mask are diagnostics only.
  weights: 'none' keeps its original weight and the total weight given to gamaka frames is unchanged, but it is split equally between the 4 types.
  mask:    boundary exclusion is +-10 ms (1 frame) around events shorter than 0.4 s; +-20 ms (2 frames) stays for longer events.
Same folds, validation groups, seeds, epochs, early stopping, filter choice, scoring and pass marks as results/stage2_main.
usage (from vocal_dataset): python results/length_fix/train_fold.py --variant both --fold P01 --out results/length_fix/both"""
import sys
from pathlib import Path
import numpy as np
VD = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(VD))
import stage2_vocal as S
ORIG_BATCH = S.sample_batch; CACHE = {}
def balanced_weights(cw, counts):
    cw = np.asarray(cw, dtype=np.float64); n = np.maximum(counts.astype(np.float64), 1.0); total = float((n[1:] * cw[1:]).sum()); out = cw.copy(); out[1:] = total / (4 * n[1:]); return out.astype(np.float32)
def batch_equal_types(train, cw, rng, *a, **k):
    if "cw" not in CACHE:
        allY = np.concatenate([d[2] for d in train if d[0] == 1.0]); CACHE["cw"] = balanced_weights(cw, np.bincount(allY, minlength=S.NCLS))
    return ORIG_BATCH(train, CACHE["cw"], rng, *a, **k)
def labels_short_mask(T, ann, short_s=0.4):
    y = np.zeros(T, np.int64); mask = np.ones(T, np.float32)
    for s, e, l in ann: y[int(round(s * S.FR)):int(round(e * S.FR))] = l
    for s, e, _ in ann:
        r = 1 if e - s < short_s else S.MASK_FR
        for b in (s, e):
            i = int(round(b * S.FR)); mask[max(0, i - r):min(T, i + r + 1)] = 0.0
    return y, mask
def apply(v):
    if v in ("weights", "both"): S.sample_batch = batch_equal_types
    if v in ("mask", "both"): S.frame_labels = labels_short_mask
if __name__ == "__main__":
    i = sys.argv.index("--variant"); v = sys.argv[i + 1]; del sys.argv[i:i + 2]; assert v in ("weights", "mask", "both"); apply(v); S.main()
