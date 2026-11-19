"""Step 1, part 1: rerun the original Stage 2 folds unchanged and save the per-frame class probabilities.
stage2_vocal.py is imported, not edited; results/stage2_main is not touched.
usage (from vocal_dataset): python results/postproc/save_probs.py --fold P01 --out results/postproc/rerun"""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; VD = HERE.parents[1]; sys.path.insert(0, str(VD))
import stage2_vocal as S

KEEP = {}
orig_train = S.train_cnn
def train_and_keep(*a, **k):
    # The trained network is normally thrown away, so keep a reference to it.
    net, best = orig_train(*a, **k); KEEP["net"] = net; return net, best
S.train_cnn = train_and_keep

fold = sys.argv[sys.argv.index("--fold") + 1]
S.main()                                  # the normal fold: same seed, same training, same output format

# Same split rule as stage2_vocal.main: validation group = next group in order.
clips = S.load_real_dataset("group"); groups = sorted({c["song"] for c in clips}); val = groups[(groups.index(fold) + 1) % len(groups)]
out = {}
for role, g in (("test", fold), ("val", val)):
    for c in clips:
        if c["song"] == g:
            out[f"{role}|{c['id']}"] = S.predict_sliding(KEEP["net"], S.feats(c, False)).astype(np.float32)   # shape [5, T]
np.savez_compressed(HERE / "probs" / f"fold_{fold}.npz", **out)
