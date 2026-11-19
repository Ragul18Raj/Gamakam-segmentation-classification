"""Step 2: train the Stage 2 CNN as gamaka vs none. Same folds, seeds, inputs, augmentation and training code as the original; only the labels are merged.
stage2_vocal.py is imported, not edited. usage (from vocal_dataset): python results/two_stage/train_binary.py P01"""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; VD = HERE.parents[1]; sys.path.insert(0, str(VD))
import stage2_vocal as S
S.P.NCLS = 2; S.NCLS = 2                    # two outputs: none, gamaka
test = sys.argv[1]; logf = open(HERE / "logs" / f"log_{test}.txt", "w")
def log(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()
clips = S.load_real_dataset("group")
for c in clips: c["ann"] = [(s, e, 1) for s, e, _ in c["ann"]]      # every marked type becomes "gamaka"
groups = sorted({c["song"] for c in clips}); val = groups[(groups.index(test) + 1) % len(groups)]; seed = groups.index(test) + 1
tr = [c for c in clips if c["song"] not in (test, val)]; va = [c for c in clips if c["song"] == val]; te = [c for c in clips if c["song"] == test]
net, best = S.train_cnn(S.prep(tr, True, False), S.prep(va, False, False), 30, seed=seed, log=log); log(f"best val frame macroF1 {best:.3f}")
np.savez_compressed(HERE / "probs" / f"fold_{test}.npz", **{f"{r}|{c['id']}": S.predict_sliding(net, S.feats(c, False)).astype(np.float32) for r, L in (("test", te), ("val", va)) for c in L})
