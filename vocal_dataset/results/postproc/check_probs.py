"""Confirm the rerun reproduces the original models: top class from saved probabilities must equal the original cnn_raw on every frame."""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; VD = HERE.parents[1]; total = bad = 0; ok_ev = True
for k in range(1, 12):
    g = f"P{k:02d}"; o = json.load(open(VD / f"results/stage2_main/fold_{g}.json")); r = json.load(open(HERE / f"rerun/fold_{g}.json")); z = np.load(HERE / f"probs/fold_{g}.npz")
    for c, d in zip(o["clips"], r["clips"]):
        a = z[f"test|{c['id']}"].argmax(0); b = np.array(c["cnn_raw"]); assert len(a) == len(b); total += len(a); bad += int((a != b).sum()); ok_ev &= c["cnn"] == d["cnn"] and c["cnn_frames"] == d["cnn_frames"]
print(f"frames compared {total}, top-class mismatches {bad}; final events and cleaned frames identical in every clip: {ok_ev}")
