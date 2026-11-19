"""Retrain one Step 3 'all' fold with the unchanged trainer, in deterministic mode, and save its weights. EXPLORATORY.
results/step3/train_variant.py is executed as written; its fold seed is the one used in Step 3.
usage (from vocal_dataset): python results/detection_limit/retrain_frozen.py P01"""
import os, random, runpy, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; VD = HERE.parents[1]; fold = sys.argv[1]; OUT = HERE / "retrained"; W = OUT / "weights" / f"fold_{fold}.pt"
if W.exists() or (OUT / "probs" / "all" / f"fold_{fold}.npz").exists(): sys.exit("Output already exists; not overwriting it.")
seed = int(fold[1:])                                   # same as groups.index(fold) + 1 in train_variant.py
os.environ["PYTHONHASHSEED"] = "0"; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
sys.argv = ["train_variant.py", "all", fold, "--out", str(OUT / "probs")]
g = runpy.run_path(str(VD / "results/step3/train_variant.py"), run_name="__main__")
torch.save(dict(state_dict=g["net"].state_dict(), fold=fold, seed=seed, best_val_frame_f1=g["best"]), W)
