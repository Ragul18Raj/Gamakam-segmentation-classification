"""Step 2 check: held-out Stage 1 DTW-3NN accuracy by marked event length (existing predictions; singer's boundaries are given)."""
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; VD = HERE.parents[1]; P = pd.read_csv(VD / "results/stage1/predictions.csv"); C = "main_4class | DTW-3NN"
fr = ((P.end_s - P.start_s) * 100).round().astype(int); L = ["<0.2 s", "0.2-0.4 s", "0.4-0.8 s", ">=0.8 s"]; P["bin"] = pd.cut(fr, [0, 20, 40, 80, 10000], right=False, labels=L); P["ok"] = P[C] == P.label
out = P.groupby("bin").agg(events=("ok", "size"), correct=("ok", "sum"), accuracy=("ok", "mean")).round(3); print(out.to_string())
print("\ncorrect/total by type"); print(P.pivot_table(index="label", columns="bin", values="ok", aggfunc=lambda s: f"{int(s.sum())}/{len(s)}", fill_value="-").to_string()); print("\nall events:", f"{int(P.ok.sum())}/{len(P)} = {P.ok.mean():.3f}")
print("\nwhat wrong events were called"); print(P[~P.ok].groupby(["bin", "label"])[C].value_counts().rename("n").reset_index().to_string(index=False))
