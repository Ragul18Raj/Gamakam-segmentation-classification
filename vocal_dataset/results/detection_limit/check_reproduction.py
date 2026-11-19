"""Reproduction check, declared before retraining: for every held-out fold, gamaka-probability correlation with the saved Step 3 output >= 0.99
AND event F1 after the same frozen Viterbi, naming and length settings within +-0.01 of the saved result. Exits with status 1 if any fold fails."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; VD = HERE.parents[1]; sys.path[:0] = [str(VD), str(VD / "results/two_stage")]
import two_stage as TS
PP = TS.PP; OUT = HERE / "retrained/reproduction_check.csv"
if OUT.exists(): sys.exit("Output already exists; not overwriting it.")
E = pd.read_csv(VD / "results/step3/events_all_B.csv"); P = pd.read_csv(VD / "results/step3/chosen_params.csv"); P = P[(P.variant == "all") & (P.cleanup == "B")].set_index("held_out_group"); rows = []
PP.viterbi(np.zeros((30, 5)), np.array([0, 2, 2, 2, 2], np.int64), 1.0)
for k, g in enumerate(TS.GROUPS):
    old = np.load(VD / f"results/step3/probs/all/fold_{g}.npz"); new = np.load(HERE / f"retrained/probs/all/fold_{g}.npz"); q = P.loc[g]; cfg = (q.D_region_s, q.switch, q.alpha); prior = TS.priors2(k); mins = {l: int(round(q[f"L_{TS.SHORT[l]}"] * TS.FR)) for l in TS.TYPES}; a, b, cl = [], [], []
    for c in TS.DATA[g]:
        po, pn = old[f"test|{c['id']}"], new[f"test|{c['id']}"]; assert po.shape == pn.shape; a.append(po[1]); b.append(pn[1]); x = E[E.track == c["id"]]
        saved = [(int(round((s - c["offset"]) * TS.FR)) / TS.FR, int(round((e - c["offset"]) * TS.FR)) / TS.FR, TS.LABEL_ID[t]) for s, e, t in zip(x.start_s, x.end_s, x.type)]
        d = dict(c); d["p2"] = pn.astype(np.float64); d["logp2"] = np.log(d["p2"].T + 1e-8); ev = []
        for s, e in TS.regions_b(d, cfg, prior):
            row = TS.dist_job((c["id"], s, e))[1]
            if row is not None: ev.append((s, e, TS.name_from_row(row, {g})))
        cl.append(dict(ref=c["ref"], saved=saved, new=PP.seconds(PP.drop_short(ev, mins))))
    a, b = np.concatenate(a), np.concatenate(b); r = float(np.corrcoef(a, b)[0, 1]); fo, fn = TS.ev_scores(cl, "saved")[0], TS.ev_scores(cl, "new")[0]; rows.append(dict(fold=g, frames=len(a), probability_correlation=r, max_abs_probability_difference=float(np.abs(a - b).max()), saved_event_f1=fo, retrained_event_f1=fn, event_f1_difference=fn - fo, correlation_pass=r >= .99, event_f1_pass=abs(fn - fo) <= .01))
T = pd.DataFrame(rows); T["pass"] = T.correlation_pass & T.event_f1_pass; T.to_csv(OUT, index=False); print(T.round(6).to_string(index=False)); print("ALL FOLDS PASS" if T["pass"].all() else "CHECK FAILED"); sys.exit(0 if T["pass"].all() else 1)
