"""Train the raga-free gamaka-vs-none detector for one fold. EXPLORATORY. Same network, folds, seeds, stretch factors, class weights, boundary mask,
batches per epoch and early stopping as the Step 3 'all three' model, without Sa-relative or swara inputs. Each training example gets a random transposition
in +-600 cents before its running median is computed. usage (from vocal_dataset): python results/raga_free/train_raga_free.py 75 P01"""
import argparse, os, random, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from rf_common import S, TV, new_net, track_base, movement_features
torch = TV["torch"]; nn = torch.nn; WIN, FR, BOUND = TV["WIN"], TV["FR"], TV["BOUND"]
A = argparse.ArgumentParser(); A.add_argument("window", type=int); A.add_argument("fold"); A.add_argument("--epochs", type=int, default=30); A.add_argument("--out", default=str(HERE)); A = A.parse_args(); ROOT = Path(A.out); W = ROOT / "weights" / f"W{A.window}" / f"fold_{A.fold}.pt"; PR = ROOT / "probs" / f"W{A.window}" / f"fold_{A.fold}.npz"
if W.exists() or PR.exists(): sys.exit("Output already exists; not overwriting it.")
W.parent.mkdir(parents=True, exist_ok=True); PR.parent.mkdir(parents=True, exist_ok=True); torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
clips = S.load_real_dataset("group"); gs = sorted({c["song"] for c in clips}); seed = gs.index(A.fold) + 1; val = gs[(gs.index(A.fold) + 1) % len(gs)]; os.environ["PYTHONHASHSEED"] = "0"; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); rng = np.random.default_rng(seed)
for c in clips: c["ann"] = [(s, e, 1) for s, e, _ in c["ann"]]
def prep(cl, aug):
    out = []
    for c in cl:
        for fac, ce, co, an in [(1.0, c["cents"], c["conf"], c["ann"])] + ([(f,) + S.stretch(c["cents"], c["conf"], c["ann"], f) for f in (.8, .9, 1.1, 1.25)] if aug else []):
            fl, v, q = track_base(ce, co); f = movement_features(fl, v, q, A.window); T = f.shape[1]; y, m = S.frame_labels(T, an); b = np.zeros((2, T), np.float32)
            for s, e, _ in an:
                for k, t in enumerate((s, e)): i = int(round(t * FR)); b[k, max(0, i - BOUND):min(T, i + BOUND + 1)] = 1
            out.append(dict(fac=fac, f=f, y=y, m=m, b=b, filled=fl, voiced=v, conf=q))
    return out
MAX_DIFF = [0.0]
def batch(data, cw, n=16):
    X, Y, W_, B = [], [], [], []; margin = A.window // 2 + 5
    for _ in range(n):
        pool = [d for d in data if d["fac"] == 1.0] if rng.random() < .5 else data; d = pool[rng.integers(len(pool))]; T = d["f"].shape[1]; shift = rng.uniform(-600, 600)
        if T <= WIN:
            x = movement_features(d["filled"] + shift, d["voiced"], d["conf"], A.window); ref = d["f"]; pad = ((0, 0), (0, WIN - T)); x, ref, b = np.pad(x, pad), np.pad(ref, pad), np.pad(d["b"], pad); y, m = np.pad(d["y"], (0, WIN - T)), np.pad(d["m"], (0, WIN - T))
        else:
            s = int(rng.integers(0, T - WIN + 1)); l, r = max(0, s - margin), min(T, s + WIN + margin); x = movement_features(d["filled"][l:r] + shift, d["voiced"][l:r], d["conf"][l:r], A.window)[:, s - l:s - l + WIN]; ref = d["f"][:, s:s + WIN]; y, m, b = d["y"][s:s + WIN], d["m"][s:s + WIN], d["b"][:, s:s + WIN]
        MAX_DIFF[0] = max(MAX_DIFF[0], float(np.abs(x - ref).max())); X.append(x); Y.append(y); W_.append(m * cw[y]); B.append(b)
    return [torch.from_numpy(np.stack(a).astype(t)) for a, t in ((X, np.float32), (Y, np.int64), (W_, np.float32), (B, np.float32))]
tr = prep([c for c in clips if c["song"] not in (A.fold, val)], True); va = prep([c for c in clips if c["song"] == val], False); vt = [(0, d["f"], d["y"], 0, 0) for d in va]; Y = np.concatenate([d["y"] for d in tr if d["fac"] == 1.0]); fq = np.maximum(np.bincount(Y, minlength=2) / len(Y), 1e-6); cw = (fq ** -.5).astype(np.float32); cw /= cw[Y].mean()
net = new_net(); opt = torch.optim.Adam(net.parameters(), 1e-3); ce = nn.CrossEntropyLoss(reduction="none"); bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.)); best, state, bad = -1, None, 0; t0 = time.time()
for ep in range(A.epochs):
    net.train(); tot = 0
    for _ in range(75):
        X, y, w, b = batch(tr, cw); lg, bd = net(X); loss = (ce(lg, y) * w).sum() / (w.sum() + 1e-8) + .5 * bce(bd, b); opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
    vs = TV["val_f1"](net, vt, False); print(f"epoch {ep + 1} loss {tot / 75:.3f} val {vs:.3f} largest input change from transposition {MAX_DIFF[0]:.2e} ({time.time() - t0:.0f}s)", flush=True)
    if vs > best: best, bad, state = vs, 0, {k: v.clone() for k, v in net.state_dict().items()}
    else:
        bad += 1
        if bad >= 5: break
assert MAX_DIFF[0] < 1e-4, f"transposition changed the input: {MAX_DIFF[0]}"
net.load_state_dict(state); P = {}
for role, g in (("test", A.fold), ("val", val)):
    for c in clips:
        if c["song"] == g: fl, v, q = track_base(c["cents"], c["conf"]); P[f"{role}|{c['id']}"] = TV["predict"](net, movement_features(fl, v, q, A.window), False)
np.savez_compressed(PR, **P); torch.save(dict(state_dict=net.state_dict(), fold=A.fold, seed=seed, window=A.window, best_val_frame_f1=best, largest_transposition_input_change=MAX_DIFF[0]), W); print(f"best val {best:.3f}; finished {time.time() - t0:.0f}s", flush=True)
