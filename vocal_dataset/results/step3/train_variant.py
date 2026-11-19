"""Step 3: gamaka-vs-none detector in PyTorch, with optional additions. EXPLORATORY.
Decided before training. Primary = all (swara distance + BiLSTM + onset/offset heads), decoded with the Step 2 Viterbi route.
base, swara, bilstm and heads are diagnostics that show what each addition does.

base: the Step 2 network rebuilt in PyTorch: 5 conv layers (48 channels, kernel 9, dilations 1, 2, 4, 8, 16; about 2.5 s of context),
      ReLU, dropout 0.2, 6 existing inputs (which already include velocity and CREPE confidence).
swara: adds distance to the nearest Kalyani swara, S R2 G3 M2 P D2 N3 in equal-tempered cents. It is recalculated after tonic jitter.
bilstm: adds a BiLSTM (32 units each way) after the convolutions.
heads: adds onset and offset outputs, trained on frames within +-30 ms of each marked boundary. They only help training; decoding does not use them.
Folds, validation groups, stretch factors, tonic jitter, class weights, boundary mask, batches per epoch and early stopping follow Stage 2.
usage (from vocal_dataset): python results/step3/train_variant.py all P01"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score
HERE = Path(__file__).resolve().parent; VD = HERE.parents[1]; sys.path.insert(0, str(VD))
import stage2_vocal as S
S.P.NCLS = 2; S.NCLS = 2
V = dict(base=(0, 0, 0), swara=(1, 0, 0), bilstm=(0, 1, 0), heads=(0, 0, 1), all=(1, 1, 1))
KALYANI = np.array([0, 200, 400, 600, 700, 900, 1100, 1200], np.float32); WIN, FR, BOUND = 300, 100, 3

def swara_channel(x):
    """x[..., 2:4] rows are sin/cos of pitch relative to Sa; returns distance to the nearest Kalyani note / 100 cents (0 when unvoiced)."""
    pc = np.mod(np.arctan2(x[2], x[3]) * 1200 / (2 * np.pi), 1200); d = np.abs(pc[..., None] - KALYANI).min(-1) / 100
    return np.where(x[4] > .5, d, 0).astype(np.float32)
def with_inputs(x, swara): return np.concatenate([x, swara_channel(x)[None]]) if swara else x
class Net(nn.Module):
    def __init__(s, cin, lstm, heads):
        super().__init__(); L = []; c = cin
        for d in (1, 2, 4, 8, 16): L += [nn.Conv1d(c, 48, 9, padding=4 * d, dilation=d), nn.ReLU(), nn.Dropout(.2)]; c = 48
        s.conv = nn.Sequential(*L); s.lstm = nn.LSTM(48, 32, batch_first=True, bidirectional=True) if lstm else None; s.drop = nn.Dropout(.2); h = 64 if lstm else 48
        s.cls = nn.Conv1d(h, 2, 1); s.bound = nn.Conv1d(h, 2, 1) if heads else None
    def forward(s, x):
        h = s.conv(x)
        if s.lstm is not None: h = s.drop(s.lstm(h.transpose(1, 2))[0].transpose(1, 2))
        return s.cls(h), (s.bound(h) if s.bound is not None else None)
def prep(clips, aug):
    out = []
    for c in clips:
        vers = [(1.0, c["cents"], c["conf"], c["ann"])] + ([(f,) + S.stretch(c["cents"], c["conf"], c["ann"], f) for f in (.8, .9, 1.1, 1.25)] if aug else [])
        for fac, ce, co, an in vers:
            f = S.feats(dict(cents=ce, conf=co), False); T = f.shape[1]; y, m = S.frame_labels(T, an); b = np.zeros((2, T), np.float32)
            for s, e, _ in an:
                for k, t in enumerate((s, e)):
                    i = int(round(t * FR)); b[k, max(0, i - BOUND):min(T, i + BOUND + 1)] = 1
            out.append((fac, f, y, m, b))
    return out
def batch(data, cw, rng, swara, n=16):
    X, Y, W, B = [], [], [], []
    for _ in range(n):
        pool = [d for d in data if d[0] == 1.0] if rng.random() < .5 else data; _, f, y, m, b = pool[rng.integers(len(pool))]; T = f.shape[1]
        if T <= WIN: s = 0; f, y, m, b = np.pad(f, ((0, 0), (0, WIN - T))), np.pad(y, (0, WIN - T)), np.pad(m, (0, WIN - T)), np.pad(b, ((0, 0), (0, WIN - T)))
        else: s = int(rng.integers(0, T - WIN + 1))
        x = f[:, s:s + WIN].copy(); dl = 2 * np.pi * rng.uniform(-30, 30) / 1200; sn, cs = x[2].copy(), x[3].copy(); v = x[4] > .5
        x[2] = np.where(v, sn * np.cos(dl) + cs * np.sin(dl), 0); x[3] = np.where(v, cs * np.cos(dl) - sn * np.sin(dl), 0); yy = y[s:s + WIN]
        X.append(with_inputs(x, swara)); Y.append(yy); W.append(m[s:s + WIN] * cw[yy]); B.append(b[:, s:s + WIN])
    return [torch.from_numpy(np.stack(a).astype(t)) for a, t in ((X, np.float32), (Y, np.int64), (W, np.float32), (B, np.float32))]
@torch.no_grad()
def val_f1(net, data, swara):
    net.eval(); ys, ps = [], []
    for _, f, y, _, _ in data: ps.append(net(torch.from_numpy(with_inputs(f, swara))[None])[0][0].argmax(0).numpy()); ys.append(y)
    net.train(); return f1_score(np.concatenate(ys), np.concatenate(ps), labels=[0, 1], average="macro", zero_division=0)
@torch.no_grad()
def predict(net, f, swara):
    """3 s windows, 0.1 s hop, softmax averaged over overlapping windows, as in pipeline.predict_sliding."""
    net.eval(); f = with_inputs(f, swara); T = f.shape[1]; fp = np.pad(f, ((0, 0), (0, max(0, WIN - T)))); st = list(range(0, max(T - WIN, 0) + 1, 10)); st += [] if st[-1] == max(T - WIN, 0) else [T - WIN]
    acc = np.zeros((2, max(T, WIN)), np.float32); cnt = np.zeros(max(T, WIN), np.float32)
    for i in range(0, len(st), 64):
        p = torch.softmax(net(torch.from_numpy(np.stack([fp[:, s:s + WIN] for s in st[i:i + 64]])))[0], 1).numpy()
        for j, s in enumerate(st[i:i + 64]): acc[:, s:s + WIN] += p[j]; cnt[s:s + WIN] += 1
    return (acc / cnt)[:, :T]
a = argparse.ArgumentParser(); a.add_argument("variant"); a.add_argument("fold"); a.add_argument("--epochs", type=int, default=30); a.add_argument("--out", default=str(HERE / "probs")); a = a.parse_args()
torch.set_num_threads(1); sw, ls, hd = map(bool, V[a.variant]); clips = S.load_real_dataset("group")
for c in clips: c["ann"] = [(s, e, 1) for s, e, _ in c["ann"]]
gs = sorted({c["song"] for c in clips}); val = gs[(gs.index(a.fold) + 1) % len(gs)]; seed = gs.index(a.fold) + 1; torch.manual_seed(seed); rng = np.random.default_rng(seed)
tr = prep([c for c in clips if c["song"] not in (a.fold, val)], True); va = prep([c for c in clips if c["song"] == val], False); te = [c for c in clips if c["song"] == a.fold]; vc = [c for c in clips if c["song"] == val]
Y = np.concatenate([d[2] for d in tr if d[0] == 1.0]); fq = np.maximum(np.bincount(Y, minlength=2) / len(Y), 1e-6); cw = (fq ** -.5).astype(np.float32); cw /= cw[Y].mean()
net = Net(6 + sw, ls, hd); opt = torch.optim.Adam(net.parameters(), 1e-3); ce = nn.CrossEntropyLoss(reduction="none"); bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.)); best, state, bad = -1, None, 0; t0 = time.time()
for ep in range(a.epochs):
    net.train(); tot = 0
    for _ in range(75):
        X, y, w, b = batch(tr, cw, rng, sw); lg, bd = net(X); loss = (ce(lg, y) * w).sum() / (w.sum() + 1e-8)
        if bd is not None: loss = loss + .5 * bce(bd, b)
        opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
    vs = val_f1(net, va, sw); print(f"epoch {ep + 1} loss {tot / 75:.3f} val {vs:.3f} ({time.time() - t0:.0f}s)", flush=True)
    if vs > best: best, bad, state = vs, 0, {k: v.clone() for k, v in net.state_dict().items()}
    else:
        bad += 1
        if bad >= 5: break
net.load_state_dict(state); out = Path(a.out) / a.variant; out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / f"fold_{a.fold}.npz", **{f"{r}|{c['id']}": predict(net, S.feats(c, False), sw) for r, L in (("test", te), ("val", vc)) for c in L}); print(f"best val {best:.3f}; finished {time.time() - t0:.0f}s", flush=True)
