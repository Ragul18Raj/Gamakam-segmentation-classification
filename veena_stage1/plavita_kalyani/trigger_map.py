"""What in the pitch line triggered the plavita-only model on one recording. Plan: end of PLAN.md. Exploratory.
Run:  python trigger_map.py   Writes nets_all14.pkl, trigger_map.npz, trigger_map.json. Drawing is in trigger_plot.py."""
import os, time
HERE = os.path.dirname(os.path.abspath(__file__)); __file__ = os.path.join(HERE, 'detector.py')
exec(open(__file__, encoding='utf-8').read().split('t0 = time.time(); res = {}')[0])     # same data, features and training code as detector.py, so nothing can drift
NAME = 'kalyani2 #03'; TAG = 'kalyani2_03'; W = 10; STEP = 5; REACH = 124; LOW = -2400; MIN_RUN = 30; WIN_HOP = 10
t0 = time.time(); P_NETS = os.path.join(HERE, 'nets_all14.pkl')
if os.path.exists(P_NETS): nets = [DilatedCNN(cin=4, seed=s) for s in SEEDS]; [setattr(n, 'params', q) for n, q in zip(nets, pickle.load(open(P_NETS, 'rb')))]
else:
    nets = [train(PLAV, s) for s in SEEDS]; pickle.dump([n.params for n in nets], open(P_NETS, 'wb'))
cents, conf = pitch(os.path.join(HERE, 'crepe', NAME + '.npy')); f0, filled, v = feats(cents, conf); T = f0.shape[1]
old = np.load(os.path.join(HERE, f'scores_{TAG}.npz')); p = predict(nets, f0); same = float(np.abs(p - old['p']).max()); print(f'retrained in {time.time() - t0:.0f}s; largest score difference from the saved run: {same:.6f}', flush=True)
vb = v > 0; sung = np.zeros(T, bool)
for a, b in (lambda e: zip(e[::2], e[1::2]))(np.flatnonzero(np.diff(np.r_[0, vb.astype(int), 0]))):
    if b - a >= MIN_RUN and np.median(filled[a:b]) > LOW: sung[a:b] = filled[a:b] > LOW
starts = list(range(0, T - WIN + 1, WIN_HOP)); starts += [T - WIN] if starts[-1] != T - WIN else []; starts = np.array(starts); cnt = np.zeros(T)
for s in starts: cnt[s:s + WIN] += 1
def margins(net, f, idx):
    out = np.zeros((len(idx), WIN), np.float32)
    for i in range(0, len(idx), 64):
        k = idx[i:i + 64]; lg = net.forward(np.stack([f[:, starts[j]:starts[j] + WIN] for j in k]))[0]; out[i:i + 64] = lg[:, 1] - lg[:, 0]
    return out
def change(kind, a, b):
    if kind == 'blank': cf = conf.copy(); cf[a:b] = 0.0; return feats(cents, cf)[0]
    c = cents.copy(); idx = np.flatnonzero(vb); L = idx[idx < a]; R = idx[idx >= b]; l, r = (c[L[-1]] if len(L) else None), (c[R[0]] if len(R) else None)
    l = r if l is None else l; r = l if r is None else r; c[a:b] = np.linspace(l, r, b - a + 2)[1:-1]; return feats(c, conf)[0]
pos = [a for a in range(0, T - W + 1, STEP) if sung[a:a + W].any()]; print(len(pos), 'pieces', flush=True)
def run(net, kind):
    idx_all = np.arange(len(starts)); base = margins(net, f0, idx_all); acc = np.zeros(T)
    for j, s in enumerate(starts): acc[s:s + WIN] += base[j]
    D = np.zeros((len(pos), T), np.float32)
    for i, a in enumerate(pos):
        f = change(kind, a, a + W); ch = np.flatnonzero((f != f0).any(axis=0))
        if len(ch) == 0: continue
        lo, hi = ch[0], ch[-1] + 1; idx = np.flatnonzero((starts < hi) & (starts + WIN > lo)); new = margins(net, f, idx); d = np.zeros(T)
        for q, j in enumerate(idx): d[starts[j]:starts[j] + WIN] += base[j] - new[q]
        D[i] = d / cnt
    return D, acc / cnt
def score(D): return np.array([D[i][max(0, a + W // 2 - REACH):a + W // 2 + REACH + 1][sung[max(0, a + W // 2 - REACH):a + W // 2 + REACH + 1]].mean() for i, a in enumerate(pos)])
out = {}; base_m = None
for kind in ('blank', 'flatten'):
    Ds = []
    for s, n in zip(SEEDS, nets):
        t1 = time.time(); D, m = run(n, kind); Ds.append(D); print(f'{kind} seed {s}: {time.time() - t1:.0f}s', flush=True); base_m = m if s == 0 and base_m is None else base_m
        if s == 0: M0 = m
    out[kind + '_seed_importance'] = np.stack([score(D) for D in Ds]); D = np.mean(Ds, axis=0); out[kind + '_importance'] = score(D)
    if kind == 'blank': out['blank_D'] = D
ms = np.mean([(lambda b: (np.add.reduce([np.pad(b[j], (starts[j], T - starts[j] - WIN)) for j in range(len(starts))]) / cnt))(margins(n, f0, np.arange(len(starts)))) for n in nets], axis=0)
t1 = time.time(); rnd = DilatedCNN(cin=4, seed=99); D, _ = run(rnd, 'blank'); out['blank_untrained_importance'] = score(D); print(f'untrained: {time.time() - t1:.0f}s', flush=True)
np.savez_compressed(os.path.join(HERE, 'trigger_map.npz'), pos=np.array(pos), W=W, p=p, margin=ms, filled=filled, voiced=v, sung=sung, label=old['label'], moving=old['moving'], **{k: x for k, x in out.items()})
print(f'done in {time.time() - t0:.0f}s', flush=True)
