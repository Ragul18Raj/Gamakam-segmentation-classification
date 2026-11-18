"""Added after seeing trigger_map.png, where red seemed to sit on rising pitch and blue on falling pitch. The pattern was spotted on kalyani2 #03, so the other two
recordings are the real test: for it to hold, rising pieces must score above zero and above falling pieces on both of them. Also repeats the table for the flatten map. Writes trigger_direction.json."""
import json, os, time
HERE = os.path.dirname(os.path.abspath(__file__)); __file__ = os.path.join(HERE, 'detector.py')
exec(open(__file__, encoding='utf-8').read().split('t0 = time.time(); res = {}')[0])
W = 10; STEP = 5; REACH = 124; LOW = -2400; SLOPE = 30.0     # 30 cents across a 0.1 s piece, 300 cents per second
nets = [DilatedCNN(cin=4, seed=s) for s in SEEDS]
for n, q in zip(nets, pickle.load(open(os.path.join(HERE, 'nets_all14.pkl'), 'rb'))): n.params = q
def sung_mask(filled, vb):
    s = np.zeros(len(vb), bool)
    for a, b in (lambda e: zip(e[::2], e[1::2]))(np.flatnonzero(np.diff(np.r_[0, vb.astype(int), 0]))):
        if b - a >= 30 and np.median(filled[a:b]) > LOW: s[a:b] = filled[a:b] > LOW
    return s
def blank_map(name):
    cents, conf = pitch(os.path.join(HERE, 'crepe', name + '.npy')); f0, filled, v = feats(cents, conf); T = f0.shape[1]; vb = v > 0; sung = sung_mask(filled, vb)
    st = list(range(0, T - WIN + 1, 10)); st += [T - WIN] if st[-1] != T - WIN else []; st = np.array(st); cnt = np.zeros(T)
    for s in st: cnt[s:s + WIN] += 1
    def mg(net, f, idx):
        o = np.zeros((len(idx), WIN), np.float32)
        for i in range(0, len(idx), 64):
            k = idx[i:i + 64]; lg = net.forward(np.stack([f[:, st[j]:st[j] + WIN] for j in k]))[0]; o[i:i + 64] = lg[:, 1] - lg[:, 0]
        return o
    pos = [a for a in range(0, T - W + 1, STEP) if sung[a:a + W].all()]; imp = np.zeros((3, len(pos)))
    for si, net in enumerate(nets):
        base = mg(net, f0, np.arange(len(st)))
        for i, a in enumerate(pos):
            cf = conf.copy(); cf[a:a + W] = 0; f = feats(cents, cf)[0]; ch = np.flatnonzero((f != f0).any(axis=0)); idx = np.flatnonzero((st < ch[-1] + 1) & (st + WIN > ch[0])); new = mg(net, f, idx); d = np.zeros(T)
            for q, j in enumerate(idx): d[st[j]:st[j] + WIN] += base[j] - new[q]
            d /= cnt; lo, hi = max(0, a + 5 - REACH), a + 5 + REACH + 1; imp[si, i] = d[lo:hi][sung[lo:hi]].mean()
    return np.array(pos), imp, filled
def table(pos, imp, x):
    y = imp.mean(axis=0); s = np.array([np.median(x[a + 5:a + 10]) - np.median(x[a:a + 5]) for a in pos]) * 2.0; g = dict(rising=s >= SLOPE, falling=s <= -SLOPE, flat=np.abs(s) < SLOPE)
    o = {k: dict(pieces=int(m.sum()), mean=round(float(y[m].mean()), 3), share_positive=round(float((y[m] > 0).mean()), 3), per_seed_mean=[round(float(imp[i][m].mean()), 3) for i in range(3)]) for k, m in g.items()}
    o['corr_with_signed_slope'] = round(float(np.corrcoef(y, np.clip(s, -150, 150))[0, 1]), 3); return o
out = {}
for name in ('kalyani3 #02', 'kalyani #02'):
    t = time.time(); pos, imp, x = blank_map(name); out[name] = table(pos, imp, x); print(name, round(time.time() - t), 's', json.dumps(out[name]), flush=True)
z = np.load(os.path.join(HERE, 'trigger_map.npz')); keep = np.array([z['sung'][a:a + W].all() for a in z['pos']]); pos = z['pos'][keep]; x = z['filled']
out['kalyani2 #03 (pattern spotted here), blank'] = table(pos, z['blank_seed_importance'][:, keep], x); out['kalyani2 #03, flatten check'] = table(pos, z['flatten_seed_importance'][:, keep], x); out['kalyani2 #03, untrained model, blank'] = table(pos, np.repeat(z['blank_untrained_importance'][keep][None], 3, axis=0), x)
for k in list(out)[2:]: print(k, json.dumps(out[k]))
D = pickle.load(open(os.path.join(HERE, '..', 'methods_explore', 'curves.pkl'), 'rb')); r = {'rising': 0, 'falling': 0, 'flat': 0}
for d in (d for d in D if 'plav' in d['file'].lower() and 'todi#03Guitar' not in d['file']):
    c = np.asarray(d['crop'], float); s = (np.array([np.median(c[a + 5:a + 10]) - np.median(c[a:a + 5]) for a in range(0, len(c) - W, STEP)])) * 2.0
    r['rising'] += int((s >= SLOPE).sum()); r['falling'] += int((s <= -SLOPE).sum()); r['flat'] += int((np.abs(s) < SLOPE).sum())
print('plavita training pieces', r); out['plavita_training_pieces'] = r; json.dump(out, open(os.path.join(HERE, 'trigger_direction.json'), 'w'), indent=1)
