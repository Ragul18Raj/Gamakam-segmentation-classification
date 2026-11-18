"""Added because the flatten check turned out too weak: a straight line across a 0.1 s rise is nearly the same curve, so it says little. This asks directly:
if whole peaks are ironed out of the pitch line, does the model stop firing? Writes trigger_iron.json."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); __file__ = os.path.join(HERE, 'detector.py')
exec(open(__file__, encoding='utf-8').read().split('t0 = time.time(); res = {}')[0])
from scipy.signal import find_peaks, medfilt
nets = [DilatedCNN(cin=4, seed=s) for s in SEEDS]
for n, q in zip(nets, pickle.load(open(os.path.join(HERE, 'nets_all14.pkl'), 'rb'))): n.params = q
def edges(b):
    e = np.flatnonzero(np.diff(np.r_[0, b.astype(int), 0])); return list(zip(e[::2], e[1::2]))
def phrases(filled, vb):
    o = []
    for a, b in edges(vb):
        if b - a >= 30 and np.median(filled[a:b]) > -2400:
            for s, e in edges(filled[a:b] > -2400):
                if e - s >= 10: o.append((a + s, a + e))
    return o
def iron(c, ph, prom):
    c = c.copy(); n = 0
    for a, b in ph:
        for _ in range(30):                                  # repeat, so peaks that sit on bigger peaks go too
            x = medfilt(c[a:b], 5); pk, pr = find_peaks(x, prominence=prom, wlen=101)
            if not len(pk): break
            i = int(np.argmax(pr['prominences'])); l, r = pr['left_bases'][i], pr['right_bases'][i]; c[a + l:a + r + 1] = np.linspace(c[a + l], c[a + r], r - l + 1); n += 1
    return c, n
def smooth(c, ph, size):
    c = c.copy()
    for a, b in ph:
        s = size if (b - a) > size else ((b - a) | 1) - 2 if (b - a) > 3 else 1; c[a:b] = median_filter(c[a:b], size=max(1, s), mode='nearest')
    return c
out = {}
for name in ('kalyani2 #03', 'kalyani3 #02', 'kalyani #02'):
    cents, conf = pitch(os.path.join(HERE, 'crepe', name + '.npy')); f0, filled, v = feats(cents, conf); vb = v > 0; ph = phrases(filled, vb); sung = np.zeros(len(vb), bool)
    for a, b in ph: sung[a:b] = True
    base = np.where(vb, filled, cents); o = {}
    def share(c): l = decide(predict(nets, feats(c, conf)[0])); return round(float(l[sung].mean()), 3)
    o['original'] = share(base); a, n = iron(base, ph, 200); o['tall peaks ironed (200 cents or more)'] = dict(marked=share(a), removed=n); a, n = iron(base, ph, 100); o['all peaks ironed (100 cents or more)'] = dict(marked=share(a), removed=n)
    a, n = iron(base, ph, 40); o['also small wobbles ironed (40 cents or more)'] = dict(marked=share(a), removed=n); o['heavy smoothing, 1 s median: slow melody line only'] = share(smooth(base, ph, 101)); y = np.zeros_like(base)
    for s, e in ph: y[s:e] = np.median(base[s:e])
    o['each phrase a single flat note'] = share(np.where(sung, y, base)); out[name] = o; print(name, json.dumps(o), flush=True)
json.dump(out, open(os.path.join(HERE, 'trigger_iron.json'), 'w'), indent=1)
