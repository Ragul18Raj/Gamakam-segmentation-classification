"""Added after the user said most high peaks in the Kalyani audio are kampita. Compares the pitch peaks in the plavita training clips with those in Kalyani,
and asks whether the model scored the peaks differently from other movement. 'Peak' is an automatic definition (prominence at least 100 cents), not an expert mark.
Writes peak_compare.json and peak_compare.png."""
import glob, json, os, pickle, numpy as np
from scipy.signal import find_peaks, medfilt
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); PROM = 100
def runs(b):
    e = np.flatnonzero(np.diff(np.r_[0, b.astype(int), 0])); return list(zip(e[::2], e[1::2]))
def analyse(segs):
    """segs: list of cents arrays, one per continuous stretch of pitch."""
    H, W, RISE, FALL, GAP, slow, flat, secs = [], [], [], [], [], [], [], 0
    for c in segs:
        c = medfilt(c, 5); secs += len(c) / 100
        v = np.abs(np.gradient(c)); flat.append(v < 0.5)                 # under 50 cents per second: pitch nearly still
        if len(c) >= 100:
            k = np.array([np.median(c[max(0, i - 50):i + 50]) for i in range(0, len(c), 10)]); slow.append(k.max() - k.min())
        pk, pr = find_peaks(c, prominence=PROM, width=1, rel_height=0.5)
        H += list(pr['prominences']); W += list(pr['widths'] * 10)
        for p, lb, rb in zip(pk, pr['left_bases'], pr['right_bases']):
            RISE.append((c[p] - c[lb]) / max(p - lb, 1) * 100); FALL.append((c[p] - c[rb]) / max(rb - p, 1) * 100)
        d = np.diff(pk); GAP += list(d[d < 100] * 10)
    q = lambda x: [round(float(np.percentile(x, a)), 0) for a in (25, 50, 75)] if len(x) else None
    return dict(seconds=round(secs, 1), peaks=len(H), peaks_per_second=round(len(H) / secs, 2), prominence_cents_q=q(H), width_ms_q=q(W), rise_cents_per_s_q=q(RISE), fall_cents_per_s_q=q(FALL),
                peak_to_peak_ms_q=q(GAP), swings_per_s_from_median_gap=round(1000 / np.median(GAP), 2) if GAP else None, slow_drift_cents_q=q(slow), share_nearly_still=round(float(np.mean(np.concatenate(flat))), 3))
D = pickle.load(open(os.path.join(HERE, '..', 'methods_explore', 'curves.pkl'), 'rb')); P = [d for d in D if 'plav' in d['file'].lower() and 'todi#03Guitar' not in d['file']]
plav = [np.asarray(d['crop'], float) for d in P]
kal = []; store = {}; sc = {'peak': [], 'moving_not_peak': []}; marked = {'peak': [], 'moving_not_peak': []}
for fn in sorted(glob.glob(os.path.join(HERE, 'scores_*.npz'))):
    z = np.load(fn); name = os.path.basename(fn)[7:-4]; x, v, p, l, mv = z['filled'], z['voiced'] > 0, z['p'], z['label'], z['moving']; store[name] = (x, v, p)
    for a, b in runs(v):
        if b - a >= 30 and np.median(x[a:b]) > -2400:
            k = x[a:b] > -2400
            for s, e in runs(k):
                if e - s < 10: continue
                c = medfilt(x[a + s:a + e], 5); kal.append(x[a + s:a + e]); pk, pr = find_peaks(c, prominence=PROM, width=1, rel_height=0.5); o = a + s
                m = np.zeros(e - s, bool)
                for q, wd in zip(pk, pr['widths']): w = max(3, int(wd / 2)); m[max(0, q - w):q + w + 1] = True
                sc['peak'] += list(p[o:o + e - s][m]); marked['peak'] += list(l[o:o + e - s][m]); rest = mv[o:o + e - s] & ~m
                sc['moving_not_peak'] += list(p[o:o + e - s][rest]); marked['moving_not_peak'] += list(l[o:o + e - s][rest])
res = dict(plavita=analyse(plav), kalyani=analyse(kal), model_on_kalyani={k: dict(frames=len(sc[k]), mean_score=round(float(np.mean(sc[k])), 3), share_marked=round(float(np.mean(marked[k])), 3)) for k in sc})
json.dump(res, open(os.path.join(HERE, 'peak_compare.json'), 'w'), indent=1)
for k, a in res.items(): print(k, json.dumps(a))
fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharey=False)
for a_, d in zip(ax[:2], (P[0], P[8])):
    c = np.asarray(d['crop'], float); t = np.arange(len(c)) / 100; a_.plot(t, c - np.median(c), color='#b5541c', lw=1.1); a_.set_xlim(0, 6); a_.set_ylim(-700, 700); a_.set_ylabel('cents, centred'); a_.set_title('Veena plavita: ' + d['file'].split('Guitar')[0], fontsize=9)
x, v, p = store['kalyani3_02']; a, b = 136, 361 + 375; c = np.where(v[a:b] & (x[a:b] > -2400), x[a:b], np.nan); t = np.arange(len(c)) / 100; ax[2].plot(t, c - np.nanmedian(c), color='#1f4e79', lw=1.1); ax[2].set_xlim(0, 6); ax[2].set_ylim(-700, 700); ax[2].set_ylabel('cents, centred'); ax[2].set_xlabel('seconds (same scale in all three)'); ax[2].set_title('Kalyani singing: kalyani3 #02, from 1.36 s', fontsize=9)
for a_ in ax: a_.grid(color='0.92')
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'peak_compare.png'), dpi=110)

# The user's remark was about the HIGH peaks, so repeat for tall peaks only: prominence at least 200 cents, a threshold chosen after seeing the numbers above.
def tall(segs, labs=None):
    H, W, G, M = [], [], [], []
    for i, c in enumerate(segs):
        c = medfilt(c, 5); pk, pr = find_peaks(c, prominence=200, width=1, rel_height=0.5); H += list(pr['prominences']); W += list(pr['widths'] * 10); d = np.diff(pk); G += list(d[d < 100] * 10)
        if labs is not None: M += [bool(labs[i][q]) for q in pk]
    q = lambda x: [round(float(np.percentile(x, a)), 0) for a in (25, 50, 75)] if len(x) else None
    return dict(peaks=len(H), per_second=round(len(H) / sum(len(c) / 100 for c in segs), 2), prominence_q=q(H), width_ms_q=q(W), gap_ms_q=q(G), gaps=len(G), share_of_peaks_marked=round(float(np.mean(M)), 3) if M else None)
ks, ls = [], []
for fn in sorted(glob.glob(os.path.join(HERE, 'scores_*.npz'))):
    z = np.load(fn); x, v, l = z['filled'], z['voiced'] > 0, z['label']
    for a, b in runs(v):
        if b - a >= 30 and np.median(x[a:b]) > -2400:
            for s, e in runs(x[a:b] > -2400):
                if e - s >= 10: ks.append(x[a + s:a + e]); ls.append(l[a + s:a + e])
res['tall_peaks_200c'] = dict(plavita=tall(plav), kalyani=tall(ks, ls)); json.dump(res, open(os.path.join(HERE, 'peak_compare.json'), 'w'), indent=1); print(json.dumps(res['tall_peaks_200c']))
