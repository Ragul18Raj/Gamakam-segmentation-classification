"""Draws the trigger map from trigger_map.npz and applies the reading rule at the end of PLAN.md. Writes trigger_map.png, trigger_map.json."""
import json, os, numpy as np
from scipy.signal import find_peaks, medfilt
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
HERE = os.path.dirname(os.path.abspath(__file__)); z = np.load(os.path.join(HERE, 'trigger_map.npz')); pos, W, p, ms, x, sung, lab, mv = z['pos'], int(z['W']), z['p'], z['margin'], z['filled'], z['sung'], z['label'].astype(bool), z['moving']; T = len(p)
def to_frames(imp):
    s = np.zeros(T); n = np.zeros(T)
    for a, q in zip(pos, imp): s[a:a + W] += q; n[a:a + W] += 1
    o = np.full(T, np.nan); k = (n > 0) & sung; o[k] = s[k] / n[k]; return o
def runs(b):
    e = np.flatnonzero(np.diff(np.r_[0, b.astype(int), 0])); return list(zip(e[::2], e[1::2]))
ib, fl, un = z['blank_importance'], z['flatten_importance'], z['blank_untrained_importance']; I = to_frames(ib)
cc = lambda a, b: float(np.corrcoef(a, b)[0, 1])
sb = z['blank_seed_importance']; sf = z['flatten_seed_importance']; pairs = ((0, 1), (0, 2), (1, 2))
rng_ = np.zeros(T); 
for i in range(T): rng_[i] = x[max(0, i - 10):i + 10].max() - x[max(0, i - 10):i + 10].min()
piece_rng = np.array([rng_[a:a + W].mean() for a in pos])
tall = np.zeros(T, bool); some = np.zeros(T, bool); pk_t = []; pk_s = []
for a, b in runs(sung):
    if b - a < 10: continue
    c = medfilt(x[a:b], 5); pk, pr = find_peaks(c, prominence=100)
    for q, h in zip(pk, pr['prominences']):
        (some)[max(a, a + q - 10):a + q + 11] = True
        if h >= 200: tall[max(a, a + q - 10):a + q + 11] = True; pk_t.append(a + q)
        else: pk_s.append(a + q)
ok = ~np.isnan(I); P = np.maximum(I, 0)
def share(m): return dict(seconds=round(float((m & ok).sum() / 100), 2), share_time=round(float((m & ok).sum() / ok.sum()), 3), share_positive=round(float(P[m & ok].sum() / P[ok].sum()), 3), mean_importance=round(float(I[m & ok].mean()), 3))
cat = dict(tall_peak=tall, smaller_peak=some & ~tall, moving_not_peak=mv & ~some, steady=~mv & ~some)
D = z['blank_D']; prof = {}; centre = pos + W // 2; dist = np.arange(-150, 151, 10); pv = []
for d in dist:
    t = centre + d; k = (t >= 0) & (t < T); vals = np.abs(D[np.flatnonzero(k), t[k]])[sung[t[k]]]; pv.append(float(vals.mean()))
pv = np.array(pv); half = pv >= pv.max() / 2; res = dict(pieces=len(pos), seed_corr_blank=round(float(np.mean([cc(sb[i], sb[j]) for i, j in pairs])), 3), seed_corr_flatten=round(float(np.mean([cc(sf[i], sf[j]) for i, j in pairs])), 3), untrained_corr=round(cc(ib, un), 3), blank_vs_flatten=round(cc(ib, fl), 3), corr_with_local_range=round(cc(ib, piece_rng), 3),
    flatten_corr_with_local_range=round(cc(fl, piece_rng), 3), categories={k: share(m) for k, m in cat.items()}, tall_enrichment=round(share(tall)['share_positive'] / share(tall)['share_time'], 2), tall_peaks=len(pk_t), smaller_peaks=len(pk_s), reach_half_height_frames=[int(dist[half][0]), int(dist[half][-1])], reach_profile=dict(zip(map(int, dist), np.round(pv, 3).tolist())),
    importance_quantiles=np.round(np.nanpercentile(I, [5, 25, 50, 75, 95]), 3).tolist(), share_pieces_positive=round(float((ib > 0).mean()), 3), margin_sung_q=np.round(np.percentile(ms[sung], [5, 50, 95]), 2).tolist())
json.dump(res, open(os.path.join(HERE, 'trigger_map.json'), 'w'), indent=1); print(json.dumps(res, indent=1))
tm = np.arange(T) / 100; fig, ax = plt.subplots(3, 1, figsize=(15, 8), sharex=True, gridspec_kw=dict(height_ratios=[3.2, 1.1, 1.1]))
lim = np.nanpercentile(np.abs(I), 98)
for a, b in runs(ok):
    pts = np.column_stack([tm[a:b], x[a:b]]); seg = np.stack([pts[:-1], pts[1:]], axis=1); lc = LineCollection(seg, cmap='coolwarm', norm=plt.Normalize(-lim, lim), linewidths=2.4); lc.set_array(I[a:b - 1]); ax[0].add_collection(lc)
for a, b in runs(sung & ~ok): ax[0].plot(tm[a:b], x[a:b], color='0.75', lw=1)
ax[0].plot(np.array(pk_t) / 100, x[pk_t] + 70, 'v', color='k', ms=5, label='tall peak (200 cents or more, automatic)')
ax[0].set_ylim(-1500, 1300); ax[0].set_ylabel('cents above 327 Hz'); ax[0].legend(loc='upper right', fontsize=8); ax[0].set_title('kalyani2 #03: red = erasing this part lowers the kampita output, blue = erasing it raises the output (model behaviour, not proof of kampita)', fontsize=10)
cb = fig.colorbar(plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(-lim, lim)), ax=ax, fraction=0.012, pad=0.01); cb.set_label('push towards kampita')
ax[1].plot(tm, np.where(sung, ms, np.nan), color='#b5541c', lw=1); ax[1].axhline(0, color='0.6', lw=0.7); ax[1].set_ylabel('raw output')
ax[2].plot(tm, np.where(sung, p, np.nan), color='#555', lw=1); ax[2].axhline(0.5, color='0.6', lw=0.7); ax[2].set_ylim(0, 1); ax[2].set_ylabel('score'); ax[2].set_xlabel('seconds'); ax[0].set_xlim(5.5, 32); ax[2].xaxis.set_major_locator(plt.MultipleLocator(2))
for a_ in ax: a_.grid(axis='x', color='0.92')
fig.savefig(os.path.join(HERE, 'trigger_map.png'), dpi=115, bbox_inches='tight')
