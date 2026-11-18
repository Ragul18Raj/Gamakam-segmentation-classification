"""Added after seeing the detector output. The model scored silence it had never been shown, and its long events ran across breath gaps,
so this limits marks to real sung phrases and lists them for checking by ear. Model scores are unchanged. Writes phrases.csv, phrase_stats.json, timeline_*.png."""
import csv, glob, json, os, pickle, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); MIN_RUN = 30; LOW = -2400   # a phrase lasts at least 0.3 s and sits above about 82 Hz
def runs(b):
    e = np.flatnonzero(np.diff(np.r_[0, b.astype(int), 0])); return list(zip(e[::2], e[1::2]))
rows = []; st = {}
for fn in sorted(glob.glob(os.path.join(HERE, 'scores_*.npz'))):
    z = np.load(fn); name = os.path.basename(fn)[7:-4]; p, l, x, v, mv = z['p'], z['label'].astype(bool), z['filled'], z['voiced'] > 0, z['moving']
    sung = np.zeros(len(p), bool)
    for a, b in runs(v):
        if b - a >= MIN_RUN and np.median(x[a:b]) > LOW:
            k = x[a:b] > LOW; sung[a:b] = k
    for a, b in runs(sung):
        c = x[a:b]; rows.append(dict(recording=name, start_s=a / 100, end_s=b / 100, length_s=round((b - a) / 100, 2), share_marked=round(float(l[a:b].mean()), 2), share_moving_rule=round(float(mv[a:b].mean()), 2), pitch_range_cents=round(float(np.percentile(c, 95) - np.percentile(c, 5)))))
    nm = sung & ~mv; st[name] = dict(sung_s=float(sung.sum() / 100), marked_share_of_sung=float(l[sung].mean()), silence_s=float((~v).sum() / 100), marked_share_of_silence=float(l[~v].mean()) if (~v).any() else None,
                                     marked_when_moving=float(l[sung & mv].mean()), marked_when_steady=float(l[nm].mean()) if nm.any() else None, steady_s=float(nm.sum() / 100), phrases=len(runs(sung)))
    t = np.arange(len(p)) / 100; fig, ax = plt.subplots(2, 1, figsize=(max(14, len(p) / 100 * 0.42), 5.2), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    ax[0].plot(t, np.where(sung, x, np.nan), color='#1f4e79', lw=0.9)
    for a, b in runs(l & sung): ax[0].axvspan(a / 100, b / 100, color='#fde0a0', zorder=0)
    ax[0].set_ylim(-1800, 1300); ax[0].set_ylabel('cents above 327 Hz'); ax[0].set_title(f'{name}: pitch of sung phrases, shaded = marked by the plavita-only model (unverified, no labels exist)', fontsize=10)
    ax[1].plot(t, p, color='#b5541c', lw=0.9); ax[1].axhline(0.5, color='0.6', lw=0.7); ax[1].set_ylim(0, 1); ax[1].set_ylabel('score'); ax[1].set_xlabel('seconds'); ax[0].set_xlim(0, t[-1]); ax[1].xaxis.set_major_locator(plt.MultipleLocator(2))
    for a in ax: a.grid(axis='x', color='0.9')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, f'timeline_{name}.png'), dpi=110); plt.close(fig)
with open(os.path.join(HERE, 'phrases.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
json.dump(st, open(os.path.join(HERE, 'phrase_stats.json'), 'w'), indent=1)
for k, s in st.items(): print(k, {a: round(b, 2) if b is not None else None for a, b in s.items()})
tot = sum(s['sung_s'] for s in st.values()); print('all: marked share of sung', round(sum(s['sung_s'] * s['marked_share_of_sung'] for s in st.values()) / tot, 3), 'sung s', round(tot, 1))
sd = sum(s['steady_s'] for s in st.values()); print('steady s', round(sd, 1), 'marked when steady', round(sum(s['steady_s'] * s['marked_when_steady'] for s in st.values()) / sd, 3), 'moving s', round(tot - sd, 1), 'marked when moving', round(sum((s['sung_s'] - s['steady_s']) * s['marked_when_moving'] for s in st.values()) / (tot - sd), 3))
print('phrases', len(rows), 'with share_marked >= 0.5:', sum(r['share_marked'] >= 0.5 for r in rows))
print(json.dumps({'hist_share_marked': np.histogram([r['share_marked'] for r in rows], bins=[0, .25, .5, .75, 1.01])[0].tolist()}))
c = json.load(open(os.path.join(HERE, 'checks.json'))); h = c['heldout_flagged_voiced_mean']; o = c['other_veena']
print('heldout pooled', {k: round(v, 3) for k, v in c['heldout_pooled'].items()}, 'heldout flagged mean', round(h, 3))
for k in ('jaru', 'ravai', 'pratyagata'): print(k, round(o[k]['flagged_voiced_mean'], 3), 'ratio', round(o[k]['flagged_voiced_mean'] / h, 3))
m = np.mean([o[k]['flagged_voiced_mean'] for k in ('jaru', 'ravai', 'pratyagata')]); print('others mean', round(m, 3), 'ratio', round(m / h, 3))
D = pickle.load(open(os.path.join(HERE, '..', 'methods_explore', 'curves.pkl'), 'rb')); P = [d for d in D if 'plav' in d['file'].lower() and 'todi#03Guitar' not in d['file']]
y = []
for d in P:
    n = len(d['raw_masked']); a, b = d['crop_range']; lab = np.zeros(n, int); lab[a:b] = 1; w = np.ones(n, bool)
    for e in (a, b): w[max(0, e - 5):min(n, e + 6)] = False
    y.append(lab[w])
y = np.concatenate(y); print('label share', round(y.mean(), 3), 'always-yes F1', round(2 * y.mean() / (1 + y.mean()), 3))
