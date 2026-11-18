"""Fourier scaling test on the annotated kampitas, all treated as one raga.
shape = peak-aligned log-spectrum (speed removed) + normalised energy (size removed)  [same as fourier_demo/fourier_scaling_solution.py]
1) measure speed (oscillation Hz) and size (cents) of each annotated kampita
2) scale them (faster/slower, wider/narrower) and check the shape still matches the kampita template, not other gamakas
3) run the shape-only detector on Kalyani and compare speed/size of what it finds against the veena kampitas
"""
import glob, os, re, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.signal import resample
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
ROOT = os.path.abspath('..'); FS = 100
NFFT = 2048; F = np.fft.rfftfreq(NFFT, 1 / FS)
G = np.exp(np.linspace(np.log(0.5), np.log(30), 160))
RATIOS = [0.5, 0.71, 1.41, 2.0, 3.0, 4.0]
def fourier_parts(c):
    x = c - c.mean(); size = np.sqrt(np.mean(x ** 2)) + 1e-6
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)), NFFT))
    spec = np.interp(G, F, X); spec = spec / (np.linalg.norm(spec) + 1e-12)
    k = int(np.argmax(spec[G > 1.0])) + int(np.sum(G <= 1.0)); peak_hz = G[k]; ls = np.log(spec + 1e-6)
    shape = [np.interp(np.log(peak_hz * r), np.log(G), ls) - ls[k] for r in RATIOS]
    return np.array(shape + [spec[k] / (spec.mean() + 1e-12), np.polyfit(np.log(G[G > 1]), ls[G > 1], 1)[0]]), peak_hz, size
def cents_of(act, thr=0.3):
    c = 1997.3794 + 20 * np.arange(360); k = act.argmax(1)
    idx = np.clip(k[:, None] + np.arange(-4, 5), 0, 359); w = np.take_along_axis(act, idx, 1)
    ce = (w * c[idx]).sum(1) / (w.sum(1) + 1e-9); v = act.max(1) >= thr
    if v.any(): ce = np.interp(np.arange(len(ce)), np.where(v)[0], ce[v])
    return ce, v
grp = lambda n: (re.match(r'(.*?#\d+)', n) or re.match(r'(.*?)[_ ]*[!\d]*Guitar', n)).group(1)

# ---- positives: annotated kampita spans
ann = pd.read_excel(os.path.join(ROOT, 'kampita_annotations_2026-11-13.xlsx'), sheet_name='All Songs')
pos = []
for _, r in ann.iterrows():
    ce, _ = cents_of(np.load(os.path.join(ROOT, 'crepe_output', r['Song Name'] + '.npy')))
    pos.append((r['Song Name'], ce[int(r['Start (sec)'] * FS):int(r['End (sec)'] * FS)]))
# ---- negatives: voiced part of every other gamaka clip, cut into ~1 s pieces
neg = []
for f in sorted(glob.glob(os.path.join(ROOT, 'crepe_output', '*.npy'))):
    n = os.path.basename(f)[:-4]
    if n.lower().startswith('kampita') or n.startswith('Sphutitam#01'): continue
    ce, v = cents_of(np.load(f)); vi = np.where(v)[0]
    if len(vi) < 40: continue
    ce = ce[vi[0]:vi[-1]]
    for s in range(0, max(1, len(ce) - 60), 80): neg.append((n, ce[s:s + 100]))
neg = [x for x in neg if len(x[1]) >= 40]

P = [fourier_parts(c) for _, c in pos]; N = [fourier_parts(c) for _, c in neg]
lines = ['=== 1) annotated kampitas, treated as one raga ===']
for (n, c), (_, hz, sz) in zip(pos, P): lines.append(f'  {n[:45]:45s} speed {hz:4.1f} Hz   size {sz:5.0f} cents RMS   dur {len(c)/FS:.2f}s')
hzs = np.array([p[1] for p in P]); szs = np.array([p[2] for p in P])
lines.append(f'  POOLED: speed {np.median(hzs):.1f} Hz (range {hzs.min():.1f}-{hzs.max():.1f}),  size {np.median(szs):.0f} cents (range {szs.min():.0f}-{szs.max():.0f})')

# ---- 2) scaling test: shape-only model trained on unscaled data, tested on scaled kampitas
Xp = np.array([p[0] for p in P]); Xn = np.array([q[0] for q in N])
X = np.vstack([Xp, Xn]); y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
g = np.array([grp(n) for n, _ in pos] + [grp(n) for n, _ in neg])
def scaled(c, s, w):                              # s>1 = faster, w = width
    m = c.mean(); return m + (resample(c - m, max(8, int(len(c) / s)))) * w
lines.append('\n=== 2) scaling test (model never sees the held-out recording or the scaled versions) ===')
lines.append(f'  {"condition":16s} kampita still recognised   shape distance to template (kampita template vs other gamakas)')
conds = [('original', 1, 1), ('speed x0.5', .5, 1), ('speed x0.8', .8, 1), ('speed x1.5', 1.5, 1), ('speed x2.5', 2.5, 1),
         ('width x0.5', 1, .5), ('width x2', 1, 2), ('Kalyani-like', 2.5, .5)]
sc = StandardScaler().fit(X); tmpl = sc.transform(Xp).mean(0); ntm = sc.transform(Xn).mean(0)
for name, s, w in conds:
    hit, dk, dn = [], [], []
    for kg in sorted(set(grp(n) for n, _ in pos)):
        tr = g != kg; m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced')).fit(X[tr], y[tr])
        for (n, c) in pos:
            if grp(n) != kg: continue
            f = fourier_parts(scaled(c, s, w))[0]; hit.append(m.predict([f])[0])
            z = sc.transform([f])[0]; dk.append(np.linalg.norm(z - tmpl)); dn.append(np.linalg.norm(z - ntm))
    lines.append(f'  {name:16s} {np.mean(hit)*100:5.0f}% of 14              {np.mean(dk):.2f} vs {np.mean(dn):.2f}  -> {"matches kampita" if np.mean(dk) < np.mean(dn) else "closer to OTHER gamakas"}')
# false alarms on other gamakas (held out by recording)
fa = []
for ng in sorted(set(grp(n) for n, _ in neg)):
    tr = g != ng; m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced')).fit(X[tr], y[tr])
    fa += list(m.predict(Xn[np.array([grp(n) == ng for n, _ in neg])]))
lines.append(f'  false alarms on other gamakas: {np.mean(fa)*100:.0f}% of {len(fa)} pieces')

# ---- 3) Kalyani: shape-only detector, 0.8 s windows
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced')).fit(X, y)
old = pd.read_csv('kalyani_kampita_segments.csv'); rows = []
files = sorted(glob.glob(os.path.join(ROOT, 'veena_stage1/plavita_kalyani/crepe/*.npy')))
fig, axs = plt.subplots(len(files), 1, figsize=(18, 3.3 * len(files)))
lines.append('\n=== 3) Kalyani with the Fourier shape detector ===')
for ax, f in zip(axs, files):
    act = np.load(f); ce, v = cents_of(act); name = os.path.basename(f)[:-4]; W = 80; hits = np.zeros(len(ce))
    for s in range(0, len(ce) - W, 10):
        if v[s:s + W].mean() < .8: continue
        fe, hz, sz = fourier_parts(ce[s:s + W])
        if m.predict_proba([fe])[0, 1] > .7 and sz > 15: hits[s:s + W] = 1; rows.append([name, s / FS, hz, sz])
    segs = []; t = 0
    while t < len(hits):
        if hits[t]:
            a = t
            while t < len(hits) and hits[t]: t += 1
            segs.append((a / FS, t / FS))
        t += 1
    hz = 10 * 2 ** (ce / 1200); hz[~v] = np.nan; tt = np.arange(len(ce)) / FS
    ax.plot(tt, hz, lw=.6, color='#333')
    for a, b in segs: ax.axvspan(a, b, color='tab:blue', alpha=.25)
    for _, r in old[old.Recording == name].iterrows(): ax.axvspan(r['Start (sec)'], r['End (sec)'], ymin=.9, color='tab:red', alpha=.8)
    ax.set_yscale('log'); ax.set_xlim(0, tt[-1]); ax.set_ylabel('Pitch (Hz)')
    ax.set_title(f'{name}: blue = Fourier shape detector ({len(segs)} regions)   red bar = earlier detector ({(old.Recording==name).sum()})', loc='left')
    lines.append(f'  {name}: {len(segs)} kampita regions (earlier detector: {(old.Recording == name).sum()})')
axs[-1].set_xlabel('Time (s)'); plt.tight_layout(); plt.savefig('kalyani_fourier_vs_old.png', dpi=110)
R = pd.DataFrame(rows, columns=['rec', 't', 'hz', 'size'])
if len(R): lines.append(f'  detected Kalyani windows: speed {R.hz.median():.1f} Hz, size {R["size"].median():.0f} cents  vs veena kampita {np.median(hzs):.1f} Hz, {np.median(szs):.0f} cents')
open('fourier_results.txt', 'w').write('\n'.join(lines)); print('\n'.join(lines))
