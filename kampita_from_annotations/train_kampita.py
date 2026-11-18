"""Train a kampita detector from the manual annotations (kampita_annotations_2026-11-13.xlsx)
and run it on the Kalyani recordings.

Input : crepe_output/<clip>.npy  (CREPE activations, 10 ms hop, already computed)
        veena_stage1/plavita_kalyani/crepe/<kalyani>.npy
Labels: Start/End (sec) from the Excel = kampita; every other frame = not kampita.
        Other gamaka clips (jaru, nokku, ravai, ...) are added as negatives.
        Kampita (leena) clips and the un-annotated todi#03 take are left out (they contain kampita but no timestamps).
Eval  : leave-one-recording-out (all .1/.2/.3 takes of a recording held out together).
Output: kalyani_kampita_segments.csv, loro_results.txt
"""
import glob, os, re, numpy as np, pandas as pd
from scipy.ndimage import median_filter
from sklearn.ensemble import RandomForestClassifier

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.dirname(os.path.abspath(__file__))
HOP = 0.01; WIN = 30          # 30 frames = 300 ms context window
CONF_MIN = 0.3

def to_cents(act):
    c = 1997.3794084376191 + 20 * np.arange(360); k = act.argmax(1)
    idx = np.clip(k[:, None] + np.arange(-4, 5), 0, 359)
    w = np.take_along_axis(act, idx, 1)
    return (w * c[idx]).sum(1) / (w.sum(1) + 1e-9), act.max(1)

def features(act):
    cents, conf = to_cents(act)
    voiced = conf >= CONF_MIN
    # fill unvoiced with nearest voiced so windows stay continuous
    if voiced.any():
        vi = np.where(voiced)[0]; cents = np.interp(np.arange(len(cents)), vi, cents[vi])
    d = np.gradient(cents); h = WIN // 2
    cp = np.pad(cents, h, mode='edge'); dp = np.pad(d, h, mode='edge'); vp = np.pad(conf, h, mode='edge')
    W = np.lib.stride_tricks.sliding_window_view(cp, WIN)[:len(cents)]
    D = np.lib.stride_tricks.sliding_window_view(dp, WIN)[:len(cents)]
    C = np.lib.stride_tricks.sliding_window_view(vp, WIN)[:len(cents)]
    w = W - W.mean(1, keepdims=True)                 # tonic-free: only local movement
    x = np.arange(WIN) - (WIN - 1) / 2
    slope = (w * x).sum(1, keepdims=True) / (x ** 2).sum()
    r = w - slope * x                                # detrended = oscillation part
    sc = (np.diff(np.sign(D), axis=1) != 0).sum(1)
    spec = np.abs(np.fft.rfft(r, 64, axis=1)); fr = np.fft.rfftfreq(64, HOP)
    band = spec[:, (fr >= 3) & (fr <= 12)].sum(1) / (spec[:, fr > 0].sum(1) + 1e-9)
    peak = fr[1:][spec[:, 1:].argmax(1)]
    return np.column_stack([r.std(1), np.ptp(r, 1), w.std(1), np.abs(D).mean(1), sc, band, peak,
                            C.mean(1), (C >= CONF_MIN).mean(1)])

# ---- labels from Excel
ann = pd.read_excel(os.path.join(ROOT, 'kampita_annotations_2026-11-13.xlsx'), sheet_name='All Songs')
ann['Gamaka Type'] = ann['Gamaka Type'].str.strip().str.lower()
spans = ann.groupby('Song Name')[['Start (sec)', 'End (sec)']].apply(lambda g: g.values.tolist()).to_dict()

def group_of(name):
    m = re.match(r'(.*?#\d+)', name)
    return m.group(1) if m else re.sub(r'[_ ]*[!\d]*Guitar.*', '', name)

X, y, g = [], [], []
for f in sorted(glob.glob(os.path.join(ROOT, 'crepe_output', '*.npy'))):
    name = os.path.basename(f)[:-4]
    if name.lower().startswith('kampita') and name not in spans:
        continue                                   # kampita without timestamps -> skip
    if name.startswith('Sphutitam#01'):
        continue                                   # one very long file, would dominate
    Fe = features(np.load(f)); lab = np.zeros(len(Fe), int)
    for s, e in spans.get(name, []):
        lab[int(round(s / HOP)):int(round(e / HOP)) + 1] = 1
    X.append(Fe); y.append(lab); g += [group_of(name)] * len(Fe)
X = np.vstack(X); y = np.concatenate(y); g = np.array(g)
print('frames', len(y), 'kampita frames', y.sum(), 'recordings', len(set(g)))

def segments(prob, thr=0.5, min_len=0.25):
    lab = median_filter((prob >= thr).astype(int), 15)
    segs, t = [], 0
    while t < len(lab):
        if lab[t]:
            s = t
            while t < len(lab) and lab[t]: t += 1
            if (t - s) * HOP >= min_len: segs.append((s * HOP, t * HOP))
        t += 1
    return segs

def mk(): return RandomForestClassifier(300, min_samples_leaf=5, class_weight='balanced', n_jobs=-1, max_samples=0.5, random_state=0)

# ---- leave-one-recording-out on the kampita recordings
lines = []
for kg in sorted({group_of(n) for n in spans}):
    tr, te = g != kg, g == kg
    p = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    pred = (median_filter((p >= .5).astype(int), 15))
    tp = ((pred == 1) & (y[te] == 1)).sum(); fp = ((pred == 1) & (y[te] == 0)).sum(); fn = ((pred == 0) & (y[te] == 1)).sum()
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    lines.append(f'{kg:35s} frame P={P:.2f} R={R:.2f} F1={2*P*R/max(P+R,1e-9):.2f}')
# false alarms on non-kampita gamakas (held out per recording would be slow; report in-sample-free via 5 groups)
neg_groups = sorted({x for x in set(g) if x not in {group_of(n) for n in spans}})
fa = []
for ng in neg_groups:
    tr, te = g != ng, g == ng
    p = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    fa.append((ng, float((median_filter((p >= .5).astype(int), 15)).mean())))
lines.append('\nfraction of frames wrongly called kampita, per non-kampita gamaka (held out):')
lines += [f'  {n:40s} {v:.2f}' for n, v in fa]
open(os.path.join(OUT, 'loro_results.txt'), 'w').write('\n'.join(lines)); print('\n'.join(lines))

# ---- final model on everything, run on Kalyani
clf = mk().fit(X, y); rows = []
for f in sorted(glob.glob(os.path.join(ROOT, 'veena_stage1', 'plavita_kalyani', 'crepe', '*.npy'))):
    p = clf.predict_proba(features(np.load(f)))[:, 1]
    for i, (s, e) in enumerate(segments(p), 1):
        rows.append([os.path.basename(f)[:-4], i, round(s, 2), round(e, 2), round(e - s, 2)])
    print(os.path.basename(f), 'kampita count', len(segments(p)))
pd.DataFrame(rows, columns=['Recording', 'S.No', 'Start (sec)', 'End (sec)', 'Duration (sec)']).to_csv(
    os.path.join(OUT, 'kalyani_kampita_segments.csv'), index=False)
