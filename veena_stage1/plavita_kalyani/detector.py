"""Plavita-only frame detector tested on the Kalyani recordings. Plan: PLAN.md. Exploratory; Kalyani has no labels.
Run after run_crepe_kalyani.py:  python detector.py   (single-threaded BLAS keeps the NumPy CNN fast on Windows)
Writes checks.json, detections.csv, timeline_<recording>.png."""
import csv, glob, json, os, pickle, sys, time
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[v] = '1'
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'sim_code_extracted', 'gamaka_sim')); sys.path.insert(0, os.path.join(ROOT, 'veena_stage1', 'local_run'))
import pipeline as PL
PL.NCLS = 2
from pipeline import DilatedCNN, loss_and_grad, Adam, predict_sliding, postprocess, runs, fill_short_gaps, softmax
from shapes import crepe_activation_to_pitch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

WIN = 300; EPOCHS = 15; BATCHES = 75; BS = 16; STRETCH = (1.0, 0.8, 0.9, 1.1, 1.25); SEEDS = (0, 1, 2); HOLD_WIN = 20; HOLD_C = 30
LBL = {0: 'none', 1: 'kampita'}

def feats(cents, conf):
    """Contour, velocity, voiced, confidence: pipeline.features without sin and cos."""
    v0 = (~np.isnan(cents)) & (conf >= 0.5)
    filled, v = fill_short_gaps(np.nan_to_num(cents), v0)
    contour = np.clip(filled - median_filter(filled, size=51, mode='nearest'), -600, 600) / 100.0
    vel = np.clip(uniform_filter1d(np.gradient(filled), 5) / 10.0, -10, 10)
    f = np.stack([contour, vel, v.astype(float), np.nan_to_num(conf)]); f[:2, ~v] = 0.0
    return f.astype(np.float32), filled, v

def pitch(path):
    hz, conf = crepe_activation_to_pitch(np.load(path))
    return 1200 * np.log2(np.maximum(hz, 1e-3) / 327.0), conf

def labels(T, a, b):
    y = np.zeros(T, np.int64); y[a:b] = 1; w = np.ones(T, np.float32)
    for e in (a, b): w[max(0, e - 5):min(T, e + 6)] = 0.0
    return y, w

def stretched(c, conf, a, b, fac):
    if fac == 1.0: return c, conf, a, b
    n = int(round(len(c) / fac)); t = np.linspace(0, len(c) - 1, n); idx = np.arange(len(c)); nan = np.isnan(c)
    x = np.interp(t, idx[~nan], c[~nan]) if (~nan).sum() > 1 else np.full(n, np.nan)
    x[np.interp(t, idx, nan.astype(float)) > 0.5] = np.nan     # stretching also scales the speed of the movement
    return x, np.interp(t, idx, conf), int(round(a / fac)), int(round(b / fac))

D = pickle.load(open(os.path.join(ROOT, 'veena_stage1', 'methods_explore', 'curves.pkl'), 'rb'))
def load(d):
    c, conf = pitch(os.path.join(ROOT, 'crepe_output', os.path.splitext(d['file'])[0] + '.npy'))
    return c, conf
def branch(d):
    f = d['file']
    return 'bhairiga#03' if 'bhairiga' in f else '#02' if '(plavita)#02' in f else 'todi#01' if 'todi#01' in f else 'todi#03'
PLAV = [d for d in D if 'plav' in d['file'].lower() and 'todi#03Guitar' not in d['file']]
assert len(PLAV) == 14

def prep(clips):
    out = []
    for d in clips:
        c, conf = load(d); a, b = d['crop_range']
        for fac in STRETCH:
            x, cf, a2, b2 = stretched(c, conf, a, b, fac); f, _, _ = feats(x, cf); y, w = labels(f.shape[1], a2, b2); out.append((f, y, w))
    return out

def batch(data, cw, rng):
    X = np.zeros((BS, 4, WIN), np.float32); Y = np.zeros((BS, WIN), np.int64); W = np.zeros((BS, WIN), np.float32)
    for i in range(BS):
        f, y, w = data[rng.integers(len(data))]; T = f.shape[1]
        if T >= WIN:
            s = int(rng.integers(0, T - WIN + 1)); X[i] = f[:, s:s + WIN]; Y[i] = y[s:s + WIN]; W[i] = w[s:s + WIN] * cw[y[s:s + WIN]]
        else:
            s = int(rng.integers(0, WIN - T + 1)); X[i, :, s:s + T] = f; Y[i, s:s + T] = y; W[i, s:s + T] = w * cw[y]    # padding has zero weight
    return X, Y, W

def train(clips, seed):
    data = prep(clips); allY = np.concatenate([y[w > 0] for _, y, w in data]); freq = np.bincount(allY, minlength=2) / len(allY)
    cw = (freq ** -0.5).astype(np.float32); cw /= cw.mean()
    net = DilatedCNN(cin=4, seed=seed); opt = Adam(net.params); rng = np.random.default_rng(seed)
    for ep in range(EPOCHS):
        tot = 0.0
        for _ in range(BATCHES):
            X, Y, W = batch(data, cw, rng); logits, cache = net.forward(X, train=True, rng=rng)
            l, d = loss_and_grad(logits, Y, W); opt.step(net.backward(d, cache)); tot += l
    print(f'  seed {seed}: last epoch loss {tot / BATCHES:.3f}, label share kampita {freq[1]:.2f}', flush=True)
    return net

def predict(nets, f):
    return np.mean([predict_sliding(n, f, win=WIN, hop=10)[1] for n in nets], axis=0)

def decide(p):
    return postprocess((p >= 0.5).astype(np.int64))

def moving_rule(filled, v):
    """At least HOLD_C cents of range inside a HOLD_WIN window, with all its frames voiced; a frame is marked if any such window covers it."""
    n = len(filled); out = np.zeros(n, bool)
    if n < HOLD_WIN: return out
    w = np.lib.stride_tricks.sliding_window_view(filled, HOLD_WIN); ok = np.lib.stride_tricks.sliding_window_view(v, HOLD_WIN).all(axis=1)
    for i in np.where(((w.max(axis=1) - w.min(axis=1)) >= HOLD_C) & ok)[0]: out[i:i + HOLD_WIN] = True
    return out

def prf(y, l):
    tp = ((y == 1) & (l == 1)).sum(); fp = ((y == 0) & (l == 1)).sum(); fn = ((y == 1) & (l == 0)).sum()
    return dict(precision=float(tp / max(tp + fp, 1)), recall=float(tp / max(tp + fn, 1)), f1=float(2 * tp / max(2 * tp + fp + fn, 1)))

t0 = time.time(); res = {}
print('Check 1: held-out branch', flush=True)
branches = sorted({branch(d) for d in PLAV}); ho = {}; Y_all, L_all = [], []
for br in branches:
    nets = [train([d for d in PLAV if branch(d) != br], 0)]; Ys, Ls, vs = [], [], []
    for d in (d for d in PLAV if branch(d) == br):
        c, conf = load(d); f, _, v = feats(c, conf); a, b = d['crop_range']; y, w = labels(f.shape[1], a, b); l = decide(predict(nets, f))
        Ys.append(y[w > 0]); Ls.append(l[w > 0]); vs.append(l[v > 0])
    Y = np.concatenate(Ys); L = np.concatenate(Ls); V = np.concatenate(vs); Y_all.append(Y); L_all.append(L)
    ho[br] = dict(prf(Y, L), flagged_voiced=float(V.mean())); print(' ', br, {k: round(x, 2) for k, x in ho[br].items()}, flush=True)
res['heldout_branch'] = ho; res['heldout_pooled'] = prf(np.concatenate(Y_all), np.concatenate(L_all))
res['heldout_flagged_voiced_mean'] = float(np.mean([ho[b]['flagged_voiced'] for b in branches]))

print('Final model on all 14 clips', flush=True)
nets = [train(PLAV, s) for s in SEEDS]

print('Check 2: steady note (SIMULATED input)', flush=True)
rs = np.random.default_rng(1); steady = {}
for jit in (0, 3, 10):
    c = 300 + rs.normal(0, jit, 1000) if jit else np.full(1000, 300.0); f, _, _ = feats(c, np.full(1000, 0.9)); steady[f'jitter_{jit}c'] = float(decide(predict(nets, f)).mean())
res['steady_note_flagged'] = steady; print(' ', steady, flush=True)

print('Check 3: other veena clips', flush=True)
oth = {}
groups = {'jaru': [d for d in D if d['cls'] == 'jaru'], 'ravai': [d for d in D if d['cls'] == 'ravai'], 'pratyagata': [d for d in D if d['cls'] == 'pratyagata'],
          'kampita leena (also a kampita)': [d for d in D if 'leena' in d['file'].lower()]}
for n, cl in groups.items():
    sh = []; anyf = []
    for d in cl:
        c, conf = load(d); f, _, v = feats(c, conf); l = decide(predict(nets, f))
        if v.sum(): sh.append(l[v > 0].mean()); anyf.append(l.any())
    oth[n] = dict(clips=len(cl), flagged_voiced_mean=float(np.mean(sh)), clips_with_a_detection=float(np.mean(anyf))); print(' ', n, {k: round(x, 2) for k, x in oth[n].items()}, flush=True)
res['other_veena'] = oth

print('Kalyani', flush=True)
rows = []; res['kalyani'] = {}
for fn in sorted(glob.glob(os.path.join(HERE, 'crepe', '*.npy'))):
    name = os.path.splitext(os.path.basename(fn))[0]; c, conf = pitch(fn); f, filled, v = feats(c, conf); pp = [predict_sliding(n, f, win=WIN, hop=10)[1] for n in nets]
    p = np.mean(pp, axis=0); l = decide(p); mv = moving_rule(filled, v); per = [decide(x) for x in pp]
    agree = np.mean([(per[i][v > 0] == per[j][v > 0]).mean() for i, j in ((0, 1), (0, 2), (1, 2))])
    ev = [(s / 100, e / 100) for s, e, k in runs(l) if k]
    for s, e in ev:
        a, b = int(s * 100), int(e * 100); x = filled[a:b][v[a:b] > 0]
        rows.append(dict(recording=name, start_s=s, end_s=e, length_s=round(e - s, 2), mean_score=round(float(p[a:b].mean()), 3), pitch_range_cents=round(float(np.percentile(x, 95) - np.percentile(x, 5)), 0) if len(x) else '', mm_ss=f'{int(s // 60)}:{s % 60:05.2f}'))
    inter = (l.astype(bool) & mv & (v > 0)).sum(); union = ((l.astype(bool) | mv) & (v > 0)).sum()
    res['kalyani'][name] = dict(seconds=len(l) / 100, voiced_s=float(v.sum() / 100), flagged_voiced=float(l[v > 0].mean()), moving_rule_voiced=float(mv[v > 0].mean()), overlap_with_rule=float(inter / max(union, 1)),
                                events=len(ev), seed_agreement=float(agree)); print(' ', name, {k: round(x, 2) for k, x in res['kalyani'][name].items()}, flush=True)
    t = np.arange(len(l)) / 100; width = max(14, len(l) / 100 * 0.42)
    fig, ax = plt.subplots(2, 1, figsize=(width, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    x = np.where(v > 0, filled, np.nan); ax[0].plot(t, x, color='#1f4e79', lw=0.8)
    for s, e in ev: ax[0].axvspan(s, e, color='#fde0a0', zorder=0)
    ax[0].set_ylabel('cents above E4 (327 Hz)'); ax[0].set_title(f'{name}: pitch, shaded = marked as kampita by the plavita-only model (unverified, no labels exist)', fontsize=10)
    ax[1].plot(t, p, color='#b5541c', lw=0.9); ax[1].axhline(0.5, color='0.6', lw=0.7); ax[1].set_ylim(0, 1); ax[1].set_ylabel('score'); ax[1].set_xlabel('seconds'); ax[0].set_xlim(0, t[-1]); ax[1].xaxis.set_major_locator(plt.MultipleLocator(2))
    for a in ax: a.grid(axis='x', color='0.9')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, f'timeline_{name.replace(" ", "_").replace("#", "")}.png'), dpi=110); plt.close(fig)
    np.savez_compressed(os.path.join(HERE, f'scores_{name.replace(" ", "_").replace("#", "")}.npz'), p=p, label=l, filled=filled, voiced=v, moving=mv)
with open(os.path.join(HERE, 'detections.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['recording', 'mm_ss', 'start_s', 'end_s', 'length_s', 'mean_score', 'pitch_range_cents']); w.writeheader(); w.writerows(rows)
json.dump(res, open(os.path.join(HERE, 'checks.json'), 'w'), indent=1); print(f'done in {time.time() - t0:.0f}s, {len(rows)} events', flush=True)
