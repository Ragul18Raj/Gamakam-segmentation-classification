"""Leave-one-take-out classifiers on the cleaned curves. Plan: PLAN.md. Exploratory.
Run from veena_stage1/methods_explore after clean.py:  python classify.py
Writes results.json, predictions.csv and prints a table."""
import csv, json, os, sys, pickle, warnings, numpy as np
from numba import njit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, confusion_matrix, recall_score
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
CL = ['jaru', 'kampita', 'ravai', 'pratyagata']; FR = 100; N_PERM = 1000; SEED = 20261030

D = pickle.load(open(os.path.join(HERE, 'curves.pkl'), 'rb'))
y = np.array([CL.index(d['cls']) for d in D]); g = np.array([d['take'] for d in D]); takes = np.unique(g)

def uf3(x):
    p = np.concatenate([x[:1], x, x[-1:]]); return (p[:-2] + p[1:-1] + p[2:]) / 3
def seg_feats(c):
    """Same 9 features as extract.py."""
    sm = uf3(c); v = np.diff(sm); sgn = np.sign(v[np.abs(v) > 1.0])
    turns = int(np.sum(sgn[1:] != sgn[:-1])) if len(sgn) > 1 else 0
    dur = len(c) / FR; base = c[0]
    return [dur, np.percentile(c, 95) - np.percentile(c, 5), turns, turns / dur, c[-3:].mean() - c[:3].mean(),
            abs(c[-3:].mean() - c[:3].mean()), c.max() - base, base - c.min(), np.abs(v).mean() * FR]
NOLEN = [1, 3, 4, 5, 6, 7, 8]                               # drop dur and turns

@njit(cache=True)
def dtw(a, b, band):
    """Absolute-difference DTW inside a Sakoe-Chiba band, normalised by n + m."""
    n, m = len(a), len(b); w = max(band, abs(n - m)); inf = 1e18
    prev = np.full(m + 1, inf); cur = np.full(m + 1, inf); prev[0] = 0.0
    for i in range(1, n + 1):
        cur[:] = inf
        for j in range(max(1, i - w), min(m, i + w) + 1):
            cur[j] = abs(a[i - 1] - b[j - 1]) + min(prev[j], prev[j - 1], cur[j - 1])
        prev, cur = cur, prev
    return prev[m] / (n + m)

def centre(c): return np.asarray(c - np.median(c), dtype=np.float64)
def resample(c, n=100): return np.interp(np.linspace(0, len(c) - 1, n), np.arange(len(c)), c)
def dist_matrix(curves, band_frac=0.2):
    N = len(curves); M = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            M[i, j] = M[j, i] = dtw(curves[i], curves[j], max(5, int(band_frac * max(len(curves[i]), len(curves[j])))))
    return M

def loto_feat(X, yy, mk):
    p = np.empty_like(yy)
    for t in takes:
        te = g == t; p[te] = mk().fit(X[~te], yy[~te]).predict(X[te])
    return p
def loto_dtw(M, yy, k):
    p = np.empty_like(yy)
    for t in takes:
        te = np.where(g == t)[0]; tr = np.where(g != t)[0]
        for i in te:
            o = tr[np.argsort(M[i, tr], kind='stable')[:k]]
            v = np.bincount(yy[o], minlength=4); best = np.flatnonzero(v == v.max())
            p[i] = next(yy[q] for q in o if yy[q] in best)  # tie: the nearest neighbour among the tied classes
    return p
def mf1(yy, p): return f1_score(yy, p, average='macro', labels=range(4))
LR = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
KN = lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(3))

def reversals(c, thr=40.0):
    """Number of direction changes of at least thr cents."""
    x = uf3(uf3(c)); hi = lo = x[0]; d = 0; n = 0
    for v in x[1:]:
        if d >= 0 and v > hi: hi = v
        if d <= 0 and v < lo: lo = v
        if d >= 0 and hi - v >= thr: d = -1; n += 1 if d else 0; lo = v
        elif d <= 0 and v - lo >= thr: d = 1; n += 1 if d else 0; hi = v
    return n
def rule(c):
    """Fixed by hand from the descriptions and plots, no fitting. Oscillation -> kampita; a large one-way move -> jaru;
    dip and return to the same note -> pratyagata; bump up then a step down -> ravai; anything else -> kampita (slow rise, leena)."""
    k = min(5, len(c)); net = np.median(c[-k:]) - np.median(c[:k]); base = np.median(c[:k]); rev = reversals(c)
    up, down = c.max() - base, base - c.min()
    if rev >= 4: return 1
    if abs(net) >= 350: return 0
    if abs(net) < 60 and down >= 50 and up < 60: return 3
    if up >= 40 and net <= -60: return 2
    return 1

raw = [d['raw'] for d in D]; clean = [d['clean'] for d in D]; crop = [d['crop'] for d in D]
F = {n: np.array([seg_feats(c) for c in v]) for n, v in (('raw', raw), ('clean', clean), ('crop', crop))}
print('DTW matrices...', flush=True)
DM = {n: dist_matrix([centre(c) for c in v]) for n, v in (('raw', raw), ('clean', clean), ('crop', crop))}
DM['crop_nolen'] = dist_matrix([centre(resample(c)) for c in crop], 0.1)
rs = np.random.default_rng(SEED)
scr = [rs.permutation(c) for c in crop]
DM['crop_scrambled'] = dist_matrix([centre(c) for c in scr]); F['crop_scrambled'] = np.array([seg_feats(c) for c in scr])

# trained configurations in the family-wise test
CFG = {}
for n in ('raw', 'clean', 'crop'):
    CFG[f'{n} | features LR'] = ('f', F[n], LR); CFG[f'{n} | features 3-NN'] = ('f', F[n], KN); CFG[f'{n} | DTW 3-NN'] = ('d', DM[n], 3)
CFG['crop | DTW 1-NN'] = ('d', DM['crop'], 1)
CFG['crop, length removed | features LR'] = ('f', F['crop'][:, NOLEN], LR)
CFG['crop, length removed | features 3-NN'] = ('f', F['crop'][:, NOLEN], KN)
CFG['crop, length removed | DTW 3-NN'] = ('d', DM['crop_nolen'], 3)
CFG['crop, length removed | DTW 1-NN'] = ('d', DM['crop_nolen'], 1)
def run(cfg, yy): return loto_feat(cfg[1], yy, cfg[2]) if cfg[0] == 'f' else loto_dtw(cfg[1], yy, cfg[2])

med = np.array([[np.median(c)] for c in raw]); dur_raw = F['raw'][:, [0]]; dur_crop = F['crop'][:, [0]]
CTRL = {'probe: median pitch LR': ('f', med, LR), 'length only: raw duration LR': ('f', dur_raw, LR), 'length only: crop duration LR': ('f', dur_crop, LR),
        'scramble: crop features LR': ('f', F['crop_scrambled'], LR), 'scramble: crop features 3-NN': ('f', F['crop_scrambled'], KN),
        'scramble: crop DTW 3-NN': ('d', DM['crop_scrambled'], 3), 'scramble: crop DTW 1-NN': ('d', DM['crop_scrambled'], 1)}

res = {}; pred = {}
for name, cfg in {**CFG, **CTRL}.items():
    p = run(cfg, y); pred[name] = p
    res[name] = dict(macro_f1=float(mf1(y, p)), recall=dict(zip(CL, recall_score(y, p, average=None, labels=range(4)).tolist())),
                     confusion=confusion_matrix(y, p, labels=range(4)).tolist(), per_take={t: float((p[g == t] == y[g == t]).mean()) for t in takes})
    print(f"{name:42s} {res[name]['macro_f1']:.3f}", flush=True)
pr = np.array([rule(c) for c in crop]); pred['rules (no p: seen the data)'] = pr
res['rules (no p: seen the data)'] = dict(macro_f1=float(mf1(y, pr)), recall=dict(zip(CL, recall_score(y, pr, average=None, labels=range(4)).tolist())),
                                          confusion=confusion_matrix(y, pr, labels=range(4)).tolist(), per_take={t: float((pr[g == t] == y[g == t]).mean()) for t in takes})
print(f"{'rules (no p: seen the data)':42s} {res['rules (no p: seen the data)']['macro_f1']:.3f}", flush=True)
maj = np.full_like(y, np.bincount(y).argmax()); res['majority'] = dict(macro_f1=float(mf1(y, maj)))

# permutation null: shuffle class across takes, same permutations for every configuration
print('permutations...', flush=True)
rng = np.random.default_rng(SEED); tcls = {t: y[g == t][0] for t in takes}; names = list(CFG); N = np.zeros((N_PERM, len(names)))
for b in range(N_PERM):
    perm = dict(zip(takes, rng.permutation([tcls[t] for t in takes]))); yp = np.array([perm[t] for t in g])
    for j, n in enumerate(names): N[b, j] = mf1(yp, run(CFG[n], yp))
    if (b + 1) % 100 == 0: print(' ', b + 1, flush=True)
mx = N.max(axis=1)
for j, n in enumerate(names):
    o = res[n]['macro_f1']
    res[n].update(p_single=float((1 + (N[:, j] >= o).sum()) / (N_PERM + 1)), p_family=float((1 + (mx >= o).sum()) / (N_PERM + 1)), null_mean=float(N[:, j].mean()), null_95=float(np.percentile(N[:, j], 95)))
res['family_null_95'] = float(np.percentile(mx, 95))

# take-level bootstrap on the best length-removed configuration and the best overall
def boot(p, n=2000):
    r = np.random.default_rng(SEED); by = {c: sorted({t for t in takes if tcls[t] == c}) for c in range(4)}; out = []
    for _ in range(n):
        idx = np.concatenate([np.where(g == t)[0] for c in range(4) for t in r.choice(by[c], len(by[c]))])
        out.append(mf1(y[idx], p[idx]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
best_all = max(names, key=lambda n: res[n]['macro_f1']); best_nl = max((n for n in names if 'length removed' in n), key=lambda n: res[n]['macro_f1'])
for n in {best_all, best_nl}: res[n]['ci95_takes'] = boot(pred[n])
res['best_overall'] = best_all; res['best_length_removed'] = best_nl
json.dump(res, open(os.path.join(HERE, 'results.json'), 'w'), indent=1)
with open(os.path.join(HERE, 'predictions.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['file', 'cls', 'take', 'branch'] + list(pred))
    for i, d in enumerate(D): w.writerow([d['file'], d['cls'], d['take'], d['branch']] + [CL[pred[n][i]] for n in pred])
print(f"\n{'config':42s} {'F1':>6s} {'p':>7s} {'p_fam':>7s}")
for n in names: print(f"{n:42s} {res[n]['macro_f1']:6.3f} {res[n]['p_single']:7.3f} {res[n]['p_family']:7.3f}")
print('family null 95th', round(res['family_null_95'], 3), 'majority', round(res['majority']['macro_f1'], 3))
for n in {best_all, best_nl}:
    print(n, 'CI', res[n]['ci95_takes'], 'recall', {k: round(v, 2) for k, v in res[n]['recall'].items()}); print(np.array(res[n]['confusion'])); print({k: round(v, 2) for k, v in res[n]['per_take'].items()})
