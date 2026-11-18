"""Follow-up, decided after seeing results.json: scrambling frames barely lowered the DTW score, so what is it using?
Tests how wide the pitch goes (extent) on its own, and shape with extent and length both removed.
Run after classify.py:  python extent_check.py   Writes extent_results.json."""
import json, os, pickle, warnings, numpy as np
from numba import njit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, recall_score, confusion_matrix
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
CL = ['jaru', 'kampita', 'ravai', 'pratyagata']; SEED = 20261030; N_PERM = 1000; N_SCR = 20; FLOOR = 30.0

D = pickle.load(open(os.path.join(HERE, 'curves.pkl'), 'rb'))
y = np.array([CL.index(d['cls']) for d in D]); g = np.array([d['take'] for d in D]); takes = np.unique(g)
crop = [d['crop'] for d in D]

@njit(cache=True)
def dtw(a, b, band):
    n, m = len(a), len(b); w = max(band, abs(n - m)); inf = 1e18
    prev = np.full(m + 1, inf); cur = np.full(m + 1, inf); prev[0] = 0.0
    for i in range(1, n + 1):
        cur[:] = inf
        for j in range(max(1, i - w), min(m, i + w) + 1):
            cur[j] = abs(a[i - 1] - b[j - 1]) + min(prev[j], prev[j - 1], cur[j - 1])
        prev, cur = cur, prev
    return prev[m] / (n + m)
def dist_matrix(curves):
    N = len(curves); M = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N): M[i, j] = M[j, i] = dtw(curves[i], curves[j], 10)
    return M
def centre(c): return np.asarray(c - np.median(c), dtype=np.float64)
def resample(c, n=100): return np.interp(np.linspace(0, len(c) - 1, n), np.arange(len(c)), c)
def extent(c): return float(np.percentile(c, 95) - np.percentile(c, 5))
def unit(c): return centre(c) / max(extent(c), FLOOR)       # extent removed; the floor stops flat clips blowing up noise

def loto_feat(X, yy, mk):
    p = np.empty_like(yy)
    for t in takes:
        te = g == t; p[te] = mk().fit(X[~te], yy[~te]).predict(X[te])
    return p
def loto_dtw(M, yy, k=3):
    p = np.empty_like(yy)
    for t in takes:
        te = np.where(g == t)[0]; tr = np.where(g != t)[0]
        for i in te:
            o = tr[np.argsort(M[i, tr], kind='stable')[:k]]
            v = np.bincount(yy[o], minlength=4); best = np.flatnonzero(v == v.max())
            p[i] = next(yy[q] for q in o if yy[q] in best)
    return p
def mf1(yy, p): return f1_score(yy, p, average='macro', labels=range(4))
LR = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
KN = lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(3))

R = np.array([[extent(c)] for c in crop])
Q = np.array([np.percentile(centre(c), [5, 25, 75, 95]) for c in crop])   # order-free signature of the pitch values
rc = [resample(c) for c in crop]
base = dist_matrix([centre(c) for c in rc]); shape = dist_matrix([unit(c) for c in rc])
CFG = {'extent only (p95-p5) | LR': ('f', R, LR), 'extent only (p95-p5) | 3-NN': ('f', R, KN), 'pitch-value percentiles, no order | 3-NN': ('f', Q, KN),
       'length removed | DTW 3-NN (from classify.py)': ('d', base), 'length and extent removed: pure shape | DTW 3-NN': ('d', shape)}
def run(c, yy): return loto_feat(c[1], yy, c[2]) if c[0] == 'f' else loto_dtw(c[1], yy)

res = {}
for n, c in CFG.items():
    p = run(c, y)
    res[n] = dict(macro_f1=float(mf1(y, p)), recall=dict(zip(CL, recall_score(y, p, average=None, labels=range(4)).tolist())),
                  confusion=confusion_matrix(y, p, labels=range(4)).tolist(), per_take={str(t): float((p[g == t] == y[g == t]).mean()) for t in takes})
rs = np.random.default_rng(SEED); sb, ss = [], []
for s in range(N_SCR):                                      # same scramble applied to both versions of the clip
    sc = [rs.permutation(c) for c in rc]
    sb.append(mf1(y, loto_dtw(dist_matrix([centre(c) for c in sc]), y))); ss.append(mf1(y, loto_dtw(dist_matrix([unit(c) for c in sc]), y)))
res['scrambled, length removed | DTW 3-NN'] = dict(mean=float(np.mean(sb)), min=float(min(sb)), max=float(max(sb)), n=N_SCR)
res['scrambled, pure shape | DTW 3-NN'] = dict(mean=float(np.mean(ss)), min=float(min(ss)), max=float(max(ss)), n=N_SCR)

rng = np.random.default_rng(SEED); tcls = {t: y[g == t][0] for t in takes}; names = list(CFG); N = np.zeros((N_PERM, len(names)))
for b in range(N_PERM):
    perm = dict(zip(takes, rng.permutation([tcls[t] for t in takes]))); yp = np.array([perm[t] for t in g])
    for j, n in enumerate(names): N[b, j] = mf1(yp, run(CFG[n], yp))
for j, n in enumerate(names):
    res[n].update(p_single=float((1 + (N[:, j] >= res[n]['macro_f1']).sum()) / (N_PERM + 1)), null_95=float(np.percentile(N[:, j], 95)))
json.dump(res, open(os.path.join(HERE, 'extent_results.json'), 'w'), indent=1)
for n in names:
    print(f"{n:52s} {res[n]['macro_f1']:.3f}  p {res[n]['p_single']:.3f}  null95 {res[n]['null_95']:.3f}  ", {k[:4]: round(v, 2) for k, v in res[n]['recall'].items()})
for n in ('scrambled, length removed | DTW 3-NN', 'scrambled, pure shape | DTW 3-NN'): print(n, {k: round(v, 3) for k, v in res[n].items()})
n = 'length and extent removed: pure shape | DTW 3-NN'; print(np.array(res[n]['confusion'])); print({k[:30]: round(v, 2) for k, v in res[n]['per_take'].items()})
print('extent by class (median):', {c: round(float(np.median(R[y == i])), 0) for i, c in enumerate(CL)})
print('extent by take (median):', {str(t)[:30]: round(float(np.median(R[g == t])), 0) for t in takes})
