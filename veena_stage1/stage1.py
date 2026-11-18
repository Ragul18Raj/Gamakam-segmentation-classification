import sys, json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, recall_score
d = pd.read_csv(sys.argv[1]); out = sys.argv[2]
CL = ['jaru', 'kampita', 'ravai', 'pratyagata']
F = ['dur','range','turns','turn_rate','net','abs_net','exc_up','exc_down','mean_abs_vel']
y = d.cls.map({c:i for i,c in enumerate(CL)}).values; g = d['take'].values; takes = np.unique(g)
def loto(X, y, mk):
    p = np.empty_like(y)
    for t in takes:
        te = g == t; m = mk().fit(X[~te], y[~te]); p[te] = m.predict(X[te])
    return p
LR = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
KN = lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(3))
X = d[F].values; Xp = d[['median_cents']].values
res = {}
for name, X_, mk in [('logreg_movement', X, LR), ('knn3_movement', X, KN), ('probe_median_pitch', Xp, LR)]:
    p = loto(X_, y, mk)
    res[name] = dict(macro_f1=f1_score(y, p, average='macro', labels=range(4)), bal_acc=balanced_accuracy_score(y, p),
                     recall={c: r for c, r in zip(CL, recall_score(y, p, average=None, labels=range(4)))},
                     confusion=confusion_matrix(y, p, labels=range(4)).tolist(),
                     per_take={t: float((p[g==t]==y[g==t]).mean()) for t in takes})
maj = np.full_like(y, np.bincount(y).argmax()); res['majority_macro_f1'] = f1_score(y, maj, average='macro', labels=range(4))
# permutation null: shuffle class labels across takes
rng = np.random.default_rng(20261030); tcls = {t: y[g==t][0] for t in takes}; null = []
for _ in range(1000):
    perm = dict(zip(takes, rng.permutation([tcls[t] for t in takes]))); yp = np.array([perm[t] for t in g])
    null.append(f1_score(yp, loto(X, yp, LR), average='macro', labels=range(4)))
null = np.array(null); obs = res['logreg_movement']['macro_f1']
res['perm'] = dict(n=1000, p=float((1 + (null >= obs).sum()) / 1001), null_mean=float(null.mean()), null_95=float(np.percentile(null, 95)))
res['n_clips'] = {c: int((y==i).sum()) for i,c in enumerate(CL)}; res['n_takes'] = {c: int(len(set(g[y==i]))) for i,c in enumerate(CL)}
res['class_means'] = d.groupby('cls')[F+['median_cents']].mean().round(2).to_dict()
lr = res['logreg_movement']; probe = res['probe_median_pitch']
res['PASS'] = bool(lr['macro_f1'] >= 0.50 and lr['macro_f1'] > probe['macro_f1'] and res['perm']['p'] < 0.05)
json.dump(res, open(f'{out}/stage1_results.json','w'), indent=1, default=float)
print(json.dumps({k: (v if not isinstance(v, dict) or k in ('perm','n_clips','n_takes') else {kk: vv for kk, vv in v.items() if kk in ('macro_f1','bal_acc','recall','confusion')}) for k, v in res.items() if k!='class_means'}, indent=1, default=lambda o: round(float(o),3)))
print('per_take logreg', {k: round(v,2) for k,v in lr['per_take'].items()})
print(pd.DataFrame(res['class_means']).T.round(1).to_string())
