"""Held-out (leave-one-recording-out) kampita segments from the random-forest detector, for eval_segments.py."""
src = open('train_kampita.py').read(); src = src[:src.index('# ---- leave-one-recording-out')]
exec(src)
import pandas as pd
names = []
for f in sorted(glob.glob(os.path.join(ROOT, 'crepe_output', '*.npy'))):
    n = os.path.basename(f)[:-4]
    if n.lower().startswith('kampita') and n not in spans: continue
    if n.startswith('Sphutitam#01'): continue
    names += [n] * len(np.load(f, mmap_mode='r'))
names = np.array(names); rows = []
for kg in sorted({group_of(n) for n in spans}):
    tr, te = g != kg, g == kg
    p = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    for n in sorted(set(names[te])):
        for s, e in segments(p[names[te] == n]): rows.append([n, round(s, 2), round(e, 2)])
pd.DataFrame(rows, columns=['Recording', 'Start (sec)', 'End (sec)']).to_csv('veena_loro_preds.csv', index=False)
print(len(rows), 'predicted segments')
