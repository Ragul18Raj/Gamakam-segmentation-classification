import csv, os, sys, numpy as np
def crepe_activation_to_pitch(act):
    """Copied from anchor_feasibility_check_v3.py (same local weighted average CREPE uses)."""
    cents_map = np.linspace(0, 7180, 360) + 1997.3794084376191
    T = act.shape[0]; cents = np.zeros(T); conf = act.max(axis=1)
    for t in range(T):
        c = int(np.argmax(act[t])); lo, hi = max(0, c - 4), min(360, c + 5)
        w = act[t, lo:hi]; cents[t] = np.sum(w * cents_map[lo:hi]) / max(np.sum(w), 1e-9)
    return 10.0 * 2 ** (cents / 1200.0), conf
FR = 100
def fill_short_gaps(c, voiced, max_gap=10):
    idx = np.arange(len(c)); v = voiced.copy()
    if v.sum() < 2: return np.zeros_like(c), v
    return np.interp(idx, idx[v], c[v]), v
def uf3(x):  # uniform filter size 3, reflect edges
    p = np.concatenate([x[:1], x, x[-1:]]); return (p[:-2] + p[1:-1] + p[2:]) / 3
def seg_feats(c):
    sm = uf3(c); v = np.diff(sm)
    sgn = np.sign(v[np.abs(v) > 1.0])
    turns = int(np.sum(sgn[1:] != sgn[:-1])) if len(sgn) > 1 else 0
    dur = len(c) / FR; base = c[0]
    return [dur, np.percentile(c, 95) - np.percentile(c, 5), turns, turns / dur,
            c[-3:].mean() - c[:3].mean(), abs(c[-3:].mean() - c[:3].mean()),
            c.max() - base, base - c.min(), np.abs(v).mean() * FR]
OCTFIX = len(sys.argv) > 1 and sys.argv[1] == '--octfix'
keep = {'jaru', 'kampita', 'ravai', 'pratyagata'}
rows = list(csv.DictReader(open('../takes.csv', encoding='utf-8-sig')))
out = []; skipped = []
for r in rows:
    cls, take = r['take'].split('|', 1)
    if cls not in keep: continue
    if '_continuous' in r['file'].lower(): skipped.append(r['file']); continue
    act = np.load(os.path.join('..', 'crepe_output', os.path.splitext(r['file'])[0] + '.npy'))
    hz, conf = crepe_activation_to_pitch(act)
    cents = 1200 * np.log2(np.maximum(hz, 1e-3) / 329.63)   # E4 reference, used only for the probe
    voiced = conf >= 0.5
    if voiced.sum() < 10: skipped.append(r['file'] + ' (too few voiced)'); continue
    idx = np.where(voiced)[0]; a, b = idx[0], idx[-1] + 1
    if OCTFIX:  # fold octave jumps back toward a running median of voiced frames (exploratory)
        vc = cents.copy()
        for _ in range(2):
            vi = np.where(voiced)[0]; x = vc[vi]
            pad = np.pad(x, 12, mode='edge'); med = np.median(np.lib.stride_tricks.sliding_window_view(pad, 25), axis=1)
            k = np.round((x - med) / 1200.0); x = x - 1200 * k; vc[vi] = x
        cents = vc
    filled, _ = fill_short_gaps(cents, voiced)
    c = filled[a:b]
    out.append([r['file'], cls, take, len(act), float(np.median(cents[voiced]))] + [float(x) for x in seg_feats(c)])
with open('clip_feats_octfix.csv' if OCTFIX else 'clip_feats.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['file', 'cls', 'take', 'frames', 'median_cents', 'dur', 'range', 'turns', 'turn_rate', 'net', 'abs_net', 'exc_up', 'exc_down', 'mean_abs_vel'])
    w.writerows(out)
print('clips', len(out), 'skipped', len(skipped)); [print('  skip', s) for s in skipped]
import collections
print(collections.Counter((o[1]) for o in out)); print('takes', len(set(o[2] for o in out)))
