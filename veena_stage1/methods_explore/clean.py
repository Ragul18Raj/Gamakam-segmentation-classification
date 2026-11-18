"""Pitch cleaning and movement cropping for the methods test. No class labels are used here.
Run from veena_stage1/methods_explore:  python clean.py
Writes curves.pkl (raw, clean, crop per clip), clean_stats.csv and check_<branch>.png."""
import csv, os, sys, wave, math, pickle, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(HERE, '..', 'local_run'))
from shapes import BRANCHES, split_name, crepe_activation_to_pitch, safe
import matplotlib.pyplot as plt

FR = 100; CONF = 0.5
JUMP = 150          # cents between adjacent frames: a larger step starts a new run
LINK = 400          # a run joins the chain only if it starts within this many cents of it
DRONE_FR = 15       # a run an octave away and at least this long (150 ms) is a drone string, not a tracker slip
TRIM_DB = -30       # end frames below this, relative to the seed run's energy, are dropped
HOLD_WIN = 20       # 0.2 s window for "is it moving"
HOLD_C = 30         # local range at or above this many cents counts as movement
MARGIN = 10         # 0.1 s kept either side of the movement
MERGE = 40          # moving stretches closer than 0.4 s are one movement
ISLAND_FR = 20      # a flat run up to 0.2 s...
ISLAND_FLAT = 15    # ...within this many cents...
ISLAND_JUMP = 80    # ...jumped down to and back up from by at least this much (an upward one can be a real flick)...
ISLAND_SAME = 60    # ...with both neighbours about level, is a tracker flip

def frame_energy(fn, n):
    w = wave.open(os.path.join(ROOT, 'Gamaka_veena_dataset', fn))
    sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64) / 32768.0
    hop, win = sr // FR, sr // 25                           # 10 ms hop, 40 ms window, centred like CREPE frames
    e = np.array([np.mean(x[max(0, t * hop - win // 2):t * hop + win // 2] ** 2) if t * hop < len(x) else 0.0 for t in range(n)])
    return 10 * np.log10(e + 1e-12)

def runs_of(cents, voiced):
    """Contiguous voiced frames with no jump above JUMP -> list of (start, end) frame ranges, end exclusive."""
    out = []; i = 0; n = len(cents)
    while i < n:
        if not voiced[i]: i += 1; continue
        j = i + 1
        while j < n and voiced[j] and abs(cents[j] - cents[j - 1]) <= JUMP: j += 1
        out.append((i, j)); i = j
    return out

def clean_curve(cents, voiced, db):
    """-> cents with NaN on rejected frames. Picks the chain of runs with the most energy, where each run starts within
    LINK cents of the previous one. Only short runs may be folded by whole octaves; a long run an octave away stays where
    it is, so it joins only if the chain is really there."""
    runs = runs_of(cents, voiced)
    if not runs: return np.full(len(cents), np.nan)
    lin = 10 ** (db / 10); w = [lin[a:b].sum() for a, b in runs]
    states = [[(i, k) for k in ((0, -2, -1, 1, 2) if b - a < DRONE_FR else (0,))] for i, (a, b) in enumerate(runs)]
    best, prev = {}, {}
    for i, (a, b) in enumerate(runs):
        for _, k in states[i]:
            start = cents[a] + 1200 * k; v, p = 0.0, None
            for j in range(i):
                for _, kj in states[j]:
                    if abs(start - (cents[runs[j][1] - 1] + 1200 * kj)) <= LINK and best[(j, kj)] > v:
                        v, p = best[(j, kj)], (j, kj)
            best[(i, k)] = w[i] + v; prev[(i, k)] = p
    st = max(best, key=best.get); chain = []
    while st is not None: chain.append(st); st = prev[st]
    chain.reverse()
    keep = [True] * len(chain)                              # flat islands below the chain: tracker flips to a drone and straight back
    for n in range(1, len(chain) - 1):
        (i, k), (ip, kp), (inx, knx) = chain[n], chain[n - 1], chain[n + 1]
        a, b = runs[i]; x = cents[a:b] + 1200 * k
        left, right = cents[runs[ip][1] - 1] + 1200 * kp, cents[runs[inx][0]] + 1200 * knx
        if b - a <= ISLAND_FR and x.max() - x.min() <= ISLAND_FLAT and abs(left - right) < ISLAND_SAME and max(x[0] - left, x[-1] - right) <= -ISLAND_JUMP:
            keep[n] = False
    out = np.full(len(cents), np.nan)
    for (i, k), ok in zip(chain, keep):
        if ok: out[runs[i][0]:runs[i][1]] = cents[runs[i][0]:runs[i][1]] + 1200 * k
    i_top = max((c[0] for c in chain), key=lambda i: w[i]); a, b = runs[i_top]
    top_db = 10 * np.log10(lin[a:b].mean() + 1e-12)
    ok = np.where(~np.isnan(out) & (db >= top_db + TRIM_DB))[0]
    if len(ok) == 0: return out
    out[:ok[0]] = np.nan; out[ok[-1] + 1:] = np.nan
    return out

def span_filled(c):
    """Voiced span with gaps interpolated, plus its first frame."""
    idx = np.where(~np.isnan(c))[0]
    if len(idx) == 0: return np.zeros(0), 0
    a, b = idx[0], idx[-1] + 1; t = np.arange(a, b)
    return np.interp(t, idx, c[idx]), a

def crop_moving(c):
    """c = gap-filled curve -> (a, b) frame range around the largest movement. Moving stretches closer than MERGE are one
    movement; a clip that never moves is kept whole."""
    n = len(c)
    if n <= HOLD_WIN: return 0, n
    w = np.lib.stride_tricks.sliding_window_view(c, HOLD_WIN)
    moving = np.where(w.max(axis=1) - w.min(axis=1) >= HOLD_C)[0]
    if len(moving) == 0: return 0, n
    groups = np.split(moving, np.where(np.diff(moving) > MERGE)[0] + 1)
    def size(g):
        x = c[g[0]:g[-1] + HOLD_WIN]; return x.max() - x.min()
    g = max(groups, key=size)
    return max(0, g[0] - MARGIN), min(n, g[-1] + HOLD_WIN + MARGIN)

def process(fn):
    act = np.load(os.path.join(ROOT, 'crepe_output', os.path.splitext(fn)[0] + '.npy'))
    hz, conf = crepe_activation_to_pitch(act)
    cents = 1200 * np.log2(np.maximum(hz, 1e-3) / 327.0); voiced = conf >= CONF
    db = frame_energy(fn, len(cents))
    raw_masked = np.where(voiced, cents, np.nan)
    cl = clean_curve(cents, voiced, db)
    raw, ra = span_filled(raw_masked); clean, ca = span_filled(cl)
    ca_, cb_ = crop_moving(clean); crop = clean[ca_:cb_]
    return dict(raw=raw, clean=clean, crop=crop, raw_masked=raw_masked, clean_masked=cl, crop_range=(ca + ca_, ca + cb_))

def rng(c): return float(np.percentile(c, 95) - np.percentile(c, 5)) if len(c) else 0.0

if __name__ == '__main__':
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'takes.csv'), encoding='utf-8-sig')))
    keep = {'jaru', 'kampita', 'ravai', 'pratyagata'}; data = []; stats = []
    for r in rows:
        cls, take = r['take'].split('|', 1)
        if cls not in keep or '_continuous' in r['file'].lower(): continue
        d = process(r['file']); stem, sl = split_name(r['file'])
        d.update(file=r['file'], cls=cls, take=take, branch=BRANCHES.get(stem, stem), slice=sl); data.append(d)
        stats.append(dict(file=r['file'], cls=cls, take=take, raw_s=len(d['raw']) / FR, clean_s=len(d['clean']) / FR, crop_s=len(d['crop']) / FR,
                          raw_range=round(rng(d['raw']), 1), clean_range=round(rng(d['clean']), 1), crop_range=round(rng(d['crop']), 1),
                          kept_frac=round(float(np.mean(~np.isnan(d['clean_masked'])) / max(np.mean(~np.isnan(d['raw_masked'])), 1e-9)), 3)))
    pickle.dump(data, open(os.path.join(HERE, 'curves.pkl'), 'wb'))
    with open(os.path.join(HERE, 'clean_stats.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0])); w.writeheader(); w.writerows(stats)
    print('clips', len(data))
    for k in ('raw_range', 'clean_range', 'crop_range'):
        v = np.array([s[k] for s in stats]); print(f'{k:12s} median {np.median(v):7.1f}  above 1200: {(v > 1200).sum()}')
    for k in ('raw_s', 'clean_s', 'crop_s'):
        v = np.array([s[k] for s in stats]); print(f'{k:12s} median {np.median(v):5.2f}  min {v.min():.2f}  max {v.max():.2f}')
    print('kept_frac median', np.median([s['kept_frac'] for s in stats]), 'min', min(s['kept_frac'] for s in stats))
    print('shortest crops', sorted((s['crop_s'], s['file']) for s in stats)[:6])
    # check plots: raw grey, clean blue, crop shaded
    branches = {}
    for d in data: branches.setdefault((d['cls'], d['branch']), []).append(d)
    for (cls, br), items in branches.items():
        items = items[:12]; ncol = min(4, len(items)); nrow = math.ceil(len(items) / ncol)
        fig, axs = plt.subplots(nrow, ncol, figsize=(3 * ncol + 0.4, 2.1 * nrow + 0.6), sharex=True, sharey=True, squeeze=False)
        for k, ax in enumerate(axs.flat):
            if k >= len(items): ax.axis('off'); continue
            d = items[k]; t = np.arange(len(d['raw_masked'])) / FR
            ax.axvspan(d['crop_range'][0] / FR, d['crop_range'][1] / FR, color='#fde0a0', zorder=0)
            ax.plot(t, d['raw_masked'], color='0.65', lw=0.8, zorder=1); ax.plot(t, d['clean_masked'], color='#1f4e79', lw=1.1, zorder=2)
            ax.set_title(f"{d['slice']} · crop {len(d['crop']) / FR:.2f}s · range {rng(d['crop']):.0f}¢", fontsize=7); ax.tick_params(labelsize=6)
        fig.suptitle(f'{cls} · {br}: raw grey, clean blue, crop shaded', fontsize=9); fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(os.path.join(HERE, f'check_{cls}__{safe(br)}.png'), dpi=130); plt.close(fig)
