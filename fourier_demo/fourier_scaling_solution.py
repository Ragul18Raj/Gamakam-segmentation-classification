"""Fourier-based answer to the scaling problem (synthetic test).
Idea: factor each gamaka into SHAPE x SPEED x SIZE.
  - size  -> divide the spectrum by its energy (removed)
  - speed -> on a log-frequency axis a tempo change is a shift; align the spectrum at its peak (removed)
  - shape -> what is left: peak-aligned spectrum + harmonic ratios (Fourier-series view)
Speed and size are kept as two separate numbers, so a model can still learn the singer's allowed range.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
FS = 100
rng = np.random.default_rng(0)
ss = lambda x: np.clip(x, 0, 1) ** 2 * (3 - 2 * np.clip(x, 0, 1))

# ---------- synthetic gamakas; s = speed factor (>1 faster), w = width factor ----------
def make(kind, s=1.0, w=1.0):
    hold = 0.1
    if kind == 'kampita':
        rate = rng.uniform(4.6, 6.4) * s; ext = rng.uniform(110, 200) * w; cyc = rng.integers(2, 6)
        dur = cyc / rate; t = np.arange(int((dur + 2 * hold) * FS)) / FS; u = np.clip(t - hold, 0, dur)
        y = ext * (0.5 - 0.5 * np.cos(2 * np.pi * rate * u))
    elif kind == 'jaru':
        dur = rng.uniform(0.38, 0.62) / s; span = rng.uniform(200, 700) * w * rng.choice([-1, 1])
        t = np.arange(int((dur + 2 * hold) * FS)) / FS; y = span * ss((t - hold) / dur)
    else:  # sphuritam: two quick touches of the lower note
        gap = rng.uniform(0.12, 0.2) / s; sig = rng.uniform(0.018, 0.026) / s; dep = rng.uniform(150, 250) * w
        dur = gap + 6 * sig; t = np.arange(int((dur + 2 * hold) * FS)) / FS; c = hold + 3 * sig
        y = -dep * (np.exp(-((t - c) / sig) ** 2) + np.exp(-((t - c - gap) / sig) ** 2))
    return y + rng.uniform(-300, 900) + rng.normal(0, 4, len(y))      # any note, CREPE-like noise

# ---------- A: time-domain features (the stage-1 set) ----------
def time_feats(c):
    sm = np.convolve(np.pad(c, 1, mode='edge'), np.ones(3) / 3, 'valid'); v = np.diff(sm)
    sg = np.sign(v[np.abs(v) > 1]); turns = int(np.sum(sg[1:] != sg[:-1])) if len(sg) > 1 else 0
    d = len(c) / FS
    return [d, np.percentile(c, 95) - np.percentile(c, 5), turns, turns / d, c[-3:].mean() - c[:3].mean(),
            abs(c[-3:].mean() - c[:3].mean()), c.max() - c[0], c[0] - c.min(), np.abs(v).mean() * FS]

# ---------- B: Fourier features ----------
NFFT = 2048; F = np.fft.rfftfreq(NFFT, 1 / FS)
G = np.exp(np.linspace(np.log(0.5), np.log(30), 160)); DL = np.log(G[1] / G[0])
RATIOS = [0.5, 0.71, 1.41, 2.0, 3.0, 4.0]                      # where to read the shape, relative to the peak
def fourier_parts(c):
    x = c - c.mean()                                          # removes the note
    size = np.sqrt(np.mean(x ** 2)) + 1e-6
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)), NFFT))
    spec = np.interp(G, F, X); spec = spec / (np.linalg.norm(spec) + 1e-12)   # removes the width
    k = int(np.argmax(spec[G > 1.0])) + int(np.sum(G <= 1.0))                 # peak above 1 Hz
    peak_hz = G[k]
    logspec = np.log(spec + 1e-6)
    shape = [np.interp(np.log(peak_hz * r), np.log(G), logspec) - logspec[k] for r in RATIOS]  # peak-aligned: removes speed
    periodicity = spec[k] / (spec.mean() + 1e-12)
    slope = np.polyfit(np.log(G[G > 1]), logspec[G > 1], 1)[0]                # power-law slope: same at any speed
    return shape + [periodicity, slope], [np.log(peak_hz), np.log(size)]
fourier_inv = lambda c: fourier_parts(c)[0]
fourier_all = lambda c: sum(fourier_parts(c), [])

# ---------- experiment: train at normal speed and width only, test on scaled versions ----------
KINDS = ['kampita', 'jaru', 'sphuritam']
def dataset(n, s_range, w_range):
    segs, y = [], []
    for k, kind in enumerate(KINDS):
        for _ in range(n):
            segs.append(make(kind, rng.uniform(*s_range), rng.uniform(*w_range))); y.append(k)
    return segs, np.array(y)
train, ytr = dataset(60, (0.9, 1.1), (0.9, 1.1))
SETS = {'A  time-domain (stage-1 features)': time_feats,
        'B  Fourier shape only (speed and size removed)': fourier_inv,
        'C  A + B': lambda c: time_feats(c) + fourier_inv(c),
        'D  B + speed and size kept as numbers': fourier_all}
models = {n: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit([f(c) for c in train], ytr) for n, f in SETS.items()}
conds = [('normal', 1, 1), ('speed x0.6', 0.6, 1), ('speed x0.8', 0.8, 1), ('speed x1.25', 1.25, 1), ('speed x1.6', 1.6, 1),
         ('width x0.6', 1, 0.6), ('width x1.6', 1, 1.6), ('both x0.6', 0.6, 0.6), ('both x1.6', 1.6, 1.6), ('both x0.5', 0.5, 0.5), ('both x2.0', 2.0, 2.0), ('speed x2.5', 2.5, 1)]
print(f"{'test condition':14s}" + ''.join(f'{n[:2]:>6s}' for n in SETS))
res = {}
for name, s, w in conds:
    te, yte = dataset(80, (s, s), (w, w))
    row = [models[n].score([f(c) for c in te], yte) for n, f in SETS.items()]; res[name] = row
    print(f'{name:14s}' + ''.join(f'{a:6.2f}' for a in row))
print('\n' + '\n'.join(SETS))
# per-class view for B at the hardest condition
from sklearn.metrics import confusion_matrix
te, yte = dataset(80, (1.6, 1.6), (1.6, 1.6))
for n in ['B  Fourier shape only (speed and size removed)', 'C  A + B']:
    print('\n', n, 'confusion at both x1.6 (rows = true kampita/jaru/sphuritam)\n', confusion_matrix(yte, models[n].predict([SETS[n](c) for c in te])))

# ---------- frame-wise version for the CNN: extra input channels ----------
def fourier_frame_features(cents, voiced, win=128):
    """cents: pitch in cents above Sa (gaps already filled), voiced: bool per frame.
    Returns [10, T]: 6 peak-aligned shape values, periodicity, spectral slope (speed- and size-free),
    then log modulation rate and log size (kept so the model can learn the singer's allowed range).
    Window 1.28 s (Hann) centred on each frame; unvoiced frames get zeros."""
    T = len(cents); h = win // 2; pad = np.pad(cents, h, mode='edge'); out = np.zeros((10, T), np.float32)
    for t in range(T):
        if not voiced[t]: continue
        inv, extra = fourier_parts(pad[t:t + win]); out[:, t] = inv + extra
    return out

if __name__ == '__main__':
    t = np.arange(300) / FS
    a = fourier_frame_features(400 + 100 * (0.5 - 0.5 * np.cos(2 * np.pi * 4 * t)), np.ones(300, bool))[:, 150]
    b = fourier_frame_features(700 + 250 * (0.5 - 0.5 * np.cos(2 * np.pi * 6 * t)), np.ones(300, bool))[:, 150]
    print('\nframe features at the centre of a kampita: 4 Hz/100 c vs 6 Hz/250 c (different note too)')
    print('shape+periodicity+slope:', np.round(a[:8], 2), '\n                        ', np.round(b[:8], 2))
    print('rate (Hz):', round(float(np.exp(a[8])), 2), round(float(np.exp(b[8])), 2), '  size ratio:', round(float(np.exp(b[9] - a[9])), 2))
