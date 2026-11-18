"""Gamaka detection pipeline: features, frame labels, dilated 1-D CNN (NumPy), post-processing,
event matching and metrics. Written to run on real CREPE output as well as the simulated set."""
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

FR = 100
K = 9
DILATIONS = (1, 2, 4, 8, 16)
NCLS = 4


# ---------------------------------------------------------------- features
def fill_short_gaps(c, voiced, max_gap=10):
    c = c.copy()
    idx = np.arange(len(c))
    v = voiced.copy()
    if v.sum() < 2:
        return np.zeros_like(c), v
    filled = np.interp(idx, idx[v], c[v])
    # short gaps become voiced
    i = 0
    while i < len(v):
        if not v[i]:
            j = i
            while j < len(v) and not v[j]:
                j += 1
            if j - i < max_gap and i > 0 and j < len(v):
                v[i:j] = True
            i = j
        else:
            i += 1
    return filled, v


def features(cents, conf):
    """6 features per frame -> array [6, T]."""
    voiced0 = (~np.isnan(cents)) & (conf >= 0.5)
    filled, voiced = fill_short_gaps(cents, voiced0)
    med = median_filter(filled, size=51, mode="nearest")
    contour = np.clip(filled - med, -600, 600) / 100.0
    vel = uniform_filter1d(np.gradient(filled), 5) / 10.0
    vel = np.clip(vel, -10, 10)
    ang = 2 * np.pi * filled / 1200.0
    f = np.stack([contour, vel, np.sin(ang), np.cos(ang), voiced.astype(float), np.nan_to_num(conf)])
    f[:4, ~voiced] = 0.0
    return f.astype(np.float32)


def frame_labels(T, ann):
    y = np.zeros(T, np.int64)
    mask = np.ones(T, np.float32)
    for s, e, l in ann:
        y[int(round(s * FR)):int(round(e * FR))] = l
    for s, e, _ in ann:                      # +-50 ms boundary mask
        for b in (s, e):
            i = int(round(b * FR))
            mask[max(0, i - 5):min(T, i + 6)] = 0.0
    return y, mask


# ---------------------------------------------------------------- CNN in NumPy
class DilatedCNN:
    def __init__(self, cin=6, ch=48, dil=DILATIONS, seed=0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.dil = dil
        self.params = []
        c = cin
        for _ in dil:
            W = rng.normal(0, np.sqrt(2.0 / (c * K)), (ch, c * K)).astype(dtype)
            self.params += [W, np.zeros(ch, dtype)]
            c = ch
        self.params += [rng.normal(0, np.sqrt(1.0 / ch), (NCLS, ch)).astype(dtype), np.zeros(NCLS, dtype)]

    def n_params(self):
        return sum(p.size for p in self.params)

    @staticmethod
    def _cols(x, d):
        T = x.shape[2]
        p = 4 * d
        xp = np.pad(x, ((0, 0), (0, 0), (p, p)))
        cols = np.stack([xp[:, :, k * d:k * d + T] for k in range(K)], axis=2)  # B,C,K,T
        return cols.reshape(x.shape[0], -1, T)

    def forward(self, x, train=False, drop=0.2, rng=None):
        cache = []
        h = x
        for li, d in enumerate(self.dil):
            W, b = self.params[2 * li], self.params[2 * li + 1]
            cols = self._cols(h, d)
            z = np.matmul(W, cols) + b[:, None]
            a = np.maximum(z, 0)
            m = None
            if train and drop > 0:
                m = (rng.random(a.shape) >= drop).astype(a.dtype) / (1 - drop)
                a = a * m
            cache.append((cols, z, m, h.shape))
            h = a
        W, b = self.params[-2], self.params[-1]
        logits = np.matmul(W, h) + b[:, None]
        cache.append(h)
        return logits, cache

    def backward(self, dlogits, cache):
        grads = [None] * len(self.params)
        h = cache[-1]
        W = self.params[-2]
        grads[-2] = np.tensordot(dlogits, h, axes=([0, 2], [0, 2]))
        grads[-1] = dlogits.sum(axis=(0, 2))
        dh = np.matmul(W.T, dlogits)
        for li in range(len(self.dil) - 1, -1, -1):
            d = self.dil[li]
            cols, z, m, xshape = cache[li]
            if m is not None:
                dh = dh * m
            dz = dh * (z > 0)
            W = self.params[2 * li]
            grads[2 * li] = np.tensordot(dz, cols, axes=([0, 2], [0, 2]))
            grads[2 * li + 1] = dz.sum(axis=(0, 2))
            if li == 0:
                break
            dcols = np.matmul(W.T, dz).reshape(xshape[0], xshape[1], K, xshape[2])
            T = xshape[2]
            p = 4 * d
            dxp = np.zeros((xshape[0], xshape[1], T + 2 * p), dz.dtype)
            for k in range(K):
                dxp[:, :, k * d:k * d + T] += dcols[:, :, k, :]
            dh = dxp[:, :, p:p + T]
        return grads


def softmax(z, axis=1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def loss_and_grad(logits, y, w):
    """weighted CE; logits B,4,T ; y B,T ; w B,T (class weight x boundary mask)"""
    p = softmax(logits)
    B, C, T = p.shape
    py = np.take_along_axis(p, y[:, None, :], axis=1)[:, 0]
    denom = w.sum() + 1e-8
    loss = -(w * np.log(py + 1e-9)).sum() / denom
    d = p.copy()
    np.put_along_axis(d, y[:, None, :], np.take_along_axis(d, y[:, None, :], axis=1) - 1, axis=1)
    d *= (w / denom)[:, None, :]
    return loss, d


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.p, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= (self.lr * mh / (np.sqrt(vh) + self.eps)).astype(p.dtype)


def predict_sliding(net, f, win=300, hop=10):
    """3 s windows, 0.1 s hop, softmax averaged over overlapping windows."""
    C, T = f.shape
    if T <= win:
        starts = [0]
        fpad = np.pad(f, ((0, 0), (0, win - T)))
    else:
        starts = list(range(0, T - win + 1, hop))
        if starts[-1] != T - win:
            starts.append(T - win)
        fpad = f
    acc = np.zeros((NCLS, max(T, win)), np.float32)
    cnt = np.zeros(max(T, win), np.float32)
    for i in range(0, len(starts), 64):
        batch = np.stack([fpad[:, s:s + win] for s in starts[i:i + 64]])
        pr = softmax(net.forward(batch)[0])
        for j, s in enumerate(starts[i:i + 64]):
            acc[:, s:s + win] += pr[j]
            cnt[s:s + win] += 1
    return (acc / cnt)[:, :T]


# ---------------------------------------------------------------- post-processing
def runs(a):
    out, s = [], 0
    for i in range(1, len(a) + 1):
        if i == len(a) or a[i] != a[s]:
            out.append((s, i, int(a[s])))
            s = i
    return out


def postprocess(lab):
    n = len(lab)
    out = lab.copy()
    for i in range(n):                       # majority over 11 frames
        w = lab[max(0, i - 5):i + 6]
        c = np.bincount(w, minlength=NCLS)
        best = out[i]
        for l in range(NCLS):
            if c[l] > c[best]:
                best = l
        out[i] = best
    for s, e, l in runs(out):                # drop runs < 0.1 s
        if l and e - s < 10:
            out[s:e] = 0
    r = runs(out)                            # close gaps < 50 ms
    for k in range(1, len(r) - 1):
        s, e, l = r[k]
        if l == 0 and e - s < 5 and r[k - 1][2] == r[k + 1][2] != 0:
            out[s:e] = r[k - 1][2]
    return out


def events_from_frames(lab):
    return [(s / FR, e / FR, l) for s, e, l in runs(lab) if l]


# ---------------------------------------------------------------- metrics
def match_events(ref, est, tol=0.2):
    """sed_eval-style event matching on onsets: same class, |onset diff| <= tol, one-to-one."""
    tp = {1: 0, 2: 0, 3: 0}
    nref = {l: sum(1 for e in ref if e[2] == l) for l in (1, 2, 3)}
    nest = {l: sum(1 for e in est if e[2] == l) for l in (1, 2, 3)}
    for l in (1, 2, 3):
        R = [e for e in ref if e[2] == l]
        E = [e for e in est if e[2] == l]
        pairs = sorted((abs(r[0] - x[0]), i, j) for i, r in enumerate(R) for j, x in enumerate(E) if abs(r[0] - x[0]) <= tol)
        ur, ue = set(), set()
        for _, i, j in pairs:
            if i not in ur and j not in ue:
                ur.add(i); ue.add(j); tp[l] += 1
    return tp, nref, nest


def f1(tp, nref, nest):
    p = tp / nest if nest else 0.0
    r = tp / nref if nref else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0
