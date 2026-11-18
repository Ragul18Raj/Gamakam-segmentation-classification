"""Leave-one-song-out evaluation on the simulated vocal-style dataset.
Usage: python run_eval.py [--folds 1,2,...] [--epochs 30] [--out results]"""
import argparse, json, time, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from scipy.ndimage import uniform_filter1d
from sim_gamaka import build_dataset, TEMPOS, OCTAVES
from pipeline import (FR, NCLS, features, frame_labels, fill_short_gaps, DilatedCNN, loss_and_grad,
                      Adam, predict_sliding, postprocess, events_from_frames, match_events, f1, runs, softmax)

WIN = 300


# ------------------------------------------------------------ segment features (stage 1, rule baseline)
def seg_feats(cents, conf, s, e):
    voiced = (~np.isnan(cents)) & (conf >= 0.5)
    filled, _ = fill_short_gaps(cents, voiced)
    a, b = int(round(s * FR)), max(int(round(e * FR)), int(round(s * FR)) + 3)
    c = filled[a:b]
    sm = uniform_filter1d(c, 3)
    v = np.diff(sm)
    sgn = np.sign(v[np.abs(v) > 1.0])
    turns = int(np.sum(sgn[1:] != sgn[:-1])) if len(sgn) > 1 else 0
    dur = (b - a) / FR
    base = np.median(filled[max(0, a - 20):a]) if a > 0 else c[0]
    return [dur, np.percentile(c, 95) - np.percentile(c, 5), turns, turns / dur,
            c[-3:].mean() - c[:3].mean(), abs(c[-3:].mean() - c[:3].mean()),
            c.max() - base, base - c.min(), np.abs(v).mean() * FR]


def candidates(f):
    contour, vel, voiced = f[0], f[1], f[4] > 0.5
    m = voiced & ((np.abs(contour) > 0.4) | (np.abs(vel) > 0.3))
    lab = m.astype(np.int64)
    r = runs(lab)
    for k in range(1, len(r) - 1):                 # close gaps < 0.15 s
        s, e, l = r[k]
        if l == 0 and e - s < 15:
            lab[s:e] = 1
    return [(s / FR, e / FR) for s, e, l in runs(lab) if l and e - s >= 10]


def overlap_label(s, e, ann):
    best, bl = 0.0, 0
    for a, b, l in ann:
        ov = max(0.0, min(e, b) - max(s, a))
        if ov > best:
            best, bl = ov, l
    return bl if best > 0.5 * (e - s) else 0


# ------------------------------------------------------------ data prep
def stretch(cents, conf, ann, factor):
    T = len(cents)
    Tn = int(round(T * factor))
    src = np.linspace(0, T - 1, Tn)
    idx = np.round(src).astype(int)
    c = np.interp(src, np.arange(T), np.nan_to_num(cents, nan=0.0)).astype(np.float32)
    c[np.isnan(cents)[idx]] = np.nan
    return c, conf[idx], [(s * factor, e * factor, l) for s, e, l in ann]


def prep(clips, augment):
    out = []
    for c in clips:
        versions = [(1.0, c["cents"], c["conf"], c["ann"])]
        if augment:
            for fac in (0.8, 0.9, 1.1, 1.25):
                versions.append((fac,) + stretch(c["cents"], c["conf"], c["ann"], fac))
        for fac, ce, co, an in versions:
            f = features(ce, co)
            y, m = frame_labels(f.shape[1], an)
            out.append((fac, f, y, m))
    return out


def sample_batch(data, cw, rng, bs=16):
    X, Y, W = [], [], []
    for _ in range(bs):
        if rng.random() < 0.5:
            cands = [d for d in data if d[0] == 1.0]
        else:
            cands = data
        _, f, y, m = cands[rng.integers(len(cands))]
        T = f.shape[1]
        if T <= WIN:
            s = 0
            f = np.pad(f, ((0, 0), (0, WIN - T))); y = np.pad(y, (0, WIN - T)); m = np.pad(m, (0, WIN - T))
        else:
            s = int(rng.integers(0, T - WIN + 1))
        x = f[:, s:s + WIN].copy()
        dlt = 2 * np.pi * rng.uniform(-30, 30) / 1200          # tonic-error augmentation
        sn, cs = x[2].copy(), x[3].copy()
        v = x[4] > 0.5
        x[2] = np.where(v, sn * np.cos(dlt) + cs * np.sin(dlt), 0)
        x[3] = np.where(v, cs * np.cos(dlt) - sn * np.sin(dlt), 0)
        yy = y[s:s + WIN]
        X.append(x); Y.append(yy); W.append(m[s:s + WIN] * cw[yy])
    return np.stack(X).astype(np.float32), np.stack(Y), np.stack(W).astype(np.float32)


def val_score(net, data):
    ys, ps = [], []
    for fac, f, y, m in data:
        p = net.forward(f[None])[0][0].argmax(0)
        ys.append(y); ps.append(p)
    return f1_score(np.concatenate(ys), np.concatenate(ps), labels=[0, 1, 2, 3], average="macro", zero_division=0)


def train_cnn(train, val, epochs, seed, log):
    rng = np.random.default_rng(seed)
    allY = np.concatenate([d[2] for d in train if d[0] == 1.0])
    freq = np.bincount(allY, minlength=NCLS) / len(allY)
    cw = (freq ** -0.5).astype(np.float32)
    cw /= (cw[allY]).mean()
    net = DilatedCNN(seed=seed)
    opt = Adam(net.params, lr=1e-3)
    best, best_p, bad = -1, None, 0
    for ep in range(epochs):
        t0 = time.time(); tot = 0
        for it in range(75):
            X, Y, W = sample_batch(train, cw, rng)
            lg, cache = net.forward(X, train=True, rng=rng)
            L, d = loss_and_grad(lg, Y, W)
            opt.step(net.backward(d, cache)); tot += L
        vs = val_score(net, val)
        log(f"    epoch {ep+1:2d} loss {tot/75:.3f} val macroF1 {vs:.3f} ({time.time()-t0:.0f}s)")
        if vs > best:
            best, best_p, bad = vs, [p.copy() for p in net.params], 0
        else:
            bad += 1
            if bad >= 5:
                break
    net.params[:] = best_p
    return net, best


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="1,2,3,4,5,6,7")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="results")
    ap.add_argument("--stress", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    clips = build_dataset(stress=a.stress)
    songs = sorted({c["song"] for c in clips})
    logf = open(os.path.join(a.out, "log.txt"), "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    for test_song in [int(x) for x in a.folds.split(",")]:
        val_song = songs[test_song % len(songs)]
        tr = [c for c in clips if c["song"] not in (test_song, val_song)]
        va = [c for c in clips if c["song"] == val_song]
        te = [c for c in clips if c["song"] == test_song]
        log(f"fold test=song{test_song} val=song{val_song} train={len(tr)} clips")
        res = {"test_song": test_song, "val_song": val_song, "clips": []}

        # stage 1: classify annotated segments (train + val songs used for fitting)
        fit = tr + va
        Xs = [seg_feats(c["cents"], c["conf"], s, e) for c in fit for s, e, l in c["ann"]]
        ys = [l for c in fit for s, e, l in c["ann"]]
        sc = StandardScaler().fit(Xs)
        lr = LogisticRegression(max_iter=2000).fit(sc.transform(Xs), ys)
        Xt = [seg_feats(c["cents"], c["conf"], s, e) for c in te for s, e, l in c["ann"]]
        yt = [l for c in te for s, e, l in c["ann"]]
        res["stage1"] = {"y": yt, "p": lr.predict(sc.transform(Xt)).tolist()}

        # rule-based baseline: movement candidates + 3-NN
        Xc, yc = [], []
        for c in fit:
            f = features(c["cents"], c["conf"])
            for s, e in candidates(f):
                Xc.append(seg_feats(c["cents"], c["conf"], s, e)); yc.append(overlap_label(s, e, c["ann"]))
        sc2 = StandardScaler().fit(Xc)
        knn = KNeighborsClassifier(3).fit(sc2.transform(Xc), yc)

        # CNN
        t0 = time.time()
        net, vbest = train_cnn(prep(tr, True), prep(va, False), a.epochs, seed=test_song, log=log)
        log(f"  CNN trained in {time.time()-t0:.0f}s, best val macroF1 {vbest:.3f}")

        for c in te:
            f = features(c["cents"], c["conf"])
            T = f.shape[1]
            y, _ = frame_labels(T, c["ann"])
            prob = predict_sliding(net, f)
            raw = prob.argmax(0)
            pp = postprocess(raw)
            ev_cnn = events_from_frames(pp)
            ev_rule = []
            for s, e in candidates(f):
                l = int(knn.predict(sc2.transform([seg_feats(c["cents"], c["conf"], s, e)]))[0])
                if l:
                    ev_rule.append((s, e, l))
            rule_frames = np.zeros(T, np.int64)
            for s, e, l in ev_rule:
                rule_frames[int(s * FR):int(e * FR)] = l
            res["clips"].append(dict(id=c["id"], tempo=c["tempo"], octave=c["octave"], ref=c["ann"],
                                     cnn=ev_cnn, rule=ev_rule, y=y.tolist(), cnn_frames=pp.tolist(),
                                     cnn_raw=raw.tolist(), rule_frames=rule_frames.tolist()))
        with open(os.path.join(a.out, f"fold{test_song}.json"), "w") as fh:
            json.dump(res, fh)
        log(f"  fold song{test_song} saved")


if __name__ == "__main__":
    main()
