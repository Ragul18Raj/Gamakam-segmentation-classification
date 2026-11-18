"""Stage 2 (PLAN.md section 7): find and name gamakas in full recordings. One fold per call.

usage: python stage2_vocal.py --fold P03 --out results/stage2_main [--split group|block6] [--nosa] [--epochs 30]

Reuses the simulation code in sim_code_extracted/gamaka_sim (features, NumPy dilated CNN, sliding window,
batch sampler, rule-based candidates) with the changes listed in PLAN.md section 7:
  5 classes (none + 4), boundary mask +-20 ms, majority filter tuned on the validation group from {3, 5, 7},
  real recordings instead of build_dataset(), group used as "song", the opening Sa cut off.
Baselines on the same fold: rule-based candidates + 3-NN, and a Kampita-only random forest
(features copied from kampita_from_annotations/train_kampita.py).
Writes <out>/fold_<test group>.json.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "sim_code_extracted" / "gamaka_sim"))
import pipeline as P  # noqa: E402

P.NCLS = 5  # DilatedCNN, predict_sliding read this at call time
from pipeline import FR, DilatedCNN, Adam, loss_and_grad, predict_sliding, features, runs  # noqa: E402
from run_eval import seg_feats, candidates, overlap_label, stretch, sample_batch  # noqa: E402

from vocal_pitch import load_cents, load_tables  # noqa: E402

NCLS = 5
LAB = {"Kampita": 1, "Nokku": 2, "Jaaru (Etra)": 3, "Jaaru (Irakka)": 4}
GCLS = (1, 2, 3, 4)
MASK_FR = 2           # +-20 ms
FILTERS = (5, 3, 7)   # 5 first: it wins ties
OPEN_PAD = 0.5        # cut the recording 0.5 s before the first annotated event (the opening Sa)


# ------------------------------------------------------------ data
def block6(rec):
    return "B7" if rec == "VARNAM" else f"B{(int(rec.split()[-1]) - 1) // 6 + 1}"


def load_real_dataset(split="group"):
    meta, ev = load_tables()
    clips = []
    for _, r in meta.iterrows():
        rec = r.recording
        t, cents, conf, voiced, _ = load_cents(rec, r.sa_hz)
        e = ev[ev.recording == rec]
        off = max(0.0, e.start_s.min() - OPEN_PAD)
        i0 = int(round(off * FR))
        cents = np.where(voiced, cents, np.nan)[i0:].astype(np.float32)
        ann = [(float(s - off), float(x - off), LAB[l]) for s, x, l in zip(e.start_s, e.end_s, e.label)]
        clips.append(dict(id=rec, song=r.group if split == "group" else block6(rec), group=r.group,
                          tempo=r.tempo or "(varnam)", offset=off, i0=i0,
                          cents=cents, conf=conf[i0:].astype(np.float32), ann=ann))
    return clips


def frame_labels(T, ann):
    y = np.zeros(T, np.int64)
    mask = np.ones(T, np.float32)
    for s, e, l in ann:
        y[int(round(s * FR)):int(round(e * FR))] = l
    for s, e, _ in ann:
        for b in (s, e):
            i = int(round(b * FR))
            mask[max(0, i - MASK_FR):min(T, i + MASK_FR + 1)] = 0.0
    return y, mask


def feats(c, nosa):
    f = features(c["cents"], c["conf"])
    if nosa:
        f[2:4] = 0.0
    return f


def prep(clips, augment, nosa):
    out = []
    for c in clips:
        versions = [(1.0, c["cents"], c["conf"], c["ann"])]
        if augment:
            for fac in (0.8, 0.9, 1.1, 1.25):
                versions.append((fac,) + stretch(c["cents"], c["conf"], c["ann"], fac))
        for fac, ce, co, an in versions:
            f = feats(dict(cents=ce, conf=co), nosa)
            y, m = frame_labels(f.shape[1], an)
            out.append((fac, f, y, m))
    return out


# ------------------------------------------------------------ post-processing and matching
def postprocess(lab, n=5):
    h = n // 2
    out = lab.copy()
    for i in range(len(lab)):                # majority over n frames (ties keep the current label)
        c = np.bincount(lab[max(0, i - h):i + h + 1], minlength=NCLS)
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
    return [(s / FR, e / FR, int(l)) for s, e, l in runs(lab) if l]


def match_iou(ref, est, thr=0.3):
    """Same label, IoU >= thr, one-to-one, best IoU first (eval_segments.py rule without the +-0.1 s alternative).
    -> tp, nref, nest per class, and the matched (ref, est) pairs."""
    tp, pairs_all = {}, []
    for l in GCLS:
        R = [e for e in ref if e[2] == l]
        E = [e for e in est if e[2] == l]
        pairs = []
        for i, (rs, re_, _) in enumerate(R):
            for j, (es, ee, _) in enumerate(E):
                inter = max(0.0, min(re_, ee) - max(rs, es))
                uni = max(re_, ee) - min(rs, es)
                if uni > 0 and inter / uni >= thr:
                    pairs.append((inter / uni, i, j))
        ur, ue, n = set(), set(), 0
        for _, i, j in sorted(pairs, reverse=True):
            if i not in ur and j not in ue:
                ur.add(i)
                ue.add(j)
                n += 1
                pairs_all.append((R[i], E[j]))
        tp[l] = n
    nref = {l: sum(e[2] == l for e in ref) for l in GCLS}
    nest = {l: sum(e[2] == l for e in est) for l in GCLS}
    return tp, nref, nest, pairs_all


def ev_macro(clips_ev, key):
    tp = {l: 0 for l in GCLS}
    nr, ne = dict(tp), dict(tp)
    for ref, est in clips_ev:
        a, b, c, _ = match_iou(ref, est[key])
        for l in GCLS:
            tp[l] += a[l]
            nr[l] += b[l]
            ne[l] += c[l]
    f = [2 * tp[l] / (nr[l] + ne[l]) for l in GCLS if nr[l] + ne[l]]
    return float(np.mean(f)) if f else 0.0


# ------------------------------------------------------------ CNN training (as run_eval.train_cnn, 5 classes)
def val_score(net, data):
    ys, ps = [], []
    for _, f, y, _ in data:
        ps.append(net.forward(f[None])[0][0].argmax(0))
        ys.append(y)
    return f1_score(np.concatenate(ys), np.concatenate(ps), labels=list(range(NCLS)), average="macro", zero_division=0)


def train_cnn(train, val, epochs, seed, log):
    rng = np.random.default_rng(seed)
    allY = np.concatenate([d[2] for d in train if d[0] == 1.0])
    freq = np.maximum(np.bincount(allY, minlength=NCLS) / len(allY), 1e-6)
    cw = (freq ** -0.5).astype(np.float32)
    cw /= (cw[allY]).mean()
    net = DilatedCNN(seed=seed)
    opt = Adam(net.params, lr=1e-3)
    best, best_p, bad = -1, None, 0
    for ep in range(epochs):
        t0, tot = time.time(), 0
        for _ in range(75):
            X, Y, W = sample_batch(train, cw, rng)
            lg, cache = net.forward(X, train=True, rng=rng)
            L, d = loss_and_grad(lg, Y, W)
            opt.step(net.backward(d, cache))
            tot += L
        vs = val_score(net, val)
        log(f"    epoch {ep + 1:2d} loss {tot / 75:.3f} val frame macroF1 {vs:.3f} ({time.time() - t0:.0f}s)")
        if vs > best:
            best, best_p, bad = vs, [p.copy() for p in net.params], 0
        else:
            bad += 1
            if bad >= 5:
                break
    net.params[:] = best_p
    return net, best


# ------------------------------------------------------------ Kampita-only random forest (train_kampita.py features)
KWIN, KCONF = 30, 0.3


def kampita_feats(act):
    c0 = 1997.3794084376191 + 20 * np.arange(360)
    k = act.argmax(1)
    idx = np.clip(k[:, None] + np.arange(-4, 5), 0, 359)
    w = np.take_along_axis(act, idx, 1)
    cents, conf = (w * c0[idx]).sum(1) / (w.sum(1) + 1e-9), act.max(1)
    voiced = conf >= KCONF
    if voiced.any():
        vi = np.where(voiced)[0]
        cents = np.interp(np.arange(len(cents)), vi, cents[vi])
    d = np.gradient(cents)
    h = KWIN // 2
    cp, dp, vp = np.pad(cents, h, mode="edge"), np.pad(d, h, mode="edge"), np.pad(conf, h, mode="edge")
    W = np.lib.stride_tricks.sliding_window_view(cp, KWIN)[:len(cents)]
    D = np.lib.stride_tricks.sliding_window_view(dp, KWIN)[:len(cents)]
    C = np.lib.stride_tricks.sliding_window_view(vp, KWIN)[:len(cents)]
    w = W - W.mean(1, keepdims=True)
    x = np.arange(KWIN) - (KWIN - 1) / 2
    r = w - (w * x).sum(1, keepdims=True) / (x ** 2).sum() * x
    sc = (np.diff(np.sign(D), axis=1) != 0).sum(1)
    spec = np.abs(np.fft.rfft(r, 64, axis=1))
    fr = np.fft.rfftfreq(64, 1 / FR)
    band = spec[:, (fr >= 3) & (fr <= 12)].sum(1) / (spec[:, fr > 0].sum(1) + 1e-9)
    peak = fr[1:][spec[:, 1:].argmax(1)]
    return np.column_stack([r.std(1), np.ptp(r, 1), w.std(1), np.abs(D).mean(1), sc, band, peak, C.mean(1), (C >= KCONF).mean(1)])


def kampita_segments(prob, thr=0.5, min_len=0.25):
    lab = median_filter((prob >= thr).astype(int), 15)
    return [(s / FR, e / FR, 1) for s, e, l in runs(lab) if l and (e - s) / FR >= min_len]


def kampita_data(c):
    act = np.load(ROOT / "crepe" / f"{c['id']}.npy")[c["i0"]:]
    X = kampita_feats(act)
    y = np.zeros(len(X), int)
    for s, e, l in c["ann"]:
        if l == 1:
            y[int(round(s * FR)):int(round(e * FR))] = 1
    return X, y


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="group", choices=["group", "block6"])
    ap.add_argument("--nosa", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / f"log_{a.fold}.txt", "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    clips = load_real_dataset(a.split)
    groups = sorted({c["song"] for c in clips})
    test = a.fold
    val = groups[(groups.index(test) + 1) % len(groups)]
    tr = [c for c in clips if c["song"] not in (test, val)]
    va = [c for c in clips if c["song"] == val]
    te = [c for c in clips if c["song"] == test]
    seed = groups.index(test) + 1
    log(f"fold test={test} val={val} train={len(tr)} recordings, nosa={a.nosa}, split={a.split}")
    res = dict(test=test, val=val, split=a.split, nosa=a.nosa, train=[c["id"] for c in tr], val_recs=[c["id"] for c in va],
               clips=[])

    # rule-based baseline: movement candidates + 3-NN (fitted on train + val, as run_eval.py)
    Xc, yc = [], []
    for c in tr + va:
        f = features(c["cents"], c["conf"])
        for s, e in candidates(f):
            Xc.append(seg_feats(c["cents"], c["conf"], s, e))
            yc.append(overlap_label(s, e, c["ann"]))
    sc2 = StandardScaler().fit(Xc)
    knn = KNeighborsClassifier(3).fit(sc2.transform(Xc), yc)

    # Kampita-only random forest (train + val)
    KX, Ky = zip(*[kampita_data(c) for c in tr + va])
    rf = RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced", n_jobs=1, max_samples=0.5,
                                random_state=0).fit(np.vstack(KX), np.concatenate(Ky))

    # CNN
    t0 = time.time()
    net, vbest = train_cnn(prep(tr, True, a.nosa), prep(va, False, a.nosa), a.epochs, seed=seed, log=log)
    log(f"  CNN trained in {time.time() - t0:.0f}s, best val frame macroF1 {vbest:.3f}")

    # majority-filter size: chosen on the validation group only (event macro-F1, IoU rule)
    vraw = [(c["ann"], predict_sliding(net, feats(c, a.nosa)).argmax(0)) for c in va]
    vs = {n: ev_macro([(ref, {"e": events_from_frames(postprocess(raw, n))}) for ref, raw in vraw], "e") for n in FILTERS}
    nf = max(FILTERS, key=lambda n: vs[n])  # max() keeps the first of equal values, so 5 wins ties
    res["filter_val_scores"], res["filter"], res["val_frame_f1"] = vs, nf, vbest
    log(f"  filter chosen on val: {nf}  {vs}")

    for c in te:
        f = feats(c, a.nosa)
        T = f.shape[1]
        y, _ = frame_labels(T, c["ann"])
        raw = predict_sliding(net, f).argmax(0)
        pp = postprocess(raw, nf)
        fr = features(c["cents"], c["conf"])
        ev_rule = []
        for s, e in candidates(fr):
            l = int(knn.predict(sc2.transform([seg_feats(c["cents"], c["conf"], s, e)]))[0])
            if l:
                ev_rule.append((s, e, l))
        rule_frames = np.zeros(T, np.int64)
        for s, e, l in ev_rule:
            rule_frames[int(round(s * FR)):int(round(e * FR))] = l
        kx, _ = kampita_data(c)
        ev_rf = kampita_segments(rf.predict_proba(kx)[:, 1])
        res["clips"].append(dict(id=c["id"], group=c["group"], tempo=c["tempo"], offset=c["offset"], ref=c["ann"],
                                 cnn=events_from_frames(pp), rule=ev_rule, kampita_rf=ev_rf, y=y.tolist(),
                                 cnn_frames=pp.tolist(), cnn_raw=raw.tolist(), rule_frames=rule_frames.tolist()))
    json.dump(res, open(out / f"fold_{test}.json", "w"))
    log(f"  fold {test} saved ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
