"""
gamaka_clips.py -- do discovered movement shapes match the named gamakas?

This is the veena branch. Each clip in Gamaka_veena_dataset is one demonstrated
gesture, and its filename carries the gamaka name. So for once we have labels.

Two questions, both answered here:
  1. UNSUPERVISED: cluster the clips by shape alone, then check whether the clusters
     line up with the names (this is the step where Aruna names the clusters).
  2. SUPERVISED: can a simple classifier tell the named gamakas apart from pitch shape?

Both are scored with GROUPED splits. Clips from one take (ahatam#01.1, #01.3, ...)
are near-duplicates; scoring them against each other inflates accuracy badly. The
ungrouped number is printed too, purely to show the size of that inflation.

Usage:
  python gamaka_clips.py --crepe_dir crepe_output --tonic 329.3 --out clip_features.csv

Needs movement_features_v2.py and anchor_feasibility_check_v3.py alongside it.
"""

import argparse
import csv
import os
import re
from collections import Counter, defaultdict

import numpy as np

import movement_features_v2 as mf

# Spelling variants that are the same gamaka. Checked against the filename stem.
# pratayagata/pratyagata: same gamaka, two spellings, confirmed from the filenames.
# sphutitam is NOT merged into sphuritam -- one file, six times longer than any
# sphuritam clip and named like a different series. Let the artist rule on it.
ALIASES = {"pratayagata": "pratyagata"}

GAMAKAS = ["ahatam", "andolita", "jaru", "kampita", "kandippu", "namitam", "nokku",
           "odukkul", "orikai", "pratyagata", "pratayagata", "ravai", "sphuritam",
           "sphutitam", "tribhinna", "vali"]


def label_of(stem):
    low = stem.lower()
    for g in sorted(GAMAKAS, key=len, reverse=True):
        if low.startswith(g):
            return ALIASES.get(g, g)
    return ""


def group_of(stem):
    """One take = one group.

    'ahatam#01.3Guitar...' and 'ahatam#01.5...' are the same take; 'kampita#01' and
    'kampita#02' are different ones. Trailing clip indexes are stripped, the '#NN'
    take number is kept. Clips of one take sit end to end on the same timeline, so
    scoring them against each other would leak.
    """
    base = stem.lower().split("guitar")[0]
    base = base.replace("!", " ")                   # "_!" and "_!2" are clip markers
    base = re.sub(r"(?<=[._\-\s])\d+(?=[._\-\s]|$)", " ", base)   # index tokens anywhere
    base = re.sub(r"[._\-\s]+\d+$", "", base)       # a trailing index with no separator
    base = re.sub(r"[._\-\s]+", "_", base)          # one separator, so " x" == "_x"
    return base.strip("_")


def union_time(rows):
    """Total time covered by these rows, counting overlaps once."""
    spans = sorted((r["start_s"], r["start_s"] + r["duration_s"]) for r in rows)
    total, cur_s, cur_e = 0.0, None, None
    for a, b in spans:
        if cur_e is None or a > cur_e:
            total += (cur_e - cur_s) if cur_e is not None else 0.0
            cur_s, cur_e = a, b
        else:
            cur_e = max(cur_e, b)
    return total + ((cur_e - cur_s) if cur_e is not None else 0.0)


def clip_features(path, tonic, hop_ms):
    """One row per clip: what shapes it contains, how big, how fast."""
    rows, note_time, held = mf.extract(path, tonic, list(np.arange(12) * 100.0), hop_ms)
    movs = [r for r in rows if r["type"] != "oscillation"]
    oscs = [r for r in rows if r["type"] == "oscillation"]
    used = [r for r in movs if r["used_in_fingerprint"]]
    total_s = sum(h["duration_s"] for h in held) + sum(r["duration_s"] for r in rows)
    ext = [r["extent_cents"] for r in used] or [0]
    turns = [r["n_turns"] for r in used] or [0]
    dirs = [r["direction"] for r in used] or [0]
    return {
        "n_movements": len(used),
        "n_oscillations": len(oscs),
        # union, not sum: contour and chain rows may overlap, and double counting
        # would push the share above 1
        "osc_time_share": round(union_time(oscs) / total_s, 3) if total_s else 0,
        "osc_extent": round(float(np.median([r["extent_cents"] for r in oscs])), 1) if oscs else 0,
        "osc_turns": round(float(np.median([r["n_turns"] for r in oscs])), 1) if oscs else 0,
        "median_extent": round(float(np.median(ext)), 1),
        "max_extent": round(float(np.max(ext)), 1),
        "median_turns": round(float(np.median(turns)), 2),
        "max_turns": int(np.max(turns)),
        "mean_direction": round(float(np.mean(dirs)), 2),
        "held_share": round(sum(h["duration_s"] for h in held) / total_s, 3) if total_s else 0,
        "n_held": len(held),
        "duration_s": round(total_s, 2),
    }


FEATS = ["n_oscillations", "osc_time_share", "osc_extent", "osc_turns", "median_extent",
         "max_extent", "median_turns", "max_turns", "mean_direction", "held_share"]


def zscore(X):
    mu, sd = X.mean(0), X.std(0)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def kmeans(X, k, seed=0, iters=100):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        newC = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return lab


def purity(clusters, labels):
    tot = 0
    for c in set(clusters):
        ls = [labels[i] for i in range(len(labels)) if clusters[i] == c]
        tot += Counter(ls).most_common(1)[0][1]
    return tot / len(labels)


def knn_score(X, y, groups, k=3, grouped=True):
    """Leave-one-out, optionally excluding every clip from the same take."""
    correct = 0
    D = ((X[:, None, :] - X[None]) ** 2).sum(-1)
    preds = []
    for i in range(len(X)):
        mask = np.ones(len(X), bool)
        mask[i] = False
        if grouped:
            mask &= np.array([g != groups[i] for g in groups])
        if not mask.any():
            preds.append(None)
            continue
        idx = np.argsort(np.where(mask, D[i], np.inf))[:k]
        vote = Counter(y[j] for j in idx).most_common(1)[0][0]
        preds.append(vote)
        correct += vote == y[i]
    return correct / len(X), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crepe_dir", required=True)
    ap.add_argument("--tonic", type=float, required=True)
    ap.add_argument("--out", default="clip_features.csv")
    ap.add_argument("--hop_ms", type=float, default=10.0)
    ap.add_argument("--clusters", type=int, default=0, help="0 = use the number of labels")
    ap.add_argument("--takes_csv", help="optional CSV with columns file,take -- real take "
                    "boundaries (e.g. from the Logic timeline). Overrides filename guessing, "
                    "which cannot see that two differently-named clips are one continuous take.")
    a = ap.parse_args()

    takes = {}
    if a.takes_csv:
        with open(a.takes_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                takes[os.path.splitext(r["file"])[0]] = r["take"]
        print(f"take boundaries read from {a.takes_csv} ({len(set(takes.values()))} takes)")

    rows = []
    for fn in sorted(os.listdir(a.crepe_dir)):
        if not fn.endswith(".npy"):
            continue
        stem = os.path.splitext(fn)[0]
        f = clip_features(os.path.join(a.crepe_dir, fn), a.tonic, a.hop_ms)
        rows.append({"file": fn, "label": label_of(stem),
                     "take": takes.get(stem, group_of(stem)), **f})

    with open(a.out, "w", newline="", encoding="utf-8") as g:
        w = csv.DictWriter(g, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} clips -> {a.out}")

    # Clips with no movement at all are KEPT. "Entirely held note, no pitch motion"
    # is the signature of a struck gamaka like ahatam, not a reason to drop the clip.
    keep = [r for r in rows if r["label"] and r["duration_s"] > 0]
    skipped = len(rows) - len(keep)
    if skipped:
        print(f"{skipped} clips skipped (no gamaka name in filename, or no pitch at all)")
    flat = sum(1 for r in keep if r["n_movements"] == 0)
    print(f"{flat} of the kept clips have no movement >= 0.05s (all held note)")
    y = [r["label"] for r in keep]
    groups = [r["take"] for r in keep]
    X = zscore(np.array([[float(r[f]) for f in FEATS] for r in keep]))

    counts = Counter(y)
    print(f"\n{len(keep)} clips, {len(counts)} gamaka types, {len(set(groups))} takes")
    print("  " + ", ".join(f"{k}:{v}" for k, v in counts.most_common()))
    per_take = defaultdict(set)
    for g_, l_ in zip(groups, y):
        per_take[l_].add(g_)
    thin = [l for l, t in per_take.items() if len(t) < 2]
    if thin:
        print(f"  WARNING: only one take for {', '.join(thin)} -- grouped scoring cannot "
              "test these at all")

    k = a.clusters or len(counts)
    clusters = kmeans(X, k)
    print(f"\nUNSUPERVISED: {k} clusters from shape alone, purity {purity(clusters, y):.2f}")
    print("  (purity 1.0 = every cluster is one gamaka; "
          f"{max(counts.values()) / len(y):.2f} = what guessing the commonest gives)")
    print("\n  cluster: what's in it")
    for c in range(k):
        ls = Counter(y[i] for i in range(len(y)) if clusters[i] == c)
        if ls:
            print(f"  {c:>3}: " + ", ".join(f"{n}x{l}" for l, n in ls.most_common(4)))

    g_acc, preds = knn_score(X, y, groups, grouped=True)
    u_acc, _ = knn_score(X, y, groups, grouped=False)
    print(f"\nSUPERVISED (3-NN on shape features)")
    print(f"  leave-one-TAKE-out : {g_acc:.2f}   <- the honest number")
    print(f"  leave-one-clip-out : {u_acc:.2f}   <- inflated by near-duplicate takes")

    print("\n  most confused pairs (true -> predicted):")
    conf = Counter((t, p) for t, p in zip(y, preds) if p and p != t)
    for (t, p), n in conf.most_common(8):
        print(f"    {t:>12} -> {p:<12} {n}")
    print("\nPitch alone cannot separate gamakas that differ by articulation "
          "(Kandippu/Jaru) or by\nplucking several strings (Tribhinna). Expect those "
          "to merge -- that is a finding, not a bug.\n")


if __name__ == "__main__":
    main()