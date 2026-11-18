"""Pitch preparation shared by stage 1 and stage 2 (PLAN.md section 4). No class labels are used here.

load_cents(rec, sa_hz) -> t, cents, conf, voiced
  cents = 1200*log2(f0 / Sa) for the whole recording, with short octave slips folded back:
  a voiced run (no adjacent step > 150 c) shorter than 150 ms that sits 1000-1400 cents from BOTH neighbouring runs,
  in the same direction, is moved by 1200 cents towards them.
event_curve(t, cents, voiced, s, e) -> curve over the event's first..last voiced frame, inner gaps linearly interpolated,
  and the number of voiced frames.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CONF = 0.5
JUMP = 150      # cents between adjacent frames: a larger step starts a new run
SLIP_FR = 15    # runs shorter than this (150 ms) may be folded
SLIP_LO, SLIP_HI = 1000, 1400


def runs_of(cents, voiced):
    out, i, n = [], 0, len(cents)
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i + 1
        while j < n and voiced[j] and abs(cents[j] - cents[j - 1]) <= JUMP:
            j += 1
        out.append((i, j))
        i = j
    return out


def fold_octave_slips(cents, voiced):
    c = cents.copy()
    runs = runs_of(c, voiced)
    n_fold = 0
    for k in range(1, len(runs) - 1):
        a, b = runs[k]
        if b - a >= SLIP_FR:
            continue
        m = np.median(c[a:b])
        dl, dr = m - c[runs[k - 1][1] - 1], m - c[runs[k + 1][0]]
        if SLIP_LO <= abs(dl) <= SLIP_HI and SLIP_LO <= abs(dr) <= SLIP_HI and np.sign(dl) == np.sign(dr):
            c[a:b] -= 1200 * np.sign(dl)
            n_fold += 1
    return c, n_fold


def load_cents(rec, sa_hz):
    d = np.load(ROOT / "crepe" / f"{rec}.npz")
    t, f, conf = d["time_s"], d["f0_hz"], d["confidence"]
    voiced = conf >= CONF
    cents = 1200 * np.log2(np.maximum(f, 1e-3) / sa_hz)
    cents, n_fold = fold_octave_slips(cents, voiced)
    return t, cents, conf, voiced, n_fold


def event_curve(t, cents, voiced, s, e):
    idx = np.where((t >= s) & (t < e))[0]
    v = idx[voiced[idx]]
    if len(v) == 0:
        return np.zeros(0), 0
    span = np.arange(v[0], v[-1] + 1)
    return np.interp(span, v, cents[v]), len(v)


def load_tables():
    meta = pd.read_csv(ROOT / "metadata.csv", dtype=str).fillna("")
    meta["sa_hz"] = meta.sa_hz.astype(float)
    ev = pd.read_csv(ROOT / "annotations" / "pass1_events.csv")
    return meta, ev
