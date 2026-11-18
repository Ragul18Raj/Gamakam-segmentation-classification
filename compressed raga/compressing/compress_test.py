#!/usr/bin/env python3
"""compress_test.py -- does the raga prediction survive lossy compression of the input audio?

Each recording is re-encoded as constant-bitrate MP3 at a ladder of bitrates (LAME via libsndfile; there is
no working ffmpeg on this box, and libsndfile's Opus output does not decode), then the COMPRESSED FILE is
given to the complete condition-D pipeline -- the recipe that built the training data (PAPER/extract_D.py):
  decode at 16 kHz   -> torchcrepe 'full', hop 186, 50-1000 Hz, Viterbi per 1024-frame batch, first 14000 frames
  decode at 44.1 kHz -> Essentia TonicIndianArtMusic over the whole recording
  -> Gaussian paint (sigma 1.2, Sa at bin 0) -> SOAR cvD (real-audio build)
The tonic is re-estimated from the compressed audio, so compression can hurt through the pitch track, the
tonic, or the classifier; the first two are reported separately (pitch_agree, tonic_dev_c).

Modes
  --audio FILE [--tonic-hz HZ]  your own recording: prediction at every bitrate. A CMD-40 file is scored by the
                                fold model that never saw it; anything else by the 5 cvD fold models averaged.
  --cmd N                       N CMD-40 recordings per raga, each scored by its out-of-fold cvD model;
                                rows go to results/results*.csv. Shard with --shard k --nshard n.
  --summarize                   merge results/results*.csv -> results/RESULT.md

Originals are only read: each source file is copied into work/ first, and every encode lands there too.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "live"))
import raga_live as rl                               # condition-D constants, paint, Ensemble, CMD index

WORK, RES = HERE / "work", HERE / "results"
LEVELS = [  # name, encode sample rate, libsndfile compression_level (CBR; kbps measured on CMD audio)
    ("original", None, None),                        # the CMD mp3 as shipped (~160 kbps)
    ("mp3_96k", 44100, 0.80),
    ("mp3_56k", 44100, 0.90),
    ("mp3_32k", 44100, 0.99),
    ("mp3_24k", 22050, 0.90),                        # below 32 kbps LAME needs a lower sample rate
    ("mp3_16k", 22050, 0.95),
    ("mp3_8k", 22050, 0.99),
]
# torchcrepe.predict decodes each 1024-frame batch on its own, so the first 14000 frames depend only on the
# audio up to batch 13's last window. Cutting there (plus a window) gives identical f0 and skips the rest.
CREPE_SAMPLES = 14 * rl.BLOCK * rl.HOP + rl.WIN
SEED = 0
FIELDS = ["stem", "raga", "fold", "level", "kbps", "ratio", "tonic_hz", "tonic_dev_c", "pitch_agree",
          "pred", "p_pred", "p_true", "rank_true", "correct", "pred_stored", "sec"]


def encode(src: Path, dst: Path, rate: int, level: float) -> float:
    """src -> CBR MP3 at dst; returns kbps measured from the file size."""
    import soundfile as sf, librosa
    y, sr = sf.read(str(src), dtype="float32", always_2d=True)          # [n, ch], channels kept
    if rate != sr:
        y = librosa.resample(y.T, orig_sr=sr, target_sr=rate).T
    sf.write(str(dst), y, rate, format="MP3", subtype="MPEG_LAYER_III",
             compression_level=level, bitrate_mode="CONSTANT")
    return os.path.getsize(dst) * 8 / (len(y) / rate) / 1000


def kbps_of(path: Path) -> float:
    import soundfile as sf
    return os.path.getsize(path) * 8 / sf.info(str(path)).duration / 1000


def pitch_track(y16: np.ndarray, device: str) -> np.ndarray:
    """extract_D.py's call. torchcrepe dithers every f0 with numpy's RNG, so it is reseeded: every level
    gets the same dither draw and differences between levels come from the audio alone."""
    import torch, torchcrepe
    np.random.seed(SEED)
    with torch.no_grad():
        f0 = torchcrepe.predict(torch.from_numpy(y16[:CREPE_SAMPLES])[None].float(), rl.SR, rl.HOP,
                                fmin=rl.FMIN, fmax=rl.FMAX, model="full", batch_size=1024, device=device,
                                return_periodicity=False).squeeze().cpu().numpy()
    return f0[:rl.MAX_FRAMES].astype(np.float32)


def tonic_of(path: Path) -> float:
    from essentia.standard import MonoLoader, TonicIndianArtMusic
    return float(TonicIndianArtMusic()(MonoLoader(filename=str(path), sampleRate=rl.TSR)()))


def pitch_agreement(f0: np.ndarray, ref: np.ndarray, max_lag: int = 8) -> float:
    """Share of frames within 50 cents of the uncompressed track (the seeded dither is at most 40 cents
    apart), best over small lags in case the decoder shifts the audio by an encoder delay."""
    a, b = 1200 * np.log2(np.maximum(f0, 1e-6)), 1200 * np.log2(np.maximum(ref, 1e-6))
    best = 0.0
    for lag in range(-max_lag, max_lag + 1):
        x, y = (a[lag:], b[:len(b) - lag]) if lag >= 0 else (a[:lag], b[-lag:])
        n = min(len(x), len(y))
        if n:
            best = max(best, float(np.mean(np.abs(x[:n] - y[:n]) <= 50)))
    return best


def cents_apart(hz: float, ref: float) -> float:
    return 1200 * math.log2(hz / ref)


def prepare(job):
    """Pool worker, the CPU-bound part of one level: encode it (unless original) and estimate its tonic."""
    copy, path, rate, lvl, want_tonic = job
    kbps = kbps_of(copy) if rate is None else encode(copy, path, rate, lvl)
    return kbps, (tonic_of(path) if want_tonic else None)


def run_levels(src: Path, ens, device: str, true_raga: str | None, tonic_fixed: float | None, keep: bool,
               pool):
    """All levels of one recording -> list of row dicts (no stem/fold yet)."""
    import librosa
    tag = src.stem
    WORK.mkdir(parents=True, exist_ok=True)
    copy = WORK / f"{tag}__src{src.suffix}"
    shutil.copy2(src, copy)                                              # never touch the original
    src_kbps = kbps_of(copy)
    paths = [copy if rate is None else WORK / f"{tag}__{name}.mp3" for name, rate, _ in LEVELS]
    jobs = [pool.submit(prepare, (copy, path, rate, lvl, tonic_fixed is None))
            for path, (_, rate, lvl) in zip(paths, LEVELS)]
    rows, ref_f0, ref_tonic = [], None, None
    for path, job, (name, rate, lvl) in zip(paths, jobs, LEVELS):
        t0 = time.time()
        kbps, tonic = job.result()
        tonic = tonic_fixed if tonic_fixed is not None else tonic
        # only the first CREPE_SAMPLES are used; decoding stops a second after them (soxr is local, so the
        # kept samples are identical to a full decode)
        y16, _ = librosa.load(str(path), sr=rl.SR, mono=True, duration=CREPE_SAMPLES / rl.SR + 1)
        f0 = pitch_track(y16.astype(np.float32), device)
        p = ens.predict(rl.paint(f0, tonic))
        if ref_f0 is None:
            ref_f0, ref_tonic = f0, tonic
        order = np.argsort(p)[::-1]
        pred = ens.idx2raga[order[0]]
        row = dict(level=name, kbps=round(kbps, 1), ratio=round(src_kbps / kbps, 2),
                   tonic_hz=round(tonic, 2), tonic_dev_c=round(cents_apart(tonic, ref_tonic), 1),
                   pitch_agree=round(pitch_agreement(f0, ref_f0), 4), pred=pred,
                   p_pred=round(float(p[order[0]]), 4), sec=round(time.time() - t0, 1))
        if true_raga is not None:
            k = ens.idx2raga.index(true_raga)
            row.update(p_true=round(float(p[k]), 4), rank_true=int(np.where(order == k)[0][0]) + 1,
                       correct=int(pred == true_raga))
        rows.append(row)
        if rate is not None and not keep:
            path.unlink()
    copy.unlink()
    return rows


# ── modes ────────────────────────────────────────────────────────────────────────────────

def mode_audio(a, device):
    src = Path(a.audio).resolve()
    hit = rl.cmd_match_path(str(src))
    per_fold = rl.MODELS["cvD"]
    if hit:
        stem, m = hit
        ens, truth = rl.Ensemble(per_fold[m["fold"]], device), m["raga"]
        print(f"CMD-40 recording ({m['raga']}, test fold {m['fold']}) -> scored by the fold model that never saw it")
    else:
        ens, truth = rl.Ensemble([q for f in per_fold for q in f], device), None
        print("not a CMD-40 recording -> 5 cvD fold models averaged")
    if a.tonic_hz is None:
        print("tonic: re-estimated by Essentia from each compressed file (pass --tonic-hz to fix it)")
    print(f"{src.name}\n")
    rows = run_levels(src, ens, device, truth, a.tonic_hz, a.keep, make_pool())
    print(f"{'level':<9s} {'kbps':>6s} {'x smaller':>9s} {'tonic Hz':>9s} {'dTonic c':>8s} {'pitch=':>7s}  prediction")
    for r in rows:
        mark = "" if truth is None else ("  OK" if r["correct"] else f"  wrong (true {truth}: rank {r['rank_true']})")
        print(f"{r['level']:<9s} {r['kbps']:6.1f} {r['ratio']:9.2f} {r['tonic_hz']:9.2f} {r['tonic_dev_c']:8.1f} "
              f"{r['pitch_agree']*100:6.1f}%  {r['pred']} ({r['p_pred']*100:.1f}%){mark}")
    print("\npitch= : frames whose pitch is within 50 cents of the uncompressed file's pitch track")


def make_pool():
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    return ProcessPoolExecutor(len(LEVELS), mp.get_context("spawn"))   # spawn: the parent holds CUDA


def pick_cmd(n_per_raga: int):
    """N recordings per raga, ordered round-robin over ragas, so a partial run is still balanced."""
    idx = json.load(open(rl.CMD_INDEX))
    by_raga = {}
    for stem, m in sorted(idx.items()):
        by_raga.setdefault(m["raga"], []).append(stem)
    rng = np.random.default_rng(SEED)
    picks = {r: list(rng.choice(v, min(n_per_raga, len(v)), replace=False)) for r, v in sorted(by_raga.items())}
    return [(p[j], idx[p[j]]) for j in range(n_per_raga) for r, p in picks.items() if j < len(p)]


def mode_cmd(a, device):
    items = pick_cmd(a.cmd)[a.shard::a.nshard]
    RES.mkdir(parents=True, exist_ok=True)
    out = RES / (f"results_s{a.shard}.csv" if a.nshard > 1 else "results.csv")
    done = set()
    if out.exists():
        done = {r["stem"] for r in csv.DictReader(open(out))}
    ens = {k: rl.Ensemble(f, device) for k, f in enumerate(rl.MODELS["cvD"])}
    pool = make_pool()
    print(f"{len(items)} recordings ({len(done)} already in {out.name}), {len(LEVELS)} levels, device {device}", flush=True)
    new = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        for i, (stem, m) in enumerate(items):
            if stem in done:
                continue
            t0 = time.time()
            e = ens[m["fold"]]
            stored = e.idx2raga[int(np.argmax(e.predict(np.load(m["npy"]).astype(np.float32))))]
            try:
                rows = run_levels(Path(m["mp3"]), e, device, m["raga"], None, False, pool)
            except Exception as ex:
                print(f"  FAIL {stem[:50]}: {type(ex).__name__}: {ex}", flush=True)
                continue
            for r in rows:
                w.writerow(dict(r, stem=stem, raga=m["raga"], fold=m["fold"], pred_stored=stored))
            fh.flush()
            print(f"  [{i+1}/{len(items)}] {m['raga']:<16s} " +
                  " ".join(("+" if r["correct"] else ".") for r in rows) +
                  f"  {time.time()-t0:.0f}s  {stem[:45]}", flush=True)


def macro_f1(truth, pred):
    labels = sorted(set(truth))
    f = []
    for c in labels:
        tp = sum(t == c and p == c for t, p in zip(truth, pred))
        fp = sum(t != c and p == c for t, p in zip(truth, pred))
        fn = sum(t == c and p != c for t, p in zip(truth, pred))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar: b = right->wrong, c = wrong->right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def mode_summarize():
    rows = [r for f in sorted(glob.glob(str(RES / "results*.csv"))) for r in csv.DictReader(open(f))]
    if not rows:
        raise SystemExit(f"no rows in {RES}/results*.csv")
    by = {}
    for r in rows:
        by.setdefault(r["level"], {})[r["stem"]] = r
    stems = sorted(set.intersection(*(set(v) for v in by.values())))      # recordings with every level
    ref = by["original"]
    L = []
    L.append(f"# Compression robustness -- condition D (cvD), {len(stems)} CMD-40 recordings\n")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `compressing/compress_test.py --summarize`. "
             f"Each recording is scored by the cvD fold model that never saw it. "
             f"Ragas: {len({ref[s]['raga'] for s in stems})}. Tonic re-estimated from each compressed file.\n")
    stored_ok = np.mean([ref[s]["pred_stored"] == ref[s]["raga"] for s in stems])
    same = np.mean([ref[s]["pred_stored"] == ref[s]["pred"] for s in stems])
    L.append(f"Check: stored realcrepe_D .npy top-1 on these recordings {stored_ok*100:.1f}%; the re-extracted "
             f"original agrees with the stored-input prediction on {same*100:.1f}% "
             f"(the stored files carry a different CREPE dither draw).\n")
    L.append("| level | kbps (median) | x smaller | top-1 % | macro-F1 | top-5 % | same pred as original % "
             "| right->wrong | wrong->right | McNemar p | tonic within 20 c % | median pitch agree % |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, _, _ in LEVELS:
        if name not in by:
            continue
        R = [by[name][s] for s in stems]
        truth, pred = [r["raga"] for r in R], [r["pred"] for r in R]
        ok = np.array([int(r["correct"]) for r in R])
        ok0 = np.array([int(ref[s]["correct"]) for s in stems])
        b, c = int(((ok0 == 1) & (ok == 0)).sum()), int(((ok0 == 0) & (ok == 1)).sum())
        L.append(f"| {name} | {np.median([float(r['kbps']) for r in R]):.0f} "
                 f"| {np.median([float(r['ratio']) for r in R]):.1f} | {ok.mean()*100:.1f} "
                 f"| {macro_f1(truth, pred):.3f} | {np.mean([int(r['rank_true']) <= 5 for r in R])*100:.1f} "
                 f"| {np.mean([r['pred'] == ref[r['stem']]['pred'] for r in R])*100:.1f} | {b} | {c} "
                 f"| {mcnemar_p(b, c):.3g} "
                 f"| {np.mean([abs(float(r['tonic_dev_c'])) <= 20 for r in R])*100:.1f} "
                 f"| {np.median([float(r['pitch_agree']) for r in R])*100:.1f} |")
    L.append("\nright->wrong / wrong->right: recordings whose correctness changed relative to the uncompressed "
             "file. pitch agree: frames within 50 cents of the uncompressed pitch track. "
             "cvD epochs were picked on the test fold, so absolute top-1 is a few points optimistic; "
             "the differences between levels are what this test measures.\n")
    worst = LEVELS[-1][0]
    if worst in by:
        flips = [s for s in stems if ref[s]["correct"] == "1" and by[worst][s]["correct"] == "0"]
        if flips:
            L.append(f"Recordings lost at {worst}:\n")
            for s in flips:
                r = by[worst][s]
                L.append(f"- {r['raga']} -> {r['pred']} (tonic {float(r['tonic_dev_c']):+.0f} c, "
                         f"pitch agree {float(r['pitch_agree'])*100:.0f}%) `{s}`")
    out = RES / "RESULT.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n-> {out}")


def main():
    ap = argparse.ArgumentParser(description="Raga prediction vs MP3 bitrate (condition-D pipeline).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--audio", help="one audio file (.mp3/.wav/.flac/...)")
    g.add_argument("--cmd", type=int, metavar="N", help="N CMD-40 recordings per raga")
    g.add_argument("--summarize", action="store_true")
    ap.add_argument("--tonic-hz", type=float, default=None, help="--audio only: fixed tonic instead of Essentia")
    ap.add_argument("--keep", action="store_true", help="--audio only: keep the compressed files in work/")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--device", default=None, help="default cuda:1 if present, else cuda:0, else cpu")
    a = ap.parse_args()
    if a.summarize:
        return mode_summarize()
    import torch
    torch.set_num_threads(4)                         # several shards share the CPU with their encode pools
    device = a.device or ("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")
    if a.audio:
        mode_audio(a, device)
    else:
        mode_cmd(a, device)


if __name__ == "__main__":
    main()
