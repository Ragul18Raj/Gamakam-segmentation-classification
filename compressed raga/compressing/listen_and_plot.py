#!/usr/bin/env python3
"""listen_and_plot.py -- hear the compression levels and see what they do to the melody the model reads.

    python3 compressing/listen_and_plot.py                      # the two examples below + summary + page
    python3 compressing/listen_and_plot.py --cmd Kamalamba      # any CMD-40 recording (name or part of it)
    python3 compressing/listen_and_plot.py --cmd X --start 40   # choose the excerpt yourself (seconds)

Per recording, in listen/<name>/:
  audio/<k>_<level>.wav  the same 20 s at every bitrate, decoded from the compressed file (what you hear is
                         exactly what the pipeline heard)
  1_spectrum.png         what each bitrate keeps of the sound (spectrogram, log frequency)
  2_melody.png           the pitch line the model receives, each bitrate drawn over the uncompressed one
  3_notes.png            share of time on each of the 12 swarasthanas over the model's whole input (162 s)
plus listen/summary.png (all 479 recordings, from results/) and listen/index.html, which shows it all.

Pitch is recomputed exactly as compress_test.py computes it (full-file encode, first 14000 frames, seeded
dither); the tonic and the prediction per level are the ones results/ recorded for that recording.
"""
from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compress_test as ct
from compress_test import rl

OUT = ct.HERE / "listen"
EXAMPLES = [  # (CMD-40 stem, folder): both right at every level down to 16 kbps, wrong at 8 kbps
    ("Sanjay_Subrahmanyan_Live_in_USA_-_Sanjay_Subrahmanyan_Shankaracharyam_Shankaracharyam",
     "1_melody_damaged"),                                 # 8 kbps: Sa kept, pitch line damaged
    ("Amritha_Murali_December_Season_2010_Pavanatmajagaccha_Pavanatmajagaccha",
     "2_sa_slipped"),                                     # 8 kbps: Sa estimated 711 cents too low
]
SWARAS = ["S", "R1", "R2", "G2", "G3", "M1", "M2", "P", "D1", "D2", "N2", "N3"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, RED, GRAY = "#2a78d6", "#eb6834", "#e34948", "#a3a29d"
RAMP = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
OFF_CENTS = 50                                            # same threshold as pitch_agree in results/


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "normal",
        "axes.titlelocation": "left", "font.size": 9, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "legend.frameon": False, "figure.dpi": 110})
    return plt


def cents(f0):
    return 1200 * np.log2(np.maximum(f0, 1e-6))


def best_lag(f0, ref, max_lag=8):
    """Decoder delay in frames: the lag at which the two tracks agree most (as pitch_agreement does)."""
    a, b = cents(f0), cents(ref)
    def agree(lag):
        x, y = (a[lag:], b[:len(b) - lag]) if lag >= 0 else (a[:lag], b[-lag:])
        n = min(len(x), len(y))
        return np.mean(np.abs(x[:n] - y[:n]) <= OFF_CENTS)
    return max(range(-max_lag, max_lag + 1), key=agree)


def shifted(x, lag):
    """x moved so that x_out[i] lines up with ref[i] (NaN where nothing lines up)."""
    out = np.full(len(x), np.nan, np.float64)
    if lag >= 0:
        out[:len(x) - lag] = x[lag:]
    else:
        out[-lag:] = x[:lag]
    return out


def load_results():
    by = {}
    for f in sorted(glob.glob(str(ct.RES / "results*.csv"))):
        for r in csv.DictReader(open(f)):
            by.setdefault(r["stem"], {})[r["level"]] = r
    return by


def pick_window(off, dur):
    """The excerpt whose off-pitch share at 8 kbps is closest to the recording's overall share
    (a typical stretch, not the worst one)."""
    n, step = int(dur * rl.FPS), int(5 * rl.FPS)
    starts = range(0, len(off) - n + 1, step)
    return min(starts, key=lambda s: (abs(off[s:s + n].mean() - off.mean()), s)) / rl.FPS


# ── per recording ────────────────────────────────────────────────────────────────────────

def render(stem, folder, res, start, dur, device, pool):
    import librosa, soundfile as sf
    m = json.load(open(rl.CMD_INDEX))[stem]
    rows = res[stem]
    d = OUT / folder
    (d / "audio").mkdir(parents=True, exist_ok=True)
    ct.WORK.mkdir(parents=True, exist_ok=True)
    copy = ct.WORK / f"{stem}__src.mp3"
    shutil.copy2(m["mp3"], copy)                                         # never touch the original
    paths = [copy if rate is None else ct.WORK / f"{stem}__{name}.mp3" for name, rate, _ in ct.LEVELS]
    jobs = [pool.submit(ct.prepare, (copy, p, rate, lvl, False)) for p, (_, rate, lvl) in zip(paths, ct.LEVELS)]
    f0 = {}
    for p, job, (name, _, _) in zip(paths, jobs, ct.LEVELS):
        job.result()
        y16, _ = librosa.load(str(p), sr=rl.SR, mono=True, duration=ct.CREPE_SAMPLES / rl.SR + 1)
        f0[name] = ct.pitch_track(y16.astype(np.float32), device)
        got, want = ct.pitch_agreement(f0[name], f0["original"]), float(rows[name]["pitch_agree"])
        assert abs(got - want) < 1e-3, f"{name}: pitch agreement {got:.4f} != results/ {want:.4f}"
    ref = f0["original"]
    lags = {k: best_lag(v, ref) for k, v in f0.items()}
    off8 = np.abs(shifted(cents(f0["mp3_8k"]), lags["mp3_8k"]) - cents(ref)) > OFF_CENTS
    if start is None:
        start = pick_window(off8, dur)
    clips = {}
    for k, (p, (name, _, _)) in enumerate(zip(paths, ct.LEVELS)):
        sr = sf.info(str(p)).samplerate
        # decode from the top and slice: seeking into a low-bitrate MP3 returns ~1 s of zeros while the
        # decoder refills its bit reservoir (measured at 8 kbps); the pipeline itself never seeks
        clip, _ = sf.read(str(p), stop=int((start + dur) * sr), dtype="float32", always_2d=True)
        clip = clip[int(start * sr):]
        ramp = np.linspace(0, 1, int(0.01 * sr), dtype=np.float32)[:, None]   # 10 ms fades: a cut mid-wave
        clip[:len(ramp)] *= ramp; clip[-len(ramp):] *= ramp[::-1]           # clicks and smears the spectrogram
        sf.write(str(d / "audio" / f"{k}_{name}.wav"), clip, sr, subtype="PCM_16")
        clips[name] = (clip.mean(1), sr)
        if p != copy:
            p.unlink()
    copy.unlink()
    tonic = {k: float(r["tonic_hz"]) for k, r in rows.items()}
    plot_spectrum(d, clips, rows)
    plot_melody(d, f0, lags, tonic, rows, start, dur)
    plot_notes(d, f0, tonic, rows)
    meta = dict(stem=stem, raga=m["raga"], start=start, dur=dur, folder=folder,
                levels=[dict(level=k, kbps=float(r["kbps"]), pred=r["pred"], correct=r["correct"] == "1",
                             pitch_agree=float(r["pitch_agree"]), tonic_dev=float(r["tonic_dev_c"]),
                             wav=f"audio/{i}_{k}.wav")
                        for i, (k, _, _) in enumerate(ct.LEVELS) for r in [rows[k]]])
    json.dump(meta, open(d / "meta.json", "w"), indent=1)
    print(f"  {folder}: {m['raga']}, excerpt {start:.0f}-{start + dur:.0f} s -> {d}")
    return meta


def level_label(name, r):
    return "original" if name == "original" else f"{float(r['kbps']):.0f} kbps"


def plot_spectrum(d, clips, rows):
    plt = style()
    from matplotlib.colors import LinearSegmentedColormap
    import librosa
    cmap = LinearSegmentedColormap.from_list("seq", RAMP)
    ref_y, ref_sr = clips["original"]
    ref_max = np.abs(librosa.stft(ref_y, n_fft=4096, hop_length=1024)).max()
    fig, axes = plt.subplots(len(ct.LEVELS), 1, figsize=(10, 13), sharex=True, sharey=True)
    for ax, (name, _, _) in zip(axes, ct.LEVELS):
        y, sr = clips[name]
        S = np.abs(librosa.stft(y, n_fft=4096 * sr // 44100, hop_length=1024 * sr // 44100))
        db = 20 * np.log10(S / ref_max + 1e-10)                         # one reference for every panel
        t = np.arange(S.shape[1]) * (1024 * sr // 44100) / sr
        fr = librosa.fft_frequencies(sr=sr, n_fft=4096 * sr // 44100)
        ax.pcolormesh(t, fr[1:], db[1:], cmap=cmap, vmin=-80, vmax=0, shading="auto", rasterized=True)
        ax.set_yscale("log"); ax.set_ylim(60, 22050); ax.grid(False)
        ax.axhline(8000, color=RED, lw=1.2, ls="--")
        ax.axhspan(100, 1000, color=ORANGE, alpha=0.08, lw=0)
        r = rows[name]
        ax.set_title(f"{level_label(name, r)}  ·  highest sound kept ≈ {top_freq(y, sr)/1000:.1f} kHz", loc="left")
        ax.set_yticks([100, 1000, 8000, 20000]); ax.set_yticklabels(["100 Hz", "1 kHz", "8 kHz", "20 kHz"])
    axes[-1].set_xlabel("seconds into the excerpt")
    fig.suptitle("What each bitrate keeps of the sound: compression cuts the top end first, "
                 "and only at 8 kbps reaches the melody band\n"
                 "red line: the pipeline discards everything above 8 kHz anyway  ·  "
                 "shaded: 100 Hz–1 kHz, where the sung melody lives", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(d / "1_spectrum.png"); plt.close(fig)


def top_freq(y, sr):
    import librosa
    S = np.abs(librosa.stft(y, n_fft=4096)) ** 2
    spec = np.median(S, 1); fr = librosa.fft_frequencies(sr=sr, n_fft=4096)   # median: one click can't set it
    return fr[np.where(10 * np.log10(spec / spec.max() + 1e-20) > -70)[0].max()]


def swara_ticks(lo, hi):
    """Ticks at S and P of every octave between lo and hi cents (relative to Sa)."""
    ticks, labels = [], []
    for octv in range(-2, 3):
        for c, s in ((0, "S"), (700, "P")):
            v = 1200 * octv + c
            if lo <= v <= hi:
                ticks.append(v)
                labels.append(s + {-2: " (lower ×2)", -1: " (lower)", 0: "", 1: " (upper)", 2: " (upper ×2)"}[octv])
    return ticks, labels


def broken(x, jump=200):
    """NaN between frames that jump more than `jump` cents, so the line does not draw spikes."""
    y = x.astype(np.float64).copy()
    y[1:][np.abs(np.diff(x)) > jump] = np.nan
    return y


def plot_melody(d, f0, lags, tonic, rows, start, dur):
    plt = style()
    a, b = int(start * rl.FPS), int((start + dur) * rl.FPS)
    t = np.arange(a, b) / rl.FPS - start
    rel_of = {n: (shifted(cents(f0[n]), lags[n]) - cents(np.array([tonic[n]])))[a:b] for n in f0}
    ref_rel = rel_of["original"]
    lo, hi = np.nanpercentile(np.concatenate(list(rel_of.values())), [0.5, 99.5])   # octave jumps stay in view
    lo, hi = lo - 150, hi + 150
    names = [n for n, _, _ in ct.LEVELS if n != "original"]
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 13), sharex=True, sharey=True)
    for ax, name in zip(axes, names):
        r = rows[name]
        aligned = shifted(cents(f0[name]), lags[name])                     # frame i lines up with original i
        rel = rel_of[name]
        off = np.abs(aligned - cents(f0["original"]))[a:b] > OFF_CENTS
        ax.plot(t, broken(ref_rel), color=GRAY, lw=2.0, label="original", solid_capstyle="round")
        ax.plot(t, broken(rel), color=BLUE, lw=1.0, label=level_label(name, r))
        ax.scatter(t[off], rel[off], s=9, color=RED, zorder=3, lw=0, label="heard on a different pitch")
        ok = "right" if r["correct"] == "1" else "WRONG"
        sa = "" if abs(float(r["tonic_dev_c"])) <= 20 else f"  ·  Sa moved {float(r['tonic_dev_c']):+.0f} cents"
        ax.set_title(f"{level_label(name, r)}  ·  {float(r['pitch_agree'])*100:.0f}% of the 162 s on the same pitch"
                     f"{sa}  ·  model: {r['pred']} ({ok})", loc="left")
        ax.set_ylim(lo, hi)
        ticks, labels = swara_ticks(lo, hi)
        ax.set_yticks(ticks); ax.set_yticklabels(labels)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8, markerscale=1.5)
    axes[-1].set_xlabel(f"seconds into the excerpt (recording {start:.0f}–{start + dur:.0f} s)")
    fig.suptitle(f"The melody line the model receives ({rows['original']['raga']}), drawn against the Sa "
                 "each version's pipeline found\ngray = uncompressed  ·  blue = compressed  ·  "
                 "red = frames heard on a different pitch", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(d / "2_melody.png"); plt.close(fig)


def plot_notes(d, f0, tonic, rows):
    plt = style()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(60) * 20
    for name, color, lw in (("original", GRAY, 3.0), ("mp3_16k", BLUE, 1.6), ("mp3_8k", ORANGE, 1.6)):
        rel = cents(f0[name]) - cents(np.array([tonic[name]]))
        h = np.bincount(np.round(rel / 20).astype(int) % 60, minlength=60) / len(rel) * 100
        r = rows[name]
        sa = "" if abs(float(r["tonic_dev_c"])) <= 20 else f", Sa moved {float(r['tonic_dev_c']):+.0f} c"
        ax.plot(np.append(x, 1200), np.append(h, h[0]), color=color, lw=lw,
                label=f"{level_label(name, r)} → {r['pred']}{sa}")
    ax.set_xticks(np.arange(12) * 100); ax.set_xticklabels(SWARAS); ax.set_xlim(-10, 1210)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("note, relative to the Sa the pipeline found (all octaves folded together)")
    ax.set_ylabel("% of the 162 s input")
    ax.legend(loc="upper center", fontsize=8)
    ax.set_title(f"Time spent on each note ({rows['original']['raga']}): the raga's note pattern, "
                 f"as the model receives it", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(d / "3_notes.png"); plt.close(fig)


# ── all 479 recordings ───────────────────────────────────────────────────────────────────

def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def plot_summary(res):
    plt = style()
    stems = [s for s, r in res.items() if len(r) == len(ct.LEVELS)]
    names = [n for n, _, _ in ct.LEVELS]
    labels = ["original\n128"] + [n.split("_")[1].replace("k", "") for n in names[1:]]
    x = np.arange(len(names))
    acc, lo, hi, pitch, sa, lost_sa, lost_pitch = [], [], [], [], [], [], []
    for n in names:
        k = sum(res[s][n]["correct"] == "1" for s in stems)
        acc.append(k / len(stems) * 100)
        l, h = wilson(k, len(stems)); lo.append(l * 100); hi.append(h * 100)
        pitch.append(np.median([float(res[s][n]["pitch_agree"]) for s in stems]) * 100)
        sa.append(np.mean([abs(float(res[s][n]["tonic_dev_c"])) <= 20 for s in stems]) * 100)
        lost = [s for s in stems if res[s]["original"]["correct"] == "1" and res[s][n]["correct"] == "0"]
        lost_sa.append(sum(abs(float(res[s][n]["tonic_dev_c"])) > 20 for s in lost))
        lost_pitch.append(len(lost) - lost_sa[-1])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    ax = axes[0]
    ax.fill_between(x, lo, hi, color=INK2, alpha=0.10, lw=0)
    ax.plot(x, acc, color=INK, lw=2, marker="o", ms=5)
    ax.annotate(f"{acc[0]:.1f}%", (x[0], acc[0]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.annotate(f"{acc[-2]:.1f}%", (x[-2], acc[-2]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.annotate(f"{acc[-1]:.1f}%", (x[-1], acc[-1]), textcoords="offset points", xytext=(-4, -14), ha="right", fontsize=8)
    ax.set_ylim(70, 95); ax.set_title("Raga identified correctly (%, band = 95% range)")
    ax = axes[1]
    ax.plot(x, pitch, color=BLUE, lw=2, marker="o", ms=5, label="melody frames on the same pitch (median)")
    ax.plot(x, sa, color=ORANGE, lw=2, marker="o", ms=5, label="recordings whose Sa stayed put")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_ylim(70, 101); ax.set_title("What survives compression (%)")
    ax = axes[2]
    ax.bar(x, lost_pitch, color=BLUE, width=0.6, edgecolor=SURFACE, lw=2, label="melody damaged (Sa kept)")
    ax.bar(x, lost_sa, bottom=lost_pitch, color=ORANGE, width=0.6, edgecolor=SURFACE, lw=2, label="Sa moved")
    for i in range(len(x)):
        ax.text(x[i], lost_pitch[i] + lost_sa[i] + 1, str(lost_pitch[i] + lost_sa[i]), ha="center", fontsize=8)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Recordings lost vs. the uncompressed file, by cause")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_xlabel("MP3 bitrate (kbps)")
    fig.suptitle(f"All {len(stems)} CMD-40 recordings: nothing changes down to 16 kbps; at 8 kbps both the "
                 "melody and the Sa estimate break", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "summary.png"); plt.close(fig)
    print(f"  summary -> {OUT / 'summary.png'}")


# ── page ─────────────────────────────────────────────────────────────────────────────────

def write_page():
    metas = [json.load(open(p)) for p in sorted(OUT.glob("*/meta.json"))]
    e = html.escape

    def row(m, l):
        name = "original" if l["level"] == "original" else l["level"].split("_")[1] + "bps"
        answer = e(l["pred"]) + (" ✓" if l["correct"] else " ✗ wrong")
        sa = "kept" if abs(l["tonic_dev"]) <= 20 else f"moved {l['tonic_dev']:+.0f} c"
        return (f"<tr><td>{e(name)}</td><td>{l['kbps']:.0f}</td>"
                f"<td><audio controls preload='none' src='{e(m['folder'])}/{e(l['wav'])}'></audio></td>"
                f"<td>{answer}</td><td>{l['pitch_agree']*100:.0f}%</td><td>{sa}</td></tr>")

    parts = []
    for m in metas:
        rows = "".join(row(m, l) for l in m["levels"])
        parts.append(f"""
<section><h2>{e(m['raga'])} — <span class="muted">{e(m['stem'])}</span></h2>
<p class="muted">Excerpt {m['start']:.0f}–{m['start'] + m['dur']:.0f} s of the recording. Listen from the top down.</p>
<table><thead><tr><th>level</th><th>kbps</th><th>listen</th><th>model's answer</th><th>melody on same pitch</th><th>Sa</th></tr></thead>
<tbody>{rows}</tbody></table>
<img src="{e(m['folder'])}/1_spectrum.png" alt="spectrogram of each bitrate">
<img src="{e(m['folder'])}/2_melody.png" alt="pitch line of each bitrate over the original">
<img src="{e(m['folder'])}/3_notes.png" alt="time spent on each note">
</section>""")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Compression Listening Test</title>
<style>
:root {{ --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --rule:#e4e3df; }}
body {{ background:var(--surface); color:var(--ink); font:15px/1.5 system-ui,sans-serif; max-width:1100px;
       margin:0 auto; padding:16px; }}
h1 {{ font-size:22px; margin:8px 0 4px; }} h2 {{ font-size:18px; margin:36px 0 4px; }}
.muted {{ color:var(--ink2); font-weight:normal; font-size:14px; }}
img {{ width:100%; height:auto; margin:14px 0; border:1px solid var(--rule); }}
table {{ border-collapse:collapse; width:100%; font-size:14px; }}
th, td {{ text-align:left; padding:4px 8px; border-bottom:1px solid var(--rule); }}
audio {{ height:32px; width:260px; max-width:100%; }}
section {{ overflow-x:auto; }}
</style></head><body>
<h1>Does MP3 compression change what the raga model hears?</h1>
<p class="muted">Each recording was compressed to 96 → 8 kbps and sent through the full pipeline
(CREPE pitch → Essentia Sa → cvD model). Spectrogram = what the file still contains; melody = the pitch line
the model actually reads; notes = the note pattern a raga is recognised from.</p>
<img src="summary.png" alt="accuracy, survival and losses for all recordings">
{''.join(parts)}
</body></html>"""
    (OUT / "index.html").write_text(page)
    print(f"  page -> {OUT / 'index.html'}")


def main():
    ap = argparse.ArgumentParser(description="Listening files + graphs for the compression levels.")
    ap.add_argument("--cmd", help="a CMD-40 recording (name or part of it); default: the two built-in examples")
    ap.add_argument("--start", type=float, default=None, help="excerpt start in seconds (default: a typical stretch)")
    ap.add_argument("--dur", type=float, default=20.0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    import torch
    device = a.device or ("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")
    res = load_results()
    todo = EXAMPLES
    if a.cmd:
        stem, _ = rl.cmd_lookup(a.cmd)
        todo = [(stem, f"x_{stem[:60]}")]
    missing = [s for s, _ in todo if s not in res or len(res[s]) != len(ct.LEVELS)]
    if missing:
        raise SystemExit(f"not in results/ yet (run compress_test.py --cmd first): {missing}")
    pool = ct.make_pool()
    for stem, folder in todo:
        render(stem, folder, res, a.start, a.dur, device, pool)
    plot_summary(res)
    write_page()


if __name__ == "__main__":
    main()
