"""Recording check PNGs, one per branch (same clips and panels as shapes.py), built from the audio itself.
Each panel: spectrogram of the WAV (y = cents above Sa, log frequency), pYIN pitch (orange, independent of CREPE)
and CREPE pitch (blue, conf >= 0.5, no octave fix). Title flags clipping and pYIN/CREPE disagreement.
Run from veena_stage1:  python local_run/spectro_check.py
Writes local_run/spectro_check/<gamaka>__<branch>.png, index.md and clip_audio_stats.csv."""
import csv, os, math, wave, numpy as np
import librosa
import matplotlib.pyplot as plt
from shapes import ROOT, SA, CONF, MAX_PANELS, BRANCHES, load_clips, crepe_activation_to_pitch, svara_label, safe

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spectro_check')
SR = 16000; HOP = 160; NFFT = 4096                          # 10 ms hop (same as CREPE), ~4 Hz bins
FMIN, FMAX = 40.0, 1400.0                                   # ,,,S (~41 Hz) up to about S'' for Sa 327 Hz
LO, HI = -3700, 2500                                        # plotted cents range
DB_RANGE = 70

def read_wav(fn):
    w = wave.open(os.path.join(ROOT, 'Gamaka_veena_dataset', fn))
    assert w.getsampwidth() == 2 and w.getnchannels() == 1, fn
    x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float32) / 32768.0
    return x, w.getframerate()

def cents(hz): return 1200 * np.log2(hz / SA)

def audio_stats(x, sr):
    """Level checks on the original-rate signal."""
    clipped = int(np.sum(np.abs(x) >= 32767 / 32768.0))
    fr = int(0.02 * sr); n = len(x) // fr
    rms = np.sqrt(np.mean(x[:n * fr].reshape(n, fr) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    return dict(dur_s=round(len(x) / sr, 2), peak_dbfs=round(float(20 * np.log10(np.max(np.abs(x)) + 1e-12)), 1),
                clipped_samples=clipped, dc_offset=round(float(np.mean(x)), 5),
                noise_floor_db=round(float(np.percentile(db, 10)), 1), loud_db=round(float(np.percentile(db, 95)), 1))

def analyse(fn):
    x, sr = read_wav(fn)
    st = audio_stats(x, sr)
    y = librosa.resample(x, orig_sr=sr, target_sr=SR)
    S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP)) ** 2
    S_db = librosa.power_to_db(S, ref=np.max, top_db=DB_RANGE)
    f = librosa.fft_frequencies(sr=SR, n_fft=NFFT); keep = f >= 30
    f0, vflag, _ = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=SR, frame_length=2048, hop_length=HOP)
    act = np.load(os.path.join(ROOT, 'crepe_output', os.path.splitext(fn)[0] + '.npy'))
    chz, cconf = crepe_activation_to_pitch(act)
    T = min(len(f0), len(chz))
    py = np.where(vflag[:T], cents(np.where(np.isnan(f0[:T]), 1.0, f0[:T])), np.nan)
    cr = np.where(cconf[:T] >= CONF, cents(chz[:T]), np.nan)
    both = ~np.isnan(py) & ~np.isnan(cr); d = np.abs(py - cr)[both]
    oct_err = np.abs(((py - cr)[both] + 600) % 1200 - 600) < 50  # within 50 cents of a whole number of octaves...
    oct_err &= d >= 600                                        # ...but at least one octave apart
    st.update(pyin_voiced_s=round(float(np.sum(~np.isnan(py)) / 100), 2), crepe_voiced_s=round(float(np.sum(~np.isnan(cr)) / 100), 2),
              both_voiced_s=round(float(both.sum() / 100), 2),
              agree_50c=round(float(np.mean(d < 50)), 3) if both.any() else '',
              octave_disagree=round(float(np.mean(oct_err)), 3) if both.any() else '',
              median_diff_c=round(float(np.median(d)), 1) if both.any() else '')
    return dict(t_spec=np.arange(S_db.shape[1]) * HOP / SR, c_spec=cents(f[keep]), S_db=S_db[keep],
                t=np.arange(T) / 100, py=py, cr=cr, stats=st)

def title(sl, st):
    bits = [sl]
    if st['agree_50c'] != '': bits.append(f"agree {100 * st['agree_50c']:.0f}%")
    if st['octave_disagree'] not in ('', 0.0): bits.append(f"oct {100 * st['octave_disagree']:.0f}%")
    if st['clipped_samples']: bits.append(f"CLIP {st['clipped_samples']}")
    return ' · '.join(bits)

def plot_branch(cls, br, items, rows):
    shown = items[:MAX_PANELS]; data = []
    for fn, sl, take in items:
        a = analyse(fn)
        rows.append(dict(gamaka=cls, branch=br, file=fn, slice=sl, take=take, plotted=len(data) < len(shown), **a['stats']))
        if len(data) < len(shown): data.append((sl, a))
    ncol = min(4, len(shown)); nrow = math.ceil(len(shown) / ncol)
    tmax = max(a['t_spec'][-1] for _, a in data)
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.4 * ncol + 0.6, 2.8 * nrow + 0.9), sharex=True, sharey=True, squeeze=False)
    ticks = [g for g in range(-3600, HI, 100) if g % 1200 in (0, 700)]
    for k, ax in enumerate(axs.flat):
        if k >= len(data): ax.axis('off'); continue
        sl, a = data[k]
        ax.pcolormesh(a['t_spec'], a['c_spec'], a['S_db'], cmap='Greys', vmin=-DB_RANGE, vmax=0, shading='nearest', rasterized=True)
        for g in ticks: ax.axhline(g, color='#9ecae1', lw=0.4, zorder=1)
        ax.plot(a['t'], a['py'], color='#ff7f0e', lw=1.4, zorder=3, label='pYIN')
        ax.plot(a['t'], a['cr'], color='#1f4e79', lw=0.7, zorder=4, label='CREPE')
        st = a['stats']
        flag = st['clipped_samples'] or (st['agree_50c'] != '' and st['agree_50c'] < 0.8)
        ax.set_title(title(sl, st), fontsize=8, color='#b30000' if flag else 'black')
        ax.set_xlim(0, tmax); ax.set_ylim(LO, HI); ax.tick_params(labelsize=7)
    for ax in axs[:, 0]:
        ax.set_yticks(ticks); ax.set_yticklabels([f'{svara_label(g)} {g:+d}' for g in ticks]); ax.set_ylabel('cents above Sa', fontsize=8)
    for ax in axs[-1, :]: ax.set_xlabel('s from clip start', fontsize=8)
    axs.flat[0].legend(fontsize=6, loc='upper right')
    cnt = f'{len(items)} clips' if len(items) <= MAX_PANELS else f'{MAX_PANELS} of {len(items)} clips'
    fig.suptitle(f'{cls} · {br} · {cnt} · spectrogram + pYIN (orange) + CREPE (blue) · Sa 327 Hz', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(OUT, f'{cls}__{safe(br)}.png'); fig.savefig(path, dpi=150); plt.close(fig)
    return os.path.basename(path)

if __name__ == '__main__':
    clips, _ = load_clips()
    order = ['jaru', 'kampita', 'ravai', 'pratyagata']
    keys = sorted(clips, key=lambda k: (order.index(k[0]), list(BRANCHES.values()).index(k[1])))
    os.makedirs(OUT, exist_ok=True); rows = []; idx = []
    for k in keys:
        png = plot_branch(k[0], k[1], clips[k], rows); idx.append((k, png, len(clips[k]))); print('wrote', png, flush=True)
    with open(os.path.join(OUT, 'clip_audio_stats.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT, 'index.md'), 'w', encoding='utf-8') as f:
        f.write('# Recording check per branch (Sa = 327 Hz)\n\n'
                'Grey: spectrogram of the WAV itself (16 kHz, 4096-point FFT, 10 ms hop, 70 dB range), y axis in cents above Sa. '
                'The fundamental and its harmonics (+1200, +1902, +2400 cents above it) are visible directly, without any pitch tracker.\n'
                'Orange: pYIN (librosa), a pitch tracker independent of CREPE. Blue: CREPE, confidence >= 0.5, no octave fix.\n'
                'Title: `agree` = share of frames voiced in both where they are within 50 cents; `oct` = share exactly an octave '
                '(or more) apart; `CLIP` = samples at full scale. Red title: clipping or agreement below 80%.\n'
                'Per-clip numbers for every clip (plotted or not): `clip_audio_stats.csv`.\n\n| Gamaka | Branch | Clips | PNG |\n|---|---|---|---|\n')
        for (cls, br), png, n in idx:
            f.write(f'| {cls} | {br} | {n}{" (12 shown)" if n > MAX_PANELS else ""} | [{png}]({png.replace(" ", "%20")}) |\n')
