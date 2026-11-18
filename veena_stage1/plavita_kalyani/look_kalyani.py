"""First look at the Kalyani audio, before any model: spectrogram with CREPE pitch on top, and pitch statistics. No gamaka labels are used."""
import glob, os, sys, wave, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'veena_stage1', 'local_run'))
from shapes import crepe_activation_to_pitch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import stft
def load24(fn):
    w = wave.open(fn); sr = w.getframerate(); b = np.frombuffer(w.readframes(w.getnframes()), dtype=np.uint8).reshape(-1, 3)
    x = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)); x = np.where(x >= 2 ** 23, x - 2 ** 24, x)
    return x / 2.0 ** 23, sr
for fn in sorted(glob.glob(os.path.join(ROOT, 'Kalyani recorded', '*.wav'))):
    name = os.path.splitext(os.path.basename(fn))[0]; x, sr = load24(fn); hz, conf = crepe_activation_to_pitch(np.load(os.path.join(HERE, 'crepe', name + '.npy')))
    v = conf >= 0.5; h = hz[v]; c = 1200 * np.log2(h / 327.0)
    hist, edges = np.histogram(c, bins=np.arange(-3000, 3000, 25)); top = np.argsort(hist)[::-1][:6]
    print(name, f'{len(x) / sr:.1f}s peak {20 * np.log10(np.abs(x).max() + 1e-12):.1f} dBFS clipped {(np.abs(x) >= 0.999).sum()} voiced {v.mean():.2f} median Hz {np.median(h):.1f} p5-p95 Hz {np.percentile(h, 5):.0f}-{np.percentile(h, 95):.0f}')
    print('   top pitch bins (Hz, share):', [(round(327 * 2 ** ((edges[i] + 12.5) / 1200), 1), round(hist[i] / hist.sum(), 3)) for i in top])
    f, t, Z = stft(x, sr, nperseg=4096, noverlap=4096 - 480); S = 20 * np.log10(np.abs(Z) + 1e-9); keep = f <= 3000
    fig, ax = plt.subplots(figsize=(max(14, len(x) / sr * 0.42), 5)); ax.pcolormesh(t[::3], f[keep], S[keep][:, ::3], vmin=S.max() - 70, vmax=S.max(), cmap='magma', shading='auto')
    ax.plot(np.arange(len(hz))[v] / 100, hz[v], '.', ms=0.8, color='#7fd4ff'); ax.set_ylim(50, 3000); ax.set_yscale('log'); ax.set_title(f'{name}: spectrogram, CREPE pitch in blue'); ax.set_xlabel('seconds'); ax.set_ylabel('Hz')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'look_' + name.replace(' ', '_').replace('#', '') + '.png'), dpi=100); plt.close(fig)
