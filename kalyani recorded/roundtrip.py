"""One Kalyani clip: audio -> CREPE activation .npy -> sound again (with sonify_crepe.py) + pitch contour picture.

Steps
1. Read `Kalyani recorded/kalyani2 #03.wav` (24-bit, 48 kHz) and run CREPE full, 10 ms hop, viterbi off
   (same settings as veena_stage1/plavita_kalyani/run_crepe_kalyani.py).
2. Save the activation (frames x 360) as kalyani2_03.activation.npy.
3. Run sonify_crepe.py on that .npy (veena tone) -> kalyani2_03.activation_sonified.wav.
4. Also write a 16 kHz copy of the original and a mix (original + sonified) to compare by ear.
5. Plot the pitch contour decoded from the .npy, over a spectrogram.
"""
import os, subprocess, sys, wave
import numpy as np
import crepe
import soundfile as sf
from scipy.signal import resample_poly
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from sonify_crepe import load_activation  # same decoding the sonifier uses

SRC = os.path.join(ROOT, "Kalyani recorded", "kalyani2 #03.wav")
NAME = "kalyani2_03"
CONF = 0.5


def load24(fn):
    w = wave.open(fn)
    sr, ch = w.getframerate(), w.getnchannels()
    assert w.getsampwidth() == 3 and ch == 1
    b = np.frombuffer(w.readframes(w.getnframes()), dtype=np.uint8).reshape(-1, 3)
    x = b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)
    x = np.where(x >= 2 ** 23, x - 2 ** 24, x)
    return (x / 256.0).astype(np.float32), sr  # 16-bit range, as fed to CREPE before


# 1-2. audio -> CREPE activation
audio, sr = load24(SRC)
_, _, _, act = crepe.predict(audio, sr, model_capacity="full", viterbi=False, step_size=10, verbose=0)
npy = os.path.join(HERE, NAME + ".activation.npy")
np.save(npy, act.astype(np.float32))
print("activation", act.shape, "->", npy)

# check against the earlier run of the same clip
old = os.path.join(ROOT, "veena_stage1", "plavita_kalyani", "crepe", "kalyani2 #03.npy")
if os.path.exists(old):
    o = np.load(old)
    if o.shape == act.shape:
        print("largest difference from earlier saved .npy:", float(np.abs(o - act).max()))

# 3. .npy -> sound, with the user's sonify_crepe.py
subprocess.run([sys.executable, os.path.join(ROOT, "sonify_crepe.py"), npy, "--tone", "veena", "--conf", str(CONF)], check=True)
son_path = os.path.join(HERE, NAME + ".activation_sonified.wav")
son, sr_son = sf.read(son_path)

# 4. original at 16 kHz, and original + sonified mix
orig16 = resample_poly(audio / 32768.0, sr_son, sr)
orig16 = 0.9 * orig16 / (np.abs(orig16).max() + 1e-9)
sf.write(os.path.join(HERE, NAME + "_original_16k.wav"), orig16, sr_son)
n = min(len(orig16), len(son))
mix = 0.6 * orig16[:n] + 0.4 * son[:n]
sf.write(os.path.join(HERE, NAME + "_original_plus_sonified.wav"), 0.9 * mix / np.abs(mix).max(), sr_son)

# 5. pitch contour picture
t, f, c = load_activation(npy, 10)
voiced = c >= CONF
fv = np.where(voiced, f, np.nan)
print(f"voiced frames {voiced.mean():.0%}, pitch range {np.nanmin(fv):.0f}-{np.nanmax(fv):.0f} Hz")

fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
ax[0].specgram(audio, NFFT=4096, Fs=sr, noverlap=4096 - 480, cmap="magma", vmin=-20)
ax[0].plot(t, fv, color="cyan", lw=1.2, label="CREPE pitch (confidence >= 0.5)")
ax[0].set_ylim(60, 1200); ax[0].set_yscale("log")
ax[0].set_ylabel("Hz (log)"); ax[0].legend(loc="upper right")
ax[0].set_title("kalyani2 #03: spectrogram of the original with the CREPE pitch contour on top")
ax[1].plot(t, fv, color="tab:blue", lw=1)
ax[1].scatter(t[~voiced], np.full((~voiced).sum(), 70), s=1, color="lightgray", label="unvoiced (silent in sonified audio)")
ax[1].set_yscale("log"); ax[1].set_ylim(65, 1200)
ax[1].set_ylabel("Hz (log)"); ax[1].set_xlabel("time (s)"); ax[1].legend(loc="upper right")
ax[1].set_title("Pitch contour from the .npy: this is all the sonified audio follows")
for a in ax:
    a.grid(alpha=0.3, which="both")
plt.tight_layout()
png = os.path.join(HERE, NAME + "_pitch_contour.png")
plt.savefig(png, dpi=110)
print("wrote", png)
