"""CREPE on the three Kalyani recordings: same settings as run_crepe.py (full, 10 ms, viterbi off). The WAVs are 24-bit, so they are
read with soundfile-free code and scaled to the 16-bit range run_crepe.py feeds to CREPE."""
import glob, os, time, wave, numpy as np, crepe
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
def load24(fn):
    w = wave.open(fn); sr = w.getframerate(); ch = w.getnchannels(); assert w.getsampwidth() == 3 and ch == 1
    b = np.frombuffer(w.readframes(w.getnframes()), dtype=np.uint8).reshape(-1, 3)
    x = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
    x = np.where(x >= 2 ** 23, x - 2 ** 24, x)
    return (x / 256.0).astype(np.float32), sr
for fn in sorted(glob.glob(os.path.join(ROOT, 'Kalyani recorded', '*.wav'))):
    a, sr = load24(fn); t0 = time.time()
    _, _, _, act = crepe.predict(a, sr, model_capacity='full', viterbi=False, step_size=10, verbose=0)
    out = os.path.join(HERE, 'crepe', os.path.splitext(os.path.basename(fn))[0] + '.npy'); np.save(out, act.astype(np.float32))
    print(os.path.basename(fn), act.shape, 'peak dBFS', round(20 * np.log10(np.abs(a).max() / 32768 + 1e-12), 1), 'clipped', int((np.abs(a) >= 32767).sum()), f'{time.time() - t0:.0f}s', flush=True)
