import glob, os, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = os.path.abspath('..'); seg = pd.read_csv('kalyani_kampita_segments.csv')
files = sorted(glob.glob(os.path.join(ROOT, 'veena_stage1/plavita_kalyani/crepe/*.npy')))
fig, axs = plt.subplots(len(files), 1, figsize=(18, 3.2 * len(files)))
for ax, f in zip(axs, files):
    act = np.load(f); c = 1997.3794 + 20 * np.arange(360); k = act.argmax(1)
    idx = np.clip(k[:, None] + np.arange(-4, 5), 0, 359); w = np.take_along_axis(act, idx, 1)
    cents = (w * c[idx]).sum(1) / (w.sum(1) + 1e-9); conf = act.max(1)
    hz = 10 * 2 ** (cents / 1200); hz[conf < 0.3] = np.nan; t = np.arange(len(hz)) * 0.01
    ax.plot(t, hz, lw=0.6, color='#333')
    name = os.path.basename(f)[:-4]; s = seg[seg.Recording == name]
    for _, r in s.iterrows():
        ax.axvspan(r['Start (sec)'], r['End (sec)'], color='tab:red', alpha=0.3)
        ax.text(r['Start (sec)'], np.nanmax(hz), f"{r['Start (sec)']:.1f}s", color='tab:red', fontsize=9, va='top')
    ax.set_yscale('log'); ax.set_xlim(0, t[-1]); ax.set_ylabel('Pitch (Hz)')
    ax.set_title(f'{name}  —  {len(s)} kampita detected (red)', loc='left')
axs[-1].set_xlabel('Time (s)'); plt.tight_layout(); plt.savefig('kalyani_kampita_locations.png', dpi=110)
