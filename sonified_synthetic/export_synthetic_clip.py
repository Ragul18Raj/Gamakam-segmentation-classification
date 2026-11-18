"""Exports one SIMULATED clip from sim_code_extracted/gamaka_sim/sim_gamaka.py as a CREPE-style CSV (time, frequency, confidence) for sonify_crepe.py,
plus the list of gamakas in it. The simulator works in cents above Sa, so a Sa has to be chosen to get Hz: 220 Hz here (an assumption)."""
import csv, os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'sim_code_extracted', 'gamaka_sim'))
from sim_gamaka import render, GAMAKAS
SA = 220.0; SONG, G, TEMPO, OCT = 1, 1, 'medium', 'low'
seed = 7919 * SONG + 104729 * G + 31 * 1 + 7 * 0          # the same seed build_dataset uses for this clip, standard condition
cents, conf, ann, true_ev = render(SONG, G, TEMPO, OCT, seed, stress=False)
name = f'synthetic_song{SONG}_G{G}_{TEMPO}_{OCT}'
hz = np.where(np.isnan(cents), 0.0, SA * 2 ** (np.nan_to_num(cents) / 1200))
with open(os.path.join(HERE, name + '.f0.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['time', 'frequency', 'confidence'])
    for i, (h, c) in enumerate(zip(hz, conf)): w.writerow([f'{i / 100:.2f}', f'{h:.3f}', f'{c:.3f}'])
names = {1: 'G1 (kampita-like)', 2: 'G2 (jaru-like)', 3: 'G3 (sphuritam-like)'}
with open(os.path.join(HERE, name + '_gamakas.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['start_s', 'end_s', 'gamaka'])
    for s, e, l in true_ev: w.writerow([f'{s:.2f}', f'{e:.2f}', names[l]])
print(name, f'{len(hz) / 100:.1f}s', 'voiced', round(float((conf >= 0.5).mean()), 2), 'Hz range', round(float(hz[hz > 0].min()), 1), '-', round(float(hz.max()), 1))
for s, e, l in true_ev: print(f'  {s:6.2f}-{e:6.2f}s  {names[l]}')
