"""EXPLORATORY (18 Nov 2026, after vocal results were seen). Cut veena test curves; no classifier here.
Test clips were fixed by rule before any classification: first clip of each plavita take, first two ascending jaru,
first clip of each descending jaru take. The 'all' set is every annotated plavita clip and every jaru clip.
Plavita: hand-marked span from kampita_annotations_2026-11-13.xlsx. Jaru: no marks exist, so the label-free crop_moving rule.
Run with /usr/bin/python3 from anywhere. Writes veena_curves.npz-style pickle and check_curves.png."""
import os, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "veena_stage1" / "methods_explore"))
import clean as C
PRIMARY = {"Kampita (plavita)#02.1": "Kampita", "Kampita (plavita) bhairiga#03.1": "Kampita",
           "Jaru_asc_1": "Jaaru (Etra)", "Jaru_asc_2": "Jaaru (Etra)",
           "Jaru_desc_shankravarnam_1": "Jaaru (Irakka)", "Jaru_desc_1_melgana": "Jaaru (Irakka)"}
SUF = "Guitar_samples_3D_OG"
ann = pd.read_excel(REPO / "kampita_annotations_2026-11-13.xlsx", sheet_name="All Songs")
span = {r["Song Name"]: (r["Start (sec)"], r["End (sec)"]) for _, r in ann.iterrows()}
files = sorted(os.listdir(REPO / "Gamaka_veena_dataset"))
items = [(f, "Kampita") for f in files if f.endswith(".wav") and f[:-4] in span]
items += [(f, "Jaaru (Etra)") for f in files if f.startswith("Jaru_asc") and f.endswith(".wav")]
items += [(f, "Jaaru (Irakka)") for f in files if f.startswith("Jaru_desc") and f.endswith(".wav")]
out = []
for f, lab in items:
    stem = f[:-4]; short = stem.replace(SUF, "", 1) if stem.endswith(SUF) else stem.replace(SUF + "_1", " [OG_1]")
    d = C.process(f); m = d["clean_masked"]
    if stem in span:
        a, b = int(round(span[stem][0] * 100)), int(round(span[stem][1] * 100)); how = "hand-marked"
    else:
        a, b = d["crop_range"]; how = "movement crop"
    idx = a + np.where(~np.isnan(m[a:b]))[0]
    cur = np.interp(np.arange(idx[0], idx[-1] + 1), idx, m[idx]) if len(idx) >= 5 else None
    out.append(dict(file=f, name=short, label=lab, how=how, primary=short in PRIMARY, a=a, b=b, curve=cur, masked=m, raw=d["raw_masked"]))
pickle.dump(out, open(HERE / "veena_curves.pkl", "wb"), protocol=4)
print("clips", len(out), pd.Series([o["label"] for o in out]).value_counts().to_dict(), "| no curve:", [o["name"] for o in out if o["curve"] is None])
print("primary found:", sorted(o["name"] for o in out if o["primary"]))
P = [o for o in out if o["primary"]]
fig, ax = plt.subplots(2, 3, figsize=(13, 5.5)); 
for x, o in zip(ax.flat, P):
    t = np.arange(len(o["raw"])) / 100; x.axvspan(o["a"] / 100, o["b"] / 100, color="#fde0a0"); x.plot(t, o["raw"], color="0.7", lw=.8); x.plot(t, o["masked"], color="#1f4e79", lw=1.1)
    c = o["curve"]; x.set_title(f'{o["name"]} = {o["label"]}\n{o["how"]}, {len(c)/100:.2f} s, net {c[-3:].mean()-c[:3].mean():.0f} c', fontsize=9); x.set_ylabel("cents re 327 Hz", fontsize=8)
fig.tight_layout(); fig.savefig(HERE / "check_curves.png", dpi=110)
for o in P:
    c = o["curve"]; print(f'{o["name"]:36s} {o["label"]:15s} {o["how"]:14s} dur {len(c)/100:.2f}  range {np.percentile(c,95)-np.percentile(c,5):.0f}  net {c[-3:].mean()-c[:3].mean():.0f}')
