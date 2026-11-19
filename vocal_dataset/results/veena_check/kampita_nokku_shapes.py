"""EXPLORATORY: pitch shapes of the Kampita and Nokku events that look alike (recorded vocal, singer's marks, pass 1).
Fig 1: the 16 events Stage 1 confused between the two types (held-out F-LR or DTW-3NN), each with its closest event of the other type from another passage set.
Fig 2: the 12 closest Kampita-Nokku pairs from different sets, each event used once. Distance = Stage 1 DTW (median-centred, real length)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; VD = HERE.parents[1]; sys.path.insert(0, str(VD))
from stage1_vocal import centre, dtw
from vocal_pitch import event_curve, load_cents, load_tables
COL = {"Kampita": "#1f5fa8", "Nokku": "#d9601a"}; FL, DT = "main_4class | F-LR", "main_4class | DTW-3NN"
meta, _ = load_tables(); sa = meta.set_index("recording").sa_hz; P = pd.read_csv(VD / "results/stage1/predictions.csv"); P = P[P.label.isin(COL)].reset_index(drop=True)
cache = {r: load_cents(r, sa[r]) for r in P.recording.unique()}; C = []
for _, r in P.iterrows():
    t, c, _, v, _ = cache[r.recording]; C.append(centre(event_curve(t, c, v, r.start_s, r.end_s)[0]))
y = P.label.values; g = P.group.values; n = len(P); D = np.full((n, n), np.inf)
for i in range(n):
    for j in range(n):
        if y[i] == "Kampita" and y[j] == "Nokku" and g[i] != g[j]:
            D[i, j] = D[j, i] = dtw(C[i], C[j], max(5, int(.2 * max(len(C[i]), len(C[j])))))
def lab(i): return f"{P.recording[i]} {P.start_s[i]:.2f}s"
def rng(i): return np.percentile(C[i], 95) - np.percentile(C[i], 5)
def draw(ax, i, j, title):
    for k, lw, ls, z in ((j, 1.3, (0, (3, 1.5)), 2), (i, 2.2, "-", 3)):
        ax.plot(np.arange(len(C[k])) / 100, C[k], color=COL[y[k]], lw=lw, ls=ls, zorder=z)
    ax.axhline(0, color=".85", lw=.7, zorder=0); ax.set_title(title, fontsize=8.2, loc="left"); ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
def finish(fig, axs, title, name):
    for a in axs.flat[len(fig._used):]: a.axis("off")
    for a in axs[-1]: a.set_xlabel("time from start of event (s)", fontsize=8)
    for a in axs[:, 0]: a.set_ylabel("cents from event median", fontsize=8)
    h = [plt.Line2D([], [], color=COL["Kampita"], lw=2.2, label="marked Kampita"), plt.Line2D([], [], color=COL["Nokku"], lw=2.2, label="marked Nokku")]
    fig.legend(handles=h, loc="upper right", ncol=2, fontsize=9, frameon=False); fig.suptitle(title, x=.01, ha="left", fontsize=11); fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(HERE / name, dpi=120); plt.close(fig)
other = {"Kampita": "Nokku", "Nokku": "Kampita"}
bad = [i for i in range(n) if P[FL][i] == other[y[i]] or P[DT][i] == other[y[i]]]; bad.sort(key=lambda i: (y[i], P.recording[i] != "VARNAM", int(P.recording[i].split()[-1]) if P.recording[i] != "VARNAM" else 0, P.start_s[i]))
fig, axs = plt.subplots(4, 4, figsize=(16, 11), sharex=True, sharey=True); fig._used = bad; rows = []
for a, i in zip(axs.flat, bad):
    j = int(np.argmin(D[i])); w = "+".join(s for s, c in (("LR", FL), ("DTW", DT)) if P[c][i] == other[y[i]])
    draw(a, i, j, f"{lab(i)} ({len(C[i])/100:.2f}s) called {other[y[i]]} by {w}\ndashed: {lab(j)} ({len(C[j])/100:.2f}s), dist {D[i,j]:.0f}c")
    rows.append(dict(figure="confused", event=lab(i), marked=y[i], length_s=len(C[i])/100, range_c=round(rng(i)), called_by=w, closest_other_type=lab(j), other_length_s=len(C[j])/100, other_range_c=round(rng(j)), dtw_c=round(D[i, j], 1)))
finish(fig, axs, "Kampita and Nokku events that Stage 1 confused (solid), with the closest event of the other type (dashed)", "shapes_confused_kampita_nokku.png")
O = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]; pairs = []; used = set()
for i, j in O:
    if not np.isfinite(D[i, j]): break
    if y[i] == "Kampita" and i not in used and j not in used: pairs.append((i, j)); used |= {i, j}
    if len(pairs) == 12: break
fig, axs = plt.subplots(3, 4, figsize=(16, 8.4), sharex=True, sharey=True); fig._used = pairs
for a, (i, j) in zip(axs.flat, pairs):
    draw(a, i, j, f"K {lab(i)} ({len(C[i])/100:.2f}s, range {rng(i):.0f}c)\nN {lab(j)} ({len(C[j])/100:.2f}s, range {rng(j):.0f}c), dist {D[i,j]:.0f}c")
    rows.append(dict(figure="closest_pair", event=lab(i), marked="Kampita", length_s=len(C[i])/100, range_c=round(rng(i)), called_by="", closest_other_type=lab(j), other_length_s=len(C[j])/100, other_range_c=round(rng(j)), dtw_c=round(D[i, j], 1)))
finish(fig, axs, "The 12 closest Kampita-Nokku pairs from different passage sets (solid Kampita, dashed Nokku)", "shapes_closest_kampita_nokku.png")
T = pd.DataFrame(rows); T.to_csv(HERE / "kampita_nokku_similar_events.csv", index=False); print(T.to_string()); 
print("typical: ", {k: (round(float(np.median([rng(i) for i in range(n) if y[i]==k]))), round(float(np.median([len(C[i])/100 for i in range(n) if y[i]==k])),2)) for k in COL}, "(median range c, median s)")
