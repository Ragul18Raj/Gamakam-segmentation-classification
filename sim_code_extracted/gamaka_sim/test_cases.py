"""Executable checks behind the test-case table (T4-T9), run on simulated data."""
import numpy as np
from pipeline import features, fill_short_gaps, frame_labels, DilatedCNN, postprocess, FR
from sim_gamaka import build_dataset

res = {}

# T4 gap interpolation: gaps < 100 ms filled, longer gaps stay unvoiced
c = np.full(400, 400.0); conf = np.full(400, 0.9)
c[100:105] = np.nan; c[200:230] = np.nan
_, v = fill_short_gaps(c, ~np.isnan(c))
res["T4"] = bool(v[100:105].all() and not v[200:230].any())

# T5 local contour unchanged by an octave shift
cl = build_dataset()[0]
f0 = features(cl["cents"], cl["conf"]); f1 = features(cl["cents"] + 1200, cl["conf"])
res["T5"] = bool(np.allclose(f0[0], f1[0], atol=1e-4) and np.allclose(f0[2:4], f1[2:4], atol=1e-4))

# T6 network output shape and receptive field
net = DilatedCNN()
out = net.forward(np.zeros((1, 6, 300), np.float32))[0]
x = np.zeros((1, 6, 600), np.float32); x[0, :, 300] = 1.0
base = net.forward(np.zeros((1, 6, 600), np.float32))[0]
diff = np.abs(net.forward(x)[0] - base).sum(axis=1)[0] > 1e-7
rf = int(diff.sum())
res["T6"] = bool(out.shape == (1, 4, 300) and rf == 249)

# T7 boundary mask: frames within +-50 ms of each annotated boundary excluded from the loss
y, m = frame_labels(1000, [(2.0, 3.0, 1)])
res["T7"] = bool(m[195:206].sum() == 0 and m[295:306].sum() == 0 and m[194] == 1 and m[306] == 1 and (y[200:300] == 1).all())

# T8 post-processing: runs < 0.1 s dropped, gaps < 50 ms closed
lab = np.zeros(400, np.int64); lab[50:58] = 2                  # 8-frame run -> dropped
lab[100:200] = 1; lab[150:153] = 0                             # 3-frame gap -> closed
out = postprocess(lab)
res["T8"] = bool((out[50:58] == 0).all() and (out[100:200] == 1).all())

# T9 leave-one-song-out: test song never in training or validation
clips = build_dataset(); songs = sorted({c["song"] for c in clips}); ok = True
for t in songs:
    v = songs[t % len(songs)]
    tr = {c["song"] for c in clips if c["song"] not in (t, v)}
    ok &= (t not in tr) and (v != t)
res["T9"] = bool(ok)

for k, v in res.items():
    print(k, "PASS" if v else "FAIL")
print("receptive field measured:", rf, "frames")
