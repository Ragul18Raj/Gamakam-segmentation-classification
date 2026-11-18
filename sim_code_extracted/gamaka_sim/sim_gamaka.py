"""Simulated vocal-style gamaka dataset for a pipeline check.

Mirrors the planned recording design: 7 songs x 3 target gamakas x 3 tempos x 2 octaves
= 126 clips of roughly 20-30 s. The musical content of a passage (notes, which gamakas,
their shapes) depends only on (song, gamaka), so the 6 recordings of one passage share
content, as in the real plan; the tempo, octave and performance details vary.

G1 = oscillation upward from a note (kampita-like)
G2 = slide across two or more scale steps (jaru-like)
G3 = two quick touches of the lower neighbour (sphuritam-like)
Non-target ornaments that resemble them are labelled none:
  vib   = small symmetric oscillation around a note
  slide = short glide between adjacent notes
  flick = one quick touch of the upper neighbour

Output: per clip a CREPE-like pitch track (cents above Sa, NaN = unvoiced) and confidence
at 10 ms frames, plus annotated events (with annotator boundary jitter).
This is SIMULATED data. It is only meant to exercise the pipeline end to end.
"""
import numpy as np

FR = 100
NOTES = [-500, -300, 0, 200, 400, 700, 900, 1200, 1400]   # Mohanam-like scale, cents above Sa
TEMPOS = {"slow": 1.4, "medium": 1.0, "fast": 0.72}
OCTAVES = {"low": 0, "high": 1200}
GAMAKAS = {1: "G1", 2: "G2", 3: "G3"}


def ss(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def neighbour(ni, step):
    return NOTES[int(np.clip(ni + step, 0, len(NOTES) - 1))]


def content_plan(song, g, stress=False):
    """Musical content of one passage: depends only on (song, gamaka)."""
    rng = np.random.default_rng(1000 * song + g)
    style = {
        "g1_rate": rng.uniform(4.6, 6.4), "g1_ext": rng.uniform(90, 200) if stress else rng.uniform(110, 200),
        "g2_dur": rng.uniform(0.38, 0.62), "g3_depth": rng.uniform(110, 190),
    }
    tokens = []
    kinds = [g] * 4
    kinds += list(rng.choice([k for k in (1, 2, 3) if k != g], size=rng.poisson(0.8)))
    kinds += list(rng.choice(["vib", "slide", "flick"], size=rng.integers(3, 6)))
    rng.shuffle(kinds)
    ni = int(rng.integers(3, 7))
    for k in kinds:
        for _ in range(int(rng.integers(1, 3))):          # held notes between ornaments
            ni = int(np.clip(ni + rng.integers(-2, 3), 1, len(NOTES) - 2))
            tokens.append(("hold", ni, rng.uniform(0.45, 1.3)))
        if k == 2:
            step = int(rng.choice([-3, -2, 2, 3]))
            nj = int(np.clip(ni + step, 0, len(NOTES) - 1))
            if abs(nj - ni) < 2:
                nj = int(np.clip(ni - step, 0, len(NOTES) - 1))
            tokens.append((2, ni, nj, rng.uniform(0.85, 1.15)))
            ni = nj
        elif k == "slide":
            stp = rng.choice([-2, -1, 1, 2]) if stress else rng.choice([-1, 1])
            nj = int(np.clip(ni + stp, 0, len(NOTES) - 1))
            tokens.append(("slide", ni, nj, rng.uniform(0.2, 0.45) if stress else rng.uniform(0.15, 0.32)))
            ni = nj
        elif k == 1:
            tokens.append((1, ni, int(rng.integers(3, 6)), rng.uniform(0.85, 1.15)))
        elif k == 3:
            tokens.append((3, ni, rng.uniform(0.85, 1.15)))
        elif k == "vib":
            tokens.append(("vib", ni, rng.uniform(0.6, 1.2), rng.uniform(80, 180) if stress else rng.uniform(50, 105)))
        elif k == "flick":
            tokens.append(("flick", ni, rng.uniform(0.08, 0.14), bool(stress and rng.random() < 0.5)))
        tokens.append(("hold", ni, rng.uniform(0.3, 0.9)))
    return style, tokens


def render(song, g, tempo, octave, seed, stress=False):
    """Render one recording of passage (song, g) at a tempo and octave."""
    style, tokens = content_plan(song, g, stress)
    tau = TEMPOS[tempo]
    rng = np.random.default_rng(seed)
    pieces, events = [], []
    t_cursor = 0.0
    prev_end = None

    def add(arr, label=0):
        nonlocal t_cursor
        if label:
            events.append((t_cursor, t_cursor + len(arr) / FR, label))
        pieces.append(arr)
        t_cursor += len(arr) / FR

    def frames(sec):
        return max(2, int(round(sec * FR)))

    since_breath = 0
    for tok in tokens:
        kind = tok[0]
        start_note = NOTES[tok[1]]
        if prev_end is not None and abs(prev_end - start_note) > 1:   # plain transition
            n = frames(rng.uniform(0.04, 0.09) * tau ** 0.5)
            u = np.linspace(0, 1, n)
            add(prev_end + (start_note - prev_end) * ss(u))
        if kind == "hold":
            n = frames(tok[2] * tau * rng.uniform(0.9, 1.1))
            arr = np.full(n, float(start_note))
            if n > 40 and rng.random() < 0.3:                         # consonant gap
                a = int(rng.integers(10, n - 20))
                arr[a:a + int(rng.integers(3, 7))] = np.nan
            add(arr)
            end = start_note
            since_breath += 1
            if since_breath >= 5 and rng.random() < 0.5:
                add(np.full(frames(rng.uniform(0.12, 0.3)), np.nan))
                since_breath = 0
        elif kind == 1:
            rate = style["g1_rate"] * tau ** -0.25 * rng.uniform(0.92, 1.08) * tok[3]
            cycles = max(2, int(round(tok[2] * tau ** 0.5 + rng.integers(-1, 1))))
            ext = style["g1_ext"] * (rng.uniform(0.7, 1.3) if stress else rng.uniform(0.85, 1.15))
            n = frames(cycles / rate)
            u = np.arange(n) / FR
            p = rng.uniform(0.7, 1.4)
            wave = (0.5 * (1 - np.cos(2 * np.pi * rate * u))) ** p
            add(start_note + ext * wave, 1)
            end = start_note
        elif kind == 2:
            target = NOTES[tok[2]]
            n = frames(style["g2_dur"] * tok[3] * tau ** 0.7 * rng.uniform(0.9, 1.1))
            u = np.linspace(0, 1, n)
            shape = ss(u) ** rng.uniform(0.8, 1.3)
            over = rng.uniform(0, 40) * np.sign(target - start_note) * np.exp(-((u - 0.88) / 0.07) ** 2)
            add(start_note + (target - start_note) * shape + over, 2)
            end = target
        elif kind == 3:
            depth = style["g3_depth"] * rng.uniform(0.85, 1.15) * tok[2]
            tw = rng.uniform(0.07, 0.11) * tau ** 0.5
            gap = rng.uniform(0.04, 0.07) * tau ** 0.5
            n = frames(2 * tw + gap + 0.06)
            u = np.arange(n) / FR
            c1, c2 = 0.03 + tw / 2, 0.03 + tw + gap + tw / 2
            dip = np.exp(-((u - c1) / (tw / 2.6)) ** 2) + np.exp(-((u - c2) / (tw / 2.6)) ** 2)
            add(start_note - depth * dip, 3)
            end = start_note
        elif kind == "vib":
            n = frames(tok[2] * tau * rng.uniform(0.9, 1.1))
            u = np.arange(n) / FR
            env = ss(u / 0.1) * ss((u[-1] - u) / 0.1)
            bias = rng.uniform(0, 0.5) * tok[3] / 2 if stress else 0.0
            add(start_note + (tok[3] / 2 * np.sin(2 * np.pi * rng.uniform(5.5, 7.0) * u) + bias) * env)
            end = start_note
        elif kind == "slide":
            target = NOTES[tok[2]]
            n = frames(tok[3] * tau ** 0.7)
            add(start_note + (target - start_note) * ss(np.linspace(0, 1, n)))
            end = target
        elif kind == "flick":
            up = neighbour(tok[1], 1) - start_note
            tw = tok[2] * tau ** 0.5
            if tok[3]:
                n = frames(2 * tw + 0.1)
                u = np.arange(n) / FR
                add(start_note + up * (np.exp(-((u - 0.03 - tw / 2) / (tw / 2.6)) ** 2)
                                       + np.exp(-((u - 0.07 - 1.5 * tw) / (tw / 2.6)) ** 2)))
            else:
                n = frames(tw + 0.06)
                u = np.arange(n) / FR
                add(start_note + up * np.exp(-((u - 0.03 - tw / 2) / (tw / 2.6)) ** 2))
            end = start_note
        prev_end = end

    cents = np.concatenate(pieces) + OCTAVES[octave]
    T = len(cents)
    t = np.arange(T) / FR
    voiced = ~np.isnan(cents)
    # CREPE-like measurement noise, slow intonation drift, tonic error, short glitches
    tonic_err = rng.uniform(-15, 15)
    drift = 6 * np.sin(2 * np.pi * rng.uniform(0.1, 0.3) * t + rng.uniform(0, 6.3))
    if stress:
        drift = drift * 2.5 + 8 * np.sin(2 * np.pi * rng.uniform(0.8, 1.6) * t + rng.uniform(0, 6.3))
    cents = cents + rng.normal(0, 4, T) + drift + tonic_err
    conf = np.where(voiced, rng.uniform(0.82, 0.96, T), rng.uniform(0.05, 0.35, T))
    for _ in range(int(T * 0.003)):
        i = int(rng.integers(0, T - 3))
        k = int(rng.integers(1, 4))
        cents[i:i + k] += rng.choice([-1, 1]) * rng.uniform(60, 150)
        conf[i:i + k] = rng.uniform(0.45, 0.7)
    cents[~voiced] = np.nan

    # annotator boundary jitter (sd 25 ms; 60 ms under stress)
    sd = 0.06 if stress else 0.025
    ann = []
    for s, e, l in events:
        s2 = max(0.0, s + rng.normal(0, sd))
        e2 = min(T / FR, e + rng.normal(0, sd))
        if e2 - s2 > 0.05:
            ann.append((round(s2, 3), round(e2, 3), l))
    return cents.astype(np.float32), conf.astype(np.float32), ann, events


def build_dataset(n_songs=7, stress=False):
    clips = []
    for song in range(1, n_songs + 1):
        for g in (1, 2, 3):
            for ti, tempo in enumerate(TEMPOS):
                for oi, octave in enumerate(OCTAVES):
                    seed = 7919 * song + 104729 * g + 31 * ti + 7 * oi
                    cents, conf, ann, true_ev = render(song, g, tempo, octave, seed, stress)
                    clips.append(dict(id=f"song{song}_G{g}_{tempo}_{octave}", song=song, target=g,
                                      tempo=tempo, octave=octave, cents=cents, conf=conf,
                                      ann=ann, true_events=true_ev))
    return clips


if __name__ == "__main__":
    clips = build_dataset()
    durs = [len(c["cents"]) / FR for c in clips]
    nev = {1: 0, 2: 0, 3: 0}
    for c in clips:
        for _, _, l in c["ann"]:
            nev[l] += 1
    print(f"{len(clips)} clips, {min(durs):.1f}-{max(durs):.1f} s (mean {np.mean(durs):.1f}), total {sum(durs)/60:.1f} min")
    print("annotated events:", {GAMAKAS[k]: v for k, v in nev.items()}, "total", sum(nev.values()))
