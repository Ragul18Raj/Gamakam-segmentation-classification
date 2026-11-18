"""Pitch-shape PNGs, one per branch (playing variant) of each stage-1 gamaka.
Run from veena_stage1:  python local_run/shapes.py  [--table-only]
Writes local_run/shapes/<gamaka>__<branch>.png, index.md and clip_shape_stats.csv."""
import csv, os, re, sys, math, collections, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shapes')
SA = 327.0; CONF = 0.5; MED_WIN = 25; MAX_PANELS = 12; FR = 100

def crepe_activation_to_pitch(act):
    """Copied from extract.py / anchor_feasibility_check_v3.py (same local weighted average CREPE uses)."""
    cents_map = np.linspace(0, 7180, 360) + 1997.3794084376191
    T = act.shape[0]; cents = np.zeros(T); conf = act.max(axis=1)
    for t in range(T):
        c = int(np.argmax(act[t])); lo, hi = max(0, c - 4), min(360, c + 5)
        w = act[t, lo:hi]; cents[t] = np.sum(w * cents_map[lo:hi]) / max(np.sum(w), 1e-9)
    return 10.0 * 2 ** (cents / 1200.0), conf

# normalised file-name stem -> branch label, per gamaka
BRANCHES = {
    'jaru asc': 'ascending',
    'jaru desc melgana': 'descending Melgana',
    'jaru desc shankravarnam': 'descending Shankarabharanam',
    'kampita plavita': 'plavita',
    'kampita plavita bhairiga': 'plavita bhairiga',
    'kampita plavitam todi': 'plavitam Todi',
    'kampita leena': 'leena',
    'ravai kalapriya': 'Kalapriya',
    'ravai mayamolagolav': 'Mayamalavagowla',
    'ravai': 'unspecified (#02)',
    'pratyagata hindola': 'Hindolam',
    'pratyagata amritvarshini': 'Amritavarshini',
}

def split_name(fn):
    """-> (normalised stem, slice label) from a wav file name."""
    s = os.path.splitext(fn)[0]
    og1 = s.endswith('_OG_1')
    s = re.sub(r'\s*Guitar_samples_3D_OG(_1)?$', '', s)
    m = re.search(r'#\d+(\.\d+)?', s)
    if m: sl = m.group(0)
    else:
        nums = re.findall(r'_\s*(!?\d+|!)(?=_|$)', s)
        sl = '_' + nums[-1] if nums else '(no number)'
    if og1: sl += ' (OG_1)'
    stem = re.sub(r'#\d+(\.\d+)?', '', s)
    stem = re.sub(r'_\s*(!?\d+|!)(?=_|$)', '', stem)       # _N, _!2, _! anywhere (Jaru_desc_1_melgana)
    stem = re.sub(r'[()_]', ' ', stem.lower())
    stem = re.sub(r'\s+', ' ', stem).strip().replace('pratayagata', 'pratyagata')
    return stem, sl

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def load_clips():
    keep = {'jaru', 'kampita', 'ravai', 'pratyagata'}
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'takes.csv'), encoding='utf-8-sig')))
    clips, unplaced = collections.defaultdict(list), []
    for r in rows:
        cls = r['take'].split('|', 1)[0]
        if cls not in keep or '_continuous' in r['file'].lower(): continue
        stem, sl = split_name(r['file'])
        br = BRANCHES.get(stem)
        if br is None or not stem.startswith(cls): unplaced.append((r['file'], cls, stem)); continue
        clips[(cls, br)].append((r['file'], sl, r['take'].split('|', 1)[1]))
    for k in clips: clips[k].sort(key=lambda c: natural_key(c[0]))
    return clips, unplaced

def clip_pitch(fn):
    act = np.load(os.path.join(ROOT, 'crepe_output', os.path.splitext(fn)[0] + '.npy'))
    hz, conf = crepe_activation_to_pitch(act)
    voiced = conf >= CONF
    cents = np.full(len(hz), np.nan); cents[voiced] = 1200 * np.log2(hz[voiced] / SA)
    vi = np.where(voiced)[0]; x = cents[vi]
    h = MED_WIN // 2                                        # window shrinks at the edges (no edge padding,
    med = np.array([np.median(x[max(0, i - h):i + h + 1]) for i in range(len(x))])  # which would repeat an end outlier)
    cand = np.stack([x - 1200, x, x + 1200])               # one-octave shift only, toward the running median
    best = np.argmin(np.abs(cand - med), axis=0)
    fixed = cand[best, np.arange(len(x))]; shifted = np.zeros(len(hz), bool); shifted[vi] = best != 1
    cents[vi] = fixed
    t = (np.arange(len(hz)) - vi[0]) / FR
    return t, cents, shifted, voiced

SVARA = ['S', 'R1', 'R2', 'G2', 'G3', 'M1', 'M2', 'P', 'D1', 'D2', 'N2', 'N3']
def svara_label(c):
    octv, i = divmod(int(round(c / 100)), 12)
    return SVARA[i] + ("'" * octv if octv > 0 else '') if octv >= 0 else ',' * -octv + SVARA[i]

def safe(s): return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')

def plot_branch(cls, br, items, stats):
    n = len(items); shown = items[:MAX_PANELS]
    data = []
    for fn, sl, take in shown:
        t, c, sh, v = clip_pitch(fn)
        cv = c[v]; rng = float(np.percentile(cv, 95) - np.percentile(cv, 5))
        data.append((sl, t, c, sh, rng))
    for fn, sl, take in items:                              # stats for every clip, plotted or not
        t, c, sh, v = clip_pitch(fn); cv = c[v]
        stats.append(dict(gamaka=cls, branch=br, file=fn, slice=sl, take=take, plotted=fn in [s[0] for s in shown],
                          voiced_s=round(v.sum() / FR, 2), span_s=round(float(t[v][-1]), 2),
                          range_p95_p5=round(float(np.percentile(cv, 95) - np.percentile(cv, 5)), 1),
                          median_cents=round(float(np.median(cv)), 1), oct_shifted=int(sh.sum()),
                          oct_shifted_frac=round(float(sh.sum() / v.sum()), 3)))
    allc = np.concatenate([d[2][~np.isnan(d[2])] for d in data])
    lo, hi = math.floor(allc.min() / 100) * 100 - 50, math.ceil(allc.max() / 100) * 100 + 50
    tmax = max(np.nanmax(np.where(np.isnan(d[2]), np.nan, d[1])) for d in data) + 0.05
    ncol = min(4, len(shown)); nrow = math.ceil(len(shown) / ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.6 * ncol + 0.6, 2.3 * nrow + 0.7), sharex=True, sharey=True, squeeze=False)
    grid = np.arange(math.ceil(lo / 100) * 100, hi + 1, 100)
    dense = (hi - lo) <= 1500
    for k, ax in enumerate(axs.flat):
        if k >= len(data): ax.axis('off'); continue
        sl, t, c, sh, rng = data[k]
        for g in grid:
            strong = int(round(g / 100)) % 12 in (0, 7)
            ax.axhline(g, color='0.55' if strong else '0.88', lw=0.8 if strong else 0.5, zorder=0)
        ax.plot(t, c, color='#1f4e79', lw=0.9, zorder=2)      # NaN at unvoiced frames -> gaps stay gaps
        if sh.any(): ax.plot(t[sh], c[sh], 'o', color='#d62728', ms=1.6, zorder=3)
        ax.set_title(f'{sl} · range {rng:.0f}¢' + (f' · {sh.sum()} oct-shifted' if sh.any() else ''), fontsize=8)
        ax.set_xlim(0, tmax); ax.set_ylim(lo, hi); ax.tick_params(labelsize=7)
    ticks = [g for g in grid if dense or int(round(g / 100)) % 12 in (0, 7)]
    for ax in axs[:, 0]:
        ax.set_yticks(ticks); ax.set_yticklabels([f'{svara_label(g)} {int(g):+d}' for g in ticks]); ax.set_ylabel('cents above Sa (svara)', fontsize=8)
    for ax in axs[-1, :]: ax.set_xlabel('s from first voiced frame', fontsize=8)
    cnt = f'{n} clips' if n <= MAX_PANELS else f'{MAX_PANELS} of {n} clips'
    fig.suptitle(f'{cls} · {br} · {cnt} · Sa 327 Hz', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(OUT, f'{cls}__{safe(br)}.png'); fig.savefig(path, dpi=150); plt.close(fig)
    return os.path.basename(path)

if __name__ == '__main__':
    clips, unplaced = load_clips()
    order = ['jaru', 'kampita', 'ravai', 'pratyagata']
    keys = sorted(clips, key=lambda k: (order.index(k[0]), list(BRANCHES.values()).index(k[1])))
    print(f'{"gamaka":<11} {"branch":<30} clips  takes')
    for k in keys:
        print(f'{k[0]:<11} {k[1]:<30} {len(clips[k]):>5}  {", ".join(sorted({c[2] for c in clips[k]}))}')
    print('total', sum(len(v) for v in clips.values()), '| unplaced', len(unplaced)); [print('  UNPLACED', u) for u in unplaced]
    if '--table-only' in sys.argv: sys.exit()
    os.makedirs(OUT, exist_ok=True); stats = []; idx = []
    for k in keys:
        png = plot_branch(k[0], k[1], clips[k], stats); idx.append((k, png, len(clips[k]))); print('wrote', png)
    with open(os.path.join(OUT, 'clip_shape_stats.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0])); w.writeheader(); w.writerows(stats)
    with open(os.path.join(OUT, 'index.md'), 'w', encoding='utf-8') as f:
        f.write('# Pitch shapes per branch (Sa = 327 Hz)\n\nVoiced frames only (CREPE confidence >= 0.5), gaps left as gaps. '
                'Red dots: frames shifted by ±1200 cents toward the 25-frame running median of the clip\'s voiced frames '
                '(window shrinks at the clip edges; one pass, one octave at most). At most 12 panels per PNG.\n'
                'y ticks: svara and cents above Sa; `S\'` = upper octave, `,S` / `,,S` = one / two octaves below. '
                'When a panel spans more than 1500 cents only S and P are labelled (light lines stay every 100 cents).\n'
                'Per-clip numbers: `clip_shape_stats.csv`.\n\n| Gamaka | Branch | Clips | PNG |\n|---|---|---|---|\n')
        for (cls, br), png, n in idx:
            f.write(f'| {cls} | {br} | {n}{" (12 shown)" if n > MAX_PANELS else ""} | [{png}]({png.replace(" ", "%20")}) |\n')
        f.write(f'\nTotal: {sum(i[2] for i in idx)} clips in {len(idx)} branches. Unplaced files: {len(unplaced) or "none"}.\n')
