# Gamaka — Project Overview: Tests, Results, Plans and Approaches

*Snapshot of the folder as of 2026-09-24 (all files dated 2026-09-23/24). Written from the scripts, logs and output files in `D:\PROJECTS\Gamaka`. Numbers are quoted from those outputs; interpretation is marked as such.*

## 1. What the project is

An exploration of whether **gamaka (ornament) shapes in Carnatic music can be found automatically from pitch contours**, and whether they carry information about the **raga** (as opposed to the singer or instrument). Two data sources:

| Source | What it is | Role |
|---|---|---|
| `Gamaka_veena_dataset/` | 222 short WAV clips (~2–6 s) of one veena/guitar-like instrument demonstrating named gamakas (performer referred to as "Aruna"). Filenames carry the gamaka name. | Labelled veena branch: *do discovered shapes match named gamakas?* |
| `saraga_pitch_csv/` + `saraga_inventory.csv` | Pitch tracks from the Saraga 1.5 Carnatic corpus (vocal/instrumental recordings, 198 inventory rows; 14 tracks extracted). | Real-music branch: *does a movement fingerprint track raga or singer?* |

The working "raga-first plan" (named in `anchor_feasibility_check.py`): **(1) find steady notes ("anchors") → (2) check they land on the raga's swaras → (3) describe the movements between anchors → (4) build a per-raga fingerprint.** Each stage is gated by a GO / NO-GO check before the next.

## 2. Pipeline and files

| Stage | Script | Main outputs |
|---|---|---|
| Pitch extraction | `run_crepe.py` — CREPE `full`, 10 ms hop, Viterbi off; saves `[T,360]` activation matrices | `crepe_output/*.npy` (222), `crepe_run.log`, `crepe_run_resume.log` |
| Dataset verification, tonic and session inference | `verify_dataset.py` | `dataset_summary.csv`, `tonic_estimates.csv`, `session_tonics.csv`, `tonic_estimates_NOTE.md`, `takes.csv` |
| Anchor feasibility (v1→v3) | `anchor_feasibility_check{,_v2,_v3}.py` | `anchor_feasibility_output{,_v2}.txt`, `kampita_v2/v3_*_output.txt`, `saraga_v2_output/`, `saraga_v3_output_raga.txt`, `anchor_plots*/`, `saraga_plots_*/` |
| Movement tables and fingerprints | `movement_features{,_v2}.py` | `features/`, `features_v2/`, `movement_features*_output.txt` |
| Raga-vs-singer test with noise floor | `fingerprint_test.py` | `fingerprint_test_output{,_v2}.txt` |
| Gamaka clip clustering and classification | `gamaka_clips.py` | `clip_features.csv`, `gamaka_clips_output{,_v2,_v3,_v4}.txt` |
| Frozen final result | — | `results/clip_features_final_2026-09-24/` (with `PROVENANCE.txt` and SHA-256 hashes) |

Input manifests: `recordings.csv`, `recordings_kampita_todi.csv`, `recordings_kampita_nontodi.csv` (veena clips + tonic), `saraga_manifest.csv` (5 tracks), `tracks.csv` (10 tracks with raga, artist, swaras).

## 3. Approaches, in the order they were tried

### 3.1 Pitch extraction
CREPE full model on all 222 veena clips. First run processed 177; the resume run processed the remaining 45 (`processed_ok=45 skipped_existing=177 failed=0`, 550 s). `verify_dataset.py` checks frame counts against clip duration (`length_ok`).

### 3.2 Tonic (Sa) estimation
No tonic metadata exists for the veena set. Approach, per `tonic_estimates_NOTE.md`:
- A **single dataset-wide tonic, Sa = E4 = 329.3 Hz**, for all 222 clips.
- Evidence: (a) E is the most common sustained pitch, 37.6% of high-confidence sustained frames, with B (Pa) next at 16.7% and F (R1) only 0.2%; (b) every session's most common pitch is on the E major scale, except one that lands on C; (c) the `odukkul (kambhoji) pa dha sa` filename gives B3, C#4, E4 = Pa, Dha, Sa.
- **Rejected:** per-clip or per-session histogram modes. The mode is the longest-held note, not necessarily Sa (e.g. andolita clips peak on Ga and Pa).
- **Status: inference, not ground truth.** Confirmation needed from the performer. E3 vs E4 is irrelevant for the anchor check because pitches are folded mod 1200 cents.

For Saraga tracks the tonic comes from the corpus metadata (`saraga_manifest.csv`, `tracks.csv`).

### 3.3 Sessions and takes
Clips are Logic Pro bounces. The `bext TimeReference` gives each clip's timeline position. Clips that tile one timeline without overlap are treated as one continuous **take** (`takes.csv`, 23 takes). This matters because clips from the same take are near-duplicates and inflate accuracy if they are split across train and test.

### 3.4 Anchor segmentation (steady-note detection)
A hysteresis rule: a note starts when frames stay within ±20 cents of their mean, and ends when the pitch leaves by more than 50 cents. Minimum 80 ms, confidence > 0.5, 50 ms median smoothing. The script itself says this is the author's own implementation of the idea and should be checked against the Rao et al. paper.
- **v1** — baseline.
- **v2** — added a max drift of 30 cents per anchor (stops slow glides becoming chains of fake notes). Added "gesture ≥ 0.15 s" share, and flagged **trivial clips** (pitch range < 200 cents) and suspected octave errors (anchors > 1000 cents from the clip median).
- **v3** — added **oscillation (kampita-like) detection** read from the pitch contour's turning points, not the anchor chain (reversal ≥ 40 cents, steps ≤ 500 cents, gap ≤ 0.35 s, ≥ 3 turning points). Reports "held notes only" on-raga rate, excluding oscillation turning points.

Gate thresholds: anchors on a raga note (≤ 40 cents) ≥ 75%; unanchored movements > 1 s ≤ 30% of voiced time.

### 3.5 Movement features and fingerprints
`movement_features_v2.py` builds one row per movement and oscillation (start/end swara, extent, turning points, direction). Movements under 0.05 s are kept in the CSV but excluded from fingerprints. Fingerprints compared across tracks: **notes** (where the notes are), **transitions** (note-to-note), **shapes** (movement shape), and in v2 of the test **svara-shapes** (which shape on which svara).

### 3.6 Raga vs singer test (`fingerprint_test.py`)
Fixes three weaknesses of the earlier whole-track comparison:
1. **Noise floor** — each track is cut into 5-minute segments, and same-track pairs set the best achievable distance.
2. **Equal sample sizes** — fixed-length segments, those with fewer than 100 movements dropped.
3. **Allied ragas** — Kambhoji, Harikambhoji and Kedaragaula share a note set, so they are reported separately from genuinely different ragas (Kharaharapriya).

### 3.7 Gamaka clip classification (`gamaka_clips.py`)
- **Unsupervised:** cluster the clips by shape, then check cluster purity against the filename gamaka names.
- **Supervised:** 3-NN on shape features, scored with **leave-one-take-out**. Leave-one-clip-out is printed only to show the inflation.
- Alias merged: `pratayagata` = `pratyagata`. `sphutitam` deliberately left separate, pending the artist's ruling.

## 4. Results

### 4.1 Anchor feasibility on veena clips

| Run | Material | Anchors on raga note | Verdict |
|---|---|---|---|
| v1 | odukkul (Kambhoji) ×5, sphuritam (Kanakangi-type) ×5 | 89% | GO |
| v2 | same, trivial clips excluded | 94% | GO |
| kampita v2/v3, non-Todi | Kampita (leena/plavita) | 100% (held notes only: 100% of 20) | GO |
| kampita v2/v3, **Todi** | Kampita (plavitam) todi | **50%** (held notes only: 40% of 10) | **NO-GO** |

Observations:
- Two sphuritam clips are trivial (≈100 cents range, one note) and were excluded.
- The 100% on non-Todi kampita is on very few anchors (about 20), so it is weak evidence.
- The **Todi NO-GO is the one failure**: anchors fall on 100/300/500 cents instead of raga notes, so either the Todi swara set or the tonic is wrong for that material, or CREPE tracking fails there. *(Interpretation: this is consistent with the tonic being an unconfirmed inference. A tonic off by a fifth or a different Sa for those takes would produce exactly this.)*
- Suspected octave errors appear at about −2400 cents in odukkul clips and −2895 cents in kampita (plavita). Octave folding hides them; they are flagged for checking by ear.
- Long unanchored movements are 0% everywhere (passes), but "real gestures ≥ 0.15 s" are only about 4–26% of voiced time, so this check is largely passing for free on short, mostly steady clips. The script warns about exactly this.
- `diagnostics/` holds two WAV excerpts (odukkul Sa reference and low segment) cut for checking the octave flags by ear.

### 4.2 Anchor feasibility on Saraga vocal tracks

| Track | Anchors on raga (v3) | 12-tone baseline | Oscillations | Note |
|---|---|---|---|---|
| Rithvik Raja – Thillana (Senchurutti) | 80% | 85% | 19 | |
| Rithvik Raja – Karunimpa Idi (Sahana) | 90% | 94% | 10 | |
| Akkarai Sisters – Karunimpa Idi | 94% | 96% | 39 | |
| S Sundar – Enadhu Manam Kavalai (Harikambhoji) | 77% | 86% | 14 | barely over the gate |
| Cherthala R Sharma – Tulasi Bilva (Kedaragaula) | 86% | 93% | 59 | |
| **Pooled** | **86% (GO)** | — | 141 (11% of voiced time) | held notes only: 87% of 2123 |

- Long unanchored movements: 5% of voiced time (GO). Real gestures are 48% of voiced time, so the segmenter is finding substantial movement here, unlike the veena clips.
- Oscillations are consistent across tracks: median ~0.9–1.0 s, 3 turning points, ~245–285 cents wide, 2–2.8 swings/s.
- The 12-tone baseline (all semitones allowed) always scores higher than the raga-restricted score, as it should. The gap (about 3–9 points) is the evidence that anchors prefer raga notes over arbitrary semitones.
- A GO here does **not** prove the tonic is right; the scripts say so explicitly.

### 4.3 Raga vs singer fingerprint (the core scientific question)

**First pass, 5 tracks** (`movement_features_output.txt`): distances between notes, transitions and shapes. Too few tracks for a conclusion; it motivated the cleaner test.

**Second pass, 10 tracks and 69 five-minute segments, 2346 pairs** (`movement_features_v2_output.txt`):
- Track-level verdict: **notes, transitions and shapes all track the singer rather than the raga** (same raga/different singer 0.305 vs same singer/different raga 0.292 for notes; 0.131 vs 0.095 for shapes). The script cautions these are averages over very few pairs.

**Third pass with noise floor** (`fingerprint_test_output*.txt`), median distances:

| Pair type | Notes | Transitions | Shapes |
|---|---|---|---|
| Same track (noise floor) | 0.324 | 0.438 | 0.189 |
| Same raga, same singer | 0.318 | 0.446 | 0.191 |
| Same raga, different singer | 0.361 | 0.467 | 0.213 |
| Different raga (allied), same singer | 0.336 | 0.442 | 0.184 |
| Different raga (allied), different singer | 0.435 | 0.518 | 0.201 |
| Different raga (not allied), same singer | 0.506 | 0.546 | 0.197 |
| Different raga (not allied), different singer | 0.453 | 0.543 | 0.202 |

Reading, from the v2 effect summaries:
- **Notes:** raga effect for *not allied* ragas is large (+0.188 vs singer effect +0.043). For *allied* ragas it is only +0.018, **smaller than the singer effect**. This matches the expectation that a note histogram cannot separate ragas sharing a swara set.
- **Transitions:** the same pattern, weaker (+0.100 not allied; −0.004 allied).
- **Shapes:** raga effect essentially zero (+0.006 not allied; −0.007 allied). **Shapes carry no raga information in this data** and are not distinguishable from noise.
- **Svara-shapes** (shape on svara): raga effect +0.064 not allied vs singer +0.026, but only +0.017 allied. Better than plain shapes, still not separating allied ragas.
- The earlier "SEPARATES / does NOT separate" lines compare against the noise floor and agree: notes and transitions separate, shapes do not.

*Interpretation:* the fingerprint works for ragas with different note sets, is not shown to work for allied ragas (the case a gamaka-based fingerprint would need to win), and the shape features add nothing so far. Singer and raga are partially confounded in the sample, and several singers have only 2–3 segments.

### 4.4 Veena gamaka classification

Latest run (v4, 222 clips, 15 gamaka types, 23 takes; `results/clip_features_final_2026-09-24/`):

| Metric | Result | Chance / baseline |
|---|---|---|
| Unsupervised cluster purity (15 clusters) | **0.35** | 0.16 (always guess commonest) |
| 3-NN, leave-one-take-out (honest) | **0.14** | — |
| 3-NN, leave-one-clip-out (inflated) | 0.36 | — |

History across versions: v1 (185 clips, 13 types) purity 0.37, take-out 0.12, clip-out 0.29; v4 purity 0.35, take-out 0.14, clip-out 0.36. Adding the take file and more clips left the honest accuracy at roughly 12–14%.

- The gap between clip-out (0.36) and take-out (0.14) shows how much near-duplicate takes inflate accuracy.
- Most confused: sphuritam→pratyagata (10), pratyagata→sphuritam (7), kampita→ravai (6), jaru→orikai (5).
- **Eleven of 15 gamaka types have only one take**, so grouped scoring cannot test them at all. 37 clips have no movement ≥ 0.05 s (all held note).
- The script's own note: pitch alone should not separate gamakas that differ by articulation (kandippu/jaru) or plucking several strings (tribhinna). This was expected, not a bug.

*Interpretation:* with this feature set and this much data, shape features recover little of the gamaka label. The result is a negative finding, but the sample (one performer, mostly one take per class) cannot say whether the features or the data are the limit.

## 5. Where things stand

**Established (with the caveats above)**
- The anchor method finds steady notes that land on raga notes: 86% pooled on real vocal tracks, 94–100% on veena clips outside Todi.
- Notes and transitions fingerprints separate ragas with different swara sets; they track singer for allied ragas.
- Honest evaluation shows weak gamaka classification from pitch shape on the veena set.

**Not established / open**
- Sa = E4 for the veena set is an inference. The Todi kampita NO-GO may be a symptom.
- Octave errors flagged in odukkul and kampita(plavita) clips have not been checked by ear.
- Shape fingerprints carry no raga signal yet.
- Most gamaka classes have a single take.
- `sphutitam` vs `sphuritam` is undecided.
- The segmentation thresholds (20/50/30 cents, 80 ms, etc.) are judgment calls, untuned and unvalidated against the Rao et al. method.
- Only 10 of the 198 Saraga inventory entries have been used; per-raga sample sizes are 1–3 singers.

## 6. Suggested next steps (from the evidence, not previously planned in the files)

1. Ask the performer to confirm Sa for the dataset and the Todi material, and re-run the Todi kampita check.
2. Listen to the `diagnostics/` clips to settle the octave flags.
3. Collect more takes per gamaka (at least 2–3 each) so grouped scoring means something; have the artist rule on `sphutitam` and name the clusters.
4. Add features that can tell articulation apart, such as amplitude/onset envelope, rather than pitch alone.
5. Expand the Saraga set (more singers per raga, more non-allied raga pairs) and re-run `fingerprint_test.py`, which already has the noise-floor and allied-raga controls.
6. Tune the segmentation thresholds on held-out material before reporting them as method results.

## 7. How to reproduce

```bash
python run_crepe.py                          # CREPE on the veena WAVs
python verify_dataset.py                     # checks, sessions, tonic estimate
python anchor_feasibility_check_v3.py --manifest recordings.csv
python movement_features_v2.py --manifest tracks.csv --out_dir features_v2/
python fingerprint_test.py --manifest tracks.csv --segment_minutes 5
python gamaka_clips.py --crepe_dir crepe_output --tonic 329.3 --out clip_features.csv --takes_csv takes.csv
```

The frozen final clip table and its checksums are in `results/clip_features_final_2026-09-24/PROVENANCE.txt`.
