# Vocal gamaka study: plan written before any model is run

**Version 1, written 17 Nov 2026, before any classifier or detector has been run on this data.**
Anything changed after a model result has been seen goes in the change log at the bottom and is labelled **exploratory**.

Source of truth for the process: `contexts 29_10/vocal_process_after_annotation.md`.

## 0. Data (frozen)

| Item | Value |
|---|---|
| Audio | `audio/VARNAM.mp3` + `audio/TRACK 1.mp3` … `TRACK 30.mp3` (31 recordings, 32.4 min, MP3 48 kHz stereo). Byte-identical copies of `vocal rec/`. |
| Labels, pass 1 | `annotations/pass1.xlsx`, sha256 `9779bc52b24bb556a876ac3e77e42f51b1645494305441c7b67bae1efc7cbb8c` |
| Events | `annotations/pass1_events.csv`, 293 events: Kampita 132, Nokku 82, Jaaru (Etra) 45, Jaaru (Irakka) 34. All checks in `results/annotation_checks.txt` passed. |
| Pitch | `crepe/<rec>.npy|.npz`: CREPE full, 10 ms, viterbi off, channels averaged (`run_crepe_vocal.py`). |
| Tonic | Measured per recording from the Sa sung at the start of every clip (singer, 17 Nov), `tonic_from_opening_sa.py` → `results/tonic.csv`. |
| Grouping | `metadata.csv`. **PROVISIONAL**: TRACK 1–3 = P01 … TRACK 28–30 = P10 (assumed one passage at slow/medium/fast), VARNAM = P11. Inferred from track order and lengths; **not yet confirmed by the singer**. |

Note: the folder holds **30** numbered tracks plus the VARNAM (TRACK 1 to TRACK 30), not 29.

## 1. The question

- **Stage 1.** Given the singer's start and end times, can the four gamaka types be told apart from the pitch line alone?
- **Stage 2.** Can a model find and name them in the full tracks without being given the boundaries?

## 2. Classes

- **Main:** Kampita, Nokku, Jaaru (Etra), Jaaru (Irakka) (+ "none" in stage 2).
- **Secondary:** Etra and Irakka merged into "Jaaru" (3 classes, + none in stage 2).

## 3. Split (both stages)

- **Leave-one-group-out**, 11 folds (P01–P11). A group's slow, medium and fast versions are always tested together and never
  seen in training. Scores are computed on the pooled held-out predictions.
- In stage 2, the validation group of each fold is the next group in order (P11's validation group is P01), as in `run_eval.py`.
  Nothing is tuned on the test group.
- **Robustness split (reported, not a pass criterion):** blocks of 6 consecutive tracks (TRACK 1–6, 7–12, …, 25–30) + VARNAM,
  6 folds. If the provisional grouping is wrong in a way that spans neighbouring triplets, this split still keeps them together.
- **With and without VARNAM:** scores also reported on the tracks only (VARNAM's held-out predictions removed), because VARNAM
  holds 30% of Nokku and 38% of Etra.

## 4. Pitch preparation (stage 1 and 2, no class labels used)

- Cents relative to the recording's measured Sa: `1200·log2(f0 / Sa)`.
- Voiced = CREPE confidence ≥ 0.5. Unvoiced gaps inside an event are linearly interpolated.
- Octave slips: a voiced run shorter than 150 ms that sits 1000–1400 cents from both neighbours is folded back by ±1200
  (same idea as `veena_stage1/methods_explore/clean.py`, simplified). This rule may be adjusted **only from the spot-check plots,
  before any classifier is run**, and any change is listed below.
- An event with fewer than 5 voiced frames is excluded from stage 1 and listed. (Decided before looking.)

## 5. Stage 1 methods (all leave-one-group-out)

Trained configurations (the "family"):
1. **F-LR**: the 9 stage-1 features of `veena_stage1/stage1.py` (dur, range p95−p5, turns, turn rate, net, |net|, excursion up,
   excursion down, mean |velocity|) → standardise → logistic regression (max_iter 2000).
2. **F-3NN**: the same features → standardise → 3-NN.
3. **DTW-3NN**: pure-shape DTW (median-centred curve, absolute difference, Sakoe-Chiba band 20% of the longer curve, normalised
   by n+m) → 3-NN, ties to the nearest of the tied classes (`methods_explore/classify.py`). **This is the confirmation on new
   data that the exploratory veena result (0.63) asked for.**
4. **DTW-1NN**: as 3 with k = 1.

Controls (no p-value needed, reported next to the family):
- **majority class**; **duration only** (LR on event length); **median pitch probe** (LR on median cents above Sa, i.e. "which note");
- **length removed**: F-LR without `dur` and `turns`, and DTW-3NN on curves resampled to 100 points. Shows how much of the score
  is the singer's chosen length rather than the movement.
- **scramble**: F-LR and DTW-3NN on curves with the frames shuffled inside each event.

Statistics:
- **Macro-F1** over the 4 classes on pooled held-out predictions (main number), plus per-class recall and the confusion matrix.
- **Permutation test**, 1000 permutations, seed 20261117: event labels are shuffled **within each group** (keeps each group's class
  mix, breaks the link between shape and label), then the whole leave-one-group-out run is repeated. p = (1 + #null ≥ observed) / 1001.
  **Family-wise p**: each permutation takes the maximum over the 4 trained configurations, so choosing the best one is paid for.
- **95% CI**: bootstrap over groups (2000 resamples) for the best configuration and for DTW-3NN.
- Breakdowns: by tempo (provisional), with/without VARNAM, 3-class merged version, robustness split.

## 6. Pass marks (proposed; to be agreed with the guides — not yet signed off)

**Stage 1 PASS** if the best of the 4 trained configurations has macro-F1 ≥ 0.50 **and** family-wise p < 0.05 **and** beats the
median-pitch probe. Majority-class macro-F1 is ≈ 0.16 (Kampita = 45%).
The veena DTW result counts as **confirmed** only if DTW-3NN itself reaches macro-F1 ≥ 0.50 with family-wise p < 0.05.

**Stage 2 PASS** (frozen now, run later): event F1 (macro over the 4 classes, IoU ≥ 0.3, same label, one-to-one, `eval_segments.py`
rule without the ±0.1 s alternative) ≥ 0.60 **and** frame macro-F1 (5 classes) ≥ 0.60 **and** the CNN is at least as good as
the rule-based baseline (paired Wilcoxon over the 11 folds, one-sided, α = 0.05). Secondary scoring: onset within ±200 ms
(`pipeline.match_events`), for comparison with the simulation. Count error per type per recording reported; its target is to be
set with the guides.

## 7. Stage 2 settings (changes from the simulation design in `sim_code_extracted/gamaka_sim/pipeline.py`)

| Change | Why |
|---|---|
| `NCLS = 5` (none + 4); `match_events` / `f1` loop over (1, 2, 3, 4) | The code hard-codes 3 gamaka classes |
| Boundary mask ±50 ms → ±20 ms | The shortest event is 0.14 s; ±50 ms would erase it |
| Majority filter 11 → 5 frames (tuned on the validation group only, from {3, 5, 7}); drop runs < 0.1 s | An 11-frame filter erases 0.14 s events |
| sin/cos note features **kept** | Sa is measured for every recording from the opening Sa. A Sa-free version (features dropped) is run as a secondary check |
| `load_real_dataset()` reading `pass1_events.csv`, `metadata.csv`, `crepe/*.npz` | Real data instead of `build_dataset()` |
| `song` = `group` in the clip dicts | So the existing fold code keeps passages together |
| The opening Sa (0 s → 0.5 s before the first event) is **excluded** from training and scoring | It is a tonic reference, not part of the performance, and is unannotated by design |

Baselines on the same folds: all-none; rule-based movement candidates + 3-NN (`run_eval.py`); Kampita-only random forest
(`kampita_from_annotations/train_kampita.py`) retrained on the vocal labels. Each fold runs as its own single-threaded process.

**Stage 2 is blocked until the singer confirms the grouping and that every gamaka of the four types was marked** (unmarked gamakas
would be trained as "none").

## 8. What counts as failure, and what follows

- **Stage 1 fails** (no configuration ≥ 0.50 with family-wise p < 0.05): stop. Look at the confusion matrix and the plots before
  stage 2; a detector cannot name what a classifier with given boundaries cannot. Report as "not separable with these features
  on this data". The veena DTW result is then *not* confirmed.
- **Stage 1 passes only through duration** (length-removed controls near chance): report that the types are separated mainly by
  the singer's chosen length, which a detector must estimate itself; stage 2 goes ahead but the claim is weaker.
- **Stage 2 fails**: report the stage 1 result plus the Kampita-only detector as the complete, honest result; the 4-class detector
  is reported as "attempted, below pass mark", with every number.
- Any result based on the provisional grouping is labelled **provisional** until the singer confirms it.

## 9. Sources of numbers

Every number in the write-up is tagged: **recorded vocal (real)**, **veena (measured)**, or **synthetic (code check only)**.

## Change log

**17 Nov 2026, ~23:55, before any classifier or detector result had been seen** (CREPE still running). These fill in
details the plan left open, as implemented in `vocal_pitch.py`, `stage1_vocal.py`, `stage2_*.py`:

1. Stage 1 event curve = the event's first to last voiced frame (unvoiced frames at the edges dropped), inner gaps
   linearly interpolated. Octave-slip folding is done on the whole recording before events are cut.
2. Stage 1 "length removed" DTW: curves resampled to 100 points, same band rule as the main DTW (20%).
3. Stage 1 permutation tests are run for all four analyses (main / block6 split × 4 / 3 classes); the pass criterion
   applies only to main, 4 classes.
4. **Stage 2 is run before the singer has confirmed the grouping and the completeness of the marking**, because of the
   30 Nov deadline. Every stage-2 number is labelled **PROVISIONAL** until confirmed; the same code is then re-run unchanged.
5. Stage 2 "CNN at least as good as rule-based": met unless the one-sided Wilcoxon test "rule better than CNN" over the
   11 per-fold event macro-F1 values gives p < 0.05. Both directions are reported. Per-fold macro-F1 averages the classes
   present in that fold (reference or prediction); the pooled main number averages all 4.
6. Stage 2 majority filter: chosen per fold from {3, 5, 7} by event macro-F1 (IoU rule) on the validation group; ties go to 5.
7. Stage 2 robustness split: the same 7 blocks as stage 1 (6 consecutive tracks each, VARNAM alone), validation = next block.
8. Kampita-only random forest: features and settings of `train_kampita.py` (n_jobs = 1 per fold process), trained on the
   train + validation groups, Kampita frames positive and every other frame (other gamakas included) negative.
9. Opening Sa cut: each recording starts 0.5 s before its first annotated event, in both training and scoring.

**18 Nov 2026, after the stage 1 and stage 2 results had been seen.** The singer answered `QUESTIONS_FOR_SINGER.md`; the answers and
the audio checks are in `SINGER_ANSWERS.md`. No method, setting, split or pass mark changes.

10. Grouping and tempo are **confirmed by the singer** and match section 0, so `group` and `tempo` in `metadata.csv` are unchanged.
    The "grouping PROVISIONAL" wording printed by `stage1_vocal.py` is out of date; the code is deliberately left unchanged.
11. Marking is confirmed only as "all the gamakas I could", and **orikai is present but unmarked**, at places the singer cannot list.
    It counts as "none" (as section 2 of the process note allows). Stage-2 precision is therefore read with this caveat.
12. Filled in `metadata.csv`: `raga` (Kalyani for TRACK 1–30 only), `complete_marking`, and `octave` (measured from pitch, not given
    by the singer). No script reads these columns.
13. Correction to item 7: the robustness split has **6 blocks, not 7** (B1–B5 and VARNAM as B7), as section 3 says. `stage2_launch.py`
    also launches a non-existent B6, which exits with an error and stops automatic scoring of that run; `stage2_score.py` is run by hand.
14. Any analysis by octave register is **exploratory**.

**18 Nov 2026, later, after all results above had been seen. Exploratory only; no pass mark or official result changes.**

15. Length checks are in `results/veena_check/`: vocal-trained stage-1 models tested on veena plavita and jaru clips, Kampita against Nokku with length matched, and stage-2 errors by gamaka length.
16. Stage 2 was retrained with equal total weight per gamaka type and a +-10 ms boundary exclusion for events under 0.4 s (`results/length_fix/`). The combined run scored event macro-F1 0.381 and frame macro-F1 0.490, against 0.395 and 0.497 originally, so it still fails. The official stage-2 result stays 0.395 / 0.497.
17. CNN output clean-up (`results/postproc/`), tuned for each held-out group on the held-out predictions of the other 10 groups. Probabilities came from an identical rerun (0 top-class differences in 147,135 frames). Method A (smooth, merge, extend, drop short): event 0.389, frame 0.470. Method B (minimum-duration Viterbi): event 0.417, frame 0.494. Both still fail; the official result is unchanged.
18. Two-stage test (`results/two_stage/`): a retrained gamaka-vs-none CNN finds regions, Step 1 style clean-up is applied, and Stage 1 DTW-3NN names them, with settings tuned on the other 10 groups. Version A: event 0.419, frame 0.493. Version B: event 0.437, frame 0.495. Both still fail; the official result is unchanged.
19. Detector additions (`results/step3/`): a PyTorch gamaka-vs-none network with swara distance, a BiLSTM and onset/offset heads, with Viterbi clean-up and Stage 1 DTW-3NN naming, was fixed as the primary run before training. Swara distance assumes Kalyani for every recording, including the VARNAM, whose raga the singer did not state. Primary run: event 0.462, frame 0.513. It still fails; the official result is unchanged.
20. Detection limit by gamaka length for that primary run, from saved held-out events only (`results/detection_limit/`): pooled X50 332 ms and X80 652 ms, but gamaka type and length are strongly confounded. The controlled insertion test has not been run because the Step 3 networks were not saved.
21. The 11 Step 3 primary networks were retrained in deterministic mode with their weights saved. The declared reproduction check passed in every fold (probability correlation 1.0, event F1 difference 0). The exploratory insertion test (`results/detection_limit/insertion/corrected/`) used 44 insertions per type and length, plus 396 flat-contour controls with 0 located. Use the `corrected` folder: the first per-type logistic fits in `insertion/` came from a solver that did not always converge.
22. Raga-free Stage 2 (`results/raga_free/`): inputs are cents minus a running median (window tuned from about 0.5, 0.75 and 1.0 s on the other 10 groups), velocity, voiced flag and CREPE confidence, with no Sa-relative sine/cosine and no swara distance; training uses random transposition within +-600 cents. Same BiLSTM and onset/offset network, Viterbi route and median-subtracted Stage 1 DTW naming. Result: event 0.387, frame 0.460, so it still fails and is below the frozen raga-aware run (0.462, 0.513). Shifting held-out tracks by +-300 cents changed nothing. A raga probe and leave-one-raga-out were not possible because TRACK 1-30 are all Kalyani and the VARNAM has no raga label; a swara probe and within-track track probe were used instead. The optional native-speed insertion test is in `results/raga_free/insertion_native/`.
23. Dashboard tool, not an evaluation (branch `classify-region`, `dashboard/classify_*.py`, `classify.js`): a "Classify a selection" section names a hand-selected stretch with the unchanged Stage 1 DTW-3NN (library = all 293 marks, own passage group left out for dataset recordings; on exact marks it reproduces the saved Stage 1 predictions 293/293). The reject distance in `dashboard/classify_config.json` is an untuned placeholder (24.49, the 99th percentile of held-out nearest distances among the marks). The optional subsequence search is rough: on marks padded by 0.5 s each side it found the mark (IoU >= 0.3) 50% of the time with the right type 41%; on exact marks 90% and 64%. Nothing in this section writes annotations.
