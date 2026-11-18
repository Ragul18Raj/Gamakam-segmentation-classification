# Veena stage-1: local reproduction (30 Oct 2026)

**Result: reproduced exactly.** Both `stage1_results.json` files written here are identical to the originals
(every key, including confusion matrices, per-take accuracy, the 1000-permutation null and class means).

## Environment

| | |
|---|---|
| Machine | Windows 11 (10.0.26200), local folder `D:\PROJECTS\Gamaka` |
| Python | 3.12.9 |
| numpy | 2.5.3 |
| pandas | 3.1.0 (installed for this run) |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 (shapes only) |

## Steps

1. Originals backed up to `original/` (`clip_feats.csv`, `clip_feats_octfix.csv`, `stage1_results.json`)
   and `original/octfix/stage1_results.json`.
2. In `veena_stage1`: `python extract.py`, `python extract.py --octfix`. Both: 112 clips, 4 `_continuous` files skipped,
   jaru 24 / kampita 26 / ravai 35 / pratyagata 27, 12 takes.
3. `python stage1.py clip_feats.csv local_run` and `python stage1.py clip_feats_octfix.csv local_run\octfix`
   (console output in `stage1_stdout.txt` and `octfix/stage1_stdout.txt`).

## Feature CSVs vs originals

Same 112 rows in the same order, same file/class/take, same integer columns. 13 cells (main) and 12 cells (octfix)
differ in the last floating-point digit: max relative difference 1.5e-15 (largest absolute 1.8e-12 cents). This is
library rounding and has no effect on any result below.

## Numbers side by side

Macro-F1 unless stated. "RESULTS.md" is the remote run as written up; "original JSON" is the backed-up file.

### Pre-registered run (`clip_feats.csv`)

| | RESULTS.md | original JSON | local | match |
|---|---|---|---|---|
| Logistic regression (main) | 0.21 | 0.2093 | 0.2093 | yes |
| 3-NN | 0.30 | 0.2967 | 0.2967 | yes |
| Probe: median pitch | 0.09 | 0.0938 | 0.0938 | yes |
| Majority class | 0.12 | 0.1190 | 0.1190 | yes |
| Permutation null mean / 95th pct | 0.15 / 0.30 | 0.1522 / 0.3025 | 0.1522 / 0.3025 | yes |
| Permutation p | 0.23 | 0.2338 | 0.2338 | yes |
| PASS | false | false | false | yes |

### Octave-corrected run (`clip_feats_octfix.csv`, exploratory)

| | RESULTS.md | original JSON | local | match |
|---|---|---|---|---|
| Logistic regression | 0.19 | 0.1926 | 0.1926 | yes |
| 3-NN | 0.33 | 0.3325 | 0.3325 | yes |
| Probe: median pitch | 0.07 | 0.0667 | 0.0667 | yes |
| Permutation p | 0.25 | 0.2478 | 0.2478 | yes |
| PASS | (fails) | false | false | yes |

## One note on RESULTS.md

The "What the model does pick up" paragraph (kampita called jaru 18/26, pratyagata called ravai 23/27, two-group
accuracy 99/112) matches the **octave-corrected** run's confusion matrix. The pre-registered run gives
kampita→jaru 22/26, pratyagata→ravai 20/27, two-group 96/112. The conclusion is the same; only the source of those
three numbers is not stated. RESULTS.md was not edited.
