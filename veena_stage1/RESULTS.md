# Veena stage-1 check: result (30 Oct 2026)

Plan fixed before running: `PREREGISTRATION.md`. Code: `extract.py` (runs on the CREPE activations), `stage1.py`.

**Data.** 112 clips in 12 takes: jaru 24 clips / 4 takes, kampita 26 / 3, ravai 35 / 3, pratyagata 27 / 2.
4 `_continuous` pratyagata files excluded. Sphuritam excluded (1 take).
**Split.** Leave-one-take-out. **Features.** 9 movement features, relative to each clip's own pitch.

## Pre-registered run: FAIL

| Model | Macro-F1 |
|---|---|
| Logistic regression, movement features (main) | **0.21** |
| 3-NN, movement features | 0.30 |
| Leakage probe: median pitch only | 0.09 |
| Majority class | 0.12 |
| Permutation null for the main model (1000 shuffles of class across takes) | mean 0.15, 95th percentile 0.30, **p = 0.23** |

Pass needed: macro-F1 >= 0.50, above the probe, and p < 0.05. Only the probe condition is met.

## Exploratory, decided after seeing the data: octave jumps corrected

39 of 112 clips had a pitch range above 1200 cents (CREPE octave errors); kampita's median range was 1206 cents.
After folding octave jumps back to a running median (kampita median range 478 cents):
logistic regression 0.19, 3-NN 0.33, probe 0.07, permutation p = 0.25. **Still fails.**

## What the model does pick up

Errors fall in two blocks: kampita is mostly called jaru (18/26) and pratyagata mostly ravai (23/27).
At the level of two groups, {jaru, kampita} vs {ravai, pratyagata}, logistic regression is right on 99/112 clips.
The two groups also differ in clip length (median voiced span 1.7–2.2 s vs 0.8 s), so this split may reflect how the
clips were cut rather than the gamaka itself. Not tested further.

## Conclusion

On takes it has not seen, pitch movement does not separate these four gamakas in this veena data. This agrees with
the earlier 15-class result (3-NN 0.14 vs 0.16 baseline) using a different feature pipeline. With 2–4 takes per
gamaka, the data, not the model, is the limit; a frame-wise CNN trained on the same takes is not expected to do better.
