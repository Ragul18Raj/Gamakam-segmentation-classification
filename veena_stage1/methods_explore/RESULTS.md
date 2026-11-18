# Veena methods test: result (Nov 2026)

**Exploratory.** Designed after the pre-registered stage-1 test failed and after seeing plots of these same clips.
The pre-registered result stays **FAIL**. Nothing here is a result on new data, on singing, or on finding gamakas inside a longer clip.
Plan: `PLAN.md`. Code: `clean.py`, `classify.py`, `extent_check.py`. Full numbers: `results.json`, `extent_results.json`, `predictions.csv`.

**Data.** The same 112 clips, 12 takes, 4 classes as stage-1, one player, leave-one-take-out, macro-F1. Majority baseline 0.12.
**Sanity check.** Raw pitch with the stage-1 features reproduces stage-1: 0.209 and 0.297 (published 0.21 and 0.30).

## Main table

| Pitch version | Features + logistic regression | Features + 3-NN | DTW + 3-NN |
|---|---|---|---|
| Raw | 0.21 | 0.30 | 0.45 |
| Clean | 0.39 | 0.49 | 0.53 |
| Clean + crop | 0.34 | 0.49 | **0.64** |
| Clean + crop, length removed | 0.21 | 0.36 | **0.63** |

Family-wise permutation test over the 14 trained configurations (1000 shuffles of class across takes): null 95th percentile 0.42.
DTW on crop: p_family 0.001, also for length removed. Other: crop DTW 1-NN 0.55, length-removed DTW 1-NN 0.60. Probe (median pitch) 0.07.
Length alone: 0.15 (raw duration), 0.06 (crop duration). Hand-written rules on crop: 0.56, no p-value because they were written after seeing the data.

## What is the score using?

Scrambling frames inside each clip left a lot of the score (0.56 for crop DTW, 0.51 for features + LR, against 0.34 unscrambled), so a follow-up,
decided after seeing that, separated the possible sources.

| Input to DTW + 3-NN (resampled to 100 points, band 10) | Macro-F1 | Note |
|---|---|---|
| Length removed | 0.63 | p 0.001 |
| The same, frames scrambled (20 scrambles) | 0.47 | range 0.43 to 0.49: how wide the pitch goes carries a real part |
| **Length and extent removed: pure shape** | **0.63** | p 0.001, 73 of 112 right, 95% CI over takes 0.46 to 0.78 |
| The same, frames scrambled (20 scrambles) | 0.27 | range 0.22 to 0.33: this version does depend on the order of the movement |

Not using DTW: extent alone 0.20 (LR) and 0.41 (3-NN); pitch-value percentiles with no order, 3-NN 0.55.
p = 0.001 is the smallest this test can give; with all 19 configurations corrected it is at most 0.019.

**Reading.** Shape in time does carry information across takes, separately from length, extent and note. The summary features did not capture it: without
length they stay at 0.21 and 0.36, and with scrambled frames they score higher than with real ones, so they should not be used.

## Where it fails (pure shape, recall)

Jaru 0.62, kampita 0.35, ravai 0.71, pratyagata 0.89. Confusion, true class in rows:

| | jaru | kampita | ravai | pratyagata |
|---|---|---|---|---|
| jaru | 15 | 3 | 5 | 1 |
| kampita | 10 | 9 | 1 | 6 |
| ravai | 3 | 2 | 25 | 5 |
| pratyagata | 2 | 0 | 1 | 24 |

Per take: jaru_asc 1.00, jaru_asc OG_1 1.00, Shankarabharanam 0.50, **Melgana 0.00**; plavita 0.71, Todi 0.50, **leena 0.00**;
ravai #02 0.73, Kalapriya 0.83, Mayamalavagowla 0.58; Hindolam 0.94, Amritavarshini 0.82.
Melgana and leena fail because no other take looks like them once they are held out: a template method cannot name a variant it has no example of.
Leena also has almost no movement in pitch (median extent 78 cents).

## Limits

- 12 takes, one player, isolated clips, so the CI is wide and says nothing about other players or gamakas inside phrases.
- The method was chosen after seeing the data, and the pure-shape run was added after seeing results. Only new data can confirm it.
- Pratyagata and ravai clips are very alike within a class in timing (the event falls near 0.4 s). Part of the score may reflect how consistently
  they were played or cut, which may not hold in real music.
- This names a clip known to hold one gamaka. It does not find where gamakas are.
- Problem clips were kept in, to keep the set unchanged: `Kampita(plavitam) todi#03` has no played note, and four pratyagata clips clip.

## Cleaning changes, made from plots before any classifier was run

1. Walking outward from the loudest run let the chain drift onto a drone (jaru_asc _1). Replaced by the chain of runs with the most total energy.
2. Added removal of short flat islands where the tracker flips to a drone and back.
3. Limited that removal to downward islands, because the first version deleted real upward ravai flicks (ravai #02.1, #02.3).
4. Crop now takes the largest movement (stretches under 0.4 s apart are merged), because small late wobbles were stretching crops.
   Known leftover: leena #01.2 and #01.7 crop to an early bump and cut the slow rise short.

These differ from the walk-outward wording in `PLAN.md`, which is left as written.
