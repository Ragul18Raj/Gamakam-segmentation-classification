# Veena methods test: plan written before any scores (Nov 2026)

**Status: exploratory.** This was designed after the pre-registered stage-1 test failed and after looking at the
pitch plots of these same clips. It cannot overturn that FAIL. A good score here is a reason to test on new data, not a result.

Question: does cleaning the pitch, cropping to the movement, and comparing whole curve shapes (DTW) separate
jaru, kampita, ravai and pratyagata better than the stage-1 summary features, on takes the classifier has not seen?

Kept the same as stage-1: the same 112 clips, 12 takes, 4 classes; leave-one-take-out; macro-F1 on pooled held-out predictions;
1000 permutations of class across takes (seed 20261030); pass thresholds macro-F1 >= 0.50, above the median-pitch probe, p < 0.05.

## Pitch versions (no class labels used in any of them)
- **raw**: stage-1 curve, conf >= 0.5, gaps interpolated.
- **clean**: split into runs at jumps > 150 cents per frame; seed on the run with the most energy; walk outward keeping runs
  within 400 cents of the chain (at the same octave, or folded by whole octaves if shorter than 150 ms; a longer
  run an octave away is treated as a drone string and dropped); trim ends below -30 dB of the seed's energy.
- **crop**: clean, cut to first and last moving frame (0.2 s local range >= 30 cents) with 0.1 s margin.

Cleaning settings may be changed only by looking at plots, before any classifier has been run. Changes are listed in RESULTS.md.

## Methods (each on raw, clean, crop unless noted)
- Stage-1 9 features: logistic regression, 3-NN.
- DTW on the median-centred curve (no note information): k = 3 (main), k = 1 on crop only.
- **crop, length removed**: DTW on curves resampled to 100 points; features without `dur` and `turns`.
- Rules (crop only): fixed by hand from gamaka descriptions and the plots. They have seen all the data, so they get no p-value
  and count only as a description.

## Checks
- Family-wise p: each permutation takes the maximum macro-F1 over all trained configurations, so picking the best one is paid for.
- Length shortcut: logistic regression on clip duration alone, raw and cropped, compared with the length-removed runs.
- Scramble: the same methods on crop with frames shuffled inside each clip. If the score holds up, the method is not using the movement.
- Probe: median absolute pitch, logistic regression.
- Reading a pass: the best length-removed config must itself clear the thresholds. Otherwise the result is reported as not separated.
