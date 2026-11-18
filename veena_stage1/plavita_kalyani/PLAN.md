# Plavita-only detector, tried on the Kalyani recordings: plan

Written before any model was trained or any detection was seen. Exploratory, following the user's request of 4 Nov 2026.

## Question
If a frame-labelling detector is trained only on the veena plavita clips, what does it mark in the three recordings in
`Kalyani recorded/`, and is what it marks specific to kampita?

## Data
- **Training:** the 15 plavita clips, minus `Kampita(plavitam) todi#03` because it has no played note, so 14 clips and 4 branches
  (#02, bhairiga #03, todi #01, todi #03), about 39 s in total.
- **Test:** `kalyani #02`, `kalyani2 #03`, `kalyani3 #02` (107 s). **They have no annotations, so no accuracy can be computed.**

## Labels, "as if annotated"
No person marked these clips. Frames inside the movement crop from `methods_explore/clean.py` are labelled kampita (1), and all other
frames in the clip are none (0). That gives 2408 voiced kampita frames, 825 voiced none frames and 660 unvoiced none frames.
Padding outside the clip gets no weight in the loss. Frames within 50 ms of a boundary get no weight, as in the project design.

## Method: the project's sliding-window design
- CREPE full, 10 ms, confidence at least 0.5, gaps under 100 ms filled (`gamaka_sim/pipeline.py`).
- The NumPy dilated CNN: 5 layers, kernel 9, 48 channels, dilations 1 to 16, 3 s window, 0.1 s hop, softmax averaged over windows.
- Two classes instead of four.

**Changes from the project design, all fixed now:**
1. Sin and cos of pitch are dropped, leaving contour, velocity, voiced and confidence. They would tie the model to the few notes in these
   clips, and the Kalyani Sa is not known.
2. The clips are shorter than most windows, so each clip is placed at a random position in the window.
3. Fixed 15 epochs instead of early stopping, because there is too little data to set aside. Time-stretch 0.8, 0.9, 1.1, 1.25 as designed.
4. Tonic-shift augmentation is dropped, since no feature depends on the tonic now.
5. The final model is the average of 3 seeds (0, 1, 2). Disagreement between seeds is reported.

## Checks that do not depend on Kalyani labels
1. **Held-out branch:** train on 3 branches, test on the 4th, and compare with the crop labels. This only shows that the model learned the
   crop rule. It is not evidence about real kampita, because that rule made the labels.
2. **Steady-note test:** a synthetic flat voiced curve with small jitter (SIMULATED input). Flagging it means "voiced" is being detected.
3. **Other veena gamakas:** the share of voiced frames flagged in the jaru, ravai and pratyagata clips, none used in training, against the
   plavita held-out share. Similar shares mean the model is not specific to kampita. Note that these clips can contain short kampita-like
   wobbles, and leena is reported separately since it is also a kampita.
4. **Movement rule:** on Kalyani, compare flagged frames with a plain rule (at least 30 cents of pitch range in a 0.2 s window, the same test
   the crop uses). High overlap means the CNN adds little beyond that rule.

## How the result will be read
- **Not usable as a kampita detector** if the steady note is flagged (above 50% of frames), or other-gamaka flagging is at least 80% of the
  held-out plavita share, or Kalyani overlap with the movement rule is above 0.8 with other-gamaka flagging also high.
- If it passes those, it is still only "a candidate". Accuracy needs labels on Kalyani, so the detections are listed with timestamps for
  checking by ear.
- Nothing here can change the stage-1 FAIL.

## Output
`RESULTS.md`, `detections.csv`, `timeline_<recording>.png`, `checks.json`.

---

# Follow-up plan: what triggered the model on `kalyani2 #03` (5 Nov 2026)

Written before the map was computed. Exploratory: it describes this model on one recording, and says nothing about whether the marked parts are really kampita.

**Model.** Retrain with the same code and seeds, then check the scores against the saved `scores_kalyani2_03.npz`. Save the model this time.

**Test.** Slide a 0.1 s piece along the sung phrases in 0.05 s steps. At each position, change only that piece, rerun the affected windows, and record the drop in the raw output (kampita logit minus none logit, averaged over windows) on sung frames within reach. A positive drop means the piece pushed towards kampita. Two kinds of change:
1. **Blank:** the piece becomes unvoiced, as agreed with the user. This is the main map.
2. **Flatten:** a straight line across the piece, so the sound stays and only the movement goes. This is a check, and it can only show moving parts.

**Reliability checks.** Agreement between the 3 seeds, and agreement with an untrained model of the same shape. If the untrained model agrees as well as the seeds do, the map shows the input and not anything learned.

**Reading rule, for the blank map.**
- **Stable** if the mean correlation between seeds is at least 0.7. Otherwise the map should not be trusted.
- **Focused on tall peaks** (at least 200 cents) if frames within 0.1 s of a tall peak hold at least twice the positive importance their share of sung time would predict.
- **Movement in general** if not focused, and importance correlates with local pitch range at 0.5 or more.
- **Reach:** how far a piece's effect spreads, from the average drop against distance. A wide spread means the model looks at long stretches and not single peaks.
