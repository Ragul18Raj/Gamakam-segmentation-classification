# Plavita-only detector on the Kalyani recordings: result (4 Nov 2026)

**Exploratory. The Kalyani recordings have no labels, so no accuracy can be given.** Plan, written before training: `PLAN.md`.
Code: `run_crepe_kalyani.py`, `detector.py`, `phrases.py`. Numbers: `checks.json`, `phrase_stats.json`, `phrases.csv`.

## Verdict
**By the rule fixed in the plan, this is not usable as a kampita detector.** It works as a "is the pitch moving?" detector.

| Plan's test | Result | Triggered? |
|---|---|---|
| Steady note flagged above 50% (simulated input, 3 jitter levels) | 0% | No |
| Other veena gamakas flagged at 80% or more of the held-out plavita share (0.81) | Mean 0.67, ratio 0.82 (jaru 0.86, ravai 0.88, pratyagata 0.74) | **Yes** |
| Kalyani overlap with the movement rule above 0.8, and other gamakas also high | 0.92, 0.93, 0.95 | **Yes** |

The two flagging shares in the second row come from slightly different models: held-out branch models for plavita (1 seed), and the final model on all 14 clips
for the others (3 seeds).

## What it marked on Kalyani

| Sung frames | Seconds | Share marked |
|---|---|---|
| Pitch moving (30 cents or more within 0.2 s) | 70.7 | 94% |
| Pitch steady | 9.9 | 9% |
| All sung frames | 80.7 | 84% |

- 24 of 25 sung phrases have at least half their frames marked, and 18 have at least 75% marked.
- Per recording, the share of sung frames marked: `kalyani #02` 81%, `kalyani2 #03` 82%, `kalyani3 #02` 92%.
- What it leaves unmarked is mostly the long held notes, for example `kalyani2 #03` 28.8 to 31.1 s.
- Seeds agree on 91% to 96% of voiced frames.

## Other findings
- **Held-out plavita branches:** F1 0.94 against the crop labels. That is only a little above the 0.86 from marking everything, because 75% of the frames are labelled kampita. It also only shows that
  the model copied the automatic crop rule. Nobody annotated these clips.
- **Kampita leena:** 0% flagged, in all 11 clips. So it misses a kampita variant with little pitch movement while marking other gamakas that do move.
- **Silence:** the model marked 57% to 92% of silent frames, because padding had no weight in training and so it never learned about long silence. Marks outside
  sung phrases were removed afterwards in `phrases.py`. This does not change the model's scores or the sung-frame shares above.

## Why this happens
With only plavita clips, the sole contrast available to learn is a moving plavita against the short still parts and silence around it. Nothing shows it a moving pitch that is *not* kampita,
so it cannot learn that difference. Without labels, this test also can't tell how much of the singing really is kampita: many marks may be right, but they can't be separated from the wrong ones.

## Limits
- No labels on Kalyani, so "marked" does not mean "correct".
- Training labels came from an automatic rule, not from a person.
- 14 clips, 4 branches, one player. Training loss reached about 0.01, so the model is close to memorising them.
- The recordings look like singing from the spectrograms (`look_*.png`), but this was not confirmed, and the model was trained on veena.
- The changes from the project design are listed in `PLAN.md`.

## Changes after the plan
`phrases.py` was added after seeing the output, because the first event list ran across breath gaps and included silence. It replaces `detections.csv` with `phrases.csv`, and it redraws
`timeline_*.png` with marks only on sung phrases. A phrase is at least 0.3 s of voiced frames above about 82 Hz.

## Follow-up: the high peaks (added after the user said most high peaks in Kalyani are kampita)
This uses an automatic definition of a peak, not expert marks. "Tall" means a prominence of at least 200 cents, a threshold chosen after seeing the first numbers. Code: `compare_peaks.py`. Numbers: `peak_compare.json`. Picture: `peak_compare.png`.

**The model did mark them:** 96% of the 52 tall peaks fall in marked frames. It does not single them out: frames around all peaks of 100 cents or more average a score of 0.84 (86% marked), and other moving frames average 0.89 (93% marked).

| Tall peaks, median (middle 50%) | Veena plavita (14 clips) | Kalyani (3 recordings) |
|---|---|---|
| Height, cents | 297 (280-309) | 376 (261-655) |
| Width at half height, ms | 254 (216-286) | 231 (134-386) |
| Time to next peak under 1 s, ms | 460 (395-498) | 520 (392-708) |
| Tall peaks per second | 1.28 | 0.64 |

Typical size and spacing are close. The Kalyani values vary much more, and the context differs: the plavita has flat-topped peaks repeating from a fixed base, with the pitch nearly still in 26% of frames against 16%,
while the singing has smooth peaks on a melody line that moves (median slow drift 553 cents against 280).

## Follow-up: what triggered the model (5 Nov 2026)
Plan: end of `PLAN.md`. Code: `trigger_map.py`, `trigger_plot.py`, `trigger_direction.py`, `trigger_iron.py`. Picture: `trigger_map.png`. Numbers: `trigger_map.json`, `trigger_direction.json`, `trigger_iron.json`. This describes the model only, and does not show that any part is really kampita. The retrained model matched the saved scores exactly (largest difference 0.000000).

**Planned reading, blank map on `kalyani2 #03`.**
- Stable: yes. Seed correlation 0.90. An untrained model correlates only 0.25, so the map reflects training.
- Focused on tall peaks: no. Frames near tall peaks hold 0.83 times the positive importance their time would predict, under the 2 times needed.
- Movement in general: no. Correlation with local pitch range is 0.07, under the 0.5 needed.
- Reach: half of the peak effect lies from 0.3 s before the piece to 0.7 s after it, and the effect ends at about 1.2 s either side.

**Unplanned finding, tested on recordings not used to spot it: the strongest push comes from rising pitch.** Mean push by piece, at a slope of 300 cents per second or more:

| Recording | Rising | Falling | Flat |
|---|---|---|---|
| `kalyani2 #03` (pattern spotted here) | 1.99 | 0.62 | -0.75 |
| `kalyani3 #02` | 1.85 | 0.38 | -0.08 |
| `kalyani #02` | 1.90 | 0.74 | -0.68 |

Falling pieces mostly still push towards kampita (78% to 85% are positive), so falls are weak rather than opposite. Only some sharp falls are blue. Plavita training clips have about as many rising as falling pieces (146 and 153), so the count does not explain this. Their peaks do rise faster than they fall (median 1505 against 934 cents per second), which may be the cause, but that was not tested.

**The flatten check was weak.** It gave little (rising 0.11, falling -0.12, flat 0.18; correlation with the blank map 0.21). This was a design error: a straight line across a 0.1 s rise is nearly the same curve. So it neither confirms nor refutes the blank map. Blanking also inserts a short gap, so part of its effect may come from the gap itself.

**Replacement test: ironing peaks out of the pitch line (added after seeing the above).** Share of sung frames marked:

| Pitch line given to the model | `kalyani2 #03` | `kalyani3 #02` | `kalyani #02` |
|---|---|---|---|
| Original | 82% | 92% | 81% |
| Tall peaks ironed (200 cents or more) | 59% | 71% | 51% |
| All peaks ironed (100 cents or more) | 48% | 48% | 28% |
| Small wobbles ironed as well (40 cents or more) | 44% | 35% | 22% |
| Slow melody line only (1 s median) | 30% | 50% | 19% |
| Each phrase a single flat note | 0% | 0% | 0% |

Peaks are a large part of what makes it fire, but not all: it still marks 19% to 50% with only the slow melody line left. There is no control here that irons non-peak parts of the same length.

The tall-peak markers in the picture are automatic. Two of them look doubtful: the end of the phrase near 12.3 s, and the one near 14.05 s, just before a tracker spike.
