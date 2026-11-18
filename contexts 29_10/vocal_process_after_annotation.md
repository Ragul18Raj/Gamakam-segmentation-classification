# Vocal recordings: what to do once the annotated Excel and the audio are in hand

Written 17 Nov 2026. This covers the vocal data that actually arrived in `vocal rec\`, which differs from the 29 Oct plan
in `context_clcode.md`. Where the two disagree, follow this file. Section 6 of `context_clcode.md` is still useful
for background on each step.

Working rules (from `context_cowork.md`): be honest, explain simply, never invent results, and label anything simulated as simulated.

---

## 0. The process in one paragraph

First check that the Excel file and the audio agree. Then settle a few facts with the singer: which tracks belong together,
whether every gamaka was marked, and what Sa is. Next, get the pitch line of every track with CREPE. **Before looking at any
model result, write down the plan and the pass marks.** Then run two tests in order:
1. **Stage 1:** given the singer's start and end times, can the four gamaka types be told apart?
2. **Stage 2:** can a model find them on its own in the full track?

Score both so that a test track is never heard in training. Report every number, met or not.

---

## 1. What arrived (measured from the files on 17 Nov 2026)

| Item | Value |
|---|---|
| Audio | `vocal rec\VARNAM.mp3` plus `TRACK 1.mp3` to `TRACK 30.mp3`, 31 files |
| Format | MP3, 48 kHz, **stereo**, about 320 kbps |
| Length | 32.4 minutes in total. VARNAM is 262.8 s; the tracks are 28.6 to 106.6 s |
| Annotations | `vocal rec\gamaka_annotations_2026-11-17.xlsx`, exported from the Gamaka Annotator |
| Sheets | `All Songs` (all 293 events), plus one sheet per recording with the same rows |
| Columns | `Song Name`, `S.No`, `Timestamp`, `Gamaka Type`, `Start (sec)`, `End (sec)`, `Duration (sec)` |
| Events | **293**: Kampita 132, Nokku 82, Jaaru (Etra) 45, Jaaru (Irakka) 34 |
| Basic checks | All starts are before their ends. No events overlap. 13 pairs touch end to start. Every event lies inside its audio file. |

**What the four labels mean** (plain language; confirm the details with the singer):

| Label | What the pitch does | Median length (range) |
|---|---|---|
| Kampita | Shakes back and forth on a note | 0.88 s (0.18 to 2.72) |
| Nokku | A quick push from a higher note onto the note | 0.29 s (0.17 to 0.72) |
| Jaaru (Etra) | A slide **up** to a note | 0.22 s (0.14 to 0.67) |
| Jaaru (Irakka) | A slide **down** to a note | 0.35 s (0.14 to 0.90) |

**Events per recording.** VARNAM has 55: 25 Nokku, 17 Etra, 9 Kampita and 4 Irakka. The tracks have 3 to 21 each.
Note that VARNAM alone holds 30% of all Nokku and 38% of all Etra.

**How this differs from the 29 Oct plan:**

| 29 Oct plan | What arrived |
|---|---|
| 126 clips, 7 songs × 3 gamakas × 3 tempos × 2 octaves | 31 recordings. Their grouping is not documented yet (see section 2) |
| 3 gamakas, G1 to G3 | 4 labels. Two are directions of the same slide (jaru up and down) |
| WAV, mono, 16-bit | MP3, stereo |
| Sa recorded every session | **No Sa recording in the folder** |
| About 10% of clips re-annotated a week later | No second pass yet |
| One CSV per clip | One Excel workbook for all recordings |

**A pattern worth checking (my inference, not confirmed):** the tracks seem to come in sets of three, going from long to short.
For example, TRACK 1 to 3 last 74, 46 and 37 s, and TRACK 28 to 30 last 107, 58 and 38 s. That looks like **10 passages, each
sung at slow, medium and fast tempo**. If so, the number of marked gamakas falls a lot with tempo:
115 in the slow tracks, 74 in the medium ones and 49 in the fast ones. Either the singer leaves gamakas out when singing fast, or
fewer were marked in the fast tracks. That is the first question for the singer, because unmarked gamakas would count as "none".

---

## 2. Settle these with the singer before any modelling

Items marked **(blocking)** change how the data is used. Without them, no result can be trusted.

1. **(blocking) Which tracks belong together?** Is each set of three one passage at three tempos? Which song is each passage from?
   Is the VARNAM a separate piece? Record the answer in `metadata.csv` (section 4, step 1).
2. **(blocking) Was every gamaka of these four types marked in every track?** Anything left unmarked becomes "none" in training,
   so a missed Kampita teaches the model that a Kampita is "none". Ask especially about the fast tracks.
3. **(blocking) Does any stretch have a gamaka that is none of these four types?** That is fine (it counts as "none"),
   but note where it happens if common.
4. **Raga** of each track, and **Sa** (for example "Sa = C#"), or a short recorded Sa. If Sa can't be had, see step 4.
5. **Octave:** were any tracks sung an octave higher? The plan had 2 octaves, but the durations suggest only tempo was varied.
6. **Stereo:** is it one microphone copied to both sides, or are the sides different (for example voice and tanpura)?
7. **Re-annotation (for reliability):** can the singer mark about 10% of the material again, at least a week after the first pass,
   without seeing it? For example, 3 tracks (one slow, one medium, one fast, from different passages) plus about 60 s of the VARNAM.

---

## 3. Rules that hold throughout

- **Never touch the originals.** Copy them into `vocal_dataset\` and work on the copies.
- **No leakage.** Recordings of the same passage (the same song at different tempos) must **always stay together**, either all in
  training or all in testing. This is the lesson of the veena study, where clips from one take made scores look much better than
  they were (0.36 against an honest 0.14).
- **Write the plan first.** The pass marks and every setting go in `PLAN.md` before any model is run. Anything decided after
  seeing results is labelled "exploratory".
- **Tune only on the validation group, never on the test group.**
- **Report macro-F1, not accuracy.** "None" is most of every track, so plain accuracy looks good even when nothing is found.
- **Say where every number comes from:** recorded vocal (real), veena (measured), or synthetic (a check of the code only).

---

## 4. Steps

### Step 1. Intake (half a day)
Create this layout, copying the files in (not moving them):
```
D:\PROJECTS\Gamaka\vocal_dataset\
  audio\                VARNAM.mp3, TRACK 1.mp3 ... TRACK 30.mp3   (copies)
  annotations\
    pass1.xlsx          copy of gamaka_annotations_2026-11-17.xlsx
    pass1_events.csv    made in step 2: recording,start_s,end_s,label
    pass2\              the re-annotated recordings, when they come
  metadata.csv          one row per recording (below)
  crepe\                step 3 output
  PLAN.md               step 5
  results\              steps 6 to 9
```
**metadata.csv**, filled from the singer's answers in section 2:
```
recording,group,song_title,raga,tempo,octave,sa_note,sa_hz,complete_marking,notes
TRACK 1,P01,<title>,<raga>,slow,low,C#,,yes,
TRACK 2,P01,<title>,<raga>,medium,low,C#,,yes,
TRACK 3,P01,<title>,<raga>,fast,low,C#,,yes,
VARNAM,P11,<title>,<raga>,,low,C#,,yes,
```
The `group` column is what keeps related recordings together in every split. Recordings of the same passage share a group.

Also note the Excel's SHA-256 hash in `PLAN.md`, so that any later edit to the labels is visible.

### Step 2. Annotation checks (script: `vocal_dataset\check_annotations.py`, to write)
Read the `All Songs` sheet. Note that **pandas needs `openpyxl` installed** to read .xlsx (`pip install openpyxl`).
Then check and print:
- [ ] The `All Songs` rows equal the union of the per-recording sheets, with no row missing or doubled.
- [ ] Labels after trimming spaces and fixing case are only `Kampita`, `Nokku`, `Jaaru (Etra)` and `Jaaru (Irakka)`.
      An earlier export had both `Kampita` and `kampita`.
- [ ] `Start < End`, and `End` is within the audio length (from `soundfile.info`).
- [ ] No overlaps. Touching events are fine, and are listed.
- [ ] Every audio file has a sheet and every sheet has an audio file.
- [ ] The count per label, per recording, per group and per tempo, so the fall from slow to fast in section 1 can be seen.
- [ ] The length per label. Flag any event under 0.1 s or over 3 s for the singer to confirm.

Write `annotations\pass1_events.csv` with the columns `recording,start_s,end_s,label`. From here on, all code reads this CSV, not the Excel.

On 17 Nov all of these passed in a first check. The per-sheet event counts match `All Songs`, but the rows were not compared one by one.
No event is under 0.14 s or over 2.72 s.

### Step 3. Pitch with CREPE
- Reuse the settings of `run_crepe.py`: model `full`, a 10 ms step, `viterbi=False`. It reads 16-bit WAV only, so for the MP3s use
  `soundfile.read` (libsndfile 1.2.2 reads MP3 here, as tested on 17 Nov). Average the two channels to mono,
  unless the singer says they differ (section 2, item 6). Also compute the channel correlation once to confirm it.
- Save per recording: `crepe\<recording>.npy`, the [T, 360] activation, plus `crepe\<recording>.npz` with `time_s`, `f0_hz` and `confidence`.
  Use the same decoding as `sonify_crepe.load_activation`.
- **Spot-check 4 recordings:** one slow, one medium, one fast, and the VARNAM.
  - Plot the pitch line over the spectrogram with the singer's events shaded on it. Adapt `kalyani recorded\roundtrip.py`, which already
    plots the pitch line on a spectrogram.
  - Listen to `sonify_crepe.py` output next to the original. Wherever the tone goes wrong, CREPE went wrong.
  - Count the octave jumps and spikes, and note them before going further.
- MP3 at 320 kbps should not be a problem. In the compression test, a raga classifier was unaffected down to 16 kbps. But that was a
  different task, so still look at the spot checks.

### Step 4. Tonic (Sa)
- **If the singer gives Sa:** convert the note to Hz in the singer's octave, then check it against the audio: the most common
  steady pitch, folded into one octave, should sit within about 30 cents of Sa, or of Pa (a fifth above). If not, ask.
- **If not:** estimate Sa from the audio (a histogram of steady pitch, or Essentia's `TonicIndianArtMusic`), and label it as
  **inferred**, as was done for the veena (E4, see `tonic_estimates_NOTE.md`).
- **The detector does not depend on Sa** if the two note-position features (sin and cos of cents) are dropped. That is what the
  plavita detector did. Keep them only if Sa is confirmed for every recording, and say which version was used.

### Step 5. Write `vocal_dataset\PLAN.md` before running any model
It must fix the following:
1. **The question:** stage 1 (tell the four types apart, given the boundaries) and stage 2 (find and name them in full tracks).
2. **Classes.** Main: the 4 labels as annotated, plus "none". Secondary: Etra and Irakka merged into "Jaaru" (3 classes plus none),
   because they differ only in direction, which is easy to read from pitch.
3. **The split:** leave-one-group-out. With 10 passages and the VARNAM, that is 11 folds. In each fold, the next group is the
   validation group, as in `run_eval.py`.
4. **Settings, and every change from the simulation design, with the reason.** See step 7.
5. **The scoring rule for events.** Recommended main rule: same label, overlap (IoU) of at least 0.3, matched one to one
   (`kampita_from_annotations\eval_segments.py`). Secondary: onset within ±200 ms (`pipeline.match_events`), so the numbers can be
   compared with the simulation. With events of 0.2 to 0.3 s, ±200 ms alone is too generous.
6. **Pass marks, agreed with the guides.** Suggested:
   - Stage 1: macro-F1 of at least 0.50, with a permutation p below 0.05. Majority-class macro-F1 is about 0.16, since Kampita is
     45% of events.
   - Stage 2: event F1 (macro over the 4 classes) of at least 0.60, and frame macro-F1 of at least 0.60, as in the 29 Oct targets.
   - The CNN must be at least as good as the rule-based baseline, with a paired Wilcoxon test over folds.
   - Counts: report the count error per type per recording and in total. Set the target with the guides, because the old
     "≤ 0.5 per 25 s clip" does not carry over to 30 to 260 s tracks.
7. **What counts as failure, and what would follow from it.**

### Step 6. Stage 1: tell the types apart, given the singer's boundaries
Cut out every annotated event using its start and end, and classify it with leave-one-group-out:
- **Features plus logistic regression and 3-NN:** duration, pitch range (p95 − p5), net change (end minus start), turning points,
  oscillation rate, and mean |velocity|. The same features as `veena_stage1\stage1.py` and the `run_eval.py` stage 1.
- **Pure-shape DTW plus 3-NN** (`veena_stage1\methods_explore\classify.py`). It scored 0.63 on veena, but that was exploratory.
  **This is the confirmation on new data that the veena result asked for.**
- **Permutation test:** shuffle the labels within the training folds 1000 times to get p.
- Report the confusion matrix, and recall per class.
- **If stage 1 fails, stop and look before stage 2.** The detector must solve this problem *and* find the boundaries.
  Likely confusions: Nokku against Etra (both short, both approach a note), and short Kampitas against Nokku.

### Step 7. Stage 2: find and name gamakas in full tracks
**Model:** the frame-wise dilated 1-D CNN in `sim_code_extracted\gamaka_sim\pipeline.py` (a 3 s window, 0.1 s hop, averaged).
Changes it needs, all of which go in `PLAN.md`:

| Change | Why |
|---|---|
| Classes 4 → 5 (none + 4 labels). `NCLS = 5`; `match_events` and `f1` loop over `(1, 2, 3, 4)` | The code hard-codes 3 gamaka classes |
| Boundary mask ±50 ms → **±20 ms** | Etra has a median of 0.22 s, and the shortest event is 0.14 s. With ±50 ms, a 0.14 s event loses all of its frames |
| Majority filter 11 → about 5 frames; drop runs < 0.1 s (keep) | An 11-frame filter can erase 0.14 s events. Tune on the validation group only |
| Sin and cos features: dropped unless Sa is confirmed | See step 4 |
| A loader `load_real_dataset()` instead of `build_dataset()` | See the sketch in `context_clcode.md`, step 4; read `pass1_events.csv`, `metadata.csv`, `crepe\*.npz` |
| `song` = `group` in the clip dicts | So the existing fold code keeps passages together |
| Long recordings (VARNAM 263 s) | No change needed; the sliding window already handles any length |

Run each fold as its own single-threaded process (`vocal_plan\eval\launch_folds.py`, with `OMP_NUM_THREADS=1`).
Run `score.py` only after all fold files exist.

**Baselines,** run on the same folds:
- all-none;
- rule-based movement candidates plus 3-NN (already in `run_eval.py`);
- **Kampita only:** the random-forest detector in `kampita_from_annotations\train_kampita.py`, retrained on the vocal labels.

**Then report:**
- event F1 per class and per fold (mean ± std), with a 95% bootstrap CI over recordings;
- precision and recall, and the onset and offset error;
- the count error;
- results by tempo;
- frame confusion matrices.

`vocal_plan\eval\deep_eval.py` produces most of this. It was written for two simulated conditions, so give it the same
results folder twice.

### Step 8. How reliable are the labels (when the second pass arrives)
On the re-annotated material, compare pass 1 with pass 2:
- **frame-level Cohen's κ** over the 5 classes (`pipeline.frame_labels` for both passes, then `sklearn.metrics.cohen_kappa_score`);
- **event F1** between the passes, with the same scoring rule as step 7.

This is the **ceiling**: the model can't be expected to agree with the singer more than the singer agrees with themself.
If κ is below about 0.6, talk to the singer before trusting the model scores.

### Step 9. Error analysis
- For 2 or 3 test groups, **listen to every miss and every false alarm.** Use the sonified pitch and the original, and the
  plot with the events shaded.
- Sort each one into a bucket:
  - look-alike movement that the singer did not mark;
  - boundary disagreement;
  - CREPE glitch or octave jump;
  - fast tempo;
  - a gamaka the singer may have missed.
- Ask the singer about the last bucket. **Do not change pass-1 labels to improve the score.** If the singer corrects
  labels, keep both versions and report both.

### Step 10. Write it up
- Add the vocal results to the report and the slides as **"recorded, 1 singer, 31 recordings, 32 min"**, next to the veena
  (measured) results and the simulation (code check).
- State the pass marks, and whether each was met.
- State the limits:
  - one singer;
  - grouping and completeness as confirmed by the singer;
  - no κ, if the second pass never came;
  - a tonic that was inferred, if it was.

---

## 5. Code to reuse

| Need | File |
|---|---|
| CREPE settings | `run_crepe.py`, `veena_stage1\plavita_kalyani\run_crepe_kalyani.py` |
| Activation → Hz and confidence | `sonify_crepe.load_activation`, `anchor_feasibility_check_v3.crepe_activation_to_pitch` |
| Pitch plot on spectrogram, sonification | `kalyani recorded\roundtrip.py`, `sonify_crepe.py` |
| Pitch cleaning (octave jumps, spikes) | `veena_stage1\methods_explore\clean.py` |
| Stage-1 features and classifiers | `veena_stage1\stage1.py`, `sim_code_extracted\gamaka_sim\run_eval.py` |
| Pure-shape DTW + 3-NN | `veena_stage1\methods_explore\classify.py` |
| Features, CNN, sliding window, post-processing | `sim_code_extracted\gamaka_sim\pipeline.py` |
| Fold runner and scoring | `run_eval.py`, `score.py`, `vocal_plan\eval\launch_folds.py`, `vocal_plan\eval\deep_eval.py` |
| Segment scoring (IoU) from Excel or CSV | `kampita_from_annotations\eval_segments.py` |
| Kampita-only baseline | `kampita_from_annotations\train_kampita.py` |
| Explaining what the CNN looks at | `veena_stage1\plavita_kalyani\trigger_map.py` |

---

## 6. Pitfalls specific to this data

- **Slow, medium and fast versions of a passage in different folds** would leak. Always split by `group`.
- **VARNAM is large.** It holds many of the Nokku and Etra events. As a test fold it makes the scores for those classes lean on
  one piece. Report results with and without it.
- **Short events.** Etra and Nokku last only 14 to 30 frames. The mask and filter settings from the simulation would erase them
  (step 7).
- **Fewer marks at fast tempo.** Until the singer confirms the marking is complete, results on fast tracks may count real gamakas
  as false alarms.
- **Kampita is 45% of events.** Use macro scores, not totals pooled over classes.
- **Touching events** (13 pairs, for example Nokku straight into Jaaru) need correct boundaries. A small timing shift moves frames
  between two labels.
- **Stereo MP3.** Check the channels before averaging them.
- **The earlier tests used veena or no labels.** The plavita detector (Kalyani) and the veena DTW score were exploratory;
  this data is the first chance to test a method on labelled singing.

---

## 7. Timeline to 30 November 2026 (13 days from 17 Nov)

| Days | Work |
|---|---|
| 18 to 19 Nov | Questions to the singer (section 2); intake; annotation checks; CREPE on all 31 recordings; spot checks |
| 20 Nov | Tonic; write and freeze `PLAN.md`; agree the pass marks with the guides |
| 21 to 22 Nov | Stage 1 (features and pure-shape DTW) |
| 23 to 26 Nov | Stage 2 CNN and baselines, 11 folds; κ if the second pass is in |
| 27 to 30 Nov | Error analysis, report, slides |

If time runs short, **stage 1 plus the Kampita-only detector** is a complete, honest result on its own. The full four-class
detector can then be reported as "planned".
