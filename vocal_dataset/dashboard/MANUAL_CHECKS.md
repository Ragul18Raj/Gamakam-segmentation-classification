# Manual checks: "Classify a selection"

Start the service with `dashboard/start_backend.sh` (from `vocal_dataset`), open `dashboard/index.html` with Live Server, and click **Classify a selection** in the header. Type the times into Start and End (or drag on the timeline), then press **Classify region**. Expected values were recorded on 2026-11-19 with the placeholder reject distance 24.49. Distances may change slightly if the Stage 1 code, the marks or the pitch files change.

## Three dataset tracks (exact singer's marks, search off, blind mode off)

| # | Recording | Start | End | Singer's mark | Expected answer | Agree | Distance | Label | Nearest examples |
|---|---|---|---|---|---|---|---|---|---|
| 1 | TRACK 4 | 56.67 | 58.24 | Kampita | Kampita | 3/3 | about 3.0 | confident | TRACK 10 57.35, TRACK 7 72.16, TRACK 25 32.73 |
| 2 | TRACK 13 | 52.46 | 52.76 | Nokku | Nokku | 3/3 | about 3.0 | confident | VARNAM 143.04, TRACK 1 54.60, TRACK 10 60.82 |
| 3 | TRACK 25 | 44.41 | 44.90 | Jaaru (Irakka) | Irakka | 3/3 | about 4.0 | confident | TRACK 10 14.44, TRACK 10 13.54, TRACK 11 14.28 |

The boxes round to 10 ms, so the distance can differ a little from the exact-mark value. For each row, check the following:
- No nearest example comes from the same passage group. TRACK 4 is in P02 with TRACK 4 to 6, TRACK 13 is in P05 with TRACK 13 to 15, and TRACK 25 is in P09 with TRACK 25 to 27. The panel says which group is left out.
- Each example's Play example button plays a short clip of the right gamaka type.
- The overlay plots show both contours, and the selection line is the one that stays the same across the three cards.

## Other behaviour

| # | What to do | Expected |
|---|---|---|
| 4 | TRACK 4, 25.04 to 46.11 s (21 s, several gamakas) | **Unknown / doesn't match** (distance about 178). The detector box says "This selection may hold 4 gamakas": Etra about 30.40 s, Kampita about 31.64 s, Kampita about 40.45 s and Irakka about 45.45 s. |
| 5 | TRACK 8, 20.00 to 21.50 s, **Search inside selection** on | Nokku 3/3, confident, distance about 2.05. The best-match stretch is 20.65 to 21.12 s and is shaded in Nokku orange. The whole-selection line says Kampita 2/3, distance about 17.7. The "Search inside selection is rough" note is shown. |
| 6 | Any track, a 50 ms selection (for example 10.00 to 10.05) | "Cannot classify this selection. The selection is 50 ms long. At least 80 ms is needed." |
| 7 | Turn on **Blind mode** and classify row 1 again | The type, neighbours and detector types are hidden. Reveal is disabled until you pick from Kampita / Nokku / Etra / Irakka / orikai / not a gamaka / not sure. After revealing, the panel shows "You: ... Model: ...". Nothing is written to `annotations/`. |
| 8 | Add a new song (any WAV or MP3 that is not a dataset file) | The section measures pitch first; this took about 4 s for a 15 s clip. "Compared with" then says all 293 marks, and a request takes roughly 0.1 to 0.2 s because the 11 detector models are averaged. |
| 9 | Switch between **Detector** and **Classify a selection** | Each section keeps its own recording and audio. Switching pauses the other section's audio, and Space plays or pauses only the visible one. |
| 9b | Zoom in, then use the **Whole song** strip under the time axis | Dragging the dark box moves the view along the song. Dragging either edge of the box zooms out or in. Clicking outside the box centres the view there. Your selection and the best match stay shaded on the strip and do not change. |
| 10 | Edit `dashboard/classify_config.json` (for example set `reject_distance` to 2.5) and classify row 1 again | The answer turns into **Unknown / doesn't match** without restarting. Set `"tuned": true` once you enter your offline value, and the "untuned" warning disappears. |

Timing for each request is written to `~/.cache/gamaka_dashboard/jobs/classify_requests.log`.
