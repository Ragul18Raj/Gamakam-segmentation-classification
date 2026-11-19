# Singer's answers to QUESTIONS_FOR_SINGER.md (received 18 Nov 2026)

## Answers, as received

1. Track 1-3 are slow, medium, and fast respectively. Just like that track 4-6 and so on.
   All the tracks are from kalyani raga.
   Varnam is a separate piece which I have sung in a single speed that is second speed (usually varnam is sung in second speed in concerts)
2. I have marked all the gamakas I could here in all the tracks. For the medium and fast ones every gamak can't be sung in that speed since it is very fast.
3. In some places there is orikai. I can't say in which audio it is present exactly.
4. I have sung swaras in both lower and higher octave. In some of the audio I have included both higher octave notes and middle octave notes. In some of them I have included lower octave notes and middle octave notes.
5. yes
6. yeah sure

## How they are used

| Question | Status | Effect |
|---|---|---|
| Grouping and tempo | **Confirmed.** Groups P01–P10 and slow/medium/fast match `metadata.csv`; VARNAM is its own group, P11 | No change to any split |
| Raga | Kalyani for TRACK 1–30. The singer did not name the VARNAM's raga | `raga` left empty for VARNAM |
| Song titles | Not given | `song_title` stays empty |
| Marking complete? | **Qualified.** "All I could", and fewer gamakas at speed is real singing, not skipped marking. This is not proof that nothing was missed | Unmarked stretches stay "none" |
| Other gamakas | **Orikai occurs and is unmarked**, at unknown places | Counts as "none". It may explain some stage-2 false alarms; this cannot be tested without marks |
| Octave | Tracks mix middle with lower or upper notes | `octave` column filled from measured pitch (below) |
| Stereo | One microphone | Averaging the channels stands |
| Second marking | Agreed | TRACK 1, TRACK 8, TRACK 30, first 60 s of VARNAM; not before 24 Nov 2026 (one week after pass 1) |

## Checks against the audio (18 Nov 2026, recorded vocal)

- **Same passage in each set of three:** the singer's answer implies this but does not say it outright. Comparing time-normalised pitch lines, each of the 30 tracks has its two set-mates as its two nearest tracks. Median DTW distance is 19 cents within a set and 111 cents between sets. The closest pair from different sets is 39 cents, about the same as the largest within-set distance (38), and several such pairs involve P08 and P09, so those passages may share material.
- **Tempo order:** performance length falls slow > medium > fast in all 10 sets. Marked events fall with speed overall (115 slow, 74 medium, 49 fast), which fits the singer's answer 2.
- **Kalyani:** 97% (median) of held singing sits on Kalyani's notes in the tracks, lowest track 89%, and 96% in the VARNAM. The sharp Ma (M2) outweighs the natural Ma (M1) in every recording.
- **Octave column:** measured, not stated by the singer. A register is listed when at least 5% of singing frames are below Sa (lower) or at or above upper Sa (upper). Result: 16 middle+upper, 9 middle, 6 lower+middle.
- **Octave-slip folding:** it changed 0 of the 17,827 frames inside marked events, so singing in several octaves did not alter any stage-1 curve.
