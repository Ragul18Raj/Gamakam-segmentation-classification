# Recording check per branch (Sa = 327 Hz)

Grey: spectrogram of the WAV itself (16 kHz, 4096-point FFT, 10 ms hop, 70 dB range), y axis in cents above Sa. The fundamental and its harmonics (+1200, +1902, +2400 cents above it) are visible directly, without any pitch tracker.
Orange: pYIN (librosa), a pitch tracker independent of CREPE. Blue: CREPE, confidence >= 0.5, no octave fix.
Title: `agree` = share of frames voiced in both where they are within 50 cents; `oct` = share exactly an octave (or more) apart; `CLIP` = samples at full scale. Red title: clipping or agreement below 80%.
Per-clip numbers for every clip (plotted or not): `clip_audio_stats.csv`.

| Gamaka | Branch | Clips | PNG |
|---|---|---|---|
| jaru | ascending | 12 | [jaru__ascending.png](jaru__ascending.png) |
| jaru | descending Melgana | 6 | [jaru__descending_Melgana.png](jaru__descending_Melgana.png) |
| jaru | descending Shankarabharanam | 6 | [jaru__descending_Shankarabharanam.png](jaru__descending_Shankarabharanam.png) |
| kampita | plavita | 3 | [kampita__plavita.png](kampita__plavita.png) |
| kampita | plavita bhairiga | 4 | [kampita__plavita_bhairiga.png](kampita__plavita_bhairiga.png) |
| kampita | plavitam Todi | 8 | [kampita__plavitam_Todi.png](kampita__plavitam_Todi.png) |
| kampita | leena | 11 | [kampita__leena.png](kampita__leena.png) |
| ravai | Kalapriya | 12 | [ravai__Kalapriya.png](ravai__Kalapriya.png) |
| ravai | Mayamalavagowla | 12 | [ravai__Mayamalavagowla.png](ravai__Mayamalavagowla.png) |
| ravai | unspecified (#02) | 11 | [ravai__unspecified_02.png](ravai__unspecified_02.png) |
| pratyagata | Hindolam | 16 (12 shown) | [pratyagata__Hindolam.png](pratyagata__Hindolam.png) |
| pratyagata | Amritavarshini | 11 | [pratyagata__Amritavarshini.png](pratyagata__Amritavarshini.png) |
