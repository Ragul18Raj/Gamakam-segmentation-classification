"""Step 2: check the pass-1 annotation workbook against the audio, then write annotations/pass1_events.csv.

Reads only the copies inside vocal_dataset/. Prints every check with PASS/FAIL and writes:
  annotations/pass1_events.csv      recording,start_s,end_s,label   (all later code reads this, not the Excel)
  results/annotation_checks.txt     the same printout
  metadata.csv                      only if it does not exist yet (PROVISIONAL grouping, see notes column)
"""
import hashlib
import re
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parent
XLSX = ROOT / "annotations" / "pass1.xlsx"
AUDIO = ROOT / "audio"
LABELS = ["Kampita", "Nokku", "Jaaru (Etra)", "Jaaru (Irakka)"]
COLS = ["Song Name", "S.No", "Timestamp", "Gamaka Type", "Start (sec)", "End (sec)", "Duration (sec)"]

out = StringIO()


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.write(s + "\n")


def check(name, ok, detail=""):
    say(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def canon_label(s):
    s = re.sub(r"\s+", " ", str(s).strip())
    for lab in LABELS:
        if s.lower() == lab.lower():
            return lab
    return None


def track_no(rec):
    m = re.fullmatch(r"TRACK (\d+)", rec)
    return int(m.group(1)) if m else None


def provisional_metadata(recs):
    """Inferred grouping: TRACK 1-3 -> P01, ... TRACK 28-30 -> P10, VARNAM -> P11. NOT confirmed by the singer."""
    rows = []
    for r in recs:
        n = track_no(r)
        if n is None:
            g, tempo = "P11", ""
        else:
            g, tempo = f"P{(n - 1) // 3 + 1:02d}", ["slow", "medium", "fast"][(n - 1) % 3]
        rows.append(dict(recording=r, group=g, song_title="", raga="", tempo=tempo, octave="", sa_note="",
                         sa_hz="", complete_marking="", notes="PROVISIONAL: group and tempo inferred from track order "
                         "and length; confirm with singer"))
    return pd.DataFrame(rows)


def main():
    all_ok = True
    say(f"workbook: {XLSX.name}  sha256={hashlib.sha256(XLSX.read_bytes()).hexdigest()}")
    sheets = pd.read_excel(XLSX, sheet_name=None)
    allsongs = sheets.pop("All Songs")
    all_ok &= check("All Songs has the expected columns", list(allsongs.columns) == COLS, list(allsongs.columns))

    # 1. All Songs == union of per-recording sheets, row by row
    per = pd.concat([df.assign(_sheet=k) for k, df in sheets.items()], ignore_index=True)
    all_ok &= check("each per-recording sheet only holds its own recording",
                    bool((per["Song Name"] == per["_sheet"]).all()))
    key = ["Song Name", "S.No", "Gamaka Type", "Start (sec)", "End (sec)"]
    a = allsongs[key].astype(str).value_counts()
    b = per[key].astype(str).value_counts()
    diff = a.subtract(b, fill_value=0)
    all_ok &= check("All Songs rows == union of per-recording sheets (row by row, multiplicity counted)",
                    bool((diff == 0).all()) and len(allsongs) == len(per),
                    f"All Songs {len(allsongs)} rows, sheets {len(per)} rows, mismatched keys {int((diff != 0).sum())}")
    all_ok &= check("no duplicated row in All Songs", not allsongs.duplicated(key).any())
    all_ok &= check("S.No runs 1..n within each recording", all(
        list(g["S.No"]) == list(range(1, len(g) + 1)) for _, g in allsongs.groupby("Song Name", sort=False)))

    # 2. labels
    raw = allsongs["Gamaka Type"].astype(str)
    canon = raw.map(canon_label)
    say("raw label spellings:", dict(raw.value_counts()))
    all_ok &= check("labels are only the 4 types after trimming/case-fixing", canon.notna().all(),
                    f"unknown: {sorted(set(raw[canon.isna()]))}")
    all_ok &= check("labels need no case/space fixing", bool((raw == canon).all()))

    ev = pd.DataFrame(dict(recording=allsongs["Song Name"].astype(str).str.strip(),
                           start_s=allsongs["Start (sec)"].astype(float),
                           end_s=allsongs["End (sec)"].astype(float), label=canon))
    all_ok &= check("Duration column == End - Start (within 2 ms)",
                    bool(np.allclose(allsongs["Duration (sec)"], ev.end_s - ev.start_s, atol=0.002)))

    # 3. audio files <-> sheets, lengths
    audio = {p.stem: p for p in AUDIO.glob("*.mp3")}
    recs = list(dict.fromkeys(ev.recording))
    all_ok &= check("every audio file has a sheet", set(audio) <= set(sheets), sorted(set(audio) - set(sheets)))
    all_ok &= check("every sheet has an audio file", set(sheets) <= set(audio), sorted(set(sheets) - set(audio)))
    info = {k: sf.info(str(p)) for k, p in audio.items()}
    dur = {k: i.frames / i.samplerate for k, i in info.items()}
    say(f"audio: {len(audio)} files, sr={sorted({i.samplerate for i in info.values()})}, "
        f"channels={sorted({i.channels for i in info.values()})}, total {sum(dur.values()) / 60:.1f} min, "
        f"VARNAM {dur.get('VARNAM', 0):.1f} s, tracks {min(v for k, v in dur.items() if k != 'VARNAM'):.1f}"
        f"-{max(v for k, v in dur.items() if k != 'VARNAM'):.1f} s")
    tracks = sorted([k for k in audio if track_no(k)], key=track_no)
    say(f"recordings: VARNAM + {len(tracks)} tracks (TRACK {track_no(tracks[0])} .. TRACK {track_no(tracks[-1])})")

    # 4. timing
    all_ok &= check("Start < End for every event", bool((ev.start_s < ev.end_s).all()))
    ev["audio_s"] = ev.recording.map(dur)
    all_ok &= check("End within audio length", bool((ev.end_s <= ev.audio_s + 1e-3).all()),
                    ev[ev.end_s > ev.audio_s + 1e-3].to_string() if (ev.end_s > ev.audio_s + 1e-3).any() else "")
    overl, touch = [], []
    for r, g in ev.groupby("recording", sort=False):
        g = g.sort_values("start_s")
        for (_, p), (_, q) in zip(g.iloc[:-1].iterrows(), g.iloc[1:].iterrows()):
            gap = q.start_s - p.end_s
            if gap < -1e-6:
                overl.append((r, p.label, p.end_s, q.label, q.start_s))
            elif gap <= 0.005:
                touch.append((r, f"{p.label} -> {q.label}", round(p.end_s, 3)))
    all_ok &= check("no overlapping events", not overl, overl)
    say(f"touching pairs (gap <= 5 ms): {len(touch)}")
    for t in touch:
        say("   ", t)

    # 5. counts
    meta_path = ROOT / "metadata.csv"
    if not meta_path.exists():
        provisional_metadata(["VARNAM"] + tracks).to_csv(meta_path, index=False)
        say("wrote metadata.csv with PROVISIONAL grouping (confirm with singer)")
    meta = pd.read_csv(meta_path, dtype=str).fillna("")
    ev = ev.merge(meta[["recording", "group", "tempo"]], on="recording", how="left")
    ev["dur"] = ev.end_s - ev.start_s
    say("\nevents per label:", dict(ev.label.value_counts()), " total", len(ev))
    say("\nevents per recording x label:")
    say(pd.crosstab(ev.recording, ev.label).reindex(["VARNAM"] + tracks).to_string())
    say("\nevents per group x label (group is PROVISIONAL unless metadata says otherwise):")
    say(pd.crosstab(ev.group, ev.label, margins=True).to_string())
    say("\nevents per tempo x label (tempo PROVISIONAL; VARNAM has no tempo):")
    say(pd.crosstab(ev.tempo.replace("", "(varnam)"), ev.label, margins=True).to_string())
    say("\nduration per label (s):")
    say(ev.groupby("label").dur.describe(percentiles=[.5]).round(3).to_string())
    flag = ev[(ev.dur < 0.1) | (ev.dur > 3.0)]
    all_ok &= check("no event under 0.1 s or over 3 s", flag.empty, flag.to_string() if len(flag) else "")
    first = ev.groupby("recording").start_s.min()
    say(f"\nfirst annotated event starts at: min {first.min():.2f} s, median {first.median():.2f} s, "
        f"max {first.max():.2f} s  (the opening Sa must end before this)")

    ev[["recording", "start_s", "end_s", "label"]].to_csv(ROOT / "annotations" / "pass1_events.csv",
                                                       index=False, float_format="%.3f")
    say(f"\nwrote annotations/pass1_events.csv ({len(ev)} events)")
    say("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "annotation_checks.txt").write_text(out.getvalue(), encoding="utf-8")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
