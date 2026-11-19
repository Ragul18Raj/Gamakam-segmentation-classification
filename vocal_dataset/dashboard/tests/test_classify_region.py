"""Tests for POST /api/classify_region. Run from vocal_dataset:
    NUMBA_CACHE_DIR=~/.cache/gamaka_dashboard/numba /usr/bin/python3 -m pytest -p no:cacheprovider dashboard/tests -q
They start the real classifier process once (about 10 s) and read only saved files; nothing is written to the project."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.dont_write_bytecode = True
DASH = Path(__file__).resolve().parents[1]; VD = DASH.parent
sys.path.insert(0, str(DASH))
import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

FULL_TO_SHORT = {"Kampita": "Kampita", "Nokku": "Nokku", "Jaaru (Etra)": "Etra", "Jaaru (Irakka)": "Irakka"}
STAGE1 = pd.read_csv(VD / "results/stage1/predictions.csv")
META = pd.read_csv(VD / "metadata.csv", dtype=str).set_index("recording")


@pytest.fixture(scope="module")
def client():
    with TestClient(server.app) as c:
        yield c
    server.WORKER.stop()


def post(client, **body):
    return client.post("/api/classify_region", json=body)


def sample20():
    """5 marked events of each type, fixed seed, from the saved Stage 1 held-out predictions."""
    rng = np.random.default_rng(20261119)
    return pd.concat([g.iloc[rng.choice(len(g), 5, replace=False)] for _, g in STAGE1.groupby("label")])


def test_matches_stage1_offline_for_20_marked_events(client):
    times = []
    for r in sample20().itertuples():
        res = post(client, track=r.recording, start_s=r.start_s, end_s=r.end_s)
        assert res.status_code == 200, res.text
        out = res.json(); times.append(out["timing_ms"]["request_ms"])
        expected = FULL_TO_SHORT[STAGE1.loc[r.Index, "main_4class | DTW-3NN"]]
        assert out["closest_type"] == expected, (r.recording, r.start_s, out["closest_type"], expected)
        assert out["type"] in (expected, None)                       # None only when the reject rule fires
        assert (out["type"] is None) == (out["distance"] > out["threshold"])
    assert np.median(times) < 1000, f"median request time {np.median(times):.0f} ms"


def test_too_short_is_rejected(client):
    res = post(client, track="TRACK 1", start_s=10.0, end_s=10.05)
    assert res.status_code == 422 and "80 ms" in res.json()["detail"]


def test_end_before_start_is_rejected(client):
    res = post(client, track="TRACK 1", start_s=10.0, end_s=9.0)
    assert res.status_code == 422


def test_unvoiced_is_rejected(client):
    """Find a 0.3 s stretch with fewer than 5 confident pitch frames in the saved CREPE track, then classify it."""
    for rec in ["TRACK 1", "TRACK 4", "VARNAM", "TRACK 10", "TRACK 20"]:
        z = np.load(VD / "crepe" / f"{rec}.npz"); voiced = z["confidence"] >= .5; t = z["time_s"]
        counts = np.convolve(voiced.astype(int), np.ones(30, int), "valid")
        hits = np.flatnonzero(counts < 5)
        if len(hits):
            s = round(float(t[hits[0]]) + .005, 2)
            res = post(client, track=rec, start_s=s, end_s=s + .25)
            assert res.status_code == 422 and "voiced" in res.json()["detail"], res.text
            return
    pytest.skip("no unvoiced 0.3 s stretch found in the checked tracks (the drone keeps CREPE confident)")


def test_own_group_and_other_tempi_are_excluded(client):
    """A TRACK 2 mark must not match TRACK 1, 2 or 3 (the same passage at three tempi, group P01)."""
    r = STAGE1[STAGE1.recording == "TRACK 2"].iloc[0]
    out = post(client, track="TRACK 2", start_s=r.start_s, end_s=r.end_s).json()
    group = META.loc["TRACK 2", "group"]; same = set(META.index[META.group == group])
    assert out["excluded_group"] == group and set(out["excluded_tracks"]) == same == {"TRACK 1", "TRACK 2", "TRACK 3"}
    assert all(n["track"] not in same and n["group"] != group for n in out["nearest"])
    assert out["library_used"] == len(STAGE1) - (STAGE1.group == group).sum()


def test_exclusion_rule_matters():
    """Without the exclusion, a marked event finds itself at distance 0; with it, it cannot (core-level check)."""
    import classify_core as C
    i = 10; rec, s, e = C.LIB_TRACK[i], C.LIB_START[i], C.LIB_END[i]
    tr = dict(C.load_track(known=rec)); g = tr["group"]
    with_rule = C.name_span(tr, s, e)
    tr["exclude"] = set()
    without = C.name_span(tr, s, e)
    assert without["dist"] == 0.0 and without["near"][0] == i
    assert with_rule["dist"] > 0 and all(C.TS.LIB_G[j] != g for j in with_rule["near"])


def test_subsequence_returns_span_inside_selection(client):
    r = STAGE1[STAGE1.recording == "TRACK 5"].iloc[2]
    s, e = round(r.start_s - .6, 2), round(r.end_s + .6, 2)
    out = post(client, track="TRACK 5", start_s=s, end_s=e, subsequence=True).json()
    bm = out["best_match"]
    assert bm and s - 1e-6 <= bm["start"] < bm["end"] <= e + .011 and bm["end"] - bm["start"] >= .08 - 1e-6


def test_nearest_examples_have_contours_and_clips(client):
    r = STAGE1.iloc[0]
    out = post(client, track=r.recording, start_s=r.start_s, end_s=r.end_s).json()
    assert len(out["nearest"]) == 3
    for n in out["nearest"]:
        assert len(n["contour"]) >= 5 and n["clip_url"].startswith("/api/clip/")
    clip = client.get(out["nearest"][0]["clip_url"])
    assert clip.status_code == 200 and clip.content[:4] == b"RIFF"
