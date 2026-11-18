"""Aggregate stage-2 folds (PLAN.md section 6, stage 2) -> <dir>/summary.json and printout.

usage: python stage2_score.py results/stage2_main
Main event rule: same label, IoU >= 0.3, one-to-one. Secondary: onset within +-200 ms (as pipeline.match_events).
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix, f1_score

from stage2_vocal import GCLS, match_iou

ROOT = Path(__file__).resolve().parent
NAMES = {1: "Kampita", 2: "Nokku", 3: "Jaaru (Etra)", 4: "Jaaru (Irakka)"}
SEED = 20261117


def match_onset(ref, est, tol=0.2):
    tp = {}
    for l in GCLS:
        R = [e for e in ref if e[2] == l]
        E = [e for e in est if e[2] == l]
        pairs = sorted((abs(r[0] - x[0]), i, j) for i, r in enumerate(R) for j, x in enumerate(E) if abs(r[0] - x[0]) <= tol)
        ur, ue, n = set(), set(), 0
        for _, i, j in pairs:
            if i not in ur and j not in ue:
                ur.add(i)
                ue.add(j)
                n += 1
        tp[l] = n
    return tp


def counts(clips, key, classes=GCLS, onset=False):
    tp = {l: 0 for l in classes}
    nr, ne = dict(tp), dict(tp)
    pairs = []
    for c in clips:
        ref = [tuple(x) for x in c["ref"] if x[2] in classes]
        est = [tuple(x) for x in c[key] if x[2] in classes] if key else []
        if onset:
            a = match_onset(ref, est)
        else:
            a, _, _, pr = match_iou(ref, est)
            pairs += pr
        for l in classes:
            tp[l] += a[l]
            nr[l] += sum(e[2] == l for e in ref)
            ne[l] += sum(e[2] == l for e in est)
    return tp, nr, ne, pairs


def prf(tp, nr, ne):
    p = tp / ne if ne else 0.0
    r = tp / nr if nr else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def ev_scores(clips, key, classes=GCLS, onset=False):
    tp, nr, ne, pairs = counts(clips, key, classes, onset)
    per = {NAMES[l]: dict(zip(("precision", "recall", "f1"), prf(tp[l], nr[l], ne[l])), n_ref=nr[l], n_est=ne[l]) for l in classes}
    present = [l for l in classes if nr[l] + ne[l]]
    macro = float(np.mean([per[NAMES[l]]["f1"] for l in present])) if present else 0.0
    return macro, per, pairs


def system(folds, clips, key, frames_key):
    r = {}
    Y = np.concatenate([c["y"] for c in clips])
    if frames_key:
        Pf = np.concatenate([c[frames_key] for c in clips])
        r["frame_macro_f1"] = float(f1_score(Y, Pf, labels=range(5), average="macro", zero_division=0))
        r["frame_cm"] = confusion_matrix(Y, Pf, labels=range(5)).tolist()
    r["event_macro_f1"], r["per_class"], pairs = ev_scores(clips, key)
    r["event_macro_f1_onset200"] = ev_scores(clips, key, onset=True)[0]
    if pairs:
        r["median_onset_err_ms"] = float(1000 * np.median([abs(a[0] - b[0]) for a, b in pairs]))
        r["median_offset_err_ms"] = float(1000 * np.median([abs(a[1] - b[1]) for a, b in pairs]))
    errs, exact, tot_err = [], 0, []
    for c in clips:
        e = [abs(sum(x[2] == l for x in (c[key] if key else [])) - sum(x[2] == l for x in c["ref"])) for l in GCLS]
        errs += e
        exact += all(v == 0 for v in e)
        tot_err.append(len(c[key] if key else []) - len(c["ref"]))
    r["count_abs_err_per_type_per_recording"] = float(np.mean(errs))
    r["count_total_err_per_recording_mean"] = float(np.mean(tot_err))
    r["recordings_all_counts_exact"] = exact / len(clips)
    r["per_fold_event_macro_f1"] = {f["test"]: ev_scores(f["clips"], key)[0] for f in folds}
    r["by_tempo"] = {t: ev_scores([c for c in clips if c["tempo"] == t], key)[0] for t in ("slow", "medium", "fast", "(varnam)")
                     if any(c["tempo"] == t for c in clips)}
    r["tracks_only_event_macro_f1"] = ev_scores([c for c in clips if c["id"] != "VARNAM"], key)[0]
    r["kampita_event_f1"] = ev_scores(clips, key, classes=(1,))[0]
    return r


def boot_recordings(clips, key, n=2000):
    rng = np.random.default_rng(SEED)
    out = [ev_scores([clips[i] for i in rng.integers(0, len(clips), len(clips))], key)[0] for _ in range(n)]
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    d = ROOT / sys.argv[1]
    folds = [json.load(open(p)) for p in sorted(glob.glob(str(d / "fold_*.json")))]
    clips = [c for f in folds for c in f["clips"]]
    Y = np.concatenate([c["y"] for c in clips])
    out = dict(n_folds=len(folds), n_recordings=len(clips), n_ref_events=sum(len(c["ref"]) for c in clips),
               frame_share_none=float((Y == 0).mean()), filters_chosen={f["test"]: f["filter"] for f in folds})
    out["all_none"] = system(folds, clips, None, None)
    out["all_none"]["frame_macro_f1"] = float(f1_score(Y, np.zeros_like(Y), labels=range(5), average="macro", zero_division=0))
    out["rule_knn"] = system(folds, clips, "rule", "rule_frames")
    out["cnn"] = system(folds, clips, "cnn", "cnn_frames")
    P_raw = np.concatenate([c["cnn_raw"] for c in clips])
    out["cnn"]["frame_macro_f1_before_postprocessing"] = float(f1_score(Y, P_raw, labels=range(5), average="macro", zero_division=0))
    out["cnn"]["event_macro_f1_ci95_recordings"] = boot_recordings(clips, "cnn")
    out["rule_knn"]["event_macro_f1_ci95_recordings"] = boot_recordings(clips, "rule")
    out["kampita_rf"] = dict(kampita_event_f1=ev_scores(clips, "kampita_rf", classes=(1,))[0],
                             per_class=ev_scores(clips, "kampita_rf", classes=(1,))[1])
    a = np.array([out["cnn"]["per_fold_event_macro_f1"][f["test"]] for f in folds])
    b = np.array([out["rule_knn"]["per_fold_event_macro_f1"][f["test"]] for f in folds])
    w = dict(cnn_mean=float(a.mean()), cnn_std=float(a.std(ddof=1)), rule_mean=float(b.mean()), rule_std=float(b.std(ddof=1)))
    if np.any(a != b):
        w["p_rule_better_one_sided"] = float(wilcoxon(b, a, alternative="greater").pvalue)
        w["p_cnn_better_one_sided"] = float(wilcoxon(a, b, alternative="greater").pvalue)
        w["p_two_sided"] = float(wilcoxon(a, b).pvalue)
    out["cnn_vs_rule_folds"] = w
    c = out["cnn"]
    out["PASS_stage2"] = dict(event_f1_ge_060=c["event_macro_f1"] >= 0.60, frame_f1_ge_060=c["frame_macro_f1"] >= 0.60,
                              cnn_not_worse_than_rule=w.get("p_rule_better_one_sided", 1.0) >= 0.05)
    out["PASS_stage2"]["all"] = all(out["PASS_stage2"].values())
    json.dump(out, open(d / "summary.json", "w"), indent=1)

    print(f"{d.name}: {len(folds)} folds, {len(clips)} recordings, {out['n_ref_events']} events; none = {out['frame_share_none']:.1%} of frames")
    print(f"filter chosen per fold: {out['filters_chosen']}")
    print(f"\n{'':10s} {'frameF1':>8s} {'eventF1':>8s} {'onset200':>8s} {'tracks':>7s} {'Kampita':>8s} {'cnt/type':>8s} {'exact':>6s}")
    for k in ("all_none", "rule_knn", "cnn"):
        s = out[k]
        print(f"{k:10s} {s['frame_macro_f1']:8.3f} {s['event_macro_f1']:8.3f} {s['event_macro_f1_onset200']:8.3f} "
              f"{s['tracks_only_event_macro_f1']:7.3f} {s['kampita_event_f1']:8.3f} {s['count_abs_err_per_type_per_recording']:8.2f} "
              f"{s['recordings_all_counts_exact']:6.0%}")
    print(f"kampita_rf Kampita event F1 {out['kampita_rf']['kampita_event_f1']:.3f}  {out['kampita_rf']['per_class']}")
    print(f"\nCNN frame macro-F1 before post-processing {c['frame_macro_f1_before_postprocessing']:.3f}; "
          f"event macro-F1 95% CI over recordings {c['event_macro_f1_ci95_recordings']}")
    for k in ("cnn", "rule_knn"):
        print(f"\n{k} per class:")
        for n, v in out[k]["per_class"].items():
            print(f"   {n:16s} P {v['precision']:.2f} R {v['recall']:.2f} F1 {v['f1']:.2f}  (ref {v['n_ref']}, est {v['n_est']})")
        print(f"   onset/offset median error {out[k].get('median_onset_err_ms', float('nan')):.0f} / "
              f"{out[k].get('median_offset_err_ms', float('nan')):.0f} ms; by tempo "
              + ", ".join(f"{t} {v:.2f}" for t, v in out[k]["by_tempo"].items()))
        print("   per fold: " + ", ".join(f"{t} {v:.2f}" for t, v in out[k]["per_fold_event_macro_f1"].items()))
    print("\nCNN frame confusion (rows = singer, cols = CNN; none, Kampita, Nokku, Etra, Irakka):")
    print(np.array(c["frame_cm"]))
    print(f"\nfolds CNN vs rule: {w}")
    print(f"PASS stage 2: {out['PASS_stage2']}")


if __name__ == "__main__":
    main()
