"""Score detected gamaka segments against manual annotations.

usage: python eval_segments.py <truth.xlsx|csv> <pred.xlsx|csv> [--type kampita] [--iou 0.3] [--tol 0.1]

Both files need: a recording column ('Song Name' or 'Recording'), 'Start (sec)', 'End (sec)'.
Optional 'Gamaka Type' column: filter with --type (case-insensitive).
A prediction is correct (true positive) if it overlaps a not-yet-matched truth segment with
IoU >= --iou  OR  both boundaries within --tol seconds. One-to-one matching, best IoU first.
Reports per recording and overall: TP, FP, FN, precision, recall, F1, count error, mean boundary error.
Only recordings that appear in the truth file are scored (predictions on unannotated files are ignored).
"""
import argparse, sys, numpy as np, pandas as pd

def load(path, gtype):
    df = pd.read_excel(path, sheet_name=0) if path.lower().endswith(('xlsx', 'xls')) else pd.read_csv(path)
    rec = 'Song Name' if 'Song Name' in df else 'Recording'
    if gtype and 'Gamaka Type' in df:
        df = df[df['Gamaka Type'].astype(str).str.strip().str.lower() == gtype.lower()]
    df = df.rename(columns={rec: 'rec'})
    df['rec'] = df['rec'].astype(str).str.strip().str.replace(r'\.(npy|wav)$', '', regex=True)
    return df[['rec', 'Start (sec)', 'End (sec)']].astype({'Start (sec)': float, 'End (sec)': float})

def match(T, P, iou_min, tol):
    pairs = []
    for i, (ts, te) in enumerate(T):
        for j, (ps, pe) in enumerate(P):
            inter = max(0, min(te, pe) - max(ts, ps)); uni = max(te, pe) - min(ts, ps)
            iou = inter / uni if uni > 0 else 0
            if iou >= iou_min or (abs(ts - ps) <= tol and abs(te - pe) <= tol):
                pairs.append((iou, i, j))
    used_t, used_p, m = set(), set(), []
    for iou, i, j in sorted(pairs, reverse=True):
        if i not in used_t and j not in used_p:
            used_t.add(i); used_p.add(j); m.append((i, j))
    return m

def score(truth, pred, iou_min=0.3, tol=0.1):
    rows = []
    for rec, tg in truth.groupby('rec'):
        T = tg[['Start (sec)', 'End (sec)']].values; P = pred[pred.rec == rec][['Start (sec)', 'End (sec)']].values
        m = match(T, P, iou_min, tol); tp = len(m)
        berr = [ (abs(T[i][0]-P[j][0]) + abs(T[i][1]-P[j][1])) / 2 for i, j in m]
        rows.append(dict(recording=rec, truth=len(T), pred=len(P), TP=tp, FP=len(P) - tp, FN=len(T) - tp,
                         count_err=len(P) - len(T), boundary_err_s=np.mean(berr) if berr else np.nan))
    R = pd.DataFrame(rows)
    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)
    R[['precision', 'recall', 'F1']] = [prf(a, b, c) for a, b, c in R[['TP', 'FP', 'FN']].values]
    tot = R[['truth', 'pred', 'TP', 'FP', 'FN']].sum()
    p, r, f = prf(tot.TP, tot.FP, tot.FN)
    overall = dict(recordings=len(R), truth=int(tot.truth), pred=int(tot.pred), TP=int(tot.TP), FP=int(tot.FP), FN=int(tot.FN),
                   precision=p, recall=r, F1=f, count_err=int(tot.pred - tot.truth),
                   mean_abs_count_err_per_rec=float(R.count_err.abs().mean()),
                   boundary_err_s=float(np.nanmean(R.boundary_err_s)) if R.TP.sum() else float('nan'))
    return R, overall

if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('truth'); a.add_argument('pred')
    a.add_argument('--type', default=None); a.add_argument('--iou', type=float, default=0.3); a.add_argument('--tol', type=float, default=0.1)
    x = a.parse_args()
    R, O = score(load(x.truth, x.type), load(x.pred, x.type), x.iou, x.tol)
    pd.set_option('display.width', 200); print(R.round(3).to_string(index=False)); print('\nOVERALL:', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in O.items()})
