"""Sanity tests for eval_segments.py on the veena kampita annotations."""
import pandas as pd, numpy as np
from eval_segments import load, score
T = load('../kampita_annotations_2026-11-13.xlsx', 'kampita')
def run(name, P, expect=None):
    _, O = score(T, P); print(f'{name:38s} P={O["precision"]:.2f} R={O["recall"]:.2f} F1={O["F1"]:.2f} count_err={O["count_err"]:+d}')
    if expect is not None: assert abs(O['F1'] - expect) < 1e-9, (name, O['F1'])
run('perfect (truth vs itself)', T.copy(), 1.0)
run('empty prediction', T.iloc[0:0].copy(), 0.0)
s = T.copy(); s[['Start (sec)', 'End (sec)']] += 0.05; run('shifted 50 ms (should still match)', s, 1.0)
s = T.copy(); s[['Start (sec)', 'End (sec)']] += 3.0; run('shifted 3 s (should all miss)', s, 0.0)
d = pd.concat([T, T.assign(**{'Start (sec)': T['Start (sec)'] + 5, 'End (sec)': T['End (sec)'] + 5})]); run('every hit + one extra false alarm', d)
h = T.copy(); h['End (sec)'] = h['Start (sec)'] + (h['End (sec)'] - h['Start (sec)']) / 2; run('half-length predictions (IoU 0.5)', h, 1.0)
dup = pd.concat([T, T]); run('duplicate predictions (1 TP + 1 FP each)', dup)
print('all asserts passed')
