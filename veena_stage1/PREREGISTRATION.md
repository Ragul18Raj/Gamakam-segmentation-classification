# Veena stage-1 check: fixed before running (2026-10-30)

Question: can Jaru, Kampita, Ravai and Pratyahatam be told apart from pitch movement alone,
on veena takes the classifier has never seen?

- Classes (fixed now): jaru (asc+desc merged), kampita (all variants), ravai, pratyagata. Sphuritam excluded (1 take).
- Files whose name contains "_continuous" are excluded (not single-gamaka clips).
- Features: the 9 stage-1 segment features from run_eval.seg_feats (duration, range p95-p5, turning points,
  turns per second, signed and absolute net change, max excursion up/down from start, mean |velocity|),
  computed over each clip's voiced span. All are relative to the clip's own pitch: no note/raga position.
- Split: leave-one-take-out (takes from takes.csv). Classifiers: logistic regression (main), 3-NN (secondary).
- Leakage probe: logistic regression on the clip's median absolute pitch alone, same split.
- Metric: macro-F1 over pooled held-out predictions; also balanced accuracy, per-class recall, confusion matrix.
- Null: 1000 permutations of class labels across takes.
- PASS if logistic-regression macro-F1 >= 0.50 AND above the leakage probe AND permutation p < 0.05.
