#!/bin/bash
# All 479 CMD-40 recordings x 7 bitrates, 3 shards on cuda:1 (~3.3 GB GPU each), then results/RESULT.md.
# Resumable: re-running skips recordings already in results/results_s*.csv.
cd "$(dirname "$0")/.."
mkdir -p compressing/logs
for k in 0 1 2; do
  python3 compressing/compress_test.py --cmd 12 --shard $k --nshard 3 --device cuda:1 \
    > compressing/logs/shard$k.log 2>&1 &
done
wait
python3 compressing/compress_test.py --summarize > compressing/logs/summarize.log 2>&1
