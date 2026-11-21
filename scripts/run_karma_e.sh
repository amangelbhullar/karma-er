#!/bin/bash
# KARMA-E (= KARMA-ER without recency, --no_rec). Shared job list; each job is
# claimed by exactly one worker via an atomic mkdir lock, so no job runs twice.
GPU=$1
cd /storage/bhull113/karma_v2
JOBS="ICEWS18-renet:42 WIKI-clean:42 ICEWS14-renet:42 YAGO15k:42 WIKIDATA12k:42 YAGO11k:42
      ICEWS18-renet:1 WIKI-clean:1 ICEWS14-renet:1 YAGO15k:1 WIKIDATA12k:1 YAGO11k:1
      ICEWS18-renet:2 WIKI-clean:2 ICEWS14-renet:2 YAGO15k:2 WIKIDATA12k:2 YAGO11k:2"
for job in $JOBS; do
  d=${job%:*}; s=${job#*:}
  mkdir runs_rec/karma_e_locks/${d}_s${s} 2>/dev/null || continue
  echo "== $d seed $s GPU $GPU $(date)"
  python karma_v2x.py train --dataset "$d" --data_dir /storage/bhull113/rhgnn/data \
    --out_dir /storage/bhull113/karma_v2/runs_rec --gpu "$GPU" --seed "$s" --no_rec || echo "FAILED $d s$s"
done
echo "DONE GPU $GPU $(date)"
