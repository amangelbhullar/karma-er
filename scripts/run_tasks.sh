#!/bin/bash
cd /storage/bhull113/karma_v2
DT=/storage/bhull113/karma_v2/data_tasks; FE=/storage/bhull113/karma_fair_eval
waitdata() { while [ ! -f $DT/$1/test.txt ]; do sleep 120; done; }
train() { waitdata $1; [ -f runs_tasks/$1_none_s$2_best_test_train.json ] && return
  python karma_v2x.py train --dataset $1 --data_dir $DT --seed $2 --epochs 60 --out_dir runs_tasks --gpu $3; }
post() { waitdata $1
  python $FE/baseline_recency.py --dataset $1 --data_dir $DT --out runs_tasks/recency_$1.json > /dev/null
  python $FE/baseline_buckets.py $1 $DT > runs_tasks/buckets_$1.txt
  for n in type random; do python karma_v2x.py factcheck --dataset $1 --data_dir $DT --ckpt runs_tasks/$1_none_s42_best.pt --neg $n --max_queries 50000 --gpu $2; done; }
case $1 in
  0) train ML-cold 42 0; post ML-cold 0; train ML-cold 1 0 ;;
  1) train STRING-PPI 42 1; post STRING-PPI 1; train STRING-PPI 1 1 ;;
  2) for d in ICEWS14-renet ICEWS18-renet WIKI-clean YAGO15k YAGO11k WIKIDATA12k; do
       python karma_v2x.py factcheck --dataset $d --ckpt runs_rec/${d}_best.pt --neg random --max_queries 50000 --gpu 2; done
     train ML-cold 2 2; train STRING-PPI 2 2 ;;
esac
echo ALL_DONE
