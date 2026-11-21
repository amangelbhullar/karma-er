#!/bin/bash
# MultiTQ QA: convert + parse, train KARMA-ER on the MultiTQ KG, answer questions.
set -e
cd /storage/bhull113/karma_v2
G=${1:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)}
echo "GPU $G  $(date)"
python make_multitq.py --src data_tasks/MultiTQ_src/data/MultiTQ --out data_tasks/MultiTQ-kg
python karma_v2x.py train --dataset MultiTQ-kg --data_dir /storage/bhull113/karma_v2/data_tasks \
  --out_dir /storage/bhull113/karma_v2/runs_tasks --gpu "$G" --seed 42 --epochs 30 --eval_every 3 --patience 3
python multitq_qa.py --ckpt runs_tasks/MultiTQ-kg_none_s42_best.pt --gpu "$G"
echo "DONE $(date)"
