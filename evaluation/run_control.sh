#!/bin/bash
# Random-data control: train each KARMA variant on random facts, then evaluate with TOIS/ACIS.
# usage: bash run_control.sh <models_dir> <work_dir> <gpu>
M=$1; W=$2; G=${3:-0}; S=$(cd "$(dirname "$0")" && pwd)
mkdir -p $W/data/RANDOM $W/ckpt $W/res
python - "$W/data/RANDOM" <<'EOF'
import sys, numpy as np
rng = np.random.default_rng(0); out = sys.argv[1]
def gen(ts, ents, n, rel=None):
    return [(rng.choice(ents), rel if rel is not None else rng.integers(0, 5), rng.choice(ents), rng.choice(ts)) for _ in range(n)]
seen, new = list(range(200)), list(range(200, 260))
tr = gen(range(0, 40), seen, 6000); va = gen(range(40, 45), seen, 500)
te = gen(range(45, 50), seen, 400) + [(rng.choice(seen), 0, rng.choice(new), rng.integers(45, 50)) for _ in range(400)]
for n, q in (('train', tr), ('valid', va), ('test', te)):
    open(f'{out}/{n}.txt', 'w').write('\n'.join(' '.join(map(str, x)) for x in q) + '\n')
print('random dataset written: 260 entities (60 never in train), 5 relations; chance MRR ~ 0.024')
EOF
for v in T base A C F R RL plus; do
  d=KARMA-$v; p=karma$(echo $v | tr 'A-Z' 'a-z')
  echo "=== $d ==="
  ( cd $M/$d && python train.py --dataset RANDOM --data_dir $W/data --save_dir $W/ckpt --epochs 30 --eval_every 5 --gpu $G > $W/res/train_$p.log 2>&1 ) || { echo "train FAILED $d"; continue; }
  for h in train observed; do
    ( cd $M/$d && python $S/eval_fair_any.py --dataset RANDOM --data_dir $W/data --ckpt $W/ckpt/${p}_RANDOM_best.pt --history $h --gpu $G --out $W/res/${p}_$h.json > $W/res/eval_${p}_$h.log 2>&1 ) || echo "eval FAILED $d $h"
  done
done
echo CONTROL_DONE
