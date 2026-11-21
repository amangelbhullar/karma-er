"""
baseline_recency.py -- model-free heuristic baselines for the KARMA-X demo table.

Same protocol as eval_fair.py: object prediction (s, r, ?, t), history = all
facts strictly before t (train+valid+test), time-aware filtered ranking,
ties at average rank, seen/unseen split = entity occurs in the training split
(with the same seeded valid carve-out as data_loader.py when valid.txt is missing).

  freq_sr  : score(o) = #times (s, r, o) occurred before t
  freq_sr+pop : freq_sr, ties broken by #times (*, r, o) occurred before t
                (entity-agnostic popularity: can rank never-trained entities)

Usage:
  python baseline_recency.py --dataset ICEWS18-renet --data_dir /storage/bhull113/rhgnn/data \
      --out /storage/bhull113/karma_fair_eval/recency_ICEWS18-renet.json
"""
import os, json, argparse
from collections import defaultdict
import numpy as np


def load(p):
    if not os.path.exists(p):
        return np.zeros((0, 4), dtype=np.int64)
    rows = []
    with open(p) as f:
        for line in f:
            x = line.split()
            if len(x) >= 4:
                try:
                    rows.append([int(x[0]), int(x[1]), int(x[2]), int(x[3])])
                except ValueError:
                    pass
    return np.array(rows, dtype=np.int64).reshape(-1, 4)


def metrics(r):
    if len(r) == 0:
        return {'n': 0, 'mrr': 0.0, 'hits1': 0.0, 'hits3': 0.0, 'hits10': 0.0}
    r = np.asarray(r, dtype=np.float64)
    return {'n': int(len(r)), 'mrr': float(np.mean(1 / r)), 'hits1': float(np.mean(r <= 1)),
            'hits3': float(np.mean(r <= 3)), 'hits10': float(np.mean(r <= 10))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--data_dir', default='/storage/bhull113/rhgnn/data')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    d = os.path.join(a.data_dir, a.dataset)
    tr, va, te = load(f'{d}/train.txt'), load(f'{d}/valid.txt'), load(f'{d}/test.txt')
    if len(va) == 0:                       # mirror data_loader.py + train.py seeding
        np.random.seed(a.seed)
        idx = np.random.permutation(len(tr)); cut = int(len(tr) * 0.9)
        va, tr = tr[idx[cut:]], tr[idx[:cut]]
    allq = np.concatenate([tr, va, te])
    E = int(allq[:, [0, 2]].max()) + 1
    R = int(allq[:, 1].max()) + 1
    seen = np.zeros(E, bool); seen[np.unique(tr[:, [0, 2]])] = True

    filt = defaultdict(set)
    for s, r, o, t in allq:
        filt[(s, r, t)].add(o)

    allq = allq[np.argsort(allq[:, 3], kind='stable')]
    sr_cnt = defaultdict(lambda: defaultdict(int))     # (s,r) -> {o: count}
    pop = np.zeros((R, E), dtype=np.float64)           # (r, o) counts
    ptr = 0
    ranks = {'freq_sr': ([], []), 'freq_sr+pop': ([], [])}   # (seen list, unseen list)

    def rank_of(sc, o, others):
        if others:
            sc[others] = -np.inf
        tgt = sc[o]
        gt = int((sc > tgt).sum()); eq = int((sc == tgt).sum()) - 1
        return gt + 1 + eq / 2.0

    for t in np.unique(te[:, 3]):
        while ptr < len(allq) and allq[ptr, 3] < t:      # history strictly before t
            s, r, o, _ = allq[ptr]
            sr_cnt[(s, r)][o] += 1; pop[r, o] += 1; ptr += 1
        for s, r, o, _ in te[te[:, 3] == t]:
            others = sorted(filt[(s, r, t)] - {o})
            base = np.zeros(E)
            for oo, c in sr_cnt.get((s, r), {}).items():
                base[oo] = c
            bucket = 0 if seen[o] else 1
            ranks['freq_sr'][bucket].append(rank_of(base.copy(), o, others))
            p = pop[r]; scale = p.max() + 1.0
            ranks['freq_sr+pop'][bucket].append(rank_of(base * scale + p, o, others))

    res = {'dataset': a.dataset,
           'protocol': 'object prediction; history before t; time-aware filtered; avg-rank ties',
           'num_test_queries': int(len(te))}
    for k, (rs, ru) in ranks.items():
        res[k] = {'all': metrics(rs + ru), 'seen_obj': metrics(rs), 'unseen_obj': metrics(ru)}
    print(json.dumps(res, indent=2))
    out = a.out or f'recency_{a.dataset}.json'
    json.dump(res, open(out, 'w'), indent=2)
    print('Saved', out)


if __name__ == '__main__':
    main()
