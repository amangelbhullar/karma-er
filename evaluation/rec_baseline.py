"""Cold-start recommendation baselines for queries whose SUBJECT is new (not in training):
  popularity : rank objects by #(*, r, o) before t  (most popular items for that relation)
  counting   : freq+pop as elsewhere ((s,r,o) counts are zero for a new user, so this ~ popularity)
Same split, filtering and average-rank ties as KARMA. Reports all relations and relation 0 (like).
usage: python rec_baseline.py <dataset> <data_dir> <out.json>"""
import os, sys, json
from collections import defaultdict
import numpy as np
d, D, out = sys.argv[1], sys.argv[2], sys.argv[3]
L = lambda p: np.array([[int(x) for x in l.split()[:4]] for l in open(p) if len(l.split()) >= 4], np.int64).reshape(-1, 4) if os.path.exists(p) else np.zeros((0, 4), np.int64)
tr, va, te = L(f'{D}/{d}/train.txt'), L(f'{D}/{d}/valid.txt'), L(f'{D}/{d}/test.txt')
if len(va) == 0:
    np.random.seed(42); i = np.random.permutation(len(tr)); c = int(len(tr) * .9); va, tr = tr[i[c:]], tr[i[:c]]
allq = np.concatenate([tr, va, te]); E = int(allq[:, [0, 2]].max()) + 1; R = int(allq[:, 1].max()) + 1
seen = np.zeros(E, bool); seen[np.unique(tr[:, [0, 2]])] = True
filt = defaultdict(set)
for s, r, o, t in allq: filt[(s, r, t)].add(o)
allq = allq[np.argsort(allq[:, 3], kind='stable')]; pop = np.zeros((R, E)); ptr = 0
res = {'all': [], 'r0': []}
for t in np.unique(te[:, 3]):
    while ptr < len(allq) and allq[ptr, 3] < t:
        pop[allq[ptr, 1], allq[ptr, 2]] += 1; ptr += 1
    for s, r, o, _ in te[te[:, 3] == t]:
        if seen[s]: continue
        sc = pop[r].copy(); oth = sorted(filt[(s, r, t)] - {o})
        if oth: sc[oth] = -np.inf
        rk = (sc > sc[o]).sum() + 1 + ((sc == sc[o]).sum() - 1) / 2
        res['all'].append(rk)
        if r == 0: res['r0'].append(rk)
m = lambda r: {'n': len(r), 'mrr': float(np.mean(1 / np.array(r))) if r else 0.0, 'hits10': float(np.mean(np.array(r) <= 10)) if r else 0.0}
j = {'dataset': d, 'method': 'popularity (cold-start subject)', 'unseen_subj': m(res['all']), 'unseen_subj_r0': m(res['r0'])}
json.dump(j, open(out, 'w'), indent=2)
print(f"[popularity {d}] new-subject all: MRR {j['unseen_subj']['mrr']:.4f} H@10 {j['unseen_subj']['hits10']:.3f} (n={j['unseen_subj']['n']}) | relation 0: MRR {j['unseen_subj_r0']['mrr']:.4f} H@10 {j['unseen_subj_r0']['hits10']:.3f} (n={j['unseen_subj_r0']['n']})")
