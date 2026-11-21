"""Adamic-Adar common-neighbour baseline for the STRING held-out-protein task, same protocol as
KARMA-ER: graph = all facts strictly before the query time (training edges + support edges),
undirected; score(s, v) = sum over shared neighbours w of 1 / log(deg(w)); time-aware filtered
ranking, average-rank ties, seen/unseen split by training entities.
usage: python common_neighbour.py <data_dir>/STRING-PPI <out.json>"""
import os, sys, json
from collections import defaultdict
import numpy as np, scipy.sparse as sp
D, OUT = sys.argv[1], sys.argv[2]
L = lambda p: np.array([[int(x) for x in l.split()[:4]] for l in open(p) if len(l.split()) >= 4], np.int64).reshape(-1, 4)
tr, va, te = L(f'{D}/train.txt'), L(f'{D}/valid.txt'), L(f'{D}/test.txt')
allq = np.concatenate([tr, va, te]); N = int(allq[:, [0, 2]].max()) + 1
seen = np.zeros(N, bool); seen[np.unique(tr[:, [0, 2]])] = True
filt = defaultdict(set)
for s, r, o, t in allq: filt[(s, r, t)].add(o)
R = {'seen': [], 'unseen': []}
for t in np.unique(te[:, 3]):
    h = allq[allq[:, 3] < t]
    A = sp.coo_matrix((np.ones(2 * len(h)), (np.r_[h[:, 0], h[:, 2]], np.r_[h[:, 2], h[:, 0]])), shape=(N, N)).tocsr()
    A.data[:] = 1.0; A.sum_duplicates(); A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel(); w = np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 2)), 0.0)
    AW = A @ sp.diags(w)
    q = te[te[:, 3] == t]
    for c in range(0, len(q), 1024):
        qb = q[c:c + 1024]
        S = (AW[qb[:, 0]] @ A).toarray()
        for i, (s, r, o, _) in enumerate(qb):
            sc = S[i]; oth = sorted(filt[(s, r, t)] - {o})
            if oth: sc[oth] = -np.inf
            rk = (sc > sc[o]).sum() + 1 + ((sc == sc[o]).sum() - 1) / 2
            R['seen' if seen[o] else 'unseen'].append(rk)
m = lambda r: {'n': len(r), 'mrr': float(np.mean(1 / np.array(r))) if r else 0.0, 'hits10': float(np.mean(np.array(r) <= 10)) if r else 0.0}
res = {'method': 'Adamic-Adar common neighbours', 'dataset': os.path.basename(D.rstrip('/')),
       'seen_obj': m(R['seen']), 'unseen_obj': m(R['unseen'])}
json.dump(res, open(OUT, 'w'), indent=2)
print(f"[common-neighbour {res['dataset']}] seen MRR {res['seen_obj']['mrr']:.4f} | unseen MRR {res['unseen_obj']['mrr']:.4f} H@10 {res['unseen_obj']['hits10']:.3f} (n={res['unseen_obj']['n']})")
