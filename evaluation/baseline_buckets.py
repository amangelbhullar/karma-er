import os, sys, json
from collections import defaultdict
import numpy as np
d = sys.argv[1]; D = (sys.argv[2] if len(sys.argv) > 2 else '/storage/bhull113/rhgnn/data') + f'/{d}'
def load(p):
    if not os.path.exists(p): return np.zeros((0, 4), np.int64)
    return np.array([[int(x) for x in l.split()[:4]] for l in open(p) if len(l.split()) >= 4], np.int64).reshape(-1, 4)
tr, va, te = load(f'{D}/train.txt'), load(f'{D}/valid.txt'), load(f'{D}/test.txt')
if len(va) == 0:
    np.random.seed(42); i = np.random.permutation(len(tr)); c = int(len(tr) * .9); va, tr = tr[i[c:]], tr[i[:c]]
allq = np.concatenate([tr, va, te]); E = int(allq[:, [0, 2]].max()) + 1; R = int(allq[:, 1].max()) + 1
seen = np.zeros(E, bool); seen[np.unique(tr[:, [0, 2]])] = True
filt = defaultdict(set)
for s, r, o, t in allq: filt[(s, r, t)].add(o)
allq = allq[np.argsort(allq[:, 3], kind='stable')]
sr = defaultdict(lambda: defaultdict(int)); pop = np.zeros((R, E)); cnt = np.zeros(E, int); ptr = 0
B = defaultdict(list)
for t in np.unique(te[:, 3]):
    while ptr < len(allq) and allq[ptr, 3] < t:
        s, r, o, _ = allq[ptr]; sr[(s, r)][o] += 1; pop[r, o] += 1; cnt[s] += 1; cnt[o] += 1; ptr += 1
    for s, r, o, _ in te[te[:, 3] == t]:
        if seen[o]: continue
        base = np.zeros(E)
        for oo, c in sr.get((s, r), {}).items(): base[oo] = c
        sc = base * (pop[r].max() + 1) + pop[r]
        oth = sorted(filt[(s, r, t)] - {o})
        if oth: sc[oth] = -np.inf
        g = (sc > sc[o]).sum(); e = (sc == sc[o]).sum() - 1
        k = min(cnt[o], 20); b = '0' if k == 0 else '1-2' if k <= 2 else '3-5' if k <= 5 else '6+'
        B[b].append(g + 1 + e / 2)
print(d, 'freq+pop unseen by prior facts:', ', '.join(f"{b}:{np.mean(1/np.array(v)):.3f}(n={len(v)})" for b, v in sorted(B.items())))
