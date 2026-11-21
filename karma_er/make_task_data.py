"""
make_task_data.py -- turn downstream tasks into cold-start TKG datasets
(train.txt / valid.txt / test.txt with "s r o t" lines) so KARMA-v2, the
counting baseline and ACIS evaluation apply unchanged.

  movielens : users + items are entities; ratings are timed facts
              (relations: 0 like >=4, 1 neutral =3, 2 dislike <=2; 3-5 = inverses).
              Weekly timestamps; temporal split 70/10/20 by time, so users and
              items that first appear after the cutoff are cold-start entities.
  string    : human STRING PPI (combined_score >= min_score). 10% of proteins
              held out for test, 5% for valid. Each held-out protein gets k
              support edges (its history, t=10, in valid.txt) and the rest of its
              edges to training proteins are queries (t=11). Training edges are
              spread over pseudo-timestamps 0..9 only to create context/target
              episodes; STRING has no real time.

usage:
  python make_task_data.py movielens --src /storage/bhull113/karma_rec/data --out .../data_tasks/ML-cold
  python make_task_data.py string --src /storage/bhull113/protein.links.v12.0.txt.gz --out .../data_tasks/STRING-PPI
"""
import os, sys, gzip, argparse
import numpy as np


def write(out, name, q):
    np.savetxt(os.path.join(out, f'{name}.txt'), np.asarray(q, np.int64).reshape(-1, 4), fmt='%d')


def find_ratings(src):
    pref = ['ratings.dat', 'ratings.csv', 'u.data']
    hits = []
    for root, _, files in os.walk(src):
        for f in files:
            if f in pref:
                hits.append(os.path.join(root, f))
    if not hits:
        sys.exit(f'no MovieLens ratings file (ratings.dat / ratings.csv / u.data) under {src}')
    hits.sort(key=lambda p: (pref.index(os.path.basename(p)), -os.path.getsize(p)))
    return hits[0]


def movielens(a):
    path = find_ratings(a.src); print('ratings file:', path)
    rows = []
    with open(path, encoding='latin-1') as f:
        for i, line in enumerate(f):
            if path.endswith('.csv') and i == 0 and not line[0].isdigit():
                continue
            x = line.strip().replace('::', '\t').replace(',', '\t').split('\t')
            if len(x) >= 4:
                rows.append((int(x[0]), int(x[1]), float(x[2]), int(float(x[3]))))
    r = np.array(rows, dtype=np.float64)
    if a.max_facts and len(r) > a.max_facts:
        r = r[np.sort(np.random.default_rng(0).choice(len(r), a.max_facts, replace=False))]
    users, ui = np.unique(r[:, 0], return_inverse=True)
    items, ii = np.unique(r[:, 1], return_inverse=True)
    U = len(users)
    rel = np.where(r[:, 2] >= 4, 0, np.where(r[:, 2] >= 3, 1, 2))
    t = (r[:, 3] // (86400 * a.days_per_step)).astype(np.int64); t -= t.min()
    q = np.stack([ui, rel, ii + U, t], 1).astype(np.int64)
    inv = q[:, [2, 1, 0, 3]].copy(); inv[:, 1] += 3
    q = np.concatenate([q, inv]); q = q[np.argsort(q[:, 3], kind='stable')]
    c1, c2 = np.quantile(q[:, 3], [0.7, 0.8])
    tr, va, te = q[q[:, 3] < c1], q[(q[:, 3] >= c1) & (q[:, 3] < c2)], q[q[:, 3] >= c2]
    os.makedirs(a.out, exist_ok=True)
    for n, x in (('train', tr), ('valid', va), ('test', te)):
        write(a.out, n, x)
    seen = set(tr[:, 0]) | set(tr[:, 2])
    new_u = len(set(te[te[:, 0] < U, 0]) - seen); new_i = len(set(te[te[:, 0] >= U, 0]) - seen)
    print(f'ML: users {U} items {len(items)} facts {len(r)} | train/valid/test {len(tr)}/{len(va)}/{len(te)} '
          f'| cold-start users in test {new_u}, items {new_i} | steps of {a.days_per_step} days')


def string(a):
    tax = f'{a.taxid}.'; ids = {}; edges = []; started = False
    op = gzip.open if a.src.endswith('.gz') else open
    with op(a.src, 'rt') as f:
        for line in f:
            if not line.startswith(tax):
                if started:
                    break                              # file is grouped by organism
                continue
            started = True
            p1, p2, sc = line.split()[:3]; sc = int(sc)
            if sc < a.min_score or p1 >= p2:           # keep each undirected edge once
                continue
            for p in (p1, p2):
                if p not in ids:
                    ids[p] = len(ids)
            edges.append((ids[p1], 0 if sc < 900 else 1, ids[p2]))
    if not edges:
        sys.exit(f'no edges found for taxid {a.taxid} with score >= {a.min_score}')
    e = np.array(edges, np.int64); N = len(ids)
    rng = np.random.default_rng(a.seed); perm = rng.permutation(N)
    n_te, n_va = int(0.10 * N), int(0.05 * N)
    role = np.zeros(N, np.int8); role[perm[:n_te]] = 2; role[perm[n_te:n_te + n_va]] = 1
    tr, sup, vq, tq = [], [], [], []
    both = lambda s, r, o, t: [(s, r, o, t), (o, r, s, t)]
    for s, r, o in e:
        rs, ro = role[s], role[o]
        if rs == 0 and ro == 0:
            tr += both(s, r, o, int(rng.integers(10)))
    for p in perm[:n_te + n_va]:
        nb = e[((e[:, 0] == p) & (role[e[:, 2]] == 0)) | ((e[:, 2] == p) & (role[e[:, 0]] == 0))]
        if len(nb) <= a.k_support:
            continue
        nb = nb[rng.permutation(len(nb))]
        for s, r, o in nb[:a.k_support]:
            sup += both(s, r, o, 10)
        tgt = tq if role[p] == 2 else vq
        for s, r, o in nb[a.k_support:]:
            other = o if s == p else s
            tgt.append((other, r, p, 11))               # object = held-out protein (cold start)
            tgt.append((p, r, other, 11))               # subject = held-out protein
    os.makedirs(a.out, exist_ok=True)
    write(a.out, 'train', tr); write(a.out, 'valid', sup + vq); write(a.out, 'test', tq)
    open(os.path.join(a.out, 'proteins.tsv'), 'w').write(
        '\n'.join(f'{i}\t{p}\t{["train", "valid", "test"][role[i]]}' for p, i in ids.items()))
    print(f'STRING {a.taxid}: proteins {N} edges(>= {a.min_score}) {len(e)} | train {len(tr)} '
          f'support {len(sup)} valid-queries {len(vq)} test-queries {len(tq)} | k_support={a.k_support}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('task', choices=['movielens', 'string'])
    p.add_argument('--src', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--days_per_step', type=int, default=7)
    p.add_argument('--max_facts', type=int, default=0)
    p.add_argument('--taxid', default='9606')
    p.add_argument('--min_score', type=int, default=700)
    p.add_argument('--k_support', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    {'movielens': movielens, 'string': string}[a.task](a)
