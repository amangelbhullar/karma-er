"""
tipnn_acis.py -- TiPNN under the same protocol as KARMA-v2 (ACIS).
Reuses TiPNN's model, graph builder, negative sampler and loss from
tipnn_clean.py UNCHANGED; only data split and evaluation differ:
  * same split as KARMA (seed-42 10% valid carve-out when valid.txt is missing)
  * evaluation on TEST timestamps only; history = facts from the previous
    history_len timestamps of the full timeline (train+valid+test), as TiPNN does
  * time-aware filtered ranking with AVERAGE-rank ties
  * every unseen-answer query is evaluated; seen-answer queries are sampled
    (up to --max_seen) because TiPNN scores all entities per query
  * buckets by #facts the answer had before t (both directions, capped 20)
usage (from the TiPNN_clean folder):
  python tipnn_acis.py --dataset ICEWS18-renet --gpu 0 --out /storage/bhull113/karma_fair_eval/tipnn/ICEWS18-renet.json
"""
import os, sys, json, argparse
from collections import defaultdict
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from tipnn_clean import TiPNN, build_graph, neg_sample

def load(p):
    if not os.path.exists(p): return np.zeros((0, 4), np.int64)
    return np.array([[int(x) for x in l.split()[:4]] for l in open(p) if len(l.split()) >= 4], np.int64).reshape(-1, 4)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True); ap.add_argument('--data_dir', default='/storage/bhull113/rhgnn/data')
    ap.add_argument('--out', required=True); ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--history_len', type=int, default=8)
    ap.add_argument('--input_dim', type=int, default=64); ap.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 64, 64, 64])
    ap.add_argument('--negative_num', type=int, default=32); ap.add_argument('--adv_temp', type=float, default=0.5)
    ap.add_argument('--message_func', default='distmult'); ap.add_argument('--max_seen', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    dev = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.enabled = False
    d = os.path.join(a.data_dir, a.dataset)
    tr, va, te = load(f'{d}/train.txt'), load(f'{d}/valid.txt'), load(f'{d}/test.txt')
    if len(va) == 0:
        np.random.seed(42); i = np.random.permutation(len(tr)); c = int(len(tr) * .9); va, tr = tr[i[c:]], tr[i[:c]]
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    allq = np.concatenate([tr, va, te]); N = int(allq[:, [0, 2]].max()) + 1; R = int(allq[:, 1].max()) + 1
    seen = np.zeros(N, bool); seen[np.unique(tr[:, [0, 2]])] = True
    filt = defaultdict(set)
    for s, r, o, t in allq: filt[(s, r, t)].add(o)
    ev = np.concatenate([allq[:, [0, 3]], allq[:, [2, 3]]]); TM = int(allq[:, 3].max()) + 2
    key = np.sort(ev[:, 0] * TM + ev[:, 1])
    prior = lambda e, t: min(20, int(np.searchsorted(key, e * TM + t) - np.searchsorted(key, e * TM)))
    times = np.unique(allq[:, 3]); by_t = {t: allq[allq[:, 3] == t] for t in times}
    tr_times = np.unique(tr[:, 3]); tr_by_t = [tr[tr[:, 3] == t] for t in tr_times]
    print(f'[data] N={N} R={R} train/valid/test={len(tr)}/{len(va)}/{len(te)} unseen-answer test queries={int((~seen[te[:, 2]]).sum())}', flush=True)

    model = TiPNN(a.input_dim, a.hidden_dims, N, R, a.message_func, history_len=a.history_len).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=5e-6)
    for ep in range(a.epochs):                         # TiPNN's own training loop, on the training split
        model.train(); losses = []
        idxs = list(range(1, len(tr_by_t))); np.random.shuffle(idxs)
        for idx in idxs:
            fut = tr_by_t[idx]; hist = np.concatenate(tr_by_t[max(0, idx - a.history_len):idx])
            g = build_graph(hist, N, R, dev); perm = np.random.permutation(len(fut))
            for b in range(0, len(fut), a.batch_size):
                trip = neg_sample(fut[perm[b:b + a.batch_size]], N, a.negative_num, dev)
                loss = model.get_loss(model(g, trip), a.adv_temp, a.negative_num)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                losses.append(loss.item())
        print(f'epoch {ep + 1}/{a.epochs} loss {np.mean(losses):.4f}', flush=True)

    model.eval(); rng = np.random.default_rng(0)
    seen_q = np.where(seen[te[:, 2]])[0]
    keep = set(np.where(~seen[te[:, 2]])[0].tolist()) | set(rng.choice(seen_q, min(a.max_seen, len(seen_q)), replace=False).tolist())
    Rk = {'seen': [], 'unseen': []}; B = defaultdict(list); tpos = {t: k for k, t in enumerate(times)}
    with torch.no_grad():
        for t in np.unique(te[:, 3]):
            qi = [i for i in np.where(te[:, 3] == t)[0] if i in keep]
            if not qi: continue
            k = tpos[t]; hist_t = times[max(0, k - a.history_len):k]
            hist = np.concatenate([by_t[x] for x in hist_t]) if len(hist_t) else np.zeros((0, 4), np.int64)
            g = build_graph(hist, N, R, dev)
            for i in qi:
                s, r, o, _ = te[i]
                trip = torch.stack([torch.full((N,), int(s), device=dev), torch.full((N,), int(r), device=dev),
                                    torch.arange(N, device=dev)], -1).unsqueeze(0)
                sc = model(g, trip).squeeze(0).float()
                oth = sorted(filt[(s, r, t)] - {o})
                if oth: sc[torch.as_tensor(oth, device=dev)] = float('-inf')
                tg = sc[o]; rk = int((sc > tg).sum()) + 1 + (int((sc == tg).sum()) - 1) / 2
                if seen[o]: Rk['seen'].append(rk)
                else:
                    Rk['unseen'].append(rk); p = prior(o, t)
                    B['0' if p == 0 else '1-2' if p <= 2 else '3-5' if p <= 5 else '6+'].append(rk)
    m = lambda r: {'n': len(r), 'mrr': float(np.mean(1 / np.array(r))) if r else 0.0, 'hits10': float(np.mean(np.array(r) <= 10)) if r else 0.0}
    res = {'method': 'TiPNN', 'dataset': a.dataset, 'protocol': 'ACIS-compatible (entity-independent), avg-rank ties, test only',
           'seen_obj': m(Rk['seen']), 'unseen_obj': m(Rk['unseen']), 'unseen_by_prior_facts': {k: m(v) for k, v in sorted(B.items())},
           'note': f'all unseen-answer queries; seen-answer queries sampled (max {a.max_seen})', 'args': vars(a)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, 'w'), indent=2)
    print(f"[TiPNN {a.dataset}] seen {res['seen_obj']['mrr']:.4f} (n={res['seen_obj']['n']}) | unseen {res['unseen_obj']['mrr']:.4f} (n={res['unseen_obj']['n']}) | "
          + ', '.join(f"{k}:{v['mrr']:.3f}" for k, v in res['unseen_by_prior_facts'].items()), flush=True)

if __name__ == '__main__':
    main()
