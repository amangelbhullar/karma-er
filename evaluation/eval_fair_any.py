"""
eval_fair_any.py -- minimal-change fair evaluation for ANY KARMA variant.

Identical to each variant's own evaluate.py EXCEPT one thing:
  original : every candidate scored from the table (entity_table, or
             neighbor_embeddings for KARMA-base); then ONLY the true unseen
             object's score is replaced by its inductive score.
  fair     : every unseen candidate gets its inductive score (from its own
             history before t); seen candidates keep the table score.
Subject = entity_embed(subject history before t, t[, s]) as in the originals
(KARMA-R paths are not used, as in the original object scoring).
Ties at average rank; time-aware filtered; object prediction.

Run from the variant's code folder, e.g.
  cd /storage/bhull113/karma/models/KARMA-C
  python eval_fair_any.py --dataset ICEWS18-renet --ckpt .../karmac_ICEWS18-renet_best.pt
"""
import os, sys, json, inspect, argparse
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from model import KARMA                                   # variant's own model
from data_loader import TKGDataset, build_history_index, build_filter_dict_normalized


def first(x):
    return x[0] if isinstance(x, (tuple, list)) else x


def hist_at(t, ents, ts, rs, nbs, start, t_min, t_range, L):
    n = len(ents)
    R = np.zeros((n, L), np.int64); N = np.zeros((n, L), np.int64)
    T = np.zeros((n, L), np.float32); Ln = np.zeros(n, np.int64)
    for i, e in enumerate(ents):
        if e + 1 >= len(start):
            continue
        lo, hi = int(start[e]), int(start[e + 1])
        if hi <= lo:
            continue
        v = np.searchsorted(ts[lo:hi], t, side='left')
        if v == 0:
            continue
        s0 = max(0, v - L); k = v - s0
        R[i, :k] = rs[lo + s0:lo + v]; N[i, :k] = nbs[lo + s0:lo + v]
        T[i, :k] = (ts[lo + s0:lo + v].astype(np.float32) - t_min) / t_range
        Ln[i] = k
    return R, N, T, Ln


def metrics(r):
    if len(r) == 0:
        return {'n': 0, 'mrr': 0.0, 'hits1': 0.0, 'hits3': 0.0, 'hits10': 0.0}
    r = np.asarray(r, np.float64)
    return {'n': int(len(r)), 'mrr': float(np.mean(1 / r)), 'hits1': float(np.mean(r <= 1)),
            'hits3': float(np.mean(r <= 3)), 'hits10': float(np.mean(r <= 10))}


def build_model(ck, E, Rn, device):
    a = dict(ck.get('args', {}))
    sig = inspect.signature(KARMA.__init__).parameters
    kw = {k: a[k] for k in sig if k in a and k not in ('self', 'num_entities', 'num_relations')}
    missing = [k for k, p in sig.items() if k not in ('self', 'num_entities', 'num_relations')
               and p.default is inspect._empty and k not in kw]
    if missing:
        raise SystemExit(f'checkpoint args missing required model params: {missing}')
    m = KARMA(num_entities=E, num_relations=Rn, **kw).to(device)
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m, kw


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--data_dir', default='/storage/bhull113/rhgnn/data')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--history', choices=['train', 'observed'], default='observed')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--out', default=None)
    args = p.parse_args()
    dev = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    a = ck.get('args', {})
    L = int(a.get('max_history', 20))
    np.random.seed(a.get('seed', 42))
    data = TKGDataset(os.path.join(args.data_dir, args.dataset), max_history=L)
    E, Rn = data.num_entities, data.num_relations
    model, kw = build_model(ck, E, Rn, dev)

    has_table = hasattr(model, 'entity_table')
    table = model.entity_table.weight if has_table else model.neighbor_embeddings
    table = model.layer_norm(table).detach()
    takes_ids = 'entity_ids' in inspect.signature(model.entity_embed).parameters and has_table

    seen = np.zeros(E, bool); seen[np.unique(data.train[:, [0, 2]])] = True
    unseen = np.where(~seen)[0]
    allq = np.concatenate([data.train, data.valid, data.test])
    filt = build_filter_dict_normalized(allq, data.t_min, data.t_range)
    if args.history == 'observed':
        ts, rs, nbs, st, _ = build_history_index(allq)
    else:
        ts, rs, nbs, st = data.ts, data.rs, data.nbs, data.ent_start

    T_ = lambda x: torch.as_tensor(x, device=dev)
    out = {'fair': ([], []), 'leaky': ([], [])}
    for t in np.unique(data.test[:, 3]):
        q = data.test[data.test[:, 3] == t]
        tau = (float(t) - data.t_min) / data.t_range
        sh = hist_at(t, q[:, 0], ts, rs, nbs, st, data.t_min, data.t_range, L)
        tt = torch.full((len(q),), tau, device=dev)
        sargs = [T_(x) for x in sh] + [tt]
        s_emb = first(model.entity_embed(*sargs, T_(q[:, 0]))) if takes_ids \
            else first(model.entity_embed(*sargs))
        sr = s_emb * model.relation_embeddings[T_(q[:, 1])]
        base = sr @ table.t()                                        # (B, E)
        fair = base.clone()
        if len(unseen):
            uh = hist_at(t, unseen, ts, rs, nbs, st, data.t_min, data.t_range, L)
            ind = model.layer_norm(first(model.inductive_embed(
                *[T_(x) for x in uh], torch.full((len(unseen),), tau, device=dev))))
            fair[:, T_(unseen)] = sr @ ind.t()
        oh = hist_at(t, q[:, 2], ts, rs, nbs, st, data.t_min, data.t_range, L)
        o_ind = model.layer_norm(first(model.inductive_embed(*[T_(x) for x in oh], tt)))
        o_ind_sc = (sr * o_ind).sum(-1)
        key = round(tau, 6)
        for i, (s, r, o, _) in enumerate(q):
            others = sorted(filt.get((int(s), int(r), key), set()) - {int(o)})
            b = 0 if seen[o] else 1
            for name, mat in (('fair', fair), ('leaky', base)):
                sc = mat[i].clone()
                if name == 'leaky' and not seen[o]:
                    sc[o] = o_ind_sc[i]
                if others:
                    sc[T_(others)] = float('-inf')
                tg = sc[o]
                gt = int((sc > tg).sum()); eq = int((sc == tg).sum()) - 1
                out[name][b].append(gt + 1 + eq / 2.0)
        print(f'[t={t}] q={len(q)}', flush=True)

    res = {'dataset': args.dataset, 'ckpt': args.ckpt, 'variant_dir': os.getcwd(),
           'history': args.history, 'model_kwargs': kw, 'n_unseen_entities': int(len(unseen))}
    for name, (rs_, ru) in out.items():
        res[name] = {'all': metrics(rs_ + ru), 'seen_obj': metrics(rs_), 'unseen_obj': metrics(ru)}
    print(json.dumps({k: res[k] for k in ('dataset', 'fair', 'leaky')}, indent=1))
    o = args.out or os.path.splitext(args.ckpt)[0] + f'_fairany_{args.history}.json'
    json.dump(res, open(o, 'w'), indent=2)
    print('Saved', o)


if __name__ == '__main__':
    main()
