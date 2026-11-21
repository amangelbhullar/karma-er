"""
multitq_qa.py -- answer MultiTQ questions with KARMA-ER (ACIS) and the counting baseline.
Each question -> query (e, r, ?, t); history = all KG facts strictly before t.
Rank of the best gold answer among all entities (other gold answers excluded,
average ties). Two parses: 'pred' (entity linker + relation classifier) and
'gold' (KG fact matching the question; upper bound for the KG-reasoning step).

usage: python multitq_qa.py --ckpt runs_tasks/MultiTQ-kg_none_s42_best.pt --gpu 0
"""
import os, sys, json, argparse
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import karma_v2x as K


def rank(sc, gold):
    g = sc[gold].max()
    m = torch.ones_like(sc, dtype=torch.bool); m[gold] = False
    o = sc[m]
    return float((o > g).sum()) + 1 + float((o == g).sum()) / 2.0


def metrics(r):
    r = np.array(r, float)
    if not len(r):
        return {'n': 0, 'mrr': 0.0, 'hits1': 0.0, 'hits10': 0.0}
    return {'n': int(len(r)), 'mrr': float(np.mean(1 / r)), 'hits1': float(np.mean(r <= 1)),
            'hits10': float(np.mean(r <= 10))}


@torch.no_grad()
def main(a):
    dev = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    d = os.path.join(a.data_dir, a.dataset)
    Q = json.load(open(os.path.join(d, 'qa_test.json')))
    data = K.Data(d)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); ar = ck['args']
    model = K.KarmaV2(ck['E'], ck['R'], ar['dim'], ar['heads'], ar['k_anchor'], ck['gap'], ar['dropout'],
                      ar.get('abl', 'none'), not ar.get('no_rec', False)).to(dev)
    model.load_state_dict(ck['state']); model.eval()
    H = K.History(data.allq, data.E, data.R); REC = K.Recency(data.allq, data.E, data.R)
    avail = K.T(data.seen, dev); unseen = np.where(~data.seen)[0]; tab = model.table()
    res = {}
    for parse in ('pred', 'gold'):
        items = []
        for x in Q['questions']:
            if parse == 'pred':
                items.append((x['e_pred'], x['r_pred'], x))
            elif x['e_gold'] is not None:
                items.append((x['e_gold'], x['r_gold'][0], x))
        by_t = defaultdict(list)
        for it in items:
            by_t[it[2]['t']].append(it)
        R = {m: defaultdict(list) for m in ('KARMA-ER', 'Counting')}
        for t, its in sorted(by_t.items()):
            for c in range(0, len(its), a.chunk):
                b = its[c:c + a.chunk]
                s = np.array([i[0] for i in b]); r = np.array([i[1] for i in b])
                C = tab.clone()
                if len(unseen):
                    h = H.get(unseen, t, ar['L'])
                    C[K.T(unseen, dev)] = model.induct(*[K.T(z, dev) for z in h], avail)
                sh = H.get(s, t, ar['L'])
                se = model.subject(K.T(s, dev), K.T(r, dev), [K.T(z, dev) for z in sh], avail)
                sc = (se * model.rel(K.T(r, dev))) @ C.t()
                Fsr, Fpop = REC.feats(s, r, t)
                Fsr, Fpop = K.T(Fsr, dev), K.T(Fpop, dev)
                sc = sc + model.rec_term(K.T(r, dev), Fsr, Fpop)
                cnt = Fsr * 1e6 + Fpop
                for i, (_, _, x) in enumerate(b):
                    g = K.T(x['answers'], dev)
                    keys = ['all'] + (['cold_entity'] if (x['cold_entity'] if parse == 'pred'
                                                          else x['cold_entity_gold']) else []) \
                        + (['cold_answer'] if x['cold_answer'] else [])
                    rk, rc = rank(sc[i], g), rank(cnt[i], g)
                    for k in keys:
                        R['KARMA-ER'][k].append(rk); R['Counting'][k].append(rc)
        res[parse] = {m: {k: metrics(v) for k, v in R[m].items()} for m in R}
        for m in R:
            print(f'[{parse} parse] {m:9s} ' + ' | '.join(
                f"{k}: H@1 {v['hits1']:.3f} MRR {v['mrr']:.3f} (n={v['n']})" for k, v in res[parse][m].items()),
                flush=True)
    res.update({'dataset': a.dataset, 'ckpt': a.ckpt, 'parser': Q['parser'], 't_cut': Q['t_cut'],
                'n_test_in_scope': Q['n_test_in_scope'], 'protocol': 'ACIS, history strictly before t'})
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f'qa_{a.dataset}_s{ar.get("seed", 42)}.json')
    json.dump(res, open(out, 'w'), indent=2); print('saved', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--dataset', default='MultiTQ-kg')
    p.add_argument('--data_dir', default='/storage/bhull113/karma_v2/data_tasks')
    p.add_argument('--out_dir', default='/storage/bhull113/karma_v2/runs_tasks')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--chunk', type=int, default=512)
    main(p.parse_args())
