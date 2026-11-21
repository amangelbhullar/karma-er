"""
KARMA-v2 + recency: v2 plus learned per-relation copy/recency terms\n(log-count of (s,r,o) and (*,r,o) before t, and a repeat indicator).\n\nKARMA-v2: cold-start TKG reasoning trained the way it is tested.

Changes vs KARMA-X:
  1. Episodic emergence training: each epoch a random 15% of training entities
     are treated as "pseudo-new" (table embedding hidden); they are encoded from
     history only and must be ranked against ALL other pseudo-new entities.
  2. Entity anchors: an entity's inductive code = multi-head attention over its
     events + attention over its top-k most similar *trained* entities.
  3. Neighbours without a usable trained embedding use a shared UNK vector
     (never an untrained table row).
  4. Subject query is conditioned on the query relation.
  5. Evaluation = ACIS (every unseen candidate scored inductively), online or
     train-only history, time-aware filtered, average-rank ties, results
     bucketed by #facts the unseen answer had before t.

Usage:
  python karma_v2.py train --dataset ICEWS18-renet --data_dir /storage/bhull113/rhgnn/data \
       --out_dir /storage/bhull113/karma_v2/runs --gpu 0
  python karma_v2.py eval  --dataset ICEWS18-renet --data_dir ... --ckpt .../ICEWS18-renet_best.pt --history online
"""
import os, json, time, argparse
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- data
def load(p):
    if not os.path.exists(p):
        return np.zeros((0, 4), np.int64)
    rows = []
    with open(p) as f:
        for line in f:
            x = line.split()
            if len(x) >= 4:
                try:
                    rows.append([int(x[0]), int(x[1]), int(x[2]), int(x[3])])
                except ValueError:
                    pass
    return np.array(rows, np.int64).reshape(-1, 4)


class Data:
    def __init__(self, d, seed=42):
        tr, va, te = load(f'{d}/train.txt'), load(f'{d}/valid.txt'), load(f'{d}/test.txt')
        if len(va) == 0:                                   # same carve-out as KARMA data_loader
            np.random.seed(seed)
            idx = np.random.permutation(len(tr)); cut = int(len(tr) * 0.9)
            va, tr = tr[idx[cut:]], tr[idx[:cut]]
        self.tr, self.va, self.te = tr, va, te
        allq = np.concatenate([tr, va, te])
        self.allq = allq
        self.E = int(allq[:, [0, 2]].max()) + 1
        self.R = int(allq[:, 1].max()) + 1
        self.seen = np.zeros(self.E, bool); self.seen[np.unique(tr[:, [0, 2]])] = True
        ut = np.unique(allq[:, 3])
        self.gap = float(np.median(np.diff(ut))) if len(ut) > 1 else 1.0
        self.filt = defaultdict(set)
        for s, r, o, t in allq:
            self.filt[(int(s), int(r), int(t))].add(int(o))


class History:
    """Both directions; inverse relation id = r + R. Vectorised lookup of the
    last L events strictly before t."""
    def __init__(self, quads, E, R):
        ents = np.concatenate([quads[:, 0], quads[:, 2]])
        rels = np.concatenate([quads[:, 1], quads[:, 1] + R])
        nbrs = np.concatenate([quads[:, 2], quads[:, 0]])
        ts = np.concatenate([quads[:, 3], quads[:, 3]])
        o = np.lexsort((ts, ents))
        self.ents, self.rels, self.nbrs, self.ts = ents[o], rels[o], nbrs[o], ts[o]
        self.TM = int(max(ts.max() if len(ts) else 0, 0)) + 2
        self.key = self.ents * self.TM + self.ts
        self.start = np.searchsorted(self.ents, np.arange(E + 1))

    def get(self, ents, t, L):
        ents = np.asarray(ents, np.int64)
        t = np.broadcast_to(np.asarray(t, np.int64), ents.shape)
        n = len(self.key)
        if n == 0:
            z = np.zeros((len(ents), L), np.int64)
            return z, z, z.astype(np.float32), z.astype(bool)
        v = np.searchsorted(self.key, ents * self.TM + np.minimum(t, self.TM - 1), 'left')
        lo = self.start[ents]
        idx = v[:, None] - L + np.arange(L)[None]
        mask = idx >= lo[:, None]
        ic = np.clip(idx, 0, n - 1)
        rel = np.where(mask, self.rels[ic], 0)
        nbr = np.where(mask, self.nbrs[ic], 0)
        dt = np.where(mask, (t[:, None] - self.ts[ic]).astype(np.float32), 0.0).astype(np.float32)
        return rel, nbr, dt, mask


class Recency:
    """Counts strictly before t: c_sr[o] = #(s, r, o) and c_pop[o] = #(*, r, o).
    Built from quads incl. inverse facts (r + R)."""
    def __init__(self, quads, E, R):
        self.E = E
        inv = quads[:, [2, 1, 0, 3]].copy(); inv[:, 1] += R
        q = np.concatenate([quads, inv]); q = q[np.argsort(q[:, 3], kind='stable')]
        self.sr, self.pop = defaultdict(lambda: ([], [])), defaultdict(lambda: ([], []))
        for s_, r_, o_, t_ in q:
            a = self.sr[(int(s_), int(r_))]; a[0].append(o_); a[1].append(t_)
            b = self.pop[int(r_)]; b[0].append(o_); b[1].append(t_)
        self.sr = {k: (np.array(v[0]), np.array(v[1])) for k, v in self.sr.items()}
        self.pop = {k: (np.array(v[0]), np.array(v[1])) for k, v in self.pop.items()}

    def feats(self, s, r, t):
        B = len(s); Fsr = np.zeros((B, self.E), np.float32); Fpop = np.zeros((B, self.E), np.float32)
        cache = {}
        for i in range(B):
            k = (int(s[i]), int(r[i]))
            if k in self.sr:
                o_, t_ = self.sr[k]; n = np.searchsorted(t_, t, 'left')
                if n: np.add.at(Fsr[i], o_[:n], 1.0)
            ri = int(r[i])
            if ri not in cache:
                v = np.zeros(self.E, np.float32)
                if ri in self.pop:
                    o_, t_ = self.pop[ri]; n = np.searchsorted(t_, t, 'left')
                    if n: v = np.bincount(o_[:n], minlength=self.E).astype(np.float32)
                cache[ri] = v
            Fpop[i] = cache[ri]
        return Fsr, Fpop


# ----------------------------------------------------------------------------- model
class KarmaV2(nn.Module):
    def __init__(self, E, R, d=200, heads=4, k_anchor=16, gap=1.0, dropout=0.1, abl='none', use_rec=True):
        super().__init__()
        self.E, self.R, self.d, self.k, self.gap = E, R, d, k_anchor, gap
        self.abl, self.use_rec = abl, use_rec
        self.ent = nn.Embedding(E, d)
        self.rel = nn.Embedding(2 * R, d)
        nn.init.xavier_uniform_(self.ent.weight); nn.init.xavier_uniform_(self.rel.weight)
        self.unk = nn.Parameter(torch.randn(d) * 0.1)
        self.cold = nn.Parameter(torch.randn(d) * 0.1)
        self.tproj = nn.Linear(8, d)
        self.ev = nn.Linear(3 * d, d)
        self.q_gen = nn.Parameter(torch.randn(d) * 0.1)
        self.q_rel = nn.Linear(d, d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.anc_q = nn.Linear(d, d)
        self.out = nn.Linear(2 * d, d)
        self.ln_i = nn.LayerNorm(d)
        self.ln_t = nn.LayerNorm(d)
        self.ln_s = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('freqs', torch.tensor([1.0, 2.0, 4.0, 8.0]))
        self.rec_w = nn.Embedding(2 * R, 3)                 # per-relation recency weights
        with torch.no_grad():
            self.rec_w.weight[:] = torch.tensor([1.0, 0.5, 1.0])

    def rec_term(self, r, Fsr, Fpop):
        if not self.use_rec:
            return 0.0
        w = self.rec_w(r)                                    # (B, 3)
        return (w[:, :1] * torch.log1p(Fsr) + w[:, 1:2] * torch.log1p(Fpop)
                + w[:, 2:3] * (Fsr > 0).float())

    def tfeat(self, dt):
        x = torch.log1p(dt.clamp(min=0) / self.gap).unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(x), torch.cos(x)], -1)

    def table(self):
        return self.ln_t(self.ent.weight)

    def induct(self, rel, nbr, dt, mask, avail, query=None, chunk=4096):
        outs = []
        for a in range(0, rel.shape[0], chunk):
            b = slice(a, a + chunk)
            outs.append(self._induct(rel[b], nbr[b], dt[b], mask[b], avail,
                                     None if query is None else query[b]))
        return torch.cat(outs, 0)

    def _induct(self, rel, nbr, dt, mask, avail, query):
        B = rel.shape[0]
        nv = self.ent(nbr) if self.abl == 'nounk' else \
            torch.where(avail[nbr].unsqueeze(-1), self.ent(nbr), self.unk.expand(*nbr.shape, self.d))
        ev = F.relu(self.ev(torch.cat([self.rel(rel), nv, self.tproj(self.tfeat(dt))], -1)))
        q = self.q_gen.expand(B, self.d) if query is None else query
        has = mask.any(1)
        kpm = ~mask
        kpm[~has, -1] = False                       # avoid all-masked rows (overridden below)
        h, _ = self.attn(q.unsqueeze(1), ev, ev, key_padding_mask=kpm, need_weights=False)
        h = torch.where(has.unsqueeze(-1), h[:, 0], self.cold.expand(B, self.d))
        if self.abl == 'noanchor':
            return self.ln_i(self.out(self.drop(torch.cat([h, torch.zeros_like(h)], -1))))
        tab = self.ent.weight
        sim = (self.anc_q(h) @ tab.t()) / self.d ** 0.5
        sim = sim.masked_fill(~avail.unsqueeze(0), float('-inf'))
        top, idx = sim.topk(min(self.k, int(avail.sum())), dim=1)
        anc = (F.softmax(top, 1).unsqueeze(-1) * tab[idx]).sum(1)
        return self.ln_i(self.out(self.drop(torch.cat([h, anc], -1))))

    def subject(self, s, r, hist, avail):
        rel, nbr, dt, mask = hist
        q = self.q_rel(self.rel(r))
        ind = self.induct(rel, nbr, dt, mask, avail, query=q)
        tab = self.ent(s) * avail[s].unsqueeze(-1).float()
        return self.ln_s(ind + tab)


def T(x, dev):
    return torch.as_tensor(x, device=dev)


# ----------------------------------------------------------------------------- eval (ACIS)
@torch.no_grad()
def evaluate(model, data, quads, hist_index, dev, L=20, max_queries=None, rec=None):
    model.eval()
    avail = T(data.seen, dev)
    unseen = np.where(~data.seen)[0]
    tab = model.table()
    ranks, buckets = {'seen': [], 'unseen': []}, defaultdict(list)
    subj = {'unseen_subj': [], 'unseen_subj_r0': []}   # new SUBJECT (e.g. cold-start user), any / relation 0
    n_done = 0
    chunks = []                                       # score at most 4096 queries at once
    for t in np.unique(quads[:, 3]):
        qa = quads[quads[:, 3] == t]
        chunks += [(t, qa[c:c + 4096]) for c in range(0, len(qa), 4096)]
    for t, q in chunks:
        C = tab.clone()
        if len(unseen):
            h = hist_index.get(unseen, t, L)
            C[T(unseen, dev)] = model.induct(*[T(x, dev) for x in h], avail)
        sh = hist_index.get(q[:, 0], t, L)
        s_emb = model.subject(T(q[:, 0], dev), T(q[:, 1], dev), [T(x, dev) for x in sh], avail)
        scores = (s_emb * model.rel(T(q[:, 1], dev))) @ C.t()
        if rec is not None:
            Fsr, Fpop = rec.feats(q[:, 0], q[:, 1], t)
            scores = scores + model.rec_term(T(q[:, 1], dev), T(Fsr, dev), T(Fpop, dev))
        o_cnt = hist_index.get(q[:, 2], t, L)[3].sum(1)   # facts before t (capped at L)
        for i, (s, r, o, _) in enumerate(q):
            sc = scores[i].clone()
            others = sorted(data.filt[(int(s), int(r), int(t))] - {int(o)})
            if others:
                sc[T(others, dev)] = float('-inf')
            tg = sc[o]
            rk = int((sc > tg).sum()) + 1 + (int((sc == tg).sum()) - 1) / 2.0
            if not data.seen[s]:
                subj['unseen_subj'].append(rk)
                if int(r) == 0:
                    subj['unseen_subj_r0'].append(rk)
            if data.seen[o]:
                ranks['seen'].append(rk)
            else:
                ranks['unseen'].append(rk)
                c = int(o_cnt[i]); bk = '0' if c == 0 else '1-2' if c <= 2 else '3-5' if c <= 5 else '6+'
                buckets[bk].append(rk)
        n_done += len(q)
        if max_queries and n_done >= max_queries:
            break
    m = lambda r: {'n': len(r), 'mrr': float(np.mean(1 / np.array(r))) if r else 0.0,
                   'hits1': float(np.mean(np.array(r) <= 1)) if r else 0.0,
                   'hits10': float(np.mean(np.array(r) <= 10)) if r else 0.0}
    return {'all': m(ranks['seen'] + ranks['unseen']), 'seen_obj': m(ranks['seen']),
            'unseen_obj': m(ranks['unseen']),
            'unseen_by_prior_facts': {k: m(v) for k, v in sorted(buckets.items())},
            **{k: m(v) for k, v in subj.items()}}


# ----------------------------------------------------------------------------- train
def train(a):
    dev = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    data = Data(os.path.join(a.data_dir, a.dataset), a.seed)
    print(f'[data] E={data.E} R={data.R} train/valid/test={len(data.tr)}/{len(data.va)}/{len(data.te)} '
          f'unseen-in-train={int((~data.seen).sum())}', flush=True)
    H_train = History(data.tr, data.E, data.R)
    H_valid = History(np.concatenate([data.tr, data.va]), data.E, data.R)   # online for valid
    REC_train = Recency(data.tr, data.E, data.R)
    REC_valid = Recency(np.concatenate([data.tr, data.va]), data.E, data.R)
    model = KarmaV2(data.E, data.R, a.dim, a.heads, a.k_anchor, data.gap, a.dropout, a.abl, not a.no_rec).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    inv = data.tr[:, [2, 1, 0, 3]].copy(); inv[:, 1] += data.R
    trq = np.concatenate([data.tr, inv])
    by_t = defaultdict(list)
    for i, t in enumerate(trq[:, 3]):
        by_t[int(t)].append(i)
    seen_ids = np.where(data.seen)[0]
    os.makedirs(a.out_dir, exist_ok=True)
    ck = os.path.join(a.out_dir, f'{a.dataset}_{a.abl}{"_norec" if a.no_rec else ""}_s{a.seed}_best.pt')
    best, bad = -1.0, 0
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time()
        P = np.array([], np.int64) if a.abl == 'noepisodic' else \
            np.random.choice(seen_ids, max(1, int(a.p_new * len(seen_ids))), replace=False)
        isP = np.zeros(data.E, bool); isP[P] = True
        avail = T(data.seen & ~isP, dev)
        tot, nb = 0.0, 0
        ts_order = list(by_t.keys()); np.random.shuffle(ts_order)
        for t in ts_order:
            idx = np.array(by_t[t]); np.random.shuffle(idx)
            for c in range(0, len(idx), a.batch):
                q = trq[idx[c:c + a.batch]]
                s, r, o = q[:, 0], q[:, 1], q[:, 2]
                pool = P if len(P) <= a.max_pool else np.unique(np.concatenate(
                    [np.random.choice(P, a.max_pool, replace=False), o[isP[o]]]))
                sh = H_train.get(s, t, a.L)
                s_emb = model.subject(T(s, dev), T(r, dev), [T(x, dev) for x in sh], avail)
                sr = s_emb * model.rel(T(r, dev))
                logits = (sr @ model.table().t()).masked_fill(~avail.unsqueeze(0), float('-inf'))
                if len(pool):
                    ph = H_train.get(pool, t, a.L)
                    p_emb = model.induct(*[T(x, dev) for x in ph], avail)
                    logits[:, T(pool, dev)] = sr @ p_emb.t()
                Fsr, Fpop = REC_train.feats(s, r, t)
                logits = logits + model.rec_term(T(r, dev), T(Fsr, dev), T(Fpop, dev))
                w = torch.where(T(isP[o], dev), a.w_new, 1.0)
                loss = (F.cross_entropy(logits, T(o, dev), reduction='none') * w).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
                tot += loss.item(); nb += 1
        msg = f'epoch {ep:3d} loss {tot / max(nb, 1):.4f} {time.time() - t0:.0f}s'
        if ep % a.eval_every == 0:
            v = evaluate(model, data, data.va, H_valid, dev, a.L, a.max_valid, rec=REC_valid)
            score = v['all']['mrr']
            msg += f" | valid ACIS MRR {score:.4f} seen {v['seen_obj']['mrr']:.4f} unseen {v['unseen_obj']['mrr']:.4f} (n={v['unseen_obj']['n']})"
            if score > best:
                best, bad = score, 0
                torch.save({'state': model.state_dict(), 'args': vars(a), 'epoch': ep,
                            'E': data.E, 'R': data.R, 'gap': data.gap}, ck)
                msg += ' *saved*'
            else:
                bad += 1
        print(msg, flush=True)
        if bad >= a.patience:
            print('early stop', flush=True); break
    a.ckpt = ck
    for h in ('online', 'train'):
        a.history = h
        run_eval(a, data)


def run_eval(a, data=None):
    dev = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    ar = ck['args']
    data = data or Data(os.path.join(a.data_dir, a.dataset), ar.get('seed', 42))
    model = KarmaV2(ck['E'], ck['R'], ar['dim'], ar['heads'], ar['k_anchor'], ck['gap'], ar['dropout'],
                    ar.get('abl', 'none'), not ar.get('no_rec', False)).to(dev)
    model.load_state_dict(ck['state'])
    hq = data.allq if a.history == 'online' else data.tr
    H = History(hq, data.E, data.R)
    res = evaluate(model, data, data.te, H, dev, ar['L'], rec=Recency(hq, data.E, data.R))
    res.update({'dataset': a.dataset, 'method': 'KARMA-v2+recency', 'protocol': 'ACIS',
                'history': a.history, 'best_epoch': ck['epoch'], 'ckpt': a.ckpt})
    out = os.path.splitext(a.ckpt)[0] + f'_test_{a.history}.json'
    json.dump(res, open(out, 'w'), indent=2)
    u = res['unseen_obj']
    print(f"[test {a.history}] all {res['all']['mrr']:.4f} | seen {res['seen_obj']['mrr']:.4f} "
          f"| unseen {u['mrr']:.4f} H@10 {u['hits10']:.4f} (n={u['n']}) | by prior facts: "
          + ', '.join(f"{k}:{v['mrr']:.3f}(n={v['n']})" for k, v in res['unseen_by_prior_facts'].items()),
          flush=True)
    print('saved', out, flush=True)


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    allv = np.concatenate([pos, neg]); order = allv.argsort(kind='mergesort')
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):                       # average ranks for ties
        m = allv == v; ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def factcheck(a):
    """Task 5: fact verification. Each test fact (s,r,o,t) is a true claim; a
    corrupted claim (s,r,o',t) uses o' drawn from objects seen with r before t
    (type-consistent), never a true answer at (s,r,t). Scores = model logits
    (ACIS candidate embeddings, online history). Reports AUROC overall and for
    claims whose object is a new (unseen-in-train) entity, vs counting."""
    dev = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); ar = ck['args']
    data = Data(os.path.join(a.data_dir, a.dataset), ar.get('seed', 42))
    model = KarmaV2(ck['E'], ck['R'], ar['dim'], ar['heads'], ar['k_anchor'], ck['gap'], ar['dropout'],
                    ar.get('abl', 'none'), not ar.get('no_rec', False)).to(dev)
    model.load_state_dict(ck['state']); model.eval()
    H = History(data.allq, data.E, data.R); REC = Recency(data.allq, data.E, data.R)
    rng = np.random.default_rng(a.seed); avail = T(data.seen, dev)
    unseen = np.where(~data.seen)[0]; tab = model.table()
    S = {'model': {'pos': [], 'neg': [], 'pos_u': [], 'neg_u': []},
         'counting': {'pos': [], 'neg': [], 'pos_u': [], 'neg_u': []}}
    done = 0
    chunks = []
    for t in np.unique(data.te[:, 3]):
        qa = data.te[data.te[:, 3] == t]
        chunks += [(t, qa[c:c + 4096]) for c in range(0, len(qa), 4096)]
    for t, q in chunks:
        C = tab.clone()
        if len(unseen):
            C[T(unseen, dev)] = model.induct(*[T(x, dev) for x in H.get(unseen, t, ar['L'])], avail)
        s_emb = model.subject(T(q[:, 0], dev), T(q[:, 1], dev),
                              [T(x, dev) for x in H.get(q[:, 0], t, ar['L'])], avail)
        sc = (s_emb * model.rel(T(q[:, 1], dev))) @ C.t()
        Fsr, Fpop = REC.feats(q[:, 0], q[:, 1], t)
        sc = (sc + model.rec_term(T(q[:, 1], dev), T(Fsr, dev), T(Fpop, dev))).cpu().numpy()
        cnt = np.log1p(Fsr) * (np.log1p(Fpop).max() + 1) + np.log1p(Fpop)
        for i, (s_, r_, o_, _) in enumerate(q):
            true = REC.pop.get(int(r_)); cands = None
            if true is not None:
                n = np.searchsorted(true[1], t, 'left'); cands = np.unique(true[0][:n]) if n else None
            bad = data.filt[(int(s_), int(r_), int(t))]
            for _ in range(10):
                o2 = int(rng.choice(cands)) if (a.neg == 'type' and cands is not None and len(cands) > 1) else int(rng.integers(data.E))
                if o2 not in bad:
                    break
            else:
                continue
            new = not data.seen[o_]
            for name, M in (('model', sc), ('counting', cnt)):
                S[name]['pos'].append(M[i, o_]); S[name]['neg'].append(M[i, o2])
                if new:
                    S[name]['pos_u'].append(M[i, o_]); S[name]['neg_u'].append(M[i, o2])
        done += len(q)
        if a.max_queries and done >= a.max_queries:
            break
    res = {'dataset': a.dataset, 'task': 'fact verification (AUROC)', 'ckpt': a.ckpt,
           'n_claims': len(S['model']['pos']), 'n_claims_new_entity': len(S['model']['pos_u'])}
    for name in S:
        res[name] = {'auroc_all': auroc(S[name]['pos'], S[name]['neg']),
                     'auroc_new_entity': auroc(S[name]['pos_u'], S[name]['neg_u'])}
    out = os.path.splitext(a.ckpt)[0] + ('_factcheck.json' if a.neg == 'type' else '_factcheck_random.json')
    res['negatives'] = a.neg
    json.dump(res, open(out, 'w'), indent=2)
    print(f"[factcheck {a.dataset}] model AUROC all {res['model']['auroc_all']:.4f} new-entity "
          f"{res['model']['auroc_new_entity']:.4f} | counting all {res['counting']['auroc_all']:.4f} "
          f"new-entity {res['counting']['auroc_new_entity']:.4f} | claims {res['n_claims']} "
          f"(new {res['n_claims_new_entity']})", flush=True)
    print('saved', out, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['train', 'eval', 'factcheck'])
    p.add_argument('--dataset', required=True)
    p.add_argument('--data_dir', default='/storage/bhull113/rhgnn/data')
    p.add_argument('--out_dir', default='/storage/bhull113/karma_v2/runs_x')
    p.add_argument('--ckpt', default=None)
    p.add_argument('--history', choices=['online', 'train'], default='online')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dim', type=int, default=200)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--k_anchor', type=int, default=16)
    p.add_argument('--L', type=int, default=20)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=0.0)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--p_new', type=float, default=0.15)
    p.add_argument('--w_new', type=float, default=2.0)
    p.add_argument('--max_pool', type=int, default=4096)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--eval_every', type=int, default=5)
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--max_valid', type=int, default=20000)
    p.add_argument('--abl', choices=['none', 'noepisodic', 'noanchor', 'nounk'], default='none')
    p.add_argument('--no_rec', action='store_true')
    p.add_argument('--max_queries', type=int, default=0)
    p.add_argument('--neg', choices=['type', 'random'], default='type')
    a = p.parse_args()
    {'train': train, 'eval': run_eval, 'factcheck': factcheck}[a.mode](a)
