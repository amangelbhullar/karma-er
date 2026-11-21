"""
make_multitq.py -- MultiTQ as a cold-start TKG QA task.

1. KG: MultiTQ kg/{train,valid,test}.txt (ICEWS05-15, time-ordered split) ->
   data_tasks/MultiTQ-kg/{train,valid,test}.txt with "s r o t" ids, t = day index
   since 2005-01-01. Entities first seen after the train cutoff are unseen.
2. Questions (light NLP, no LLM): entity-answer, qtype 'equal', day-level questions.
   - entity linking: longest n-gram match against KG names ("X (Y)" also as
     "X in Y", "X of Y", "Y X");
   - time: regex for the date;
   - relation + direction: TF-IDF + linear classifier trained on TRAIN questions,
     weakly labelled with the KG fact (e, r, answer, t) or (answer, r, e, t) -> r + R.
   Test questions are kept if their date is after the KG train cutoff, so the
   answer fact is never seen in training and history is strictly before t.
   Output: data_tasks/MultiTQ-kg/qa_test.json (predicted and gold-parse queries).

usage: python make_multitq.py --src .../MultiTQ_src/data/MultiTQ --out .../data_tasks/MultiTQ-kg
"""
import os, re, json, argparse, datetime, collections
import numpy as np

MONTHS = {m: i + 1 for i, m in enumerate(
    ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
     'september', 'october', 'november', 'december'])}
D0 = datetime.date(2005, 1, 1)
MONTHS3 = {k[:3]: v for k, v in MONTHS.items()}
MON = r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?'
PATS = [
    (re.compile(r'(\d{4})-(\d{1,2})-(\d{1,2})'), lambda m: (int(m[1]), int(m[2]), int(m[3]))),
    (re.compile(MON + r'\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})', re.I),
     lambda m: (int(m[3]), MONTHS3[m[1].lower()[:3]], int(m[2]))),
    (re.compile(r'(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?' + MON + r',?\s+(\d{4})', re.I),
     lambda m: (int(m[3]), MONTHS3[m[2].lower()[:3]], int(m[1]))),
]


def day(y, m, d):
    try:
        return (datetime.date(y, m, d) - D0).days
    except ValueError:
        return None


def parse_date(q):
    for p, f in PATS:
        m = p.search(q)
        if m:
            t = day(*f(m))
            if t is not None:
                return t, m.span()
    return None, None


def toks(s):
    return re.findall(r"\w+|[^\w\s]", s.lower().replace('_', ' '))


def surfaces(name):
    n = name.replace('_', ' ').strip()
    out = {n}
    m = re.match(r'^(.*\S)\s*\((.*)\)$', n)
    if m:
        x, y = m[1], m[2]
        out |= {f'{x} in {y}', f'{x} of {y}', f'{y} {x}', f"{y}'s {x}", f'{x} from {y}'}
    return out


class Linker:
    def __init__(self, ent2id):
        self.d, self.maxn = {}, 1
        for name, i in ent2id.items():
            for s in surfaces(name):
                k = tuple(toks(s))
                if k:
                    self.d.setdefault(k, set()).add(i); self.maxn = max(self.maxn, len(k))

    def link(self, q):
        """Non-overlapping matches, longest first: list of (entity ids, (start, end) token span)."""
        w = toks(q); found = []
        for n in range(min(self.maxn, len(w)), 0, -1):
            for a in range(len(w) - n + 1):
                k = tuple(w[a:a + n])
                if k in self.d and not any(a < e and a + n > s for _, (s, e) in found):
                    found.append((self.d[k], (a, a + n)))
        return w, found


def kg(src, out, ent2id, rel2id):
    os.makedirs(out, exist_ok=True); res = {}
    for split in ('train', 'valid', 'test'):
        rows = []
        for line in open(os.path.join(src, 'kg', f'{split}.txt'), encoding='utf-8'):
            x = line.rstrip('\n').split('\t')
            if len(x) < 4:
                continue
            y, m, d = map(int, x[3].split('-'))
            rows.append((ent2id[x[0]], rel2id[x[1]], ent2id[x[2]], day(y, m, d)))
        q = np.array(rows, np.int64)
        np.savetxt(os.path.join(out, f'{split}.txt'), q, fmt='%d'); res[split] = q
    return res


def main(a):
    ent2id = json.load(open(os.path.join(a.src, 'kg', 'entity2id.json')))
    rel2id = json.load(open(os.path.join(a.src, 'kg', 'relation2id.json')))
    E, R = max(ent2id.values()) + 1, max(rel2id.values()) + 1
    K = kg(a.src, a.out, ent2id, rel2id)
    seen = np.zeros(E, bool); seen[np.unique(K['train'][:, [0, 2]])] = True
    t_cut = int(K['train'][:, 3].max())
    print(f'KG: E={E} R={R} train/valid/test={len(K["train"])}/{len(K["valid"])}/{len(K["test"])} '
          f'train cutoff day {t_cut} ({D0 + datetime.timedelta(t_cut)}) unseen entities {int((~seen).sum())}',
          flush=True)
    allq = np.concatenate([K['train'], K['valid'], K['test']])
    at = collections.defaultdict(list)                 # (entity, day) -> [(r', other)]
    for s, r, o, t in allq:
        at[(int(s), int(t))].append((int(r), int(o)))
        at[(int(o), int(t))].append((int(r) + R, int(s)))
    L = Linker(ent2id)
    norm = lambda s: ent2id.get(s.replace(' ', '_'), ent2id.get(s))

    def prep(Q, need_label):
        out, stat = [], collections.Counter()
        for q in Q:
            if q['answer_type'] != 'entity' or q['qtype'] != 'equal' or q['time_level'] != 'day':
                continue
            stat['in_scope'] += 1
            t, span = parse_date(q['question'])
            if t is None:
                stat['no_date'] += 1
                if stat['no_date'] <= 3: print('  [no date]', q['question'])
                continue
            text = q['question'][:span[0]] + ' ' + q['question'][span[1]:]
            w, found = L.link(text)
            if not found:
                stat['no_entity'] += 1
                if stat['no_entity'] <= 3: print('  [no entity]', q['question'])
                continue
            ans = [x for x in (norm(s) for s in q['answers']) if x is not None]
            if not ans:
                stat['answer_not_in_kg'] += 1; continue
            # predicted entity: longest match (first in list); gold parse: a match with a KG fact to the answer
            ids0, sp0 = found[0]
            gold = None
            for ids, sp in found:
                for e in sorted(ids):
                    rs = sorted({r for r, o in at.get((e, t), []) if o in ans})
                    if rs:
                        gold = (e, rs, sp); break
                if gold: break
            stat['linked'] += 1
            if gold:
                stat['gold_parse'] += 1
            ent_pred = sorted(ids0)[0]
            sp_mask = gold[2] if (need_label and gold) else sp0
            masked = ' '.join(w[:sp_mask[0]] + ['ENT'] + w[sp_mask[1]:])
            rec = {'quid': q['quid'], 'q': q['question'], 'text': masked, 't': t,
                   'answers': ans, 'e_pred': ent_pred,
                   'e_gold': gold[0] if gold else None, 'r_gold': gold[1] if gold else None}
            if need_label and not (gold and len(gold[1]) == 1):
                continue
            out.append(rec)
        return out, stat

    tr, st = prep(json.load(open(os.path.join(a.src, 'questions', 'train.json'))), True)
    print('train questions:', dict(st), '-> labelled', len(tr), flush=True)
    te, st2 = prep(json.load(open(os.path.join(a.src, 'questions', 'test.json'))), False)
    n_all = len(te)
    te = [x for x in te if x['t'] > t_cut]
    print('test questions:', dict(st2), f'| after train cutoff {len(te)} of {n_all}', flush=True)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform([x['text'] for x in tr]); y = [x['r_gold'][0] for x in tr]
    clf = LinearSVC(C=0.5).fit(X, y)
    pred = clf.predict(vec.transform([x['text'] for x in te])) if te else []
    acc_e, acc_r, n_g = 0, 0, 0
    for x, p in zip(te, pred):
        x['r_pred'] = int(p)
        x['cold_entity'] = bool(not seen[x['e_pred']])
        x['cold_entity_gold'] = None if x['e_gold'] is None else bool(not seen[x['e_gold']])
        x['cold_answer'] = bool(all(not seen[o] for o in x['answers']))
        if x['e_gold'] is not None:
            n_g += 1; acc_e += x['e_pred'] == x['e_gold']; acc_r += int(p) in x['r_gold']
        del x['text']
    print(f'parser on test (where gold parse exists, n={n_g}): entity acc {acc_e / max(n_g, 1):.3f} '
          f'relation acc {acc_r / max(n_g, 1):.3f}', flush=True)
    print(f'cold-start (entity unseen in KG train): {sum(x["cold_entity"] for x in te)} questions; '
          f'answer unseen: {sum(x["cold_answer"] for x in te)}', flush=True)
    json.dump({'E': E, 'R': R, 't_cut': t_cut, 'n_test_in_scope': st2['in_scope'],
               'parser': {'n_gold': n_g, 'entity_acc': acc_e / max(n_g, 1), 'relation_acc': acc_r / max(n_g, 1)},
               'questions': te}, open(os.path.join(a.out, 'qa_test.json'), 'w'))
    print('saved', os.path.join(a.out, 'qa_test.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True)
    p.add_argument('--out', required=True)
    main(p.parse_args())
