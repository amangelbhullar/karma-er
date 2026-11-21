"""Collect every result file on Luna into one summary.json for the demo's Benchmarks tab.
usage: python export_demo_results.py /storage/bhull113 /path/to/karma-x/demo/results/summary.json
Missing pieces are skipped, never invented."""
import os, sys, json, glob, re
import numpy as np

B, OUT = sys.argv[1], sys.argv[2]
J = lambda p: json.load(open(p)) if os.path.exists(p) else None
DS = ['ICEWS14-renet', 'ICEWS18-renet', 'WIKI-clean', 'YAGO15k', 'YAGO11k', 'WIKIDATA12k']
r4 = lambda x: None if x is None else round(float(x), 4)
S = {'note': 'All numbers from runs of the released code; ACIS = all-candidate inductive scoring, '
             'TOIS = target-only inductive scoring. Unseen = entity absent from the training split.'}


def buckets_txt(p):
    if not os.path.exists(p):
        return None
    txt = open(p).read().split(':', 1)[-1]
    return {k: float(v) for k, v in re.findall(r'([0-9+\-]+):([0-9.]+)\(n=', txt)}


# (a) link prediction
rows = []
for d in DS:
    c = J(f'{B}/karma_fair_eval/recency_{d}.json')
    er = [J(f) for f in glob.glob(f'{B}/karma_v2/runs_rec/{d}_best_test_online.json')
          + glob.glob(f'{B}/karma_v2/runs_x/{d}_none_s*_best_test_online.json')]
    kx = [J(f)['fair']['unseen_obj']['mrr'] for f in glob.glob(f'{B}/karma_fair_eval/all/karma*_karma*_{d}.json')]
    tl = J(f'{B}/TLogic/acis/tlogic_{d}.json'); tp = J(f'{B}/karma_fair_eval/tipnn_acis/{d}.json')
    u = [x['unseen_obj']['mrr'] for x in er]; s = [x['seen_obj']['mrr'] for x in er]
    rows.append({'dataset': d,
                 'counting_unseen': r4(c['freq_sr+pop']['unseen_obj']['mrr']) if c else None,
                 'counting_seen': r4(c['freq_sr+pop']['seen_obj']['mrr']) if c else None,
                 'karmax_best_ACIS_unseen': r4(max(kx)) if kx else None,
                 'karma_er_unseen_mean': r4(np.mean(u)) if u else None, 'karma_er_unseen_std': r4(np.std(u)) if u else None,
                 'karma_er_seen_mean': r4(np.mean(s)) if s else None, 'karma_er_seeds': len(er),
                 'tlogic_unseen': r4(tl['unseen_obj']['mrr']) if tl else None,
                 'tipnn_unseen': r4(tp['unseen_obj']['mrr']) if tp else None,
                 'n_unseen_queries': er[0]['unseen_obj']['n'] if er else None})
S['link_prediction'] = rows

# per-history-length buckets (KARMA-ER seed 42 vs counting)
bk = []
for d in DS:
    j = J(f'{B}/karma_v2/runs_rec/{d}_best_test_online.json'); c = buckets_txt(f'{B}/karma_fair_eval/buckets_{d}.txt')
    if j:
        for k, v in j['unseen_by_prior_facts'].items():
            bk.append({'dataset': d, 'prior_facts': k, 'n': v['n'], 'karma_er': r4(v['mrr']),
                       'counting': r4(c.get(k)) if c else None})
S['buckets'] = bk

# protocol audit: old KARMA-X variants, TOIS vs ACIS
aud = []
for mode, pat in (('online', f'{B}/karma_fair_eval/all/karma*_karma*_*.json'), ('train', f'{B}/karma_fair_eval/all_train/*.json')):
    for f in glob.glob(pat):
        j = J(f); v = re.match(r'(karma[a-z]+)', os.path.basename(f)).group(1)
        aud.append({'dataset': j['dataset'], 'variant': v, 'history': mode, 'seen': r4(j['fair']['seen_obj']['mrr']),
                    'unseen_TOIS': r4(j['leaky']['unseen_obj']['mrr']), 'unseen_ACIS': r4(j['fair']['unseen_obj']['mrr'])})
S['protocol_audit'] = sorted(aud, key=lambda x: (x['dataset'], x['variant'], x['history']))
S['random_control'] = [{'run': os.path.basename(f)[:-5], 'unseen_TOIS': r4(J(f)['leaky']['unseen_obj']['mrr']),
                        'unseen_ACIS': r4(J(f)['fair']['unseen_obj']['mrr'])}
                       for f in sorted(glob.glob(f'{B}/karma_fair_eval/control/res/*.json'))]

# ablations
ab = []
for f in glob.glob(f'{B}/karma_v2/runs_x/*_test_online.json'):
    m = re.match(r'(.+?)_(none|noepisodic|noanchor|nounk)_s(\d+)_best', os.path.basename(f))
    if m and m.group(2) != 'none':
        j = J(f); ab.append({'dataset': m.group(1), 'removed': m.group(2), 'unseen': r4(j['unseen_obj']['mrr']), 'seen': r4(j['seen_obj']['mrr'])})
S['ablations'] = sorted(ab, key=lambda x: (x['dataset'], x['removed']))

# (b) fact verification
fv = []
for f in sorted(glob.glob(f'{B}/karma_v2/runs_rec/*_factcheck*.json') + glob.glob(f'{B}/karma_v2/runs_tasks/*_s42_best_factcheck*.json')):
    j = J(f); fv.append({'dataset': j['dataset'], 'negatives': j.get('negatives', 'type'),
                         'karma_er_new': r4(j['model']['auroc_new_entity']), 'counting_new': r4(j['counting']['auroc_new_entity']),
                         'karma_er_all': r4(j['model']['auroc_all']), 'counting_all': r4(j['counting']['auroc_all'])})
S['fact_verification'] = fv

# (c) recommendation
p = J(f'{B}/karma_v2/runs_tasks/popularity_ML-cold.json'); rec = []
if p:
    rec.append({'method': 'Popularity', 'mrr': r4(p['unseen_subj_r0']['mrr']), 'hits10': r4(p['unseen_subj_r0']['hits10']), 'n': p['unseen_subj_r0']['n']})
for f in sorted(glob.glob(f'{B}/karma_v2/runs_tasks/ML-cold_none_s*_best_test_online.json')):
    j = J(f)
    if 'unseen_subj_r0' in j:
        rec.append({'method': 'KARMA-ER seed ' + f.split('_s')[-1].split('_')[0], 'mrr': r4(j['unseen_subj_r0']['mrr']),
                    'hits10': r4(j['unseen_subj_r0']['hits10']), 'n': j['unseen_subj_r0']['n']})
S['recommendation'] = rec

# (d) biomedical
bio = []
c = J(f'{B}/karma_v2/runs_tasks/recency_STRING-PPI.json')
if c:
    bio.append({'method': 'Counting', 'mrr': r4(c['freq_sr+pop']['unseen_obj']['mrr']), 'hits10': r4(c['freq_sr+pop']['unseen_obj']['hits10'])})
cn = J(f'{B}/karma_v2/runs_tasks/common_neighbour_STRING-PPI.json')
if cn:
    bio.append({'method': 'Common neighbours (Adamic-Adar)', 'mrr': r4(cn['unseen_obj']['mrr']), 'hits10': r4(cn['unseen_obj']['hits10'])})
tl = J(f'{B}/TLogic/acis/tlogic_STRING-PPI.json')
if tl:
    bio.append({'method': 'TLogic', 'mrr': r4(tl['unseen_obj']['mrr']), 'hits10': r4(tl['unseen_obj']['hits10'])})
for f in sorted(glob.glob(f'{B}/karma_v2/runs_tasks/STRING-PPI_none_s*_best_test_online.json')):
    j = J(f); bio.append({'method': 'KARMA-ER seed ' + f.split('_s')[-1].split('_')[0], 'mrr': r4(j['unseen_obj']['mrr']), 'hits10': r4(j['unseen_obj']['hits10'])})
S['biomedical'] = bio

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(S, open(OUT, 'w'), indent=1)
print('wrote', OUT, '|', {k: len(v) for k, v in S.items() if isinstance(v, list)})
