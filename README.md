# KARMA-ER: Cold-Start Reasoning in Temporal Knowledge Graphs

Code, evaluation protocol, results and demo for the AAAI-27 Demonstration paper

> **Inspecting Cold-Start Reasoning in Temporal Knowledge Graphs: KARMA-ER Across Four Tasks**
> Amangel Bhullar and Ziad Kobti, University of Windsor

- **Live demo:** _link to be added after deployment_

## What is here

| Folder | Contents |
|---|---|
| `demo/` | Streamlit app (`demo/app.py`). The **Benchmarks** tab reads `demo/results/summary.json`; the other tabs are labelled interface prototypes with illustrative outputs. |
| `karma_er/` | KARMA-ER model, training and ACIS evaluation (`karma_v2x.py`), task converters (`make_task_data.py`, `make_multitq.py`), question answering (`multitq_qa.py`). |
| `evaluation/` | TOIS/ACIS evaluator (`eval_fair_any.py`), counting, popularity and common-neighbour baselines, random-data control, TiPNN adapter. |
| `scripts/` | Run scripts and `export_demo_results.py` (builds `summary.json`). |

## Two evaluation protocols

Unseen = entity absent from the training split. For a query `(s, r, ?, t)` whose answer is unseen:

- **TOIS (target-only inductive scoring):** only the true answer is scored by the inductive encoder; other unseen candidates keep their untrained embeddings. A model trained on random facts reaches MRR 1.000 under TOIS.
- **ACIS (all-candidate inductive scoring):** every unseen candidate is scored by the same encoder from its own facts strictly before `t`. All results below use ACIS, filtered ranks and average-rank ties.

## Model

KARMA-ER encodes an entity from its last L facts before `t` (relation, neighbour, time gap), attends over them with a relation-conditioned query, and adds anchors to its top-k most similar trained entities. Neighbours that are unavailable use a shared UNK vector. Training hides 15% of seen entities per epoch and ranks them against each other (episodic emergence). A learned per-relation recency term uses earlier `(s, r, o)` and `(·, r, o)` counts. Scoring is DistMult.

## Tasks and data

| Task | Dataset | Cold-start entities |
|---|---|---|
| Link prediction | ICEWS14, ICEWS18, WIKI-clean, YAGO15k, YAGO11k, WIKIDATA12k | entities first seen after training |
| Fact verification | the six above, MovieLens-1M, STRING | claims involving a new entity (AUROC) |
| Recommendation | MovieLens-1M (weekly steps, like/neutral/dislike) | users first seen after the time cutoff |
| Biomedical | STRING human PPI (score >= 700) | 10% held-out proteins, 3 known interactions each |
| Question answering (prototype) | MultiTQ | answers that are new entities |

Datasets are not redistributed. Download the public benchmarks, MovieLens-1M (GroupLens), STRING v12 `protein.links` (9606) and MultiTQ, then build the task graphs:

```bash
python karma_er/make_task_data.py movielens --src <ml-1m dir> --out data_tasks/ML-cold
python karma_er/make_task_data.py string --src <protein.links.v12.0.txt.gz> --out data_tasks/STRING-PPI
python karma_er/make_multitq.py --src <MultiTQ>/data/MultiTQ --out data_tasks/MultiTQ-kg
```

## Reproduce

```bash
# train + ACIS test (writes *_test_online.json / *_test_train.json)
python karma_er/karma_v2x.py train --dataset WIKI-clean --data_dir <data> --out_dir runs --seed 42
# fact verification
python karma_er/karma_v2x.py factcheck --dataset WIKI-clean --data_dir <data> --ckpt runs/WIKI-clean_none_s42_best.pt --neg type
# without the recency term (KARMA-E) / ablations
python karma_er/karma_v2x.py train ... --no_rec
python karma_er/karma_v2x.py train ... --abl noepisodic   # or noanchor, nounk
# demo results
python scripts/export_demo_results.py && streamlit run demo/app.py
```

## Main results (ACIS)

Unseen-answer MRR, link prediction (KARMA-ER: mean of 3 seeds):

| | ICEWS14 | ICEWS18 | WIKI | YAGO15k | YAGO11k | WD12k |
|---|---|---|---|---|---|---|
| Counting | **.037** | **.029** | .403 | .450 | .002 | .038 |
| KARMA-ER | .032 | .017 | **.495** | **.538** | **.045** | **.071** |

Fact verification (AUROC, new-entity claims, type-consistent negatives): KARMA-ER .739 / .565 / .828 / .935 on ICEWS14 / ICEWS18 / WIKI / YAGO15k, .911 on MovieLens-1M, .977 on STRING. Recommendation (new users): KARMA-ER .030 MRR, popularity .030. Held-out proteins: KARMA-ER .201 MRR (3 seeds), common neighbours .004. Full tables, per-seed values, history-length buckets and ablations are in `demo/results/summary.json` and the app.

KARMA-ER trails counting on the two ICEWS datasets; it is at chance on YAGO11k fact verification (5 claims) and only matches popularity for new users. Comparisons with TLogic and TiPNN under ACIS are in progress.

## Citation

```bibtex
@inproceedings{bhullar2027karmaer,
  title     = {Inspecting Cold-Start Reasoning in Temporal Knowledge Graphs: {KARMA-ER} Across Four Tasks},
  author    = {Bhullar, Amangel and Kobti, Ziad},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (Demonstration Track)},
  year      = {2027}
}
```
