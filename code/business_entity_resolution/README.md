# Business Entity Resolution Pipeline (ML Challenge 2026)

Matches every Source 1 business to its Source 2 / Source 3 records (zero, one or many),
optimising the precision-weighted macro F0.5 metric. Runs end to end on a 16 GB / 16-thread
Windows or Linux machine with CPU only.

```
normalise (undo the noise) -> TF-IDF top-k blocking per country -> 74 pair + 6 context features
-> LightGBM stage 1 (cross-fitted) -> group / one-to-one / cluster features -> LightGBM stage 2
-> one-to-one assignment -> per-entity expected-F0.5 decision -> output TSVs -> validator
```

## 1. Environment

- Python 3.11 (tested with 3.11.9), 16 GB RAM, ~8 GB free disk for the work directory.
- `pip install -r requirements.txt` (numpy, scipy, polars, pyarrow, rapidfuzz, lightgbm,
  numba, anyascii - all MIT / BSD / Apache licensed). No pretrained model, no external
  data or API is used: every dictionary is either learned from the training labels or a
  static normalisation table in `src/normalize.py`.

## 2. Reproduce `output/matching_results.tsv` and `output/candidate_pairs.tsv`

From the repository root (paths below assume the student resource layout; point
`--train-dir/--test-dir` wherever the TSVs live):

```bash
python code/business_entity_resolution/src/run.py --mode all \
    --train-dir dataset/train --test-dir dataset/test \
    --output-dir output --work-dir work \
    --model-dir code/business_entity_resolution/model \
    --validator utils/validate_submission.py
```

`--mode all` runs every stage in order; each stage caches its results in `--work-dir` so
it can also be run separately:

| mode | what it does | measured time* |
| --- | --- | --- |
| `prepare` | learn the transliteration dictionary, normalise all six source files, block train and test | 40 min |
| `train` | stage-1 / stage-2 LightGBM (cross-fitted), held-out evaluation, decision tuning -> `model/` | 90 min |
| `predict` | score the test candidates, one-to-one assignment, decisions, write both TSVs | 50 min |
| `validate` | run the official `validate_submission.py` on `output/` | 1 min |

\* `--mode all` took 2.9 h on a 16-thread laptop CPU (16 GB RAM). Pair features are computed on
the fly (~11-25 µs per pair), so nothing heavier than the candidate tables (~2.5 GB) is written
to the work directory.

Held-out macro F0.5 of the shipped models (tuned on one half of the training entities,
evaluated on the other): **0.9829** (India 0.9824-0.9825, US 0.9832).

The trained artefacts in `model/` (`stage1.pkl`, `stage2.pkl`, `thresholds.json`,
`translit.json`, `train_summary.json`) let `--mode predict` run without retraining.

`evaluate_holdout.py` re-runs the held-out evaluation / decision tuning from the saved
out-of-fold scores (no retraining) and prints macro F0.5 overall, by country, for
singletons and the blocking recall. `test_pipeline.py` holds unit tests.

## 3. Layout

```
src/
  normalize.py   name / address canonicalisation (transliteration, alias phrases, legal
                 forms, OCR swaps, domains, states, street types, house numbers, ...)
  translit.py    learns the native-script -> Latin word dictionary from training pairs
  data.py        TSV loading + parallel normalisation, cached as parquet
  blocking.py    per-country TF-IDF key features + numba sparse top-k retrieval
  features.py    74 pair features (rapidfuzz cpdist, IDF cosine / containment, flags, sibling offsets)
  model.py       LightGBM fold ensemble (cross-fitting by Source 1 entity)
  decide.py      group / one-to-one / cluster-support features, decision rules
  metrics.py     official macro F0.5 + breakdowns + blocking recall
  pipeline.py    stage orchestration (per country, chunked, memory bounded)
  run.py         command line entry point
evaluate_holdout.py, test_pipeline.py, requirements.txt, model/
```

See `Documentation_template.md` at the repository root for the full methodology and
results.
