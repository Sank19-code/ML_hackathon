# Business Entity Resolution Pipeline (ML Challenge 2026)

Matches every Source 1 business to its Source 2 / Source 3 records (zero, one or many),
optimising the precision-weighted macro F0.5 metric. Runs end to end on a 16 GB / 16-thread
Windows or Linux machine with CPU only.

```
normalise (undo the noise) -> TF-IDF top-k blocking per country -> 78 pair + 6 context features
-> LightGBM stage 1 (cross-fitted) -> group / one-to-one / cluster / twin features -> LightGBM stage 2
-> one-to-one assignment -> calibration -> per-entity exact expected-F0.5 decision -> output TSVs
-> validator
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
| `prepare` | learn the transliteration dictionary, normalise all six source files, block train and test | 40 min (+25 min normalisation on first run) |
| `train` | stage-1 / stage-2 LightGBM (cross-fitted), held-out evaluation, decision tuning -> `model/` | ~2.5 h |
| `predict` | score the test candidates, one-to-one assignment, decisions, write both TSVs | ~1 h |
| `validate` | run the official `validate_submission.py` on `output/` | 1 min |

\* measured on a 16-thread laptop CPU with 16 GB RAM (98 M training / 78 M test candidate pairs).
Pair features are computed on the fly (~11-25 µs per pair); only the sampled training rows are
written, as float32 memory-mapped blocks in `<work-dir>/tmp` (~7 GB for stage 1, ~9 GB for
stage 2, deleted after each fit), which LightGBM bins without stacking them in RAM. On a 16 GB machine run `train` and `predict` as
separate processes (a long-lived process keeps the training matrices' memory and starts paging),
and keep Windows from power-throttling the background process.

Countries are an open set: a test country without training labels (France) uses the pooled
calibration with the unlabelled-country temperature / missed-match settings in
`model/thresholds.json` (chosen on leave-one-country-out rehearsals); no country name appears in
the pipeline logic (the deeper India blocking list is triggered by the share of non-Latin-script
records).

Optional flags: `--predict-countries India US` scores only some test countries,
`--train-countries`, `--stage1-sample` / `--stage2-sample` change what the models are fitted on,
`--reuse-stage1` keeps the stage-1 model and its saved out-of-fold scores.

Held-out macro F0.5 of the shipped models (tuned on one half of the training entities,
evaluated on the other): **0.9861** (India 0.9843-0.9845, US 0.9871-0.9872), trained with
`--stage1-sample 0.20 --stage2-sample 0.20`.

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
  features.py    78 pair features (rapidfuzz cpdist, IDF cosine / containment, flags, sibling offsets,
                 content core), locality -> region imputation
  model.py       LightGBM fold ensemble (cross-fitting by Source 1 entity)
  decide.py      group / one-to-one / cluster-support / twin features, calibration, exact
                 expected-F0.5 decision (numba Poisson-binomial DP), decision rules
  metrics.py     official macro F0.5 + breakdowns + blocking recall
  pipeline.py    stage orchestration (per country, chunked, memory bounded)
  run.py         command line entry point
evaluate_holdout.py, test_pipeline.py, requirements.txt, model/
```

See `Documentation_template.md` at the repository root for the full methodology and
results.
