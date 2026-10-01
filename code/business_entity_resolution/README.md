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

- Python 3.11 (tested with 3.11.9), 16 GB RAM, about 6 GB of free disk to regenerate the outputs
  with the shipped model (4.6 GB work directory + 1.1 GB outputs) and about 25 GB to retrain
  (temporary memory-mapped training matrices).
- `pip install -r requirements.txt` (numpy, scipy, polars, pyarrow, rapidfuzz, lightgbm, numba,
  anyascii - MIT / BSD / Apache / ISC licensed).
- No pretrained model and no external database, API or downloaded data is used. Every dictionary
  is either learned from the training labels (`translit.json`), fitted without labels on each
  country's own records (at prediction time the test records: the locality -> region table, the
  TF-IDF / IDF weights and the name / address duplicate counts of the context features), or one of
  these hand-written static tables: in `src/normalize.py` US state codes and names, Indian states
  (including native-script names) and historic city names, French regions and 16 departments, French
  street types and legal forms, and generic word lists (street types, directionals, ordinals,
  honorifics, inserted filler words including "india" and "france", address unit / filler words
  including French articles, unit words and ordinal suffixes); in `src/blocking.py` a street-type
  stop list. France has no training records, so the French tables and words were written from the
  format of the unlabelled test records.

## 2. Reproduce `output/matching_results.tsv` and `output/candidate_pairs.tsv`

Run from the folder that contains `code/` (the unzipped package root). `DATA` is the student
resource folder (it contains `dataset/train`, `dataset/test` and `utils/validate_submission.py`).
Run each command as its own process: on a 16 GB machine a single long-lived process keeps the
training matrices' memory and starts paging.

**A. Regenerate the submitted files with the shipped model (`model/`, no retraining, about 2 h):**

```bash
DATA=/path/to/student_resource
python code/business_entity_resolution/src/run.py --mode prepare --train-dir $DATA/dataset/train \
    --test-dir $DATA/dataset/test --work-dir work --model-dir code/business_entity_resolution/model
python code/business_entity_resolution/src/run.py --mode predict --train-dir $DATA/dataset/train \
    --test-dir $DATA/dataset/test --work-dir work --model-dir code/business_entity_resolution/model \
    --output-dir output_repro --validator $DATA/utils/validate_submission.py
```

`output_repro/` then holds the same candidate pairs and the same match sets as `output/` (IDs in a
list are ordered by score; exact score ties may be listed in another order). `prepare` does not
touch `model/`: the shipped `translit.json` is reused.

**B. Retrain end to end (about 5 h; never point it at the shipped `model/` or `output/`):**

```bash
python code/business_entity_resolution/src/run.py --mode prepare --train-dir $DATA/dataset/train \
    --test-dir $DATA/dataset/test --work-dir work_retrain --model-dir model_retrained
python code/business_entity_resolution/src/run.py --mode train --train-dir $DATA/dataset/train \
    --test-dir $DATA/dataset/test --work-dir work_retrain --model-dir model_retrained \
    --stage1-sample 0.20 --stage2-sample 0.20
python code/business_entity_resolution/src/run.py --mode predict --train-dir $DATA/dataset/train \
    --test-dir $DATA/dataset/test --work-dir work_retrain --model-dir model_retrained \
    --output-dir output_retrained --validator $DATA/utils/validate_submission.py
```

A fresh model folder relearns `translit.json` from the training labels. LightGBM retraining is not
guaranteed to be bit-identical across machines, so recipe A is the one that reproduces the
submission. `--mode all` runs prepare, train and predict in one process and writes into
`--model-dir`, which defaults to the shipped `model/`: always pass a new folder when training.

| mode | what it does | measured time* |
| --- | --- | --- |
| `prepare` | learn the transliteration dictionary (skipped if `<model-dir>/translit.json` exists), normalise all six source files, block train and test | 41 min |
| `train` | stage-1 / stage-2 LightGBM (cross-fitted), held-out evaluation, decision tuning -> `<model-dir>` | ~2.5 h |
| `predict` | score the test candidates, one-to-one assignment, decisions, write both TSVs, run the validator | 78 min |
| `validate` | run the official `validate_submission.py` on `<output-dir>` | 1-3 min |

\* measured on a 16-thread laptop CPU with 16 GB RAM (98 M training / 78 M test candidate pairs);
the work directory of recipe A takes 4.6 GB. Pair features are computed on the fly (~11-25 µs per
pair); only the sampled training rows are written, as float32 memory-mapped blocks in
`<work-dir>/tmp` (~7 GB for stage 1, ~9 GB for stage 2, deleted after each fit), which LightGBM
bins without stacking them in RAM. On Windows, keep the OS from power-throttling the background
process.

Countries are an open set. The model, features, blocking depth and decision path never branch on a
country name: the deeper blocking list is triggered by the share of non-Latin-script records (India
18 %, US and France 0 %), and a country without training labels (France) uses the pooled calibration
with the unlabelled-country temperature / missed-match settings in `model/thresholds.json`, chosen
on leave-one-country-out rehearsals. Only the address normaliser branches on the country string: it
selects the US / India state tables or the French region / department and street-type tables, and
for France applies two parsing rules (a "postcode city" component is a locality; a component that
starts with a French street type is a street). Any other country gets the generic street rules and
no state table. Every test entity is written out.

Optional flags: `--predict-countries India US` scores only some test countries,
`--train-countries`, `--stage1-sample` / `--stage2-sample` change what the models are fitted on,
`--reuse-stage1` keeps the stage-1 model and its saved out-of-fold scores.

Held-out macro F0.5 of the shipped models (tuned on one half of the training entities,
evaluated on the other): **0.9861** (India 0.9843-0.9845, US 0.9871-0.9872), trained with
`--stage1-sample 0.20 --stage2-sample 0.20` (the defaults).

The trained artefacts in `model/` (`stage1.pkl`, `stage2.pkl`, `thresholds.json`,
`translit.json`, `train_summary.json`) let recipe A run `--mode predict` without retraining.

`src/evaluate_holdout.py` re-runs the held-out evaluation / decision tuning from the out-of-fold
scores that `--mode train` saves in its work directory, and prints macro F0.5 overall, by country,
for singletons and the blocking recall; it rewrites `<model-dir>/thresholds.json`, so point it at a
retrained model folder (`--model-dir` is required). `src/test_pipeline.py` holds unit tests
(`cd code/business_entity_resolution/src && python test_pipeline.py`).

## 3. Layout

```
src/
  normalize.py         name / address canonicalisation (transliteration, alias phrases, legal
                       forms, OCR swaps, domains, states, street types, house numbers, ...)
  translit.py          learns the native-script -> Latin word dictionary from training pairs
  data.py              TSV loading + parallel normalisation, cached as parquet
  blocking.py          per-country TF-IDF key features + numba sparse top-k retrieval
  features.py          78 pair features (rapidfuzz cpdist, IDF cosine / containment, flags, sibling
                       offsets, content core), locality -> region imputation
  model.py             LightGBM fold ensemble (cross-fitting by Source 1 entity)
  decide.py            group / one-to-one / cluster-support / twin features, calibration, exact
                       expected-F0.5 decision (numba Poisson-binomial DP), decision rules
  metrics.py           official macro F0.5 + breakdowns + blocking recall
  pipeline.py          stage orchestration (per country, chunked, memory bounded)
  run.py               command line entry point
  evaluate_holdout.py  held-out evaluation from saved out-of-fold scores
  test_pipeline.py     unit tests
model/                 trained models, decision rules, transliteration dictionary
requirements.txt       pinned dependencies
```

See `Documentation_template.md` at the package root for the full methodology and results.
