# Business Entity Resolution Pipeline (ML Challenge 2026)

This repository contains an end-to-end, high-performance Machine Learning pipeline for multi-source entity resolution, specifically designed to solve the **Amazon ML Challenge 2026**.

The system matches deduplicated reference entities from **Source 1** to zero, one, or many corresponding records in **Source 2** and **Source 3**, optimizing for the precision-weighted macro **$F_{0.5}$ metric**.

---

## 1. System Requirements & Environment

- **Python**: 3.9+ (tested on Python 3.9.18)
- **OS**: Linux / macOS / Windows
- **Key Dependencies**:
  - `polars` (Ultra-fast parallel DataFrame engine)
  - `rapidfuzz` (C++ SIMD-accelerated string similarity)
  - `lightgbm` (Histogram-based Gradient Boosted Trees)
  - `numpy`, `scipy`, `scikit-learn`, `tqdm`

Install all dependencies via:

```bash
pip install -r requirements.txt
```

---

## 2. Directory Structure

```text
code/business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── normalize.py       # Text canonicalization, legal suffixes, and token cleaning
│   ├── blocking.py        # Country-partitioned multi-pass inverted index blocking
│   ├── features.py        # Pairwise string similarity and structural feature extraction
│   ├── metrics.py         # Official macro F_0.5 evaluation metric with singleton handling
│   ├── model.py           # LightGBM classifier with F_0.5 threshold optimization
│   ├── pipeline.py        # End-to-end training and inference execution
│   └── run.py             # CLI entry point
├── requirements.txt       # Pinned dependencies
└── README.md              # Reproduction guide
```

---

## 3. End-to-End Reproduction Instructions

### Step 1: Run Full Pipeline (Training + Inference)

From the project root:

```bash
python3 code/business_entity_resolution/src/run.py \
    --mode all \
    --train-dir dataset/train \
    --test-dir dataset/test \
    --output-dir output
```

This will:
1. Load training records from `dataset/train/` (`train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`).
2. Run multi-pass blocking and candidate generation.
3. Extract pairwise string similarity and structural features.
4. Train the LightGBM classifier and optimize the decision threshold $\tau^*$ on a holdout validation set to directly maximize macro $F_{0.5}$.
5. Run candidate generation and model inference on `dataset/test/`.
6. Generate both required output files in `output/`:
   - `output/candidate_pairs.tsv`
   - `output/matching_results.tsv`
7. Automatically run `utils/validate_submission.py` to confirm zero format violations.

### Step 2: Separate Training & Prediction (Optional)

To train and save the model:

```bash
python3 code/business_entity_resolution/src/run.py \
    --mode train \
    --train-dir dataset/train \
    --model-path code/business_entity_resolution/model.pkl
```

To run inference using the trained model on test data:

```bash
python3 code/business_entity_resolution/src/run.py \
    --mode predict \
    --test-dir dataset/test \
    --output-dir output \
    --model-path code/business_entity_resolution/model.pkl
```

---

## 4. Methodology Highlights

1. **Strict Country Partitioning**:
   - Analysis of ground truth confirmed that true matches **never** cross national boundaries ($0\%$ cross-country matching).
   - All blocking and inference are partitioned by country (`US`, `India`, `France`), drastically reducing search space and eliminating cross-country false merges.
2. **Multi-Pass Inverted Index Blocking**:
   - Clean compact names (legal suffixes and domain extensions stripped).
   - Sorted name tokens (order-invariant matching).
   - Address numbers combined with street tokens and distinctive name tokens.
3. **High-Speed Pairwise Feature Engineering**:
   - SIMD-accelerated Levenshtein, Token Sort Ratio, Token Set Ratio, Partial Ratio, and Jaro-Winkler via `rapidfuzz`.
   - Structural address overlap, door/plot/postal number intersections, and empty address indicators.
4. **Precision-Weighted Macro $F_{0.5}$ Optimization**:
   - Precision is weighted $2\times$ over recall to heavily penalize false merges.
   - Threshold $\tau^*$ is selected via exhaustive grid search on validation data to maximize the exact competition macro $F_{0.5}$ metric while identifying singletons ($11.2\%$ of records).
