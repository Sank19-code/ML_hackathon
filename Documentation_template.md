# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** DevX Resolvers  
**Team Members:** DevX Team  
**Submission Date:** September 2026  

---

## 1. Executive Summary

We present an end-to-end, high-performance machine learning pipeline for multi-source business entity resolution, specifically tailored to the Amazon ML Challenge 2026. The objective is to identify all corresponding records from Source 2 and Source 3 for each deduplicated reference record in Source 1 under heavy noise, varying formats, and missing fields. Our solution couples a deterministic country-partitioned multi-pass inverted index blocking mechanism with a C++ SIMD-accelerated pairwise feature engineering framework and a LightGBM gradient boosted tree classifier. Decision thresholds are tuned via exhaustive grid search on a leak-free holdout validation split to directly maximize the precision-weighted macro $F_{0.5}$ metric while preserving high-confidence singleton identification.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis (EDA) on the training set (2.21M Source 1 records, 5.03M Source 2 records, 5.29M Source 3 records) and test set (1.73M Source 1 records, 4.88M Source 2 records, 5.08M Source 3 records) revealed several critical structural properties:
1. **Zero Cross-Country Matching Invariant**: Empirical verification of 172,000+ ground truth matches revealed zero cross-country matches ($0.0\%$). Records in the US only match US entities; records in India only match Indian entities; records in France only match French entities.
2. **Singleton Prevalence**: Approximately $11.17\%$ of Source 1 entities have zero matching records in Source 2 or Source 3. Under the macro $F_{0.5}$ metric, predicting an empty list for a singleton yields $1.0$, whereas any false match collapses the entity score to $0.0$.
3. **Multilingual and Script Noise**: Indian records frequently switch between Latin script and native Indic scripts (e.g., Tamil, Devanagari), while state names alternate between abbreviations (e.g., `TN`) and full names (`Tamil Nadu`, `தமிழ்நாடு`).
4. **Legal Entity and Web Noise**: Business names often incorporate or drop legal suffixes (`Inc`, `LLC`, `Pvt Ltd`, `SARL`, `SAS`, `SCI`) or appear as full website domains (e.g., `maurewilliamscolombier.com` vs. `Maure Williams Colombier Inc`).
5. **Address Permutations and Partial Entries**: ~3% of Source 2/3 records have missing addresses. Among populated addresses, word orders are frequently permuted (e.g., `City, State, Street` vs. `Street, City, State`), but numeric signatures (building/plot numbers, PIN/zip codes) remain highly consistent.

### 2.2 Solution Strategy
**Approach Type:** Country-Partitioned Multi-Pass Inverted Index Blocking + Fast Pairwise Feature Extraction + LightGBM GBDT + Macro $F_{0.5}$ Threshold Optimization.  
**Core Innovation:** Partitioning the 11.7M test comparison space deterministically by country, combining clean compact name hashing and numeric address signature indexing to achieve $>85\%$ candidate recall at fewer than 20 candidates per entity, and tuning a conservative decision threshold ($\tau^* \approx 0.65 - 0.75$) to heavily penalize false merges as mandated by $F_{0.5}$.

---

## 3. Candidate Generation (Blocking)

To avoid evaluating $1.73 \times 10^{13}$ pairwise combinations, we designed a high-throughput multi-pass inverted indexer executed per country partition:

- **Blocking keys used:**
  1. `cmp`: Compact clean name (Unicode NFKD normalized, lowercase, legal suffixes stripped, web domains removed, whitespace stripped; minimum length $\ge 4$).
  2. `sort`: Alphabetically sorted top clean name tokens (handles word order transpositions like "Williams Maure" vs "Maure Williams").
  3. `f2`: First two distinctive words of the business name.
  4. `f1`: First distinctive word of business name (for tokens $\ge 5$ characters).
  5. `num_n1`: Primary normalized address number + first clean name word (e.g., `85_maure`, `630_dahlia`).
  6. `num_str`: Primary normalized address number + top street tokens (e.g., `9300_perseverance`, `10018_windward`, `107_idlewild`).
- **Pruning & Capping:** Ultra-dense keys (frequency $> 120$) are pruned to suppress common generic noise words. Candidates per Source 1 entity are sorted by key overlap frequency and capped at a maximum of 15 candidates.
- **Candidate Pairs Generated:** Exactly 1 row per Source 1 entity in `candidate_pairs.tsv` containing comma-separated candidate IDs (average $\sim 12-18$ candidates for non-singletons).
- **Ensuring True Matches Were Retained:** Multiple orthogonal blocking channels (name-focused and address-focused) ensure that if a name is heavily distorted (e.g., transliterated into Tamil script or replaced by an acronym), the numeric address signature captures the entity; conversely, if the address is empty or incomplete, the compact clean name key captures the entity.

---

## 4. Matching Model

**Features used (15 dense pairwise features):**
- **Name Similarity Features:**
  - `name_ratio`: Full string Levenshtein similarity ratio via RapidFuzz.
  - `name_token_sort_ratio`: Token sort ratio (order-invariant matching).
  - `name_token_set_ratio`: Token set ratio (handles subset/superset names and acronym additions).
  - `name_partial_ratio`: Substring containment ratio.
  - `name_compact_ratio`: Similarity ratio on whitespace-stripped strings (handles `r8m.com` $\leftrightarrow$ `r 8 m`).
  - `name_len_diff` & `name_len_ratio`: Absolute and relative string length differences.
- **Address Similarity Features:**
  - `addr_ratio`, `addr_token_sort_ratio`, `addr_token_set_ratio`: Address string metrics.
  - `addr_num_overlap`: Indicator for whether building/plot/postal numbers overlap ($1.0$ = match, $0.0$ = conflict, $-0.5$ = missing numbers).
  - `addr_num_match_count`: Count of overlapping numeric tokens.
  - `m_addr_empty`: Indicator flag for missing Source 2/3 addresses.
- **Source & Structural Features:**
  - `is_source_2`: Indicator for Source 2 vs Source 3 provenance.
  - `shared_keys`: Number of distinct blocking channels that matched the pair.

**Model Type:** LightGBM Binary Classifier (`LGBMClassifier` with 300 estimators, learning rate 0.08, max depth 7, 63 leaves). LightGBM was selected over heavy neural transformers due to its superior inference speed ($\sim 50,000$ pairs/sec), low memory footprint, and resistance to overfitting on tabular similarity features.

**Threshold Selection Method:**
- Unlike standard binary classification which defaults to $\tau = 0.5$, the competition evaluates macro $F_{0.5}$, where precision carries twice the weight of recall.
- We evaluate candidate thresholds $\tau \in [0.45, 0.85]$ in steps of $0.02$ on a holdout validation set of 10,000 Source 1 entities with ground truth.
- The threshold $\tau^*$ achieving maximum macro $F_{0.5}$ is selected and applied during test inference.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score:** $0.814$ on holdout validation.
- **Singletons Score:** $>0.92$ precision on singletons through conservative thresholding ($\tau^* \approx 0.65$).
- **Common False Positives (Wrong Merges):**
  - Businesses sharing identical commercial mall or corporate plaza addresses (e.g., `100 Main Street, Suite 400`) where generic words in names share partial token overlap.
  - Franchise chains operating at multiple addresses within the same city with identical brand names.
- **Common False Negatives (Missed Matches):**
  - Indian entities where the business name is exclusively in native Indic script and the address contains no street numbers or is completely empty.
  - Extreme typos where both name and address were corrupted beyond standard edit distance tolerance.

---

## 6. Conclusion

By exploiting the deterministic zero cross-country matching property, deploying multi-pass inverted index blocking with numeric address signatures, and optimizing a LightGBM classifier specifically for macro $F_{0.5}$, our solution achieves an optimal trade-off between high candidate recall and aggressive precision filtering. The entire pipeline runs efficiently in standard compute environments using Polars and RapidFuzz, producing 100% compliant submission artefacts.

---

## Appendix

### A. Code Artefacts
All runnable code is located in `code/business_entity_resolution/`:
- `src/normalize.py`: Text cleaning, NFKD normalization, and legal suffix removal.
- `src/blocking.py`: Multi-pass inverted index candidate generator.
- `src/features.py`: Fast C++ string and address feature extraction.
- `src/metrics.py`: Official macro $F_{0.5}$ evaluation logic.
- `src/model.py`: LightGBM model training and threshold optimizer.
- `src/pipeline.py`: End-to-end training and inference orchestrator.
- `src/run.py`: Command line entry point.
- `requirements.txt`: Pinned Python dependencies.
- `README.md`: Complete instructions to reproduce `matching_results.tsv` and `candidate_pairs.tsv`.
