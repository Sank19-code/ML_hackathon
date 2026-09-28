---
title: "Entity resolution: plan to 0.988+ (algorithm v6 and execution)"
subtitle: "DevX Resolvers · Amazon ML Challenge 2026 · 26 Sep 2026"
---

Live version: https://claude.ai/code/artifact/5ec5c662-6153-4b08-b5c3-1a908f038250

Your pipeline scores 0.9854 held-out but 0.9792 on the leaderboard. Most of that gap is France (about 15% of test entities, estimated at 0.95 F0.5), so fixing France is the first job. Clearing 0.988 also needs India and the US to reach about 0.988 on test, which is at your pipeline's own measured ceiling. So 0.989–0.990 is unlikely without leakage; 0.984–0.986 is realistic.

# 1. Where the 0.021 is lost

France costs about a third of the total loss while holding 15% of the entities. Figures are derived from the v5 logs, `tl_tune_v3.log` and the test entity counts, assuming the public leaderboard mixes countries in the same proportion as the full test set.

| Segment | Test S1 entities | Est. test F0.5 | Share of total loss | Basis |
| --- | --- | --- | --- | --- |
| France | 259,452 (15.0%) | ~0.953 (0.947–0.955) | ~0.0070 | Implied by 0.9792 once India/US are fixed at their estimate |
| India | 809,986 (46.8%) | ~0.983 | ~0.0080 | Held-out 0.9835, minus ~0.001 train-to-test shift |
| US | 663,106 (38.3%) | ~0.985 | ~0.0058 | Held-out 0.9866, minus ~0.001 shift |
| **All** | 1,732,544 | **0.9792** | 0.0208 | Leaderboard |

- **France errors are confident, not borderline.** The model self-estimates France at 0.981, yet the leaderboard implies about 0.95. Threshold tuning can't fix confident mistakes, which is why France floor tuning moved nothing (0.97161 to 0.97162).
- **India/US loss is recall.** Held-out pair precision is 0.9975 but pair recall is 0.962. About 1.8% of true pairs never reach the model (blocking) and roughly 2% more are empty-address records under names shared by many entities.
- **What each target needs.** France at India/US level gives about 0.984. Reaching 0.988 needs every country at about 0.988 on test.

# 2. Can 0.991 happen? The error budget

No, not realistically. Held-out loss is 0.0146, and about 0.006–0.007 of it comes from records nothing in the data can place. Even a perfect pipeline scores about 0.992 held-out.

| Loss source | Share of true pairs | Est. held-out F0.5 loss | Recoverable? |
| --- | --- | --- | --- |
| Ambiguous by construction (empty address, name shared by many entities) | ~2.0% | ~0.006–0.007 | Mostly no; step 10 (E2) may win back a slice |
| Blocking misses that aren't ambiguous | ~1.2% | ~0.003–0.004 | Partly, via blocking changes |
| Classifier misses that aren't ambiguous | ~0.5% | ~0.001–0.002 | Yes: better model, calibration |
| False merges (0.25% of predicted pairs) | — | ~0.002–0.003 | Partly |
| **Held-out total** | | **0.0146** (0.9854) | Realistic floor ~0.010 → ~0.989–0.990 |

The leaderboard then adds the India/US shift (about −0.001 to −0.003) and France (about −0.005 today). A realistic best case is **0.986–0.988**.

# 3. Target changes vs your v5 pipeline

| # | Step | v5 now | Change | Basis | Est. LB gain |
| --- | --- | --- | --- | --- | --- |
| A0 | Leave-one-country-out proxy | None | Train US-only, score 10% India sample, and the reverse. Stand-in for France with labels | DGER (Xu & Wang 2024) | Enables A3–A6 |
| A1 | France leaderboard probes | No France-specific rules | Two submissions changing only France rows: drop same-house + one-word-swap pairs (~25.6k), then sibling-offset pairs (~7.9k) | `fr_rules.log` | 0 to +0.002 |
| A2 | France dev set | None | Hand-label 300–400 France pairs, chosen by entropy × divergence | DUAL (Xu & Wang 2024) | Enables A3–A4 |
| A3 | France self-training | US/India model + floor 0.40 | Refit stage 2 with France pseudo-labels (p ≥ 0.99 winners / p ≤ 0.01 or losers), ≤ 2 rounds | DADER; Noisy Student; TTA survey | +0.001 to +0.003 |
| A4 | LLM teacher for France | — | Qwen3-8B (Apache-2.0) labels 20–50k uncertain France pairs; keep only if ≥ 97% agreement with A2 | arXiv 2606.28823 | +0.001 to +0.002 (overlaps A3) |
| A5 | Synthetic French pairs | — | Re-dress US/India training pairs French-style (Rue/Bd/Allée, SARL/SAS/SA, region ↔ department) | Augmentation surveys | within France total |
| A6 | Per-country experts | One global model | US expert + India expert + global, blended for France with weights chosen on A0 | DAME (WSDM '22) | within France total |
| B1 | Shift-weighted training | Test-like weights only for evaluation | Adversarial train-vs-test density ratio as sample weights | DMM (AAAI-18); Park et al. 2020 | +0.0005 to +0.001 |
| B2 | Sibling augmentation | Sibling features only | Inject synthetic sibling negatives until the rate matches test (~1.9x) | Rotom (SIGMOD 2021) | +0.0005 to +0.0015 |
| B3 | Shift-invariant features | All 81 features | Drop/coarsen features that separate train from test (`n_cands`, rank, ambiguity counts) if A0 improves | DANN (Ganin et al. 2016) | with B1 |
| C1 | Cross-encoder (stage 3) | LightGBM only | xlm-roberta-base / mdeberta-v3-base on the uncertain band, stacked into stage 2 | Ditto (VLDB 2020) | +0.0005 to +0.002 |
| C2 | More LightGBM data | 12% entities, 2 folds | 20–40% entities, 3–5 folds | — | +0.0003 to +0.001 |
| D1 | Exact F0.5 decision | γ-power plug-in rule | Calibrate, then exact general F-measure maximizer per entity | Waegeman et al. JMLR 2014 | +0.0001 to +0.0004 |
| E2 | Collective resolution | Ambiguous records left out | Assign an empty-address record to the entity still missing a copy, only if posterior ≥ 0.8 | Structural inference | up to +0.001 |

Stay away from row-order or ID-pattern leakage; the top teams' packages are audited.

# 4. Algorithm v6: formal design

For every Source 1 entity e, output a set M(e) of Source 2/3 records maximising macro F0.5. France has no labels and borrows from the leave-one-country-out (LOCO) stand-in.

**Objective.** Per entity with true set T and predicted set S (an empty prediction for a singleton scores 1):

$$F_{0.5}(S,T) = \frac{1.25\,|S \cap T|}{0.25\,|T| + |S|}, \qquad \text{score} = \frac{1}{|E|}\sum_{e \in E} F_{0.5}\big(M(e), T(e)\big)$$

**Exact decision.** Sort an entity's calibrated probabilities p1 ≥ … ≥ pn. P_k is the Poisson-binomial distribution of true matches among the top k; Q_k the same for the rest plus one Bernoulli(m) for matches blocking missed. Pick the k with the highest value (k = 0 means predict nothing). Two O(n²) dynamic programmes per entity, n ≤ 60.

$$\mathbb{E}[F_k] = \sum_{a=0}^{k} \sum_{b} P_k(a)\,Q_k(b)\,\frac{1.25\,a}{0.25\,(a+b) + k}, \qquad \mathbb{E}[F_0] = (1-m)\prod_{i=1}^{n}(1-p_i)$$

**Shift weights and calibration.** d(e) is an adversarial classifier's probability that entity e looks like test; weights are clipped to [0.2, 5]. Each country gets a temperature T_c; France's is fitted on the LOCO stand-in.

$$w(e) = \frac{d(e)}{1-d(e)} \cdot \frac{n_{\text{train}}}{n_{\text{test}}}, \qquad \hat p = \sigma\!\left(\frac{\operatorname{logit} p}{T_c}\right), \qquad T_c = \arg\min_T \sum_{(e,r)} w(e)\,\ell\big(y_{er},\, \sigma(\operatorname{logit} p_{er}/T)\big)$$

**Pseudocode.**

```text
INPUT   S1, S2, S3 per country c;  ground truth G (train only)
OUTPUT  M(e) for every test Source 1 entity e

# ---------- shared ----------
 1  R <- Normalise(records)          rules + learned transliteration + libpostal expand (extra field)
 2  C(e) <- union of top-k lists, ranked by combined cosine, cap 60
            combined 25 (India 40) | name 10 | address 20 | house+street 5 (new)
 3  X1(e,r) <- pair features, minus the shift-sensitive set D (adversarial validation)

# ---------- training (India, US) ----------
 4  w(e) <- density ratio from adversarial train-vs-test classifier, clipped [0.2, 5]
 5  T_aug <- synthetic sibling negatives (offsets {1,2,3,4,5,7,9,11,13,21})
            + French-dressed copies of training pairs
 6  f1 <- LightGBM, k folds by entity, sampled entities + T_aug, weights w, early stop
    p1 <- out-of-fold f1
 7  X2 <- [p1, group features(p1), one-to-one margins];  f2 <- LightGBM;  p2 <- OOF
 8  U <- {(e,r) : 0.02 < p2 < 0.98}                (about 0.25 pairs per entity)
    g <- cross-encoder fine-tuned on U;  p3 <- stack(p2, logit g) on U, p2 elsewhere
 9  T_c <- temperature per country (weighted log-loss, weights w)
10  tune floor and reassignment margin on fold A, report F0.5 on fold B (and reverse)

# ---------- France adaptation ----------
11  LOCO: rerun 6-9 on US only -> score India sample, and India only -> US sample
          -> choose France temperature, expert weights and which steps to keep
12  pseudo-labels: P+ = one-to-one winners with p >= 0.99, P- = losers or p <= 0.01
    refit f2 with France rows at weight 0.5; <= 2 rounds; stop if dev F0.5 drops
13  experts: f_US, f_India, f_global;  p_FR = sum_k alpha_k f_k (alpha from LOCO)

# ---------- inference (per country) ----------
14  p <- sigma(logit p3 / T_c) for every candidate pair
15  one-to-one: owner(r) = argmax_e p(e,r); keep only (owner(r), r)
16  for each e: S*(e) <- argmax_k E[F_k]; k = 0 -> empty
17  reassign: r rejected by its owner goes to second-best e' if p(e',r) >= 0.9
              and it still beats every other entity; rerun 16 for e'
18  collective (empty-address r, name shared by n entities):
        pi(e|r) proportional to p(e,r) * Pr[e still lacks a copy from r's source]
        add r to M(e) only if pi >= 0.8 and E[F] of e rises
19  write matching_results.tsv + candidate_pairs.tsv; run validate_submission.py
```

# 5. Pipeline v6

Twelve cached stages, each writing to `work/v6/` so any stage can be rerun alone. New stages are marked **(new)**.

Flow: Raw TSVs → Normalise → Block → Pair features → Stage 1 GBDT → Stage 2 GBDT → Stage 3 cross-encoder **(new)** → Calibrate **(new)** → One-to-one → Exact F0.5 decision **(new)** → Collective E2 **(new)** → Write + validate. France adaptation feeds stages 5 and 7.

| # | Stage | Module | Input → output | Time (laptop) | Gate to keep it |
| --- | --- | --- | --- | --- | --- |
| 1 | Normalise | `normalize.py` (+ new `postal.py`) | TSVs → `norm/*.parquet` | cached from v5 | Held-out F0.5 ≥ +0.0002 |
| 2 | Block | `blocking.py` (+ house+street list) | norm → `cands/*.parquet` | cached / ~40 min | Recall up at same candidate count |
| 3 | Pair features | `features.py` (drop set D) | cands → feature chunks | on the fly | Adversarial AUC drops, held-out holds |
| 4 | Stage 1 GBDT | `model.py` (folds, weights, aug) | features → `p1` | ~3–3.5 h | Held-out and LOCO both up |
| 5 | Stage 2 GBDT | `decide.py` + `model.py` | p1 + group features → `p2` | ~1.5 h | Held-out up |
| 6 | Stage 3 cross-encoder (new) | new `xenc.py` | uncertain pairs as text → `p3` | GPU only; skip on CPU tonight | Held-out up on uncertain band |
| 7 | Calibrate (new) | new `calibrate.py` | OOF p3 + weights → `T_c` | minutes | Weighted log-loss down |
| 8 | One-to-one | `decide.one_to_one` (+ reassignment) | p → owner per record | minutes | Held-out up |
| 9 | Exact F0.5 decision (new) | new `decide.gfm_select` | owners → M(e) | minutes | Beats γ rule on fold B |
| 10 | Collective E2 (new) | new `collective.py` | M(e) + ambiguous records → M(e) | minutes | Held-out up, no singleton loss |
| 11 | France adaptation | `pipeline.py` (LOCO, pseudo-labels, experts) | feeds stages 5 and 7 | ~2 h | LOCO and dev set both up |
| 12 | Write + validate | `run.py --mode predict` | → `output/*.tsv` | ~1 h | Validator PASS |

# 6. Laptop-only execution

The whole plan runs on the laptop (16 threads, 16 GB RAM, CPU). Run one heavy job at a time.

| Stage | AWS version | Laptop version | Cost |
| --- | --- | --- | --- |
| Stage 1 GBDT | 40% of entities, 5 folds, 4–6 h | 20% of entities, 3 folds, lr 0.05, `max_bin` 63, ~3–3.5 h | About half of this step's gain |
| Stage 3 cross-encoder | xlm-roberta-base on ~500k pairs, GPU | Skip tonight, or multilingual MiniLM-L12 (Apache-2.0) on 50k pairs, int8, ~2 h | Smaller gain, ≤ +0.0005 |
| Everything else | — | Unchanged | None |

**Overnight schedule (IST).**

| Time | Job | Output |
| --- | --- | --- |
| 22:30–23:00 | A1 France probe files, then submit | Two leaderboard scores |
| 23:00–00:30 | A0 leave-one-country-out on 10% samples + B3 adversarial feature check | France proxy score, drop set D |
| 00:30–04:00 | Stage 1 retrain (B1 weights, B2 sibling augmentation, drop set D) | New p1 |
| 04:00–05:30 | Stage 2 retrain with France pseudo-labels (A3) | New p2 |
| 05:30–06:00 | Calibrate, exact F0.5 decision, collective E2; tune on held-out | Held-out F0.5 |
| 06:00–07:00 | Predict test + validator | Submission files |

**Before starting.** Set the Windows power plan to Best performance, keep the charger in, turn off sleep (the README warns Windows throttles background processes). Run each stage as its own process, following `work/v5_chain.sh`, with outputs in `work/v6/`.

**Not written yet.** `xenc.py`, `calibrate.py`, `decide.gfm_select`, `collective.py` and the A0/A1/B1–B3 scripts still need code before the run.

**Checklist**

- [ ] A1 France probes submitted
- [ ] A0 leave-one-country-out proxy + B3 feature check
- [ ] B1 + B2 + B3 + A5 stage-1/2 refit launched
- [ ] A2 France dev set labelled
- [ ] A3 + A6 + D1 + E2, final submission (pick on held-out + A0 + A2, not public LB alone)

Open question: how many leaderboard submissions per day are allowed?

# 7. Sources

- Folder: `Documentation_template.md`, `work/v5_train.log`, `work/v5_predict.log`, `work/fr_rules.log`, `work/tl_tune_v3.log`, `work/tl_tunefr_v3.log`, problem statement PDF; code audited: `model.py`, `decide.py`, `features.py`, `blocking.py`, `pipeline.py`
- Papers in `researchpapers/`: Xu & Wang, Neurocomputing 2024 (DGER/DUAL); Ganin et al., JMLR 2016 (DANN); Cao, Long & Wang, AAAI-18 (DMM); Trabelsi, Heflin & Cao, WSDM '22 (DAME); Baktashmotlagh et al., JMLR 2016; Liang, He & Tan, TTA survey (arXiv 2303.15361); Wang et al., data augmentation survey (arXiv 2405.09591); Zhong, Zhou & Wang, Neural Processing Letters 2025
- Tu et al., Domain Adaptation for Deep Entity Resolution (SIGMOD 2022): https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/entity-sigmod-2022.pdf
- DADER (VLDB 2022): https://www.vldb.org/pvldb/vol15/p3666-fan.pdf
- Labeling Training Data for EM Using LLMs (arXiv 2606.28823): https://arxiv.org/html/2606.28823
- Fine-tuning LLMs for Entity Matching (arXiv 2409.08185): https://arxiv.org/abs/2409.08185
- Entity Matching using LLMs (arXiv 2310.11244): https://arxiv.org/html/2310.11244v4
- Ditto (VLDB 2020): https://arxiv.org/abs/2004.00584
- Rotom (SIGMOD 2021): https://www.miaozhengjie.com/assets/pdf/rotom-sigmod21.pdf
- On the Bayes-Optimality of F-Measure Maximizers (JMLR 2014): https://jmlr.org/papers/volume15/waegeman14a/waegeman14a.pdf
- Optimizing F-measures: A Tale of Two Approaches (ICML 2012): https://www.comp.nus.edu.sg/~leews/publications/fscore.pdf
- Sparkly (VLDB 2023): https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf
- libpostal (MIT): https://github.com/openvenues/libpostal
- Park et al., Calibrated Prediction with Covariate Shift (AISTATS 2020): https://proceedings.mlr.press/v108/park20b/park20b.pdf
- One-to-one matching algorithms for ER (VLDB Journal 2023): https://arxiv.org/abs/2112.14030v1
- Noisy Student (CVPR 2020): https://arxiv.org/abs/1911.04252
- Qwen3-8B licence (Apache-2.0): https://huggingface.co/Qwen/Qwen3-8B/blob/main/LICENSE
