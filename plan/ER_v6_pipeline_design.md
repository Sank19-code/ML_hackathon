---
title: "Entity resolution v6: research-backed pipeline and algorithm"
subtitle: "DevX Resolvers · Amazon ML Challenge 2026 · 26 Sep 2026 · design only, no code yet"
---

Starting point: v5, held-out 0.9854, public leaderboard 0.979211. This document is based on the
8 papers in `researchpapers/`, a targeted web search (about 40 sources) and a check of the v5
code. It answers the problems raised against `ER_v6_execution_plan.md` (P1-P9 below).

**Realistic target:**
- Held-out: 0.987-0.988. The held-out ceiling with the current information is 0.989-0.990.
- Leaderboard: **0.981-0.984**, if the leaderboard probes confirm the France distractor
  hypothesis; otherwise about 0.980-0.981.
- 0.99 on the leaderboard is still not reachable legitimately.

# 0. The problems this design has to solve

| # | Problem | Fix in this design (section) |
|---|---|---|
| P1 | France (about 15% of test, implied F0.5 0.95-0.97) has no labels, and its errors are *confident*: the model rates France at 0.981 | Unlabelled-country pathway (§4.9), distractor augmentation (§4.5), content-core features (§4.4), leaderboard probes (§4.10) |
| P2 | Hand-labelling France pairs (A2 / DUAL) breaks fair play, so there is no France dev set | Validation harness with label-free proxies (§4.1); leaderboard probes run under a Ladder discipline (§4.10) |
| P3 | Naive self-training copies confident errors | Self-training is dropped from the core plan; a guarded variant is optional and gated by a LOCO rehearsal (§4.11) |
| P4 | An 8B LLM teacher cannot run at pair scale on this CPU | Dropped. Its job goes to content-core features and a cheap counterfactual audit |
| P5 | libpostal and hand-typed French lists are external data or knowledge | Token roles learned from each country's own records (§4.3) |
| P6 | About 1.2% resolvable blocking misses plus about 2% ambiguous records | Copy-anchored second-hop retrieval (§4.2); greedy reassignment (§4.8); each is gated by the 0.8-precision triage rule |
| P7 | The γ-power plug-in rule is only an approximation; the probabilities are not calibrated | Per-country isotonic calibration (§4.7) and an exact Poisson-binomial expected-F0.5 DP with a learned blocking-miss term (§4.8) |
| P8 | The stage-1 fit commits about 19 GB, so no larger sample and no more folds | Memory-safe binned datasets (§4.6): zero-copy list of arrays or `lgb.Sequence`, then `save_binary` |
| P9 | Open-set country rule; shift-sensitive top features (rank, n_cands); over-fitting the public leaderboard | No country literals in the logic; one "no labels" code path; shifted features get extra relative copies and are never dropped unless LOCO shows it helps (§4.4, §4.9); Ladder discipline |

**Compliance finding in the current v5 code (fix in v6):** `src/pipeline.py:49` has
`"block_country": {"India": {"k_comb": 40, "max_candidates": 60}}`. This keys pipeline logic on
a country name, which the rules forbid ("treat country as an open set... do not hard-code").
`normalize.py` also has hand-typed state and region maps for US/India/France (lines ~255-330).
§4.3 replaces both with rules learned from the data.

# 1. What the research actually says (and what it rules out)

1. **Neural domain adaptation does not transfer to a GBDT.** DANN's gradient reversal, the
   DistilBERT experts in DAME/DGER, DME/DMM subspace learning and the DADER aligners all align
   a *learned representation*. LightGBM on 81 hand-built features has no such representation.
   DADER even reports that feature-level adaptation (the part a GBDT cannot do) beats
   instance-level adaptation. Three things do transfer:
   - leave-one-domain-out simulated targets (DGER §4.2.1, DAME §4.2.3);
   - a domain classifier used for diagnostics and PAD (DANN §3.2);
   - output-level experts (DGER Fig. 8).
2. **With only 2 labelled countries, experts must not be the backbone.** DAME Fig. 2: a
   mixture of experts beats the global model only with 5 or more experts. Experts are
   therefore used only inside the unlabelled-country head.
3. **Zero-shot entity matching fails by losing precision, not recall** (DAME Table 4, DGER
   Table 4: precision 0.30-0.74 with recall kept). This is exactly France's pattern:
   confident false merges.
4. **Label-free methods cannot fix a changed labelling function.** Ben-David/DANN Theorem 2:
   target risk is bounded by source risk + divergence + β. Only label information reduces β.
   Covariate reweighting, prior correction (Saerens EM / BBSE), entropy-based test-time
   adaptation and pseudo-labelling all reproduce the source concept. If France's
   sibling/word-swap pairs are *different businesses* while the same shapes are 96-99.5%
   matches in train, only three things can help:
   - (a) synthetic negatives built from train;
   - (b) leaderboard feedback;
   - (c) contrasts supported by labels (US versus India conventions).
   This explains the two null results already measured: France-weighted rule tuning and
   per-bucket prior correction.
5. **Reweighting is weak for trees.** DMM Tables 2-4: KMM gains only +0.2 to +0.6 over the
   baseline. Zadrozny 2004: trees are "global" learners, so reweighting moves *where* the
   splits go, not p(y|x). Decision-level reweighting is provably near-inert because the
   per-entity F-optimal set is local (it explains 0.98177 -> 0.98174). Use it only as
   entity *resampling*, after a go/no-go test.
6. **Self-training under a large shift can have unbounded error** (Kumar et al. ICML 2020;
   Ruder & Plank 2018). Tri-training with agreement filters and negative-heavy labels is the
   safest form, and it still cannot fix confident errors.
7. **Exact expected-F maximisation beats thresholding when calibration is good** (Ye et al.
   ICML 2012; Jansche 2007). Example: for p = (0.6, 0.6) the plug-in scores k=2 at 0.652, but
   the exact value is 0.627. The plug-in overstates larger sets.
8. **The marginal-inclusion rule gives cheap triage for every recall idea.** Adding a pair
   raises expected F0.5 only if its precision is above F/1.25, about 0.78 at F 0.97. That
   explains why the 75%-precise legal-form disambiguation gave nothing, and it gates
   §4.2 and §4.8.

# 2. v6 pipeline at a glance

```
            ┌──────────────── S0 harness: held-out · LOCO×2 · leave-one-shape-out · reweighted · S1-S1 FP probe ───────────────┐
records ─► S1 normaliser v6 (rules + induced token roles, no country literals) ─► S2 blocking (+ data-driven depth,
           optional copy-anchored 2nd hop) ─► S3 pair features (+ content-core, exact-twin, relative copies)
        ─► S4 training data (+ rate-matched synthetic orphan distractors, label-safe real negatives)
        ─► S5 stage 1 (binned cache, 20-25% entities, 3 folds) ─► stage 2 (group features)
        ─► S6 calibration (isotonic per labelled country · pooled + LOCO temperature for unlabelled)
        ─► S7 decision: blocking-miss Poisson · exact Poisson-binomial E[F0.5] DP · exclusivity-aware greedy assignment
        ─► S8 unlabelled-country head (θ_u from worst-case LOCO; expert disagreement; out-of-support shrink)
        ─► S9 probe-gated bucket offsets (≤ 20 binary leaderboard decisions) ─► matching_results.tsv
```

# 3. Order of work and gates

Each phase ends in a submission or a go/no-go decision. The next phase starts only after its
gate passes.

| Phase | Work | Compute | Gate / output | Expected LB |
|---|---|---|---|---|
| **P0 diagnostics** (no retrain) | §4.1 harness; PAD/Hellinger shift map; S1-S1 FP probe; conflict rate; count-prior monitor; counterfactual `s_house` audit; transliteration coverage check; token-role induction dry run; second-hop recoverability; exact-twin co-occurrence | ~2-3 h | Tells which of §4.2-4.5 are worth a retrain | 0 |
| **P0b France probes** | 2-3 differential submissions: France-only, drop suspect bucket (sibling / same-house word swap / both) | minutes each | **Key decision:** are the France distractor buckets really errors? | +0 to +0.002 (keep the winning drop) |
| **P1 decision layer** (no retrain) | §4.7 calibration + §4.8 exact DP + greedy reassignment on saved v5 scores; tuned on held-out | ~1-2 h | Held-out ≥ +0.0003, else keep plug-in | +0.0003 to +0.001 |
| **P2 engineering** | §4.6 memory-safe cache; learning curve 6/12/24% | ~3 h | Identical predictions on 12%; choose sample size | 0 |
| **P3 retrain #1** | §4.3 normaliser v6 + §4.4 features + §4.5 distractors (only if P0b confirms) + larger sample | ~4-5 h | Harness acceptance rule (§4.1) | +0.001 to +0.003 |
| **P4 LOCO + unlabelled head** | US-only and India-only models (6%), θ_u, expert disagreement, out-of-support shrink, stage2u | ~4-5 h | Both LOCO directions improve | +0.0003 to +0.002 (France) |
| **P5 optional** | §4.2 second hop, §4.11 guarded self-training | ~4-6 h | Only if P0 measurements pass | 0 to +0.001 |

**Timeline:** about 3-4 days of wall-clock time, running one heavy job at a time (16 GB RAM)
with overnight runs. P0/P0b/P1 fit in the first day and already produce a better submission.

# 4. Algorithms

## 4.1 Validation harness (P2, P9)

Every change is scored on five proxies, and all results for a change go into one table:

1. **In-domain held-out** per labelled country: entity split, as now.
2. **LOCO ×2.** Train stage 1+2 on US only (6% entity sample, ~1.5 h) and score India
   *through the unlabelled-country code path*. Repeat India → US. This is the only labelled
   "unseen country" we have (DGER §6.2.1, DomainBed).
3. **Leave-one-shape-out.** Hold out train entities whose candidate lists contain sibling-offset
   or same-house word-swap candidates, or more than the median number of uncertain candidates.
   This simulates a test population rich in distractors.
4. **Reweighted held-out.** Entity-level density ratio from a cross-fitted train-vs-test domain
   classifier, clipped at 8, with Kish ESS ≥ 30% reported. It is used for *model* selection,
   not for decision tuning.
5. **Label-free precision probes on test:**
   - **S1-S1 false-positive rate.** Near-duplicate S1-S1 pairs are distinct businesses by
     construction (Auto-FuzzyJoin reference-table assumption). Put the second S1 record in
     the S2 slot, score it and count acceptances per shape and per country. This compares
     variants and countries, not absolute values.
   - **Conflict rate.** Records that pass the inclusion test for 2 or more entities give a
     lower bound on false positives.
   - **Count-prior monitor.** Accepted pairs per entity per source compared with train
     (3.37 vs 3.35 confident candidates suggests the generator keeps per-entity match
     counts).

**Acceptance rule** (Gulrajani & Lopez-Paz: "a DG method is incomplete without its selection
rule"):
- held-out Δ ≥ −0.0002;
- mean LOCO Δ > 0, with neither direction below −0.001;
- the France S1-S1 false-positive rate on suspect shapes does not rise.

Entity bootstrap (1000 resamples) gives the SDs. A change is accepted only if its gain exceeds
2 SD on at least one proxy.

**Sanity check of the harness:** the LOCO drop (in-country minus cross-country F0.5) should be
at least the observed held-out → leaderboard gap for India/US (about 0.006). If it is much
smaller, LOCO is too easy and more weight goes on the probes.

## 4.2 Blocking (P6, P9)

- **Data-driven depth replaces the `"India"` literal.** Per country, choose `k_comb` from a
  statistic of the country's own lists, for example the share of S1 entities whose top-25
  combined list is saturated by records sharing the S1 name, or a cosine-gap rule: stop when
  the score drops below `α ×` the top score. Calibrate on train so India still gets 40/60 and
  the US 25/50.
- **Copy-anchored second hop** (optional, P5 phase). For each S1 entity take its confident
  copies C(e) (owner = e, p ≥ 0.9, about 3.3 per entity). Query the existing S2/S3 index with
  each copy's name, address and combined strings (top 5-10), and add retrieved records not yet
  in e's list, with the features "retrieved via copy" and "similarity to that copy". Exclude
  sibling-offset and one-token-swap shapes relative to the copy: those neighbours are exactly
  the distractors.
  - **Gate (measure on train first):** among blocking-missed true pairs, the share recovered
    at top 10 must be ≥ 25%, and the added pairs must reach held-out precision ≥ 0.8.
  - Expected +0.0005 to +0.001 if the gate passes; the cost is about 1-2 h of queries plus
    15-25% more scoring.

## 4.3 Normaliser v6: induced token roles (P5, P9, P1)

For every country string, run the same code on its own S1/S2/S3 records (train + test, no
labels):

- **Legal forms:** tokens with document frequency above the 99.5th percentile that sit in the
  last 1-2 name positions for many distinct names (position concentration ≥ 0.8). Should find
  SARL/SAS/EURL, PVT/LTD, LLC/INC.
- **Street types:** high-DF tokens adjacent to the house number or street name (rue, avenue,
  boulevard, chemin / road, nagar / st, ave).
- **Generic words:** name tokens with high DF across distinct entities in the same locality.
  They are down-weighted in a name-similarity variant, never deleted.
- **Region/locality tables:** the most frequent value in the last address component, grouped
  by co-occurrence. This replaces the hand-typed state/department maps where it reproduces
  them.
- **Abbreviation pairs** (simplified Arasu et al. 2009): token alignments (a, b) mined from
  train-labelled pairs and from near-duplicate S2↔S3 records within a country. Keep a pair if
  a is a prefix or subsequence of b, support ≥ 20 and consistency ≥ 0.9.

**Validation without labels:**
- On US/India the induced tables must recover ≥ 90% of the current hand lists.
- On France, the per-feature Hellinger distance to train must fall.
- LOCO: induce India tables while training on US, and check India F.
- Also run the **transliteration cross-fit check**. `translit.json` is learned from *all*
  train pairs, including the held-out ones, so held-out may be optimistic for native-script
  India. If the coverage rate differs between India train (OOF) and India test, learn the
  dictionary per fold.

## 4.4 Features (P1, P9)

- **Content-core name features.** core(name) = tokens minus induced legal forms minus tokens
  equal to either record's locality. Add:
  - core Jaccard;
  - the IDF mass of non-shared core tokens;
  - a flag "both cores non-empty and disjoint";
  - the group feature "another candidate at the same house has a matching core".
  This targets `Tourcoing Ecole SARL` vs `Tourcoing Amis SARL`.
- **Exact-twin group feature.** For a sibling-offset or word-swap candidate: does the entity
  also have an exact-house / exact-core candidate *from the same source*? This separates "a
  second copy with a typo" (India) from "a neighbour business" (US, France).
- **Relative copies of shift-sensitive features.** Add n_conf (p1 ≥ 0.5, which is invariant:
  3.37 vs 3.35), n_unc, rank among non-sibling candidates, rank within the same source, and
  within-country percentiles of IDF cosines. The raw rank/n_cands **stay**: dropping top-3
  features costs accuracy, and their shift is partly informative. The relative copies are kept
  only if LOCO and leave-one-shape-out do not get worse.
- **Optional `s_house` counterfactual.** Re-score an accepted pair with the S2 house number set
  equal to the S1 number and with an offset applied. If the probability barely moves, the
  model ignores the house number (a shortcut acceptance). Use it as an audit first; if it
  separates true from false siblings on held-out, make it a stage-2 feature.

## 4.5 Training data: rate-matched synthetic distractors (P1, P3, P9)

This is the only legitimate way to feed β-type (concept) information to the model. **It runs
only if the P0b probes show that France's suspect buckets are worse than the model thinks.**
Otherwise it teaches the opposite of the train labels.

```
for each sampled train entity e, each true copy r of e (with house number / ≥2 core tokens):
    with prob ρ_sib : r~ = copy(r); house += δ, δ ~ test-observed offset histogram {1,2,3,4,5,7,9,11,13,21}
    with prob ρ_swap: r~ = copy(r); replace one generic core token by another generic token
                      of the same country's TRAIN vocabulary with similar DF
    add the light character noise that real copies show; new id; same source
    r~ is an ORPHAN: label 0 for every entity; insert it into every list that contains r
recompute rank / n_cands / context / group features on the augmented lists
ρ per country chosen so the per-entity sibling / swap candidate counts match test (≈2.2× siblings)
```

- **Real negatives with guaranteed labels.** Deeper-list orphans (train S2/S3 records not in
  the ground truth are non-matches for every entity) move n_cands, rank and the margins toward
  the test range. This is safe even without the probe.
- **Realism gate.** The adversarial AUC of synthetic vs test pairs of the same shape must be
  ≤ 0.6, and the Fréchet/KS distance per feature is reported.
- **Model-centric affinity.** Held-out F with synthetic distractors injected, divided by plain
  held-out F, should reproduce about the 0.006 held-out → test drop *before* retraining, and
  approach 1 after it.
- **Rules:** use the train vocabulary only (no test tokens in generated records); cap the
  augmented entities at 20-30% of the sample.
- **Do not:** use feature-level mixup, SMOTE or table masking. Those vectors are not
  realisable pairs, and the classes are not scarce (7.6 M positives).

## 4.6 Memory-safe training (P8)

Culprits in v5:
- `pipeline.py:368` calls `np.vstack(Xs)` while `Xs` is still alive, so two full float32 copies
  exist at once;
- `pipeline.py:395` does the same for stage 2;
- `pipeline.py:237` and `features.py:231` each make an extra copy through `.astype(np.float32)`.

Fix:
1. Featurise into preallocated float32 C-order blocks, using `astype(..., copy=False)`.
2. `lgb.Dataset(data=[X_in, X_us, ...], label=..., free_raw_data=True)`. LightGBM 4.7 sends a
   list of arrays to `LGBM_DatasetCreateFromMats` without concatenating it. Alternatively, wrap
   `np.load(mmap_mode='r')` chunks in an `lgb.Sequence` (batch 200k,
   `bin_construct_sample_cnt` 1e6).
3. Call `.construct()`, `save_binary('stage1.bin')`, then `del` the raw arrays. Fold models use
   `full.subset(idx)`, one fold at a time.
4. Stage 2: `base.add_features_from(G20)` instead of re-stacking 81+20 columns.
5. `force_col_wise=True` and `max_bin` 127 (1 byte per cell). 30 M rows × 81 features is then
   about 2.4 GB binned instead of 9.7 GB raw.

Check: identical predictions on the 12% sample (max |Δp| < 1e-6), with peak memory logged. Then
run a learning curve at 6/12/24% to decide whether 2-3× rows or 3 folds are worth it (expected
+0.0003 to +0.001 held-out).

## 4.7 Calibration (P7)

- For each labelled country string, fit **isotonic** regression of y on OOF stage-2 p over
  held-out entities; use Platt on the logit if there are fewer than 5k pairs. Store it keyed by
  the data's own country string.
- **Pooled calibrator** fitted on all labelled countries.
- **Unlabelled country:** p′ = σ(logit(pooled(p)) / T_u). T_u is the mean of the two LOCO
  temperatures (US-only model on India, and the reverse), clipped to [1, 1.5]. Optionally use
  one T per suspect shape if LOCO miscalibration is concentrated there.
- Optional shift-aware temperature (AdapTable-style): z′ = a0 + a1·z + a2·d + a3·z·d, where d
  is the domain-classifier logit, fitted on LOCO. It is near-identity for train-like pairs.
  Use it only if the P0 diagnostic shows that France's confident errors have high d. If the
  distractors look train-like, no calibration can catch them.

## 4.8 Decision: exact expected-F0.5 DP + exclusivity-aware assignment (P7, P6)

For each S1 entity with calibrated, sorted probabilities p1 ≥ p2 ≥ ... (p ≥ 1e-3; the rest of
the tail is summed into a Poisson term):

```
M_e ~ Poisson(λ_e)            # true copies outside the candidate list (blocking misses)
λ_e = small LightGBM Poisson regression on TRAIN entities
      (target = #true matches not in candidates; features: n_cands, max/mean p1, c1, c2,
       empty-address flag, address count); keyed by labelled country, pooled for unlabelled
E[F | k=0] = Π_i (1 - p_i) · exp(-λ_e)                     # empty prediction scores 1 only if truly empty
E[F | k≥1] = Σ_a Σ_c P(A_k = a) · P(B_k + M_e = c) · 1.25a / (k + 0.25(a + c))
     A_k ~ PoissonBinomial(p_1..p_k)   (prefix pmfs by repeated Bernoulli convolution)
     B_k ~ PoissonBinomial(p_k+1..p_n) (suffix pmfs), convolved with Poisson(λ_e), pmfs truncated at 25
k* = argmax_{k=0..15} E[F | k]          → predict the top-k*
```

- Under independence the optimal set is a top-k (Ye et al. Theorem 9). The DP runs in numba,
  `@njit(parallel=True)` over entities, in under a minute for about 2 M entities.
- It removes the three inconsistencies in `decide.py:161-187`:
  - the floor drops probability mass from the expected true count;
  - `_p0` ignores floored candidates and the missed term;
  - γ is standing in for calibration.
- **Fallback:** keep the plug-in and choose per country on held-out. Ye et al. show that
  threshold tuning is more robust when calibration is poor.

**Exclusivity-aware greedy assignment.** This replaces "argmax entity, then decide", where a
record rejected by its best entity is lost:

```
edges = all (record, entity) pairs with calibrated p
gain(r,e) = E[F_e | S_e ∪ {r}] - E[F_e | S_e]      # from the DP
max-heap on gain; pop the best edge; accept if r unassigned and gain > 0;
lazily recompute e's remaining gains (greedy submodular style); orphans stay unassigned
```

- Expected +0.0001 to +0.0005, on entities with rival candidates.
- Gate: reassigned records must reach held-out precision ≥ 0.8.
- **Component assignment** (optional): join S2/S3 records into components when
  support_sim ≥ 0.9 and the house number is the same (no offsets), and assign each component
  as a unit. Only if the component purity measured on train is ≥ 0.99.
- The old E2 idea ("the entity missing a copy") is **dropped**: entities have several copies
  per source (3.46 matches on average), so a capacity argument is invalid.

## 4.9 Unlabelled-country pathway (P1, P3, P9)

The trigger is "this country string has no training labels", never a country name.

1. **θ_u from worst-case LOCO.** Tune T_u, λ-model choice, the DP fallback and the minimum p on
   the two LOCO runs, with the objective min(F_India | US model, F_US | India model). Freeze
   the result as `params['__unlabelled__']`. This replaces "max of labelled params + 0.05
   floor".
2. **Foreign-expert head `stage2u`** (GBDT version of DGER's STL / DAME's meta-target):
   - Score each labelled country with the *other* country's stage-1 expert.
   - Recompute the 20 group features from those foreign probabilities.
   - Fit a head on the foreign rows with true labels, with monotone +1 constraints on p and
     the margins.
   - For an unlabelled country: q_u = mean over experts, and
     q = β·q_u + (1 − β)·q_global, with β ∈ {0.5, 1} set by one probe.
   - Adopt only if both LOCO directions improve.
3. **Expert disagreement.** δ(x) = max_c p_c(x) − min_c p_c(x) over the labelled-country
   experts. Shapes whose label is a country convention (siblings: India says match, US says
   not) get a large δ. Use p′ = p − κ·δ with κ ∈ {0, 0.1, 0.25}, or add δ as a feature of
   `stage2u`. LOCO diagnostic first: do foreign errors concentrate at high δ?
4. **Out-of-support shrink.** A pair-level domain classifier (test country vs labelled train,
   similarity features only, no script or format flags) marks a pair out of support when
   P(T|x) ≥ 0.95. Then p′ = p^κ, κ > 1. It is inert for India/US by construction.
   - Go/no-go test: in LOCO, precision of p ≥ 0.9 predictions by decile of the divergence
     score. If that precision is flat across deciles, drop the idea.
   - Divergence cannot catch distractors that *look* like train positives (DAME Fig. 3), so
     this is a secondary lever.

**Expected:** France +0.005 to +0.015 F0.5, i.e. +0.001 to +0.002 on the total leaderboard,
together with §4.5 and §4.10.

## 4.10 Leaderboard probes under Ladder discipline (P1, P2, P9)

- **Differential submissions.** Two files that differ only on a known entity set S. With a
  random public split, mean ΔF_S ≈ ΔLB · N_test / |S|, so which entities are public does not
  need to be known.
- **Pre-registered buckets** (fixed before any upload), each ≥ 1,000 entities / ≥ 5k pairs:
  - France sibling-offset;
  - France same-house one-generic-word swap;
  - France empty-address shared name;
  - the same three buckets in US/India.
- **Simulate before uploading.** Re-run the decision with the bucket's p set to 0. Monte-Carlo
  the remaining truth from calibrated p over a grid of bucket precisions q to get the
  predicted ΔLB(q). Invert the observed ΔLB to an estimate q̂ (a BBSE analogue).
  - Validate the simulator on LOCO first: predict India's ΔF from dropping India siblings,
    then compare with the true value.
  - Then set a single bucket offset b_s so that the mean σ(z + b_s) = q̂, or choose the
    expert blend whose implied precision matches.
- **Discipline:**
  - accept only if |ΔLB| > 2 SE and the sign agrees with LOCO and the S1-S1 FP probe;
  - binary decisions only, about 15-20 in total;
  - never tune continuous parameters on the leaderboard;
  - log every probe in the documentation.
  - The private leaderboard decides the ranking, and adaptive probing over-fits the public
    subset (Blum & Hardt 2015).

## 4.11 Optional: guarded tri-training for the unlabelled country (P3)

This is only for the "harder cases within a type" part of the France gap (normalisation
quirks). By design it does nothing for the distractor shapes.

- **Teachers:** T_full, T_name (address block dropped), T_addr (name block dropped).
- **Positive** only if all of these hold:
  - min p ≥ 0.97 across the three teachers;
  - margin over the best rival entity ≥ 0.5;
  - the pair is **not** a suspect shape;
  - the pair lies in the source-like half of the domain classifier.
- **Negative:** max p ≤ 0.03, or the loser of a ≥ 0.98 one-to-one owner in all teachers;
  about 3× as many negatives as positives.
- **Training:** stage 2 only, with source replay, target weight 0.2-0.5 and soft labels; one
  round, at most 3 with a throttled curriculum (Kim et al. 2023).
- **Gate:** the identical procedure run on LOCO must reach pseudo-label precision ≥ 0.995 and
  improve both directions. Otherwise skip it.
- **Cost / gain:** about 10-15 h in total, for 0 to +0.0015 overall. Lowest priority.

# 5. Dropped from the previous plan, and why

| Item | Verdict | Reason |
|---|---|---|
| A2 hand-label 300-400 France pairs (DUAL) | **Dropped** | Human labelling of test data breaks fair play; the same goes for reading test pairs to write rules |
| A4 Qwen3-8B teacher | **Dropped** | About 1-2 s/pair on CPU gives about 20-40k pairs/night; it needs A2 to validate; content-core features cover its job for free |
| A5 French-dressed synthetic pairs | **Replaced** by §4.3 + §4.5 | Positives are not scarce; the useful part is rate-matched *negatives* and learned vocabularies |
| A6 experts as backbone | **Restricted** to §4.9 | Needs ≥ 5 experts (DAME Fig. 2); here there are 2 |
| B1 density-ratio sample weights | **Downgraded** to entity resampling behind a go/no-go test | Trees plus local decisions are near-inert to weights (DMM, Zadrozny) |
| B3 drop shifted features | **Changed** to *add* relative copies | Dropping top-3 features costs accuracy; shift is partly signal |
| C1 cross-encoder | **Dropped** (low priority) | Fine-tuned on train, it learns train's conventions (word swaps = match); 3-6 h of CPU fine-tuning for ≤ +0.001 |
| Self-training at p ≥ 0.99 (A3) | **Gated** (§4.11) | Copies confident errors; unbounded error under a large shift |
| Saerens EM / BBSE prior correction | **Dropped** | The shift is new negative x's (p(x|y=0) changes), not label shift; already measured as a null |
| libpostal / OSM / gazetteers | **Dropped** | External data; §4.3 learns the same roles from the provided records |
| Feature mixup / SMOTE / masking | **Dropped** | Produces unrealisable pair vectors; the classes are not scarce |
| E2 "entity missing a copy" | **Replaced** by §4.8 greedy reassignment and component assignment | Entities have several copies per source |

# 6. Expected gains (realistic, not additive at the top end)

| Lever | Held-out | Leaderboard | Confidence |
|---|---|---|---|
| P0b France bucket drop (if confirmed) | 0 | +0.000 to +0.002 | medium; decided by probe |
| Calibration + exact DP + greedy reassignment | +0.0003 to +0.0015 | +0.0003 to +0.001 | high |
| Memory-safe build → 2-3× rows / 3 folds | +0.0003 to +0.001 | similar | medium-high |
| Normaliser v6 + content-core + exact-twin features | ~0 to +0.0005 | +0.0003 to +0.002 (France) | medium |
| Rate-matched distractors (only if the probe confirms) | −0.0005 to 0 | +0.001 to +0.003 | medium-low |
| Unlabelled-country head (θ_u, δ, OOS, stage2u) | 0 | +0.0005 to +0.002 | medium-low |
| Second hop / tri-training (optional) | +0 to +0.001 | +0 to +0.001 | low |
| **Total** | **0.9854 → ~0.987-0.988** | **0.9792 → ~0.981-0.984** | |

# 7. Rule compliance

**Transductive uses to declare in the documentation:**
- TF-IDF fitted on test records;
- token-role induction and percentiles on test records;
- domain classifiers fitted on unlabelled test features;
- S1-S1 probe scoring;
- optional self-training;
- leaderboard probes (discrete, logged).

**Excluded:**
- human labels or reading test pairs to write rules;
- libpostal/OSM or any external list;
- models that are not MIT/Apache or are above 8B;
- country literals in the pipeline logic (grep audit: no `"US"`, `"India"` or `"France"` in
  `src/` logic);
- row-order or ID leakage.

**Pre-submission check:** exact TSV format, `validate_submission.py --check-ids`, the zip
rebuilt from the same code.
