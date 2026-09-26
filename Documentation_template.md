# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** DevX Resolvers
**Team Members:** DevX Team
**Submission Date:** September 2026

---

## 1. Executive Summary

We undo the synthetic noise with a rule-based normaliser (transliteration learned from the
training labels, alias/legal/OCR/domain/state/street canonicalisation), retrieve candidates per
country with an exact TF-IDF nearest-neighbour search over blocking keys (97.4 % blocking
recall at ~34 candidates per entity), and score pairs with a two-stage LightGBM: stage 1 on 74
pairwise + 6 ambiguity-context features, stage 2 adding one-to-one competition and
cluster-consistency features built from stage-1 scores. A per-entity expected-F0.5 decision on
top of a one-to-one assignment gives a **held-out macro F0.5 of 0.9829** on the labelled training
universe (tuning and evaluation on disjoint halves of the Source 1 entities). Features that detect
the generator's "sibling" distractor businesses (same name and street, house number shifted by a
fixed offset) target the main difference between the training and test sets.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from the training ground truth (2.21 M Source 1, 5.03 M Source 2, 5.29 M Source 3
records; 7.64 M true pairs):

| Property | Value | Consequence |
| --- | --- | --- |
| Source 2/3 records matched to more than one Source 1 entity | **0** | strict many-to-one: one-to-one assignment and competition features |
| Cross-country matches | **0** | everything is partitioned by country |
| Singleton Source 1 entities | 5.6 % | empty predictions must be earned |
| Matches per Source 1 entity | 3.46 on average (up to 11), both from S2 and S3 | several noisy copies per entity |
| Unmatched Source 2/3 records | 26 % / 25 % | noisy copies of businesses absent from Source 1 = hard negatives |
| Repeated Source 1 names | e.g. 46 x "Golden Care Private Limited" (India), 253 x "Primary Care Group" (US) | the name alone rarely identifies the entity; the address must confirm |
| Non-Latin Source 2/3 names (India) | 11 % of S2, 6.5 % of S3 (Devanagari, Telugu, Kannada, Tamil, Gujarati, Bengali, Malayalam, Odia, Gurmukhi) | transliteration needed, otherwise they become empty strings |
| Postcodes | essentially absent (<1 %) | house number + street / name keys instead |
| "Sibling" distractors | same name and street, house number = entity number + an offset from {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}, usually another legal form; 99.9 % of small positive offsets among negatives, while real number noise is symmetric; about 2x more frequent in test | signed house-number offset features |

Noise catalogue measured on 520 K matched pairs (share of Latin-script S2/S3 names):
upper case 12 %, double spaces 12 %, lower case 8 %, brackets 9 %, hyphenation 5 %,
domain form 4.4 % (`primarycaregroup.com`), junk prefixes (`#`, `@`, `>>`, `...`, `***`, `--`)
~2 %, alias phrases 1 % (`<random word> dba|d/b/a|t/a|fka|f/k/a|aka|a/k/a|formerly (known as)|
doing business as|née <true name>`), `(ID: 12345)`, `#12345` and `- [2735036094]` tags; inserted
words (Center, Services, Partners, Labs, Sys, One, The), honorifics (Shri, Sri, Smt, Mr, Dr, M/s),
legal-form swaps/moves (Inc -> LP, `Llc Boda Solana`), OCR swaps (`lnc`, `5ervices`, `c0m`,
`8rothers`, `hea1th`), character typos/scrambles, repeated words, word re-ordering and full name
replacement by a random alias (the address is then the only signal). Addresses: component
re-ordering, S2 uses state codes while S3 uses full names or native script (`TN` / `Tamil Nadu`
/ `தமிழ்நாடு`), street types abbreviated (`St/Street`, `Rd/Road`, `R./Rue`, `Bd`, `All.`),
decorated / mutated house numbers (`#111`, `##906`, `0733`, `3503.`, `61A`, `4-47/16` for
`47/16`), injected numbers (`H.NO 764 202`, `Door No 86`, `PO BOX 2280`), fillers (`null`,
`<NULL>`, `N/A`), city aliases (Bombay/Mumbai, Poona/Pune, St-Nazaire) and ~3 % empty addresses.

The test set has a third country (France, 15 % of test Source 1) with no labels, French street
types, and regions and departments used interchangeably (Gironde -> Nouvelle-Aquitaine).

### 2.2 Solution Strategy

**Approach Type:** Normalisation + TF-IDF top-k blocking + two-stage gradient boosting +
one-to-one assignment + expected-F0.5 decision (per country).
**Core Innovation:** (1) a normaliser that undoes each catalogued noise operation, including a
native-script dictionary learned from the labels; (2) blocking as exact sparse nearest-neighbour
search where every classic blocking key is an IDF-weighted feature and the name, address and
character blocks are normalised separately (no hard key caps); (3) exploiting the many-to-one
structure: stage-2 features describe how an S2/S3 record's score compares with every other
Source 1 entity that retrieved it, and records are assigned to their best entity only.

---

## 3. Candidate Generation (Blocking)

Per country, every Source 1 record is queried against all Source 2 and Source 3 records of that
country (exact search, numba parallel inverted-index accumulation, ~10 min for 1.3 M x 6.2 M).

- **Blocking keys (sparse TF-IDF features, three separately L2-normalised blocks):**
  - *name block:* core-name tokens, sorted token pairs (word order), compact name / domain stem
    (`wilfordhancock.com` -> `wilfordhancock`);
  - *address block:* house number + street word, house number + name token (the postcode+name
    key of the brief - postcodes are nearly absent here), house number + locality word, compound
    ids (`D-12` -> `d12`, `22/235`), consecutive number pairs (`47_16`), street words, locality
    words, numbers; house keys use the first two numbers because noise injects numbers in front;
  - *character block:* 3-grams of the core name (typos, scrambles, glued words).
- **Ranking instead of cutting:** three top-k lists from one pass - combined
  0.40 name + 0.45 address + 0.15 char cosine (top 25), name+char (top 10, catches records with
  missing or changed addresses), address only (top 15, catches records renamed to an alias) - are
  unioned (at most 50 per entity) and ranked by the combined cosine. Only features with
  df > 5 000 (words) / 2 000 (3-grams) are left out of the index (they carry almost no IDF
  weight and dominate the cost).
- **Candidate pairs generated:** train 75.4 M (India 31.0 M, US 44.4 M; 34 per entity);
  test 58.9 M (India 28.3 M, US 22.2 M, France 8.5 M; 33-35 per entity). `candidate_pairs.tsv`
  holds exactly the pairs the models score.
- **Blocking recall (train, all 7.64 M true pairs):** India 97.25 % of pairs / 97.3 % per
  entity, US 97.4 % / 97.5 %. Separately normalising the name and address blocks was worth +2
  points, second house-number and number-pair keys +1.8 points, a deeper address-only list +0.15.
- **How true matches were kept:** measured recall after every change; remaining misses are mostly
  empty-address records whose (repeated) name matches dozens of entities, and aliases with
  mutated numbers.

---

## 4. Matching Model

**Features used (74 pair features + 6 context features; stage 2 adds 20 group features):**
- Name features: ratio, token sort / set, partial ratio and Jaro-Winkler on the core name;
  ratios on the full cleaned name (keeps "services", "trading", legal forms) and on the raw name;
  compact-string ratios and cross partial ratios for domain / glued names; acronym prefix match
  (`jiprivate.com`); alias comparison; IDF cosine and IDF-weighted containment of name tokens in
  both directions; character 3-gram TF-IDF cosine / containment; token counts, first/last token
  agreement, legal-form agreement.
- Address features: full / street / locality string similarities; separate match and conflict
  flags for house number (both directions), number set, compound id, locality (city) and state;
  number-set intersections and extra numbers; IDF cosine / containment of address tokens;
  sibling detectors - signed house-number difference, whether it is one of the generator's
  sibling offsets, and whether any candidate number equals a Source 1 number plus such an offset
  (covers compound numbers such as `116-47 -> 116-49`).
- Other: blocking cosines and rank, number of candidates, source (S2/S3), domain /
  transliterated / alias flags, missing-address flags; ambiguity context (how many Source 1 and
  Source 2/3 records share the name and the normalised address).
- Stage-2 group features (from stage-1 probability p1): rank / gap / count of strong candidates
  inside the entity and inside the same source; for the S2/S3 record: number of competing
  entities, rank and margin over the best competing entity (p1, address cosine, name cosine);
  cluster support = similarity to the entity's other strong candidates (S2 <-> S3 consistency);
  sibling-cluster size (other candidates sharing the same sibling-offset number) and the number
  of confident candidates confirming the entity's own house number.

**Model type:** two LightGBM binary classifiers (MIT licence; 700 / 500 rounds, 127 leaves,
trained on the candidates of a 12 % sample of training entities, 9.0 M pairs), each a 2-fold
ensemble cross-fitted by Source 1 entity - every training pair gets an out-of-fold score and no
candidate of an entity leaks across the split; the fold models are averaged on test.
Stage-1 top features (gain): combined blocking score, address cosine, blocking rank, address
token-set ratio, domain partial ratio, the two sibling-offset flags. Stage-2 top features: p1
(87 %) and the one-to-one margin p1 - best rival entity (10 %).

**Threshold selection method:** each S2/S3 record is first assigned to the entity where it scores
highest (one-to-one). Then, per entity, the top-k candidates maximising the expected F0.5
(1.25 sum_topk p^g / (0.25 sum p^g + k)) are kept, or none when the singleton probability
prod(1 - p) is higher. g and a probability floor are tuned per country on half of the entities
and evaluated on the other half; per-(country, source) thresholds were evaluated the same way
(0.9827) and lost to this rule (0.9829). France has no labels and uses the stricter of the tuned
rules plus a 0.05 margin on the probability floor.

---

## 5. Results & Error Analysis

Held-out macro F0.5 on the labelled training universe (all 2.21 M Source 1 entities, fold A tunes
/ fold B evaluates and vice versa):

| | evaluated on fold 0 | evaluated on fold 1 | singletons | non-singletons | pair precision | pair recall | blocking entity recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| India | 0.9824 | 0.9825 | 0.979-0.985 | 0.982-0.983 | 0.997 | 0.956 | 0.973 |
| US | 0.9832 | 0.9832 | 0.988 | 0.983 | 0.998 | 0.956 | 0.974-0.975 |
| **Overall** | **0.9829** | **0.9829** | 0.984-0.987 | 0.983 | 0.9975 | 0.956 | 0.974 |

- **F_0.5 Score (macro):** 0.9829 held-out on the training universe (iterations: 0.9808, 0.9816,
  0.9829). Public leaderboard: 0.9737 for the 0.9816 model (submitted by the team).
- **Ceiling:** a perfect classifier on the current candidates would score 0.991-0.992 on the
  training universe (blocking misses cost 0.0086). About 2 % of all true pairs are Source 2/3
  records with an empty address whose name is shared by several Source 1 entities (e.g. 253 x
  "Primary Care Group"); nothing in the data identifies their entity, so they are left unmatched
  (precision first). Leakage such as row order or identifier patterns was deliberately not used.
- **Test set (no labels):** 94 % of test entities receive matches in every country (training
  truth: 94.4 % non-singletons); France, which has no training labels, shows the same match rate
  and matches-per-entity profile as India and the US, and manual inspection of French pairs in
  every probability band looked correct above the decision floor.
- **Train -> test shift (why the leaderboard is below held-out):** the test set has more Source
  2/3 records per entity (5.8 vs 4.7). Exact same-address matches per entity are unchanged (1.68
  US, 1.25 India), but candidates with the same name and street and a house number shifted by a
  sibling offset are 1.9x (US) and more (France: 96 % of its nearby-number pairs) as frequent -
  the extra test records are sibling distractors, often with several copies each, a configuration
  that is mostly a true match in training (where the Source 1 number is the noisy one). The
  model's own uncertainty per entity is 1.6x (India), 2.8x (US) and 5x (France) the training
  level. The sibling features (third iteration) cut training false merges by 23 % (sibling-type
  false merges by 80 %) and raise the model's self-estimated test F0.5 in every country (India
  0.9917 -> 0.9929, US 0.9885 -> 0.9902, France 0.979 -> 0.981).
- **Common false positives (wrong merges):** same-name businesses whose S2/S3 record belongs to
  an entity that is not in Source 1 (e.g. another "Red Consulting Private Limited" in the same
  district); alias-named records at multi-tenant addresses (two companies at "C-28, Second Floor
  Panchsheel Enclave"); one-letter name variants at the same address ("Jarliq" / "Jarluq York
  Company"); in France, siblings that swap a generic name word ("Merignac Amis SA" / "Merignac
  Club S.A.").
- **Common false negatives (missed matches):** records with an empty address whose name is
  shared by several Source 1 entities (genuinely ambiguous); names replaced by random aliases
  with mutated house numbers; acronym domains (`bindia.com`); heavy typos in both fields.

---

## 6. Conclusion

Most of the gain came from understanding the generator: undoing its noise operations, making
blocking robust to repeated names, and exploiting the fact that each Source 2/3 record belongs to
exactly one Source 1 entity, and, for the test set, recognising the generator's sibling
distractors. Precision is 0.9975; the remaining loss is blocking recall (97.4 %) and records that
are ambiguous by construction (empty addresses under names shared by many businesses).

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` (entry point `src/run.py`, see its README):

```
python code/business_entity_resolution/src/run.py --mode all --train-dir dataset/train \
    --test-dir dataset/test --output-dir output --work-dir work \
    --model-dir code/business_entity_resolution/model --validator utils/validate_submission.py
```

`normalize.py` (noise undoing), `translit.py` (learned native-script dictionary), `data.py`
(parallel normalisation), `blocking.py` (TF-IDF keys + numba top-k), `features.py` (pair features),
`model.py` (LightGBM fold ensembles), `decide.py` (group features, one-to-one, decisions),
`metrics.py` (macro F0.5 + breakdowns), `pipeline.py` (stages), `run.py` (CLI);
`evaluate_holdout.py` re-evaluates from saved out-of-fold scores; `test_pipeline.py` unit tests.
Only MIT / BSD / Apache libraries; no external data, APIs or pretrained models.

### B. Additional Results

Blocking recall progression on India training data: single combined TF-IDF index 93.3 % ->
separately normalised name / address / char blocks 95.3 % -> + second house number and number-pair
keys 97.1 % -> deeper address-only list 97.25 %.

Held-out macro F0.5 by iteration (both folds): first full model (thresholds 0.9805, expected-F
0.9808); + numeric-tag stripping, acronym/domain features, record-level address/name competition
margins, deeper address list, 12 % training sample (thresholds 0.9814, expected-F 0.9816,
public leaderboard 0.9737); + sibling-offset pair and group features (thresholds 0.9827,
expected-F **0.9829**). Expected-F0.5 with an additional learned entity-level no-match model:
+0.00002 (not kept). A per-bucket prior-shift correction of the probabilities was also tested
in simulation and did not help (not kept).
