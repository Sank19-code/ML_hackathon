# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** AnimeWatchers
**Team Members:** Sankar Sriram (team leader), Renuka Bandepalli, Jishnu Nambiar
**Submission Date:** October 2026

---

## 1. Executive Summary

We undo the synthetic noise with a rule-based normaliser (transliteration learned from the
training labels, alias/legal/OCR/domain/state/street canonicalisation), retrieve candidates per
country with an exact TF-IDF nearest-neighbour search over blocking keys (98.1 % blocking
recall at 39-53 candidates per entity, including keys for house numbers clipped by one digit), and
score pairs with a two-stage LightGBM: stage 1 on 78 pairwise + 6 ambiguity-context features,
stage 2 adding 25 one-to-one competition, cluster-consistency and same-source "twin" features
built from stage-1 scores. Probabilities are calibrated (isotonic, per country) and each entity
keeps the top-k candidates that maximise its exact expected F0.5 (Poisson-binomial dynamic
programme) after a one-to-one assignment: **held-out macro F0.5 0.9861** on the labelled training
universe (tuning and evaluation on disjoint halves of the Source 1 entities). A country without
labels (France) goes through one data-driven path whose settings were chosen by
leave-one-country-out rehearsals (a model trained on one labelled country, scored on the other).
Features that detect the generator's "sibling" distractor businesses (same name and street, house
number shifted by a fixed offset) target the main difference between the training and test sets.

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
| Non-Latin Source 2/3 names (India) | 23.5 % of S2, 13 % of S3, 18 % combined (Devanagari, Telugu, Kannada, Tamil, Gujarati, Bengali, Malayalam, Odia, Gurmukhi) | transliteration needed, otherwise they become empty strings |
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
/ `தமிழ்நாடு`), street types abbreviated (`St/Street`, `Rd/Road`), decorated / mutated house
numbers (`#111`, `##906`, `0733`, `3503.`, `61A`, `4-47/16` for `47/16`), house numbers clipped
by one digit (`2424` -> `424`, `17177` -> `1717`) or dropped, injected numbers (`H.NO 764 202`,
`Door No 86`, `PO BOX 2280`), fillers (`null`, `<NULL>`, `N/A`), city aliases (Bombay/Mumbai,
Poona/Pune) and ~3 % empty addresses.

The test set has a third country (France, 15 % of test Source 1) with no labels. Its address format
- seen only in the unlabelled test records - uses French street types written in full or abbreviated
(`Rue`/`R.`, `Boulevard`/`Bd`, `Allée`/`All.`), French legal forms (SARL, SAS, ...), postcode + city
components (`44600 St-Nazaire`), and regions and departments used interchangeably (Gironde ->
Nouvelle-Aquitaine). The normaliser handles these with hand-written static tables and word lists
(French street types, legal forms, articles, unit words and ordinal suffixes, the 13 regions and 16
departments) written from that format; no test pair was labelled and no external list was
downloaded. The noise drops the region from 35 % of the French Source 2/3 addresses (the address
ends with the city) against 3 % in the US and India; the pair features fill a missing region from a
locality -> region table learned from the country's own records (97 % coverage afterwards), so
`state_match` means the same thing in every country.

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
    house number with its first or last digit dropped + street word (the noise clips numbers:
    `2424 Peach Ave` -> `424 Peach Ave`; crossed so that only an exact number on one side meets a
    clipped variant on the other), street word + locality word (records whose number was dropped);
  - *character block:* 3-grams of the core name (typos, scrambles, glued words).
- **Ranking instead of cutting:** three top-k lists from one pass - combined 0.40 name + 0.45
  address + 0.15 char cosine (top 25), name+char (top 10, catches records with missing or changed
  addresses), address only (top 20, catches records renamed to an alias) - are unioned (at most 50
  per entity) and ranked by the combined cosine. A country whose Source 2/3 records are partly
  written in a non-Latin script (share >= 5 %; India 18 %, US and France 0 %) gets a deeper combined
  list (top 40, at most 60 per entity): transliterated and renamed businesses with fragmentary
  addresses rank just below the top 25 there. The rule is a statistic of each country's own records.
  The model, features, blocking depth and decision path never branch on a country name. Only the
  address normaliser branches on the country string: it selects the US / India state tables or the
  French region / department and street-type tables, and for France applies two parsing rules (a
  "postcode city" component is a locality; a component that starts with a French street type is a
  street). Any other country gets the generic street rules and no state table. Only features with df
  > 5 000 (words) / 2 000 (3-grams) are left out of the index (they carry almost no IDF weight and
  dominate the cost).

- **Candidate pairs generated:** train 98.2 M (India 47.1 M, US 51.1 M; 53 / 39 per entity);
  test 78.2 M (India 43.0 M, US 25.3 M, France 9.9 M; 53 / 38 / 38 per entity). `candidate_pairs.tsv`
  holds exactly the pairs the models score.
- **Blocking recall (train, all 7.64 M true pairs):** India 97.7 % of pairs / 97.7 % per
  entity, US 98.4 % / 98.4 % (98.1 % overall). Separately normalising the name and address blocks
  was worth +2 points, second house-number and number-pair keys +1.8 points, a deeper address-only
  list +0.15, the clipped-number and street x locality keys +1.0 (US) / +0.2 (India), India's
  deeper combined list +0.25.
- **How true matches were kept:** measured recall after every change and categorised the misses
  (`empty address + shared name`, `name differs`, `address differs`, `domain`); exact name or
  address keys recover almost none of the rest (the names are shared by dozens of entities, the
  addresses differ). Remaining misses: empty-address records whose repeated name matches dozens of
  entities (a third of them), renamed businesses with empty or fragmentary addresses (India),
  aliases with mutated numbers.

---

## 4. Matching Model

**Features used (78 pair features + 6 context features; stage 2 adds 25 group features):**
- Name features: ratio, token sort / set, partial ratio and Jaro-Winkler on the core name;
  ratios on the full cleaned name (keeps "services", "trading", legal forms) and on the raw name;
  compact-string ratios and cross partial ratios for domain / glued names; acronym prefix match
  (`jiprivate.com`); alias comparison; IDF cosine and IDF-weighted containment of name tokens in
  both directions; character 3-gram TF-IDF cosine / containment; token counts, first/last token
  agreement, legal-form agreement; content-core features - the core name minus the locality
  words of either record: Jaccard, disjoint flag, IDF share of the non-shared words (illustration:
  the pair shape `Tourcoing Ecole SARL` / `Tourcoing Amis SARL` -> `ecole` vs `amis`, seen among
  unlabelled French test candidates; the feature's weight is learned from training labels only).
- Address features: full / street / locality string similarities; separate match and conflict
  flags for house number (both directions), number set, compound id, locality (city) and state;
  number-set intersections and extra numbers; IDF cosine / containment of address tokens;
  sibling detectors - signed house-number difference, whether it is one of the generator's
  sibling offsets, and whether any candidate number equals a Source 1 number plus such an offset
  (covers compound numbers such as `116-47 -> 116-49`); a clipped-number flag (one house number
  equals the other with its first or last digit dropped).
- Other: blocking cosines and rank, number of candidates, source (S2/S3), domain /
  transliterated / alias flags, missing-address flags; ambiguity context (how many Source 1 and
  Source 2/3 records share the name and the normalised address).
- Stage-2 group features (from stage-1 probability p1): rank / gap / count of strong candidates
  inside the entity and inside the same source; for the S2/S3 record: number of competing
  entities, rank and margin over the best competing entity (p1, address cosine, name cosine);
  cluster support = similarity to the entity's other strong candidates (S2 <-> S3 consistency);
  sibling-cluster size (other candidates sharing the same sibling-offset number) and the number
  of confident candidates confirming the entity's own house number; "twin" features - whether a
  confident candidate from the same source confirms the house number or has an identical content
  core (a second noisy copy vs a neighbouring business); uncertain-candidate count and rank among
  non-sibling candidates.

**Model type:** two LightGBM binary classifiers (MIT licence; 700 / 500 rounds, 127 leaves,
trained on the candidates of a 20 % sample of training entities, 19.6 M pairs each), each a 2-fold
ensemble cross-fitted by Source 1 entity - every training pair gets an out-of-fold score and no
candidate of an entity leaks across the split; the fold models are averaged on test. The sampled
rows are featurised chunk by chunk into float32 memory-mapped files and binned by LightGBM from
those blocks (no in-RAM stacking), which is what lets a 20 % sample fit in 16 GB (12 % before).
Stage-1 top features (gain): address cosine, combined blocking score, blocking rank, address
token-set ratio, domain partial ratio, the sibling-offset flags. Stage-2 top features: p1 (59 %)
and the one-to-one margin p1 - best rival entity (38 %).

**Threshold selection method:** each S2/S3 record is first assigned to the entity where it scores
highest (one-to-one). Its probability is calibrated (isotonic regression on out-of-fold scores,
per labelled country; a pooled map for other countries). Per entity, the number k of top
candidates to keep is the one that maximises the exact expected F0.5 under independent Bernoulli
matches plus a Poisson term for true matches outside the candidate list (dynamic programme over
Poisson-binomial count distributions, verified against brute-force enumeration); k = 0 when
P(no true match) is higher. Tuned on half of the entities and evaluated on the other half, this
scores 0.986060 against 0.986057 for the previous plug-in rule (+0.00002 on the fifth model's
scores) and 0.98594 for per-(country, source) thresholds - a tie in accuracy, kept because its
probabilities are calibrated, which the unlabelled-country path relies on. A country without
labels uses the pooled calibration softened by a temperature of 1.5 and no missed-match term: on
the leave-one-country-out
rehearsals this was the only setting that improved both directions over the previous rule (US
model on India 0.96774 -> 0.96807, India model on US 0.97766 -> 0.97769).

---

## 5. Results & Error Analysis

Held-out macro F0.5 on the labelled training universe (all 2.21 M Source 1 entities, fold A tunes
/ fold B evaluates and vice versa):

| | evaluated on fold 0 | evaluated on fold 1 | singletons | non-singletons | pair precision | pair recall | blocking entity recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| India | 0.9845 | 0.9843 | 0.982-0.984 | 0.984-0.985 | 0.997 | 0.960 | 0.977 |
| US | 0.9871 | 0.9872 | 0.986-0.987 | 0.987 | 0.998 | 0.966 | 0.984 |
| **Overall** | **0.9861** | **0.9860** | 0.985 | 0.986 | 0.998 | 0.964 | 0.982 |

- **F_0.5 Score (macro):** 0.9861 held-out on the training universe (iterations: 0.9808, 0.9816,
  0.9829, 0.9852, 0.9854, 0.9861). Public leaderboard (submitted by the team): 0.9737 for the
  0.9816 model, 0.9772 for the 0.9829 model, 0.9792 for the 0.9854 model.
- **Ceiling:** on the previous candidates (97.4 % recall) a perfect classifier would score
  0.991-0.992, 0.987-0.988 when the ambiguous records below are excluded, and 0.993-0.994 with
  perfect blocking as well - i.e. blocking was the largest avoidable loss, which the fourth
  iteration attacked. About 2 % of all true pairs are Source 2/3
  records with an empty address whose name is shared by several Source 1 entities (e.g. 253 x
  "Primary Care Group"); nothing in the data identifies their entity, so they are left unmatched
  (precision first). Leakage such as row order or identifier patterns was deliberately not used.
- **Test set (no labels):** 94 % of test entities receive matches in every country (training
  truth: 94.4 % non-singletons); France, which has no training labels, shows the same match rate
  and a similar matches-per-entity profile (3.24 vs 3.34 / 3.40 predicted pairs per entity in
  India / the US).
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
- **Test-like holdout:** test entities have the same number of confident candidates as training
  entities (3.37 vs 3.35 with p1 >= 0.5) but more uncertain ones (p1 in [0.1, 0.9): 0.21 -> 0.27 /
  0.29 per entity in India / US, 0.75 in France) and 2.2x more same-name sibling candidates (US).
  Re-weighting the held-out entities by these label-free descriptors to the test mix predicts
  0.9802 for the third model against 0.9772 on the leaderboard: about half of the gap is a change
  in the mix of entity types, the rest is harder cases within a type - most likely France, which
  the leaderboard total implies is around 0.95-0.97. Re-tuning the decision rule on the re-weighted
  holdout gave nothing (0.98177 -> 0.98174), neither did tuning France's rule on training entities
  weighted to France's mix (0.97161 -> 0.97162), so the tuned rules are kept.
- **Common false positives (wrong merges):** same-name businesses whose S2/S3 record belongs to
  an entity that is not in Source 1 (e.g. another "Red Consulting Private Limited" in the same
  district); alias-named records at multi-tenant addresses (two companies at "C-28, Second Floor
  Panchsheel Enclave"); one-letter name variants at the same address ("Jarliq" / "Jarluq York
  Company"); in France (no labels, so a suspected pattern seen in unlabelled predictions rather
  than a confirmed error), siblings that swap a generic name word ("Merignac Amis SA" / "Merignac
  Club S.A."). In training, the corresponding pair shapes (one generic word swapped at the same
  address, sibling offset) are 96-99.5 % precise among predicted pairs, so no France-specific
  rejection rule is applied without labels to justify it.
- **Validation without test labels (sixth iteration).** Only aggregate statistics of the
  unlabelled test records were used to choose settings: no test record was labelled, no test label
  was inferred and no leaderboard feedback set any parameter. (The hand-written French address
  and name word tables of section 2.1 were written from the format of the unlabelled test
  records.)
  - *Leave-one-country-out rehearsals:* a stage-1 + stage-2 model trained on one labelled country
    (6 % entity sample) scores the other one through the unlabelled-country path. Treating a
    labelled country as unseen costs 0.016 (US model on India: 0.9677 vs 0.9836 in-domain) and
    0.009 (India model on US: 0.9777 vs 0.9866); the foreign model mostly loses recall (true
    matches at calibrated p ~0.15 are really 0.26). These runs chose the unlabelled settings above.
    A shift classifier on entity-level score summaries separates France from training less well
    (proxy A-distance 1.23) than it separates India from the US (1.73), so the rehearsals are a
    conservative stand-in for France.
  - *Known non-matches:* Source 1 holds each business once, so near-duplicate Source 1 / Source 1
    pairs are different businesses by construction. The stage-1 model accepts (p >= 0.5) 0.0085 %
    of them in the US and 0.036 % in India (the same as on the training sets: 0.0097 % / 0.038 %)
    but 1.2 % in France, mostly same-name businesses at another house number - label-free
    evidence that France's remaining errors are confident false merges between look-alike
    businesses rather than a calibration problem. The sixth model lowers France's rate to 0.95 %
    (siblings 0.76 -> 0.51 %, same name at another house 1.40 -> 0.96 %; US 0.0085 -> 0.0071 %).
  - *Rejected by these checks:* a head trained on each labelled country's pairs as scored by the
    other country's model (it lost in both directions: 0.955 vs 0.969, 0.969 vs 0.977); offering
    records rejected by their best entity to the second-best one (no held-out gain); a
    fold-wise transliteration dictionary (train 97.8 % vs test 96.4 % token coverage - too small
    a gap to justify re-normalising).
  - *Not attempted* (excluded by the fair-play rules or too costly on CPU): self-training /
    pseudo-labelling on test records, LLM labelling, external gazetteers or libpostal.
- **Common false negatives (missed matches):** records with an empty address whose name is
  shared by several Source 1 entities (genuinely ambiguous); names replaced by random aliases
  with mutated house numbers; acronym domains (`bindia.com`); heavy typos in both fields.

---

## 6. Conclusion

Most of the gain came from understanding the generator: undoing its noise operations, making
blocking robust to repeated names, and exploiting the fact that each Source 2/3 record belongs to
exactly one Source 1 entity, and, for the test set, recognising the generator's sibling
distractors. Precision is 0.998; the remaining loss is blocking recall (98.1 %, mostly India) and
records that are ambiguous by construction (empty addresses under names shared by many businesses).
The sixth iteration's gains came from a larger training sample made possible by a memory-safe
data path and from same-source "twin" features; for the unlabelled country the most useful tools
were label-free: leave-one-country-out rehearsals and known non-matches.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`, entry point `src/run.py`. Its README gives two exact recipes:
(A) regenerate the submitted files with the shipped model in `model/` (`--mode prepare`, then
`--mode predict` into a new output folder, about 2 h); (B) retrain end to end (`--mode prepare`,
`--mode train --stage1-sample 0.20 --stage2-sample 0.20`, `--mode predict`) into a separate model
folder.

All source is in `src/`: `normalize.py` (noise undoing), `translit.py` (learned native-script
dictionary), `data.py` (parallel normalisation), `blocking.py` (TF-IDF keys + numba top-k),
`features.py` (pair features), `model.py` (LightGBM fold ensembles), `decide.py` (group features,
one-to-one, decisions), `metrics.py` (macro F0.5 + breakdowns), `pipeline.py` (stages), `run.py`
(CLI), `evaluate_holdout.py` (held-out evaluation from saved out-of-fold scores) and
`test_pipeline.py` (unit tests).

Only MIT / BSD / Apache / ISC (anyascii) libraries; no external databases, APIs, downloaded data or
pretrained models. The pipeline contains hand-written static tables: in `src/normalize.py` US state
codes and names, Indian states (including native-script names) and historic city names, French
regions and 16 departments, French street types and legal forms, and generic word lists (street
types, directionals, ordinals, honorifics, inserted filler words including "india" and "france",
address unit / filler words including French articles, unit words and ordinal suffixes); in
`src/blocking.py` a street-type stop list. France has no training records, so the French tables and
words were written from the format of the unlabelled test records. Transductive but label-free uses
of the test records, as in any unsupervised preprocessing: TF-IDF / IDF statistics, the locality ->
region table and the name / address duplicate counts of the six ambiguity-context features are
computed per country on that country's records.

### B. Additional Results

Blocking recall progression on India training data: single combined TF-IDF index 93.3 % ->
separately normalised name / address / char blocks 95.3 % -> + second house number and number-pair
keys 97.1 % -> deeper address-only list 97.25 % -> clipped-number and street x locality keys
97.45 % (US: 97.4 % -> 98.4 %; on a 60 K-entity US sample the new keys gave +0.9 points at the
same list depth, while doubling the list depths alone gave +0.5 points at 1.8x the candidates).

Held-out macro F0.5 by iteration (both folds): first full model (thresholds 0.9805, expected-F
0.9808); + numeric-tag stripping, acronym/domain features, record-level address/name competition
margins, deeper address list, 12 % training sample (thresholds 0.9814, expected-F 0.9816,
public leaderboard 0.9737); + sibling-offset pair and group features (thresholds 0.9827,
expected-F 0.9829, public leaderboard 0.9772); + clipped-number / street x locality blocking keys,
address list 20, clipped-number pair feature (thresholds 0.9850, expected-F 0.9852); + deeper
combined list for India (top 40 of at most 60; India recall 97.45 -> 97.7 %; on a 60 K-entity
sample a deeper name list gave +0.03 and a deeper address list +0.17 points), stage 1 kept and
stage 2 retrained (thresholds 0.9852, expected-F 0.9854: India 0.9831/0.9832 -> 0.9835/0.9836,
public leaderboard 0.9792); + content-core, same-source twin and uncertain-count features, missing
regions filled from localities, 20 % training sample via memory-mapped datasets, calibrated exact
expected-F0.5 decision (thresholds 0.9859, plug-in 0.98606, exact **0.98606**: India 0.9843/0.9845,
US 0.9871/0.9872). Expected-F0.5 with an additional learned entity-level no-match model:
+0.00002 (not kept). A per-bucket prior-shift correction of the probabilities was also tested
in simulation and did not help (not kept).
