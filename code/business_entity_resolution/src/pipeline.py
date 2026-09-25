import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import polars as pl
from tqdm import tqdm

try:
    from .blocking import (
        build_country_inverted_index,
        retrieve_candidates_for_s1,
    )
    from .features import compute_pair_features
    from .metrics import compute_macro_f05
    from .model import EntityResolutionModel
except ImportError:
    from blocking import (
        build_country_inverted_index,
        retrieve_candidates_for_s1,
    )
    from features import compute_pair_features
    from metrics import compute_macro_f05
    from model import EntityResolutionModel


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_tsv(path: str, **kwargs) -> pl.DataFrame:
    """Load a TSV file tolerating null bytes and mixed encodings."""
    return pl.read_csv(path, separator="\t", null_values=["", "null", "NULL", "None"], **kwargs)


def _s23_lookup(df: pl.DataFrame) -> Dict[str, Tuple[str, str]]:
    """Build entity_id → (name, addr) dict from a Polars DataFrame (fast tuple iteration)."""
    d = {}
    for eid, name, addr in df.select(
        ['entity_id', 'business_name', 'business_address']
    ).iter_rows():
        d[eid] = (name or '', addr or '')
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def run_train(
    train_dir: str,
    model_save_path: str = "code/business_entity_resolution/model.pkl",
    train_sample_size: int = 80000,
    val_sample_size: int = 20000,
    max_candidates: int = 20,
):
    """
    Full training pipeline:
      1. Load GT + sources (full S2/S3 per country for index quality)
      2. Multi-pass blocking → candidate pool with injected positives
      3. Feature extraction
      4. Train LightGBM
      5. Threshold optimisation on hold-out validation set
      6. Save model
    """
    print("=" * 60)
    print("TRAINING PIPELINE")
    print("=" * 60)
    t0 = time.time()

    # ── 1. Ground truth ────────────────────────────────────────────
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    print(f"Loading ground truth from {gt_path} …")
    gt_df = _load_tsv(gt_path)
    total_gt = len(gt_df)
    print(f"  Total GT rows: {total_gt:,}")

    # Shuffle to get a balanced country mix (file is sorted by country)
    gt_df = gt_df.sample(fraction=1.0, seed=42, shuffle=True)

    total_sample = min(train_sample_size + val_sample_size, total_gt)
    sampled_gt = gt_df.head(total_sample)

    all_s1_ids = sampled_gt['source1_entity_id'].to_list()
    train_s1_set = set(all_s1_ids[:train_sample_size])
    val_s1_ids = all_s1_ids[train_sample_size:]
    val_s1_set = set(val_s1_ids)

    # Build GT pair set and validation GT dict
    all_gt_pairs: Set[Tuple[str, str]] = set()
    val_ground_truth: Dict[str, Set[str]] = {}
    needed_s23: Set[str] = set()

    for s1_id, raw_m in sampled_gt.select(
        ['source1_entity_id', 'matched_entity_ids']
    ).iter_rows():
        mids: Set[str] = set()
        if raw_m:
            mids = set(raw_m.split(','))
            needed_s23.update(mids)
            for mid in mids:
                all_gt_pairs.add((s1_id, mid))
        if s1_id in val_s1_set:
            val_ground_truth[s1_id] = mids

    print(f"  Train S1: {len(train_s1_set):,} | Val S1: {len(val_s1_ids):,}")
    print(f"  Positive pairs: {len(all_gt_pairs):,}")

    # ── 2. Load sources ────────────────────────────────────────────
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")

    print("Loading S1 …")
    s1_df = _load_tsv(s1_path).filter(pl.col('entity_id').is_in(set(all_s1_ids)))
    s1_dict: Dict[str, Tuple[str, str, str]] = {}
    for eid, name, addr, country in s1_df.select(
        ['entity_id', 'business_name', 'business_address', 'country']
    ).iter_rows():
        s1_dict[eid] = (name or '', addr or '', country or '')

    print("Loading S2 (full) …")
    s2_df = _load_tsv(s2_path)
    print("Loading S3 (full) …")
    s3_df = _load_tsv(s3_path)
    print(f"  S2: {len(s2_df):,} | S3: {len(s3_df):,}")

    # Build s23 lookup for ALL needed positives (guaranteed presence)
    all_s23_df = pl.concat([
        s2_df.filter(pl.col('entity_id').is_in(needed_s23)),
        s3_df.filter(pl.col('entity_id').is_in(needed_s23)),
    ])
    s23_needed_dict = _s23_lookup(all_s23_df)

    # Build per-country inverted index (full S2/S3 for high recall)
    print("\nBuilding per-country inverted indices …")
    countries_in_train = list(s1_df['country'].unique())
    country_indices = {}
    country_s23_dict: Dict[str, Dict[str, Tuple[str, str]]] = {}

    for country in countries_in_train:
        t_idx = time.time()
        c_s2 = s2_df.filter(pl.col('country') == country)
        c_s3 = s3_df.filter(pl.col('country') == country)
        country_indices[country] = build_country_inverted_index(c_s2, c_s3, max_key_density=150)
        country_s23_dict[country] = _s23_lookup(pl.concat([c_s2, c_s3]))
        print(f"  [{country}] index built in {time.time()-t_idx:.1f}s "
              f"| S2={len(c_s2):,} S3={len(c_s3):,} "
              f"| keys={len(country_indices[country]):,}")

    # ── 3. Candidate generation + feature extraction ───────────────
    print("\nGenerating candidates and features …")
    X_train, y_train = [], []
    X_val, val_cand_pairs = [], []

    # Pre-build GT lookup: s1_id → set of matching ids (for fast injection)
    gt_by_s1: Dict[str, Set[str]] = defaultdict(set)
    for s1_id, mid in all_gt_pairs:
        gt_by_s1[s1_id].add(mid)

    for s1_id in tqdm(all_s1_ids, desc="Processing S1", unit="ent"):
        if s1_id not in s1_dict:
            continue

        s1_name, s1_addr, country = s1_dict[s1_id]
        idx = country_indices.get(country, {})
        c_s23 = country_s23_dict.get(country, {})
        is_train = s1_id in train_s1_set

        # Retrieve candidates via blocking
        cands = retrieve_candidates_for_s1(s1_name, s1_addr, idx, max_candidates=max_candidates)
        cand_set = set(cands)

        # For training: inject any missed true positives so the model
        # always sees positive examples (critical for recall learning)
        if is_train:
            for mid in gt_by_s1.get(s1_id, set()):
                if mid not in cand_set:
                    cands.append(mid)
                    cand_set.add(mid)

        for mid in cands:
            # Name+addr lookup: candidate may be in country dict or needed dict
            if mid in c_s23:
                m_name, m_addr = c_s23[mid]
            elif mid in s23_needed_dict:
                m_name, m_addr = s23_needed_dict[mid]
            else:
                continue

            label = 1 if (s1_id, mid) in all_gt_pairs else 0
            feat = compute_pair_features(s1_name, s1_addr, m_name, m_addr, mid)

            if is_train:
                X_train.append(feat)
                y_train.append(label)
            else:
                X_val.append(feat)
                val_cand_pairs.append((s1_id, mid))

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    X_val = np.array(X_val, dtype=np.float32)

    print(f"\nX_train: {X_train.shape} | Positives: {int(np.sum(y_train)):,}")
    print(f"X_val:   {X_val.shape}")

    # ── 4. Train & optimise threshold ──────────────────────────────
    model = EntityResolutionModel()
    model.fit(X_train, y_train)

    if len(X_val) > 0:
        val_probs = model.predict_proba(X_val)
        model.optimize_threshold(val_s1_ids, val_cand_pairs, val_probs, val_ground_truth)
    else:
        print("Warning: no validation data; using default threshold 0.65")

    model.save(model_save_path)
    print(f"\nTotal training time: {time.time()-t0:.1f}s")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_predict(
    test_dir: str,
    output_dir: str,
    model_path: str = "code/business_entity_resolution/model.pkl",
    max_candidates: int = 20,
):
    """
    Full inference pipeline:
      1. Load model + test sources
      2. For each country: build index → retrieve candidates → score with model
      3. Apply optimised threshold
      4. Write matching_results.tsv + candidate_pairs.tsv
    """
    print("=" * 60)
    print("INFERENCE PIPELINE")
    print("=" * 60)
    t0 = time.time()

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load model ──────────────────────────────────────────────
    print(f"Loading model from {model_path} …")
    model = EntityResolutionModel.load(model_path)
    tau = model.optimal_threshold
    print(f"  Threshold τ* = {tau:.3f}")

    # ── 2. Read test sources ───────────────────────────────────────
    print("Reading test sources …")
    s1_df = _load_tsv(os.path.join(test_dir, "test_source1.tsv"))
    s2_df = _load_tsv(os.path.join(test_dir, "test_source2.tsv"))
    s3_df = _load_tsv(os.path.join(test_dir, "test_source3.tsv"))
    print(f"  S1: {len(s1_df):,} | S2: {len(s2_df):,} | S3: {len(s3_df):,}")

    s1_ordered_ids = s1_df['entity_id'].to_list()
    final_candidates: Dict[str, List[str]] = {sid: [] for sid in s1_ordered_ids}
    final_matches: Dict[str, List[str]] = {sid: [] for sid in s1_ordered_ids}

    # ── 3. Per-country processing ──────────────────────────────────
    countries = sorted(s1_df['country'].drop_nulls().unique().to_list())
    print(f"Countries: {countries}\n")

    for country in countries:
        print(f"── Country: {country} ──")
        c_s1 = s1_df.filter(pl.col('country') == country)
        c_s2 = s2_df.filter(pl.col('country') == country)
        c_s3 = s3_df.filter(pl.col('country') == country)
        print(f"  S1={len(c_s1):,} S2={len(c_s2):,} S3={len(c_s3):,}")

        t_idx = time.time()
        index = build_country_inverted_index(c_s2, c_s3, max_key_density=150)
        print(f"  Index built in {time.time()-t_idx:.1f}s | keys={len(index):,}")

        # Build s23 text lookup for this country
        t_lk = time.time()
        s23_lk = _s23_lookup(pl.concat([c_s2, c_s3]))
        print(f"  Lookup dict built in {time.time()-t_lk:.1f}s | {len(s23_lk):,} entries")

        # ── Chunked scoring ────────────────────────────────────────
        chunk_size = 10000
        total_s1 = len(c_s1)

        for start in range(0, total_s1, chunk_size):
            chunk = c_s1.slice(start, chunk_size)
            chunk_pairs: List[Tuple[str, str]] = []
            chunk_feats: List[List[float]] = []

            for eid, s1_name, s1_addr in chunk.select(
                ['entity_id', 'business_name', 'business_address']
            ).iter_rows():
                s1_name = s1_name or ''
                s1_addr = s1_addr or ''

                cands = retrieve_candidates_for_s1(
                    s1_name, s1_addr, index, max_candidates=max_candidates
                )
                final_candidates[eid] = cands

                for mid in cands:
                    if mid in s23_lk:
                        m_name, m_addr = s23_lk[mid]
                        feat = compute_pair_features(s1_name, s1_addr, m_name, m_addr, mid)
                        chunk_pairs.append((eid, mid))
                        chunk_feats.append(feat)

            if chunk_feats:
                X_batch = np.array(chunk_feats, dtype=np.float32)
                probs = model.predict_proba(X_batch)
                for (eid, mid), prob in zip(chunk_pairs, probs):
                    if prob >= tau:
                        final_matches[eid].append(mid)

            done = min(start + chunk_size, total_s1)
            print(f"  [{country}] {done:,}/{total_s1:,} processed …", end='\r')

        print(f"  [{country}] Done.                              ")

    # ── 4. Write output files ──────────────────────────────────────
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    match_path = os.path.join(output_dir, "matching_results.tsv")

    print(f"\nWriting {cand_path} …")
    with open(cand_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_ordered_ids:
            f.write(f"{sid}\t{','.join(final_candidates[sid])}\n")

    print(f"Writing {match_path} …")
    with open(match_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_ordered_ids:
            valid_cands = set(final_candidates[sid])
            # Deduplicate and ensure matches are a strict subset of candidates
            seen: Set[str] = set()
            m_list: List[str] = []
            for mid in final_matches[sid]:
                if mid in valid_cands and mid not in seen:
                    m_list.append(mid)
                    seen.add(mid)
            f.write(f"{sid}\t{','.join(m_list)}\n")

    elapsed = time.time() - t0
    print(f"\nInference complete in {elapsed:.1f}s")
    print(f"  → {match_path}")
    print(f"  → {cand_path}")
