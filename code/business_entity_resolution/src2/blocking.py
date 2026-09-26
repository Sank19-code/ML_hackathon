import re
from collections import defaultdict
from typing import Dict, List, Tuple

import polars as pl

try:
    from .normalize import clean_name, clean_compact_name, extract_addr_numbers, extract_addr_words
except ImportError:
    from normalize import clean_name, clean_compact_name, extract_addr_numbers, extract_addr_words


def generate_blocking_keys(name: str, addr: str) -> List[Tuple[str, str]]:
    """
    Generate high-recall blocking keys for a business record.

    Key strategy (multi-pass, each from a different signal):
      cmp      – compact clean name (handles domain names, whitespace noise)
      cmp6     – 6-char prefix of compact clean name
      sort     – sorted top-4 clean tokens (handles word-order transpositions)
      sort3    – sorted top-3 tokens (shorter businesses)
      f2       – first 2 clean tokens
      f1       – first clean token (len >= 3, captures short brands like IBM, GAP, SONY)
      num_n1   – primary address number + first name word
      num_str  – primary address number + each significant street word (top 3)
      num_pair – first two address numbers (e.g., plot+zip)
      pin_n1   – 5 or 6 digit postal PIN/ZIP + first name word
    """
    keys = []
    cname = clean_name(name)
    tokens = cname.split()

    # --- Name-based keys ---
    compact = cname.replace(' ', '')
    if len(compact) >= 4:
        keys.append(('cmp', compact))
        if len(compact) >= 6:
            keys.append(('cmp6', compact[:6]))

    if tokens:
        # Sorted top-4 tokens: catches permuted word orders
        top_sorted = "_".join(sorted(tokens[:4]))
        keys.append(('sort', top_sorted))

        # Sorted top-3 (cheaper, better for short names)
        if len(tokens) >= 2:
            top3_sorted = "_".join(sorted(tokens[:3]))
            keys.append(('sort3', top3_sorted))
            keys.append(('f2', f"{tokens[0]}_{tokens[1]}"))

        # First word (allow >= 3 to capture short brands like Sony, Ford, Gap, Nike, Dell, Tata, IBM)
        if len(tokens[0]) >= 3:
            keys.append(('f1', tokens[0]))

    # --- Address-based keys ---
    nums = extract_addr_numbers(addr)
    words = extract_addr_words(addr)

    if nums:
        primary_num = nums[0]
        # number + first name word (most robust key)
        if tokens:
            keys.append(('num_n1', f"{primary_num}_{tokens[0]}"))
            # Check for 5-digit US or 6-digit India postal code
            for n in nums:
                if len(n) in (5, 6):
                    keys.append(('pin_n1', f"{n}_{tokens[0]}"))
                    if compact and len(compact) >= 3:
                        keys.append(('pin_c3', f"{n}_{compact[:3]}"))
                    break
        # number + each significant street word
        for w in words[:3]:
            keys.append(('num_str', f"{primary_num}_{w}"))

        # number + sorted top 2 words
        if len(tokens) >= 2:
            keys.append(('num_sort2', f"{primary_num}_{'_'.join(sorted(tokens[:2]))}"))

        # pair of address numbers (plot number + area code)
        if len(nums) >= 2:
            keys.append(('num_pair', f"{nums[0]}_{nums[1]}"))

    return keys


def build_country_inverted_index(
    s2_df: pl.DataFrame,
    s3_df: pl.DataFrame,
    max_key_density: int = 150
) -> Dict[Tuple[str, str], List[str]]:
    """
    Build key → [entity_ids] inverted index for S2 and S3 within a country.
    Uses tuple iteration (fastest Polars path) and prunes ultra-common keys.
    """
    index = defaultdict(list)

    # Use iter_rows() (unnamed tuple) for maximum speed
    for df in (s2_df, s3_df):
        for eid, name, addr in df.select(
            ['entity_id', 'business_name', 'business_address']
        ).iter_rows():
            name = name or ''
            addr = addr or ''
            for k in generate_blocking_keys(name, addr):
                index[k].append(eid)

    # Prune ultra-dense keys (common stop-words / generic names pollute results)
    return {k: v for k, v in index.items() if len(v) <= max_key_density}


def retrieve_candidates_for_s1(
    name: str,
    addr: str,
    index: Dict[Tuple[str, str], List[str]],
    max_candidates: int = 20
) -> List[str]:
    """
    Retrieve candidate entity IDs for a single Source 1 record.
    Ranks candidates by number of distinct blocking keys they share with S1
    (more shared keys → higher confidence → ranked first).
    """
    cand_freq: Dict[str, int] = defaultdict(int)

    for k in generate_blocking_keys(name, addr):
        bucket = index.get(k)
        if bucket:
            for eid in bucket:
                cand_freq[eid] += 1

    # Stable deterministic ordering matters for ties at the candidate cap.
    sorted_cands = sorted(cand_freq, key=lambda eid: (-cand_freq[eid], eid))
    return sorted_cands[:max_candidates]


def retrieve_candidates_with_counts(
    name: str,
    addr: str,
    index: Dict[Tuple[str, str], List[str]],
    max_candidates: int = 20,
) -> List[Tuple[str, int]]:
    """Return top candidates together with their blocking-key overlap count."""
    cand_freq: Dict[str, int] = defaultdict(int)
    for k in generate_blocking_keys(name, addr):
        for eid in index.get(k, []):
            cand_freq[eid] += 1
    ranked = sorted(cand_freq, key=lambda eid: (-cand_freq[eid], eid))
    return [(eid, cand_freq[eid]) for eid in ranked[:max_candidates]]


def get_shared_key_count(
    name: str,
    addr: str,
    candidate_id: str,
    index: Dict[Tuple[str, str], List[str]]
) -> int:
    """Return the number of blocking keys shared between an S1 and a candidate."""
    count = 0
    for k in generate_blocking_keys(name, addr):
        bucket = index.get(k)
        if bucket and candidate_id in bucket:
            count += 1
    return count
