from typing import List
import numpy as np
from rapidfuzz import fuzz

try:
    from .normalize import clean_name, clean_compact_name, clean_address, extract_addr_numbers
except ImportError:
    from normalize import clean_name, clean_compact_name, clean_address, extract_addr_numbers


FEATURE_NAMES = [
    # Name similarity features (7 + 2)
    'name_ratio',             # Levenshtein similarity on cleaned names
    'name_token_sort_ratio',  # Token sort ratio (order-invariant)
    'name_token_set_ratio',   # Token set ratio (handles subset/superset)
    'name_token_ratio',       # Token ratio
    'name_wratio',            # Weighted ratio (handles different lengths well)
    'name_partial_ratio',     # Substring containment score
    'name_compact_ratio',     # Similarity on whitespace-stripped cleaned names
    'name_token_jaccard',     # Token-level Jaccard similarity
    'name_first_token_match', # 1.0 if first non-trivial token matches
    'name_len_diff',          # Absolute token length difference
    'name_len_ratio',         # Relative length ratio
    # Address similarity features (4 + 4)
    'addr_ratio',             # Raw address similarity
    'addr_wratio',            # Weighted ratio for addresses
    'addr_token_sort_ratio',  # Address token sort ratio
    'addr_token_set_ratio',   # Address token set ratio
    'addr_token_jaccard',     # Address token Jaccard similarity
    'addr_num_overlap',       # 1=numbers overlap, 0=conflict, -0.5=missing
    'addr_num_conflict',      # 1.0 if both have numbers but 0 overlap
    'addr_num_exact',         # 1.0 if primary numbers match exactly
    'addr_num_count',         # Count of shared address numbers
    'addr_is_empty',          # 1 if candidate address is empty
    # Cross-field features (2)
    'name_in_addr',           # S1 name token overlap with candidate address
    'addr_in_name',           # Candidate address tokens overlapping S1 name
    # Source / structural features (2)
    'is_source_2',            # 1 if candidate is from S2
    'shared_key_count',       # Number of distinct blocking keys shared
]


def compute_pair_features(
    s1_name: str,
    s1_addr: str,
    m_name: str,
    m_addr: str,
    m_id: str,
    shared_key_count: int = 1
) -> List[float]:
    """
    Compute pairwise features for a (S1, candidate) pair.
    All string metrics use C++ SIMD-accelerated rapidfuzz functions.
    """
    # --- Clean inputs ---
    s1_nc = clean_name(s1_name)
    m_nc = clean_name(m_name)
    s1_cmp = s1_nc.replace(' ', '')
    m_cmp = m_nc.replace(' ', '')

    # --- Name similarity features ---
    n_rat = fuzz.ratio(s1_nc, m_nc)
    n_sort = fuzz.token_sort_ratio(s1_nc, m_nc)
    n_set = fuzz.token_set_ratio(s1_nc, m_nc)
    n_tok = fuzz.token_ratio(s1_nc, m_nc)
    n_wrat = fuzz.WRatio(s1_nc, m_nc)
    n_part = fuzz.partial_ratio(s1_nc, m_nc)
    n_cmp = fuzz.ratio(s1_cmp, m_cmp)

    s1_tokens = s1_nc.split()
    m_tokens = m_nc.split()
    s1_tok_set = set(s1_tokens)
    m_tok_set = set(m_tokens)

    # Name Jaccard
    union_len = len(s1_tok_set | m_tok_set)
    n_jaccard = float(len(s1_tok_set & m_tok_set)) / float(union_len) if union_len > 0 else 0.0

    # First token match (crucial brand signal)
    first_tok_match = 1.0 if (s1_tokens and m_tokens and s1_tokens[0] == m_tokens[0]) else 0.0

    l1, l2 = len(s1_nc), len(m_nc)
    n_len_diff = float(abs(l1 - l2))
    n_len_rat = float(min(l1, l2)) / float(max(l1, l2) + 1e-5)

    # --- Address features ---
    s1_a = clean_address(s1_addr)
    m_a = clean_address(m_addr)
    m_empty = 1.0 if len(m_a) == 0 else 0.0

    if s1_a and m_a:
        a_rat = fuzz.ratio(s1_a, m_a)
        a_wrat = fuzz.WRatio(s1_a, m_a)
        a_sort = fuzz.token_sort_ratio(s1_a, m_a)
        a_set = fuzz.token_set_ratio(s1_a, m_a)

        s1_a_toks = set(s1_a.split())
        m_a_toks = set(m_a.split())
        a_union = len(s1_a_toks | m_a_toks)
        a_jaccard = float(len(s1_a_toks & m_a_toks)) / float(a_union) if a_union > 0 else 0.0

        nums1 = extract_addr_numbers(s1_a)
        nums2 = extract_addr_numbers(m_a)
        set_nums1 = set(nums1)
        set_nums2 = set(nums2)

        if set_nums1 and set_nums2:
            inter = len(set_nums1 & set_nums2)
            num_overlap = 1.0 if inter > 0 else 0.0
            num_conflict = 1.0 if inter == 0 else 0.0
            num_exact = 1.0 if nums1[0] == nums2[0] else 0.0
            num_count = float(inter)
        else:
            num_overlap = -0.5
            num_conflict = 0.0
            num_exact = 0.0
            num_count = 0.0
    else:
        a_rat = 0.0
        a_wrat = 0.0
        a_sort = 0.0
        a_set = 0.0
        a_jaccard = 0.0
        num_overlap = -0.5
        num_conflict = 0.0
        num_exact = 0.0
        num_count = 0.0

    # --- Cross-field features ---
    if s1_tok_set and m_a:
        m_addr_tokens = set(m_a.split())
        name_in_addr = float(len(s1_tok_set & m_addr_tokens)) / float(len(s1_tok_set))
    else:
        name_in_addr = 0.0

    if m_tok_set and s1_a:
        s1_addr_tokens = set(s1_a.split())
        addr_in_name = float(len(m_tok_set & s1_addr_tokens)) / float(len(m_tok_set))
    else:
        addr_in_name = 0.0

    is_s2 = 1.0 if m_id.startswith('S2-') else 0.0

    return [
        float(n_rat),
        float(n_sort),
        float(n_set),
        float(n_tok),
        float(n_wrat),
        float(n_part),
        float(n_cmp),
        n_jaccard,
        first_tok_match,
        n_len_diff,
        n_len_rat,
        float(a_rat),
        float(a_wrat),
        float(a_sort),
        float(a_set),
        a_jaccard,
        num_overlap,
        num_conflict,
        num_exact,
        num_count,
        m_empty,
        name_in_addr,
        addr_in_name,
        is_s2,
        float(shared_key_count),
    ]
