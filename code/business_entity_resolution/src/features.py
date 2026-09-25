from typing import List
import numpy as np
from rapidfuzz import fuzz

try:
    from .normalize import clean_name, clean_compact_name, extract_addr_numbers
except ImportError:
    from normalize import clean_name, clean_compact_name, extract_addr_numbers


FEATURE_NAMES = [
    # Name similarity features (5 + 2)
    'name_ratio',           # Levenshtein similarity on cleaned names
    'name_token_sort_ratio',# Token sort ratio (order-invariant)
    'name_token_set_ratio', # Token set ratio (handles subset/superset)
    'name_partial_ratio',   # Substring containment score
    'name_compact_ratio',   # Similarity on whitespace-stripped cleaned names
    'name_len_diff',        # Absolute token length difference
    'name_len_ratio',       # Relative length ratio
    # Address similarity features (3 + 3)
    'addr_ratio',           # Raw address similarity
    'addr_token_sort_ratio',# Address token sort ratio
    'addr_token_set_ratio', # Address token set ratio
    'addr_num_overlap',     # 1=numbers overlap, 0=conflict, -0.5=missing
    'addr_num_count',       # Count of shared address numbers
    'addr_is_empty',        # 1 if candidate address is empty
    # Cross-field features (2)
    'name_in_addr',         # S1 name token overlap with candidate address
    'addr_in_name',         # Candidate address tokens overlapping S1 name
    # Source / structural features (2)
    'is_source_2',          # 1 if candidate is from S2
    'shared_key_count',     # Number of distinct blocking keys shared
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
    Compute 17 pairwise features for a (S1, candidate) pair.
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
    n_part = fuzz.partial_ratio(s1_nc, m_nc)
    n_cmp = fuzz.ratio(s1_cmp, m_cmp)

    l1, l2 = len(s1_nc), len(m_nc)
    n_len_diff = float(abs(l1 - l2))
    n_len_rat = float(min(l1, l2)) / float(max(l1, l2) + 1e-5)

    # --- Address features ---
    s1_a = (s1_addr or "").strip().lower()
    m_a = (m_addr or "").strip().lower()
    m_empty = 1.0 if len(m_a) == 0 else 0.0

    if s1_a and m_a:
        a_rat = fuzz.ratio(s1_a, m_a)
        a_sort = fuzz.token_sort_ratio(s1_a, m_a)
        a_set = fuzz.token_set_ratio(s1_a, m_a)

        nums1 = set(extract_addr_numbers(s1_a))
        nums2 = set(extract_addr_numbers(m_a))
        if nums1 and nums2:
            inter = len(nums1 & nums2)
            num_overlap = 1.0 if inter > 0 else 0.0
            num_count = float(inter)
        else:
            num_overlap = -0.5
            num_count = 0.0
    else:
        a_rat = 0.0
        a_sort = 0.0
        a_set = 0.0
        num_overlap = -0.5
        num_count = 0.0

    # --- Cross-field features ---
    # Do S1 name tokens appear in candidate address?
    s1_name_tokens = set(s1_nc.split())
    m_addr_tokens = set(m_a.split())
    if s1_name_tokens and m_addr_tokens:
        name_in_addr = float(len(s1_name_tokens & m_addr_tokens)) / float(len(s1_name_tokens))
    else:
        name_in_addr = 0.0

    # Do candidate address tokens appear in S1 name?
    m_nc_tokens = set(m_nc.split())
    s1_addr_tokens = set(s1_a.split())
    if m_nc_tokens and s1_addr_tokens:
        addr_in_name = float(len(m_nc_tokens & s1_addr_tokens)) / float(len(m_nc_tokens))
    else:
        addr_in_name = 0.0

    is_s2 = 1.0 if m_id.startswith('S2-') else 0.0

    return [
        float(n_rat),
        float(n_sort),
        float(n_set),
        float(n_part),
        float(n_cmp),
        n_len_diff,
        n_len_rat,
        float(a_rat),
        float(a_sort),
        float(a_set),
        num_overlap,
        num_count,
        m_empty,
        name_in_addr,
        addr_in_name,
        is_s2,
        float(shared_key_count),
    ]
