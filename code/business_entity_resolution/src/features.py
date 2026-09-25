"""
features.py — High-performance feature engineering for pairwise Entity Resolution.

Uses RapidFuzz (C++ SIMD-accelerated) for sub-millisecond string metrics.
Includes explicit address missingness, street number conflicts, and exact location indicators.
"""

import numpy as np
from rapidfuzz import fuzz

FEATURE_NAMES = [
    'name_token_sort_ratio',
    'name_token_set_ratio',
    'name_levenshtein_ratio',
    'name_partial_ratio',
    'name_len_diff',
    'name_len_ratio',
    'name_token_count_diff',
    'name_exact_match',
    'address_token_sort_ratio',
    'address_token_set_ratio',
    'address_partial_ratio',
    'number_overlap',
    'postal_code_match',
    'country_match',
    'is_source_2',
    'is_addr_missing',
    'street_num_conflict',
    'street_num_match',
    'addr_exact_location'
]


def extract_pairwise_features(s1_meta: dict, cand_meta: dict) -> list:
    """
    Computes a 19-dimensional numeric feature vector for a pair of entities.
    Expects normalized metadata dicts containing:
    - norm_name (str)
    - norm_addr (str)
    - pin (str)
    - nums (set)
    - country (str)
    - eid (str)
    """
    n1 = s1_meta.get('norm_name', '')
    n2 = cand_meta.get('norm_name', '')
    a1 = s1_meta.get('norm_addr', '')
    a2 = cand_meta.get('norm_addr', '')
    p1 = s1_meta.get('pin', '')
    p2 = cand_meta.get('pin', '')
    nums1 = s1_meta.get('nums', set())
    nums2 = cand_meta.get('nums', set())
    c1 = s1_meta.get('country', '')
    c2 = cand_meta.get('country', '')
    cand_id = cand_meta.get('eid', '')

    # --- Name metrics ---
    f_name_ts = fuzz.token_sort_ratio(n1, n2) / 100.0
    f_name_set = fuzz.token_set_ratio(n1, n2) / 100.0
    f_name_ratio = fuzz.ratio(n1, n2) / 100.0
    f_name_part = fuzz.partial_ratio(n1, n2) / 100.0

    len1, len2 = len(n1), len(n2)
    f_name_len_diff = abs(len1 - len2)
    f_name_len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0

    toks1 = n1.split()
    toks2 = n2.split()
    f_tok_diff = abs(len(toks1) - len(toks2))
    f_name_exact = 1.0 if n1 == n2 and len1 > 0 else 0.0

    # --- Address metrics ---
    has_a1 = bool(a1 and a1.strip() and a1.strip() != 'null')
    has_a2 = bool(a2 and a2.strip() and a2.strip() != 'null')

    f_addr_ts = fuzz.token_sort_ratio(a1, a2) / 100.0 if (has_a1 and has_a2) else 0.0
    f_addr_set = fuzz.token_set_ratio(a1, a2) / 100.0 if (has_a1 and has_a2) else 0.0
    f_addr_part = fuzz.partial_ratio(a1, a2) / 100.0 if (has_a1 and has_a2) else 0.0

    # Explicit indicator for missing address (prevents penalizing name matches when address is empty)
    f_is_addr_missing = 1.0 if (not has_a1 or not has_a2) else 0.0

    # --- Street number metrics ---
    has_nums1 = bool(nums1)
    has_nums2 = bool(nums2)

    if has_nums1 and has_nums2:
        intersect = nums1.intersection(nums2)
        f_num_overlap = 1.0 if intersect else 0.0
        f_street_num_match = 1.0 if intersect else 0.0
        f_street_num_conflict = 1.0 if not intersect else 0.0  # Different street numbers!
    elif not has_nums1 and not has_nums2:
        f_num_overlap = 0.5  # neutral
        f_street_num_match = 0.0
        f_street_num_conflict = 0.0
    else:
        f_num_overlap = 0.2  # one has numbers, other doesn't
        f_street_num_match = 0.0
        f_street_num_conflict = 0.0

    # Exact location match (same street number AND highly similar address) — strong signal for DBA
    f_addr_exact_location = 1.0 if (f_street_num_match == 1.0 and f_addr_part >= 0.80) else 0.0

    # --- Postal code match ---
    if p1 and p2:
        f_pin_match = 1.0 if p1 == p2 else 0.0
    else:
        f_pin_match = -1.0  # missing

    # --- Country match ---
    f_country_match = 1.0 if c1 == c2 else 0.0

    # --- Source indicator ---
    f_is_s2 = 1.0 if cand_id.startswith('S2-') else 0.0

    return [
        f_name_ts,
        f_name_set,
        f_name_ratio,
        f_name_part,
        f_name_len_diff,
        f_name_len_ratio,
        f_tok_diff,
        f_name_exact,
        f_addr_ts,
        f_addr_set,
        f_addr_part,
        f_num_overlap,
        f_pin_match,
        f_country_match,
        f_is_s2,
        f_is_addr_missing,
        f_street_num_conflict,
        f_street_num_match,
        f_addr_exact_location
    ]


if __name__ == '__main__':
    meta1 = {
        'norm_name': 'orelee s barbershop',
        'norm_addr': '1795 westchester drive high point north carolina',
        'pin': '',
        'nums': {'1795'},
        'country': 'US',
        'eid': 'S1-001'
    }
    meta2 = {
        'norm_name': 'orelee s barbershop',
        'norm_addr': '1795 westchester drive nc high point',
        'pin': '',
        'nums': {'1795'},
        'country': 'US',
        'eid': 'S2-001'
    }
    feat = extract_pairwise_features(meta1, meta2)
    print("Feature vector:", feat)
    print("Feature count:", len(feat))
    assert len(feat) == len(FEATURE_NAMES)
    print("Assert passed: 19 features.")
