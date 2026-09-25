"""
blocking.py — High-Recall Multi-Pass Candidate Generator for Entity Resolution.

Combines:
1. Country Partitioning (US, India, France)
2. Primary Name Token Inverted Index
3. Character 3-Gram Inverted Index (typo recovery — Fix 1)
4. Address Token & Street Number Index (DBA/trade name recovery — Fix 2b)
5. Postal Code Index
6. Domain-stripped name matching (Fix 2a handled in normalizer.py)

Target: >95% Candidate Recall with average < 30 candidates per S1 entity.
"""

import sys
import os
import time
from collections import defaultdict

# Relative import
from normalizer import (
    normalize_business_name,
    normalize_address,
    extract_postal_code,
    extract_street_numbers
)

# Common generic stop tokens that should NOT be used as blocking keys
GENERIC_STOP_TOKENS = {
    'pvt_ltd', 'ltd', 'inc', 'corp', 'llc', 'co', 'and', 'the', 'of', 'in', 'for',
    'services', 'service', 'company', 'enterprises', 'solutions', 'technologies',
    'trading', 'traders', 'group', 'industries', 'international', 'global',
    'sarl', 'sasu', 'sas', 'eurl', 'consulting', 'management', 'center',
    'street', 'road', 'avenue', 'drive', 'lane', 'court', 'parkway', 'highway',
    'floor', 'suite', 'unit', 'north', 'south', 'east', 'west', 'near', 'opposite',
    'pvt', 'private', 'limited', 'india', 'new', 'old'
}

# Address stop words for address-only blocking
ADDRESS_STOP_TOKENS = {
    'street', 'road', 'avenue', 'drive', 'lane', 'court', 'parkway', 'highway',
    'floor', 'suite', 'unit', 'north', 'south', 'east', 'west', 'near', 'opposite',
    'no', 'plot', 'flat', 'block', 'sector', 'phase', 'null', 'india', 'us'
}


def _extract_char_ngrams(text: str, n: int = 3) -> set:
    """Extracts character n-grams from normalized text for fuzzy blocking."""
    text = text.replace(' ', '')  # Remove spaces for character-level matching
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def _extract_addr_city_tokens(norm_addr: str) -> set:
    """Extracts likely city/locality tokens from normalized address."""
    tokens = norm_addr.split()
    # Filter out numbers, very short tokens, and common stop words
    return {t for t in tokens if len(t) >= 4 and not t.isdigit() and t not in ADDRESS_STOP_TOKENS}


class RecordStore(dict):
    """Memory-efficient store that stores tuples but yields dicts on access."""
    def __getitem__(self, key):
        val = super().__getitem__(key)
        if isinstance(val, tuple):
            return {
                'eid': key,
                'norm_name': val[0],
                'norm_addr': val[1],
                'pin': val[2],
                'nums': val[3],
                'country': val[4]
            }
        return val

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default


class CandidateBlocker:
    """
    Inverted Index Candidate Blocker with multi-pass recall recovery.
    Partitioned by country to eliminate cross-country comparisons.
    Optimized for high-speed, bounded-memory test set inference.
    """

    def __init__(self, max_candidates=30):
        self.max_candidates = max_candidates
        # country -> token -> list of entity_ids
        self.name_indices = defaultdict(lambda: defaultdict(list))
        # country -> char_ngram -> list of entity_ids
        self.ngram_indices = defaultdict(lambda: defaultdict(list))
        # country -> street_number -> list of entity_ids
        self.number_indices = defaultdict(lambda: defaultdict(list))
        # country -> postal_code -> list of entity_ids
        self.postal_indices = defaultdict(lambda: defaultdict(list))
        # country -> (street_number, city_token) -> list of entity_ids
        self.addr_combined_indices = defaultdict(lambda: defaultdict(list))
        # store entity normalized metadata efficiently
        self.records = RecordStore()

    def add_candidate(self, entity_id: str, name: str, address: str, country: str):
        """Preprocesses and indexes a candidate record (from S2 or S3)."""
        country = country.strip() if country else 'US'
        norm_name = normalize_business_name(name)
        norm_addr = normalize_address(address)
        pin = extract_postal_code(address)
        nums = extract_street_numbers(address)

        # Store compact tuple to save ~2.5GB RAM across 10M records
        self.records[entity_id] = (norm_name, norm_addr, pin, nums, country)

        # === Pass 1 Index: Primary name tokens ===
        tokens = [t for t in norm_name.split() if len(t) >= 3 and t not in GENERIC_STOP_TOKENS]
        for t in set(tokens[:4]):
            lst = self.name_indices[country][t]
            if len(lst) < 600:
                lst.append(entity_id)

        # === Pass 2 Index: Character 3-grams (typo recovery) ===
        core_name = ' '.join(tokens[:3])
        if len(core_name) >= 4:
            ngrams = _extract_char_ngrams(core_name, 3)
            for ng in list(ngrams)[:7]:
                lst = self.ngram_indices[country][ng]
                if len(lst) < 250:
                    lst.append(entity_id)

        # === Pass 3 Index: Street numbers ===
        for num in nums:
            lst = self.number_indices[country][num]
            if len(lst) < 400:
                lst.append(entity_id)

        # === Pass 4 Index: Postal code ===
        if pin:
            lst = self.postal_indices[country][pin]
            if len(lst) < 400:
                lst.append(entity_id)

        # === Pass 5 Index: Address combined (number + city) for DBA/trade name recovery ===
        city_tokens = _extract_addr_city_tokens(norm_addr)
        for num in nums:
            for city in city_tokens:
                key = f"{num}|{city}"
                lst = self.addr_combined_indices[country][key]
                if len(lst) < 150:
                    lst.append(entity_id)

    def retrieve_candidates(self, query_id: str, query_name: str, query_address: str, query_country: str) -> list:
        """
        Retrieves top candidate entity_ids for a query record (from S1).
        Uses weighted multi-pass scoring to rank candidate plausibility.
        """
        country = query_country.strip() if query_country else 'US'
        norm_name = normalize_business_name(query_name)
        norm_addr = normalize_address(query_address)
        pin = extract_postal_code(query_address)
        nums = extract_street_numbers(query_address)

        cand_scores = defaultdict(float)

        # === Pass 1: Name Token Inverted Index ===
        tokens = [t for t in norm_name.split() if len(t) >= 3 and t not in GENERIC_STOP_TOKENS]
        for t in set(tokens[:4]):
            matched_cands = self.name_indices[country].get(t, [])
            token_weight = 3.0 if len(matched_cands) < 500 else 1.0
            for cid in matched_cands:
                cand_scores[cid] += token_weight

        # === Pass 2: Character 3-Gram Overlap (typo recovery) ===
        core_name = ' '.join(tokens[:3])
        if len(core_name) >= 4:
            query_ngrams = _extract_char_ngrams(core_name, 3)
            ngram_cand_hits = defaultdict(int)
            for ng in list(query_ngrams)[:8]:
                for cid in self.ngram_indices[country].get(ng, []):
                    ngram_cand_hits[cid] += 1
            # Only boost candidates with >= 3 ngram overlaps (reduces noise)
            min_overlap = max(2, len(query_ngrams) // 4)
            for cid, hit_count in ngram_cand_hits.items():
                if hit_count >= min_overlap:
                    cand_scores[cid] += 1.0 + (hit_count / max(len(query_ngrams), 1)) * 2.0

        # === Pass 3: Street Number overlap ===
        for num in nums:
            matched_num_cands = self.number_indices[country].get(num, [])
            if len(matched_num_cands) < 1000:
                for cid in matched_num_cands:
                    cand_scores[cid] += 1.5

        # === Pass 4: Postal Code matches ===
        if pin:
            matched_pin_cands = self.postal_indices[country].get(pin, [])
            if len(matched_pin_cands) < 500:
                for cid in matched_pin_cands:
                    cand_scores[cid] += 1.0

        # === Pass 5: Address-only blocking (DBA / trade name recovery) ===
        # If entities share same (street_number + city), they're strong candidates
        # even with completely different business names
        city_tokens = _extract_addr_city_tokens(norm_addr)
        for num in nums:
            for city in city_tokens:
                key = f"{num}|{city}"
                matched_addr_cands = self.addr_combined_indices[country].get(key, [])
                if len(matched_addr_cands) < 200:
                    for cid in matched_addr_cands:
                        cand_scores[cid] += 3.0  # Strong signal — same physical location

        if not cand_scores:
            return []

        # Sort candidates by combined overlap score
        sorted_cands = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in sorted_cands[:self.max_candidates]]


if __name__ == '__main__':
    blocker = CandidateBlocker(max_candidates=10)
    # Test 1: Standard name match
    blocker.add_candidate("S2-001", "Orelee's Barbershop", "1795 Westchester Drive, High Point, NC", "US")
    blocker.add_candidate("S3-002", "Orelee's (Barbershop)", "#1795 Westchester Dr, High Point", "US")
    # Test 2: DBA / trade name (different name, same address)
    blocker.add_candidate("S3-003", "Solkeloquo", "6114 Tenth Ave, Spokane Valley, Washington", "US")
    # Test 3: Domain name
    blocker.add_candidate("S2-004", "summithealth.com", "123 Main St, New York, NY", "US")
    # Test 4: Typo
    blocker.add_candidate("S2-005", "Payne Enterprises LLC", "456 Oak Rd, Dallas, TX", "US")
    blocker.add_candidate("S2-006", "Tata Motors Ltd", "Bombay House, Fort, Mumbai", "India")

    print("Test 1 - Standard name:")
    print(" ", blocker.retrieve_candidates("S1-100", "Orelee's Barbershop", "1795 Westchester Drive, NC", "US"))

    print("Test 2 - DBA / trade name (address-only match):")
    print(" ", blocker.retrieve_candidates("S1-101", "Uptown Pub", "6114 10th Avenue, Spokane Valley, WA", "US"))

    print("Test 3 - Domain name match:")
    print(" ", blocker.retrieve_candidates("S1-102", "Summit Health LLC", "123 Main Street, New York, NY", "US"))

    print("Test 4 - Typo recovery:")
    print(" ", blocker.retrieve_candidates("S1-103", "Payne Enterprises", "456 Oak Road, Dallas, TX", "US"))

    print("Test 5 - India:")
    print(" ", blocker.retrieve_candidates("S1-200", "Tata Motors Limited", "Fort, Mumbai 400001", "India"))
