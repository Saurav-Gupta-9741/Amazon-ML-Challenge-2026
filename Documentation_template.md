# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Team CognitiveAI  
**Problem Track:** Business Entity Resolution across Multi-Source Heterogeneous Catalogues  
**Evaluation Metric:** Macro-averaged $F_{0.5}$ (Precision-weighted)  
**Submission Date:** September 25, 2026  

---

## 1. Executive Summary

We developed an ultra-high precision, two-stage scalable Entity Resolution architecture combining multi-pass inverted-index blocking with a gradient-boosted decision tree (LightGBM) pairwise matcher. Our key innovations include an authentic 8-script Indic transliteration engine (handling Hindi, Tamil, Telugu, Kannada, Gujarati, Bengali, Gurmukhi, and Malayalam), domain-name canonicalization, and location-aware feature engineering (street number conflict detection and missing address invariance). On local validation against 2,000 independent ground-truth queries with 150,000 background distractors, our solution achieved **96.29% Macro $F_{0.5}$** with **99.62% non-singleton precision** and **94.22% candidate blocking recall ceiling**.

---

## 2. Methodology

### 2.1 Problem Analysis
Through deep error analysis across 2.2M training entities, we discovered five fundamental data traps:
1. **Multi-Script Heterogeneity:** Records in Source 2 and Source 3 frequently appear in regional Indic scripts (Devanagari, Tamil, Telugu, Kannada, Gujarati, Bengali, etc.), while Source 1 is primarily Latin-scripted.
2. **Domain Concatenation:** Many corporate records in S2/S3 represent businesses by web domains (e.g. `summithealth.com` vs `Summit Health LLC`), rendering standard word tokenization ineffective.
3. **Empty Address Ambiguity:** Approximately 5% of true matches possess blank or null addresses in S2/S3. Standard similarity metrics score address distance as 0, artificially depressing overall match probability.
4. **Street Number Disambiguation:** Businesses on the same avenue often share 90%+ address string overlap (e.g., `103 Crickettown Rd` vs `112 Crickettown Rd`), requiring explicit street number conflict detection to prevent catastrophic false merges.
5. **Singleton Dominance:** ~5.5% of Source 1 entities have zero corresponding records in S2/S3, penalizing uncontrolled recall under Macro $F_{0.5}$.

### 2.2 Solution Strategy
**Approach Type:** Country-Partitioned Multi-Pass Blocking + 19-Dimensional SIMD Feature Engineering + LightGBM Pairwise Classification + Calibrated Dual-Threshold Gating.  
**Core Innovation:** Dynamic Dravidian/Indo-Aryan phonetic romanization with 200+ canonical business terms, paired with a target-guaranteed hard-negative training sampler and exact location conflict detection.

---

## 3. Candidate Generation (Blocking)

To reduce the $1.73\text{M} \times 10\text{M}$ candidate space to an average of $<25$ candidates per entity without losing true matches:
- **Country Partitioning:** Partitioned into US, India, and France sub-indices, eliminating 70% of cross-country comparisons.
- **Pass 1 (Distinctive Token Index):** Inverted index over non-generic name tokens (IDF-weighted).
- **Pass 2 (Character 3-Gram Inverted Index):** Substring n-gram indexing to catch near-misses and OCR typos.
- **Pass 3 (Physical Location & Street Number Index):** Inverted index mapping `(street_number + city)` to capture DBAs/trade names sharing identical physical premises.
- **Pass 4 (Postal Code Index):** PIN/ZIP code grouping for local business disambiguation.
- **Recall Ceiling:** Achieved **94.22% blocking recall** on challenging 150,000 distractor pools.

---

## 4. Matching Model

**19-Dimensional SIMD Pairwise Features (RapidFuzz C++ SIMD):**
1. **Name Similarity (8 dims):** Token Sort Ratio, Token Set Ratio, Levenshtein Ratio, Partial Ratio, Length Difference, Length Ratio, Token Count Difference, Exact Match Indicator.
2. **Address Similarity (4 dims):** Token Sort Ratio, Token Set Ratio, Partial Ratio, Missing Address Indicator (`is_addr_missing`).
3. **Street Number & Location Logic (4 dims):** Street Number Overlap, Street Number Conflict Indicator (`num_conflict`), Street Number Match (`num_match`), Exact Location Match (`addr_exact_location`).
4. **Metadata & Country (3 dims):** Postal Code Match, Country Match, Source Indicator (`is_source_2`).

**Model Architecture:** LightGBM GBDT (300 estimators, max depth 8, num leaves 63, learning rate 0.08, subsample 0.8).  
**Threshold Selection:** Fine-grained grid search directly maximizing Macro $F_{0.5}$ on an 80/20 validation split, yielding calibrated Singleton Threshold = 0.70 and Match Threshold = 0.65.

---

## 5. Results & Error Analysis

### Benchmark Performance Progression

| Metric | Baseline | Pass 1 | Final System |
|---|---|---|---|
| **Macro $F_{0.5}$ (Leaderboard Metric)** | 82.64% | 88.58% | **96.29%** |
| **Candidate Blocking Recall Ceiling** | 80.07% | 94.13% | **94.22%** |
| **Non-Singleton Precision** | 99.69% | 99.61% | **99.62%** |
| **Non-Singleton Recall** | 65.77% | 73.97% | **91.38%** |
| **Singleton Accuracy (Correctly Blank)** | 99.1% | 99.1% | **98.2%** |
| **US Macro $F_{0.5}$** | 88.63% | 91.53% | **97.39%** |
| **India Macro $F_{0.5}$** | 73.39% | 84.03% | **94.59%** |

- **False Merges (False Positives):** Only 0.38% of predictions were false merges, mitigated by the street number conflict detector.
- **Missed Matches (False Negatives):** Primarily constrained by candidate blocking recall ceiling (5.78% of unindexed edge cases).

---

## 6. Conclusion

By addressing the root causes of error—regional Indic script diversity, domain name noise, street number collisions, and training distribution imbalance—our system achieved a **96.29% Macro $F_{0.5}$** score. The pipeline is zero-API, fully data-driven, and processes over 90 entities per second, making it fully ready for full-scale test inference.

---

## Appendix

### A. Code Artefacts
- `src/config.py`: Central configuration, paths, and hyperparameters.
- `src/transliterate.py`: Multi-script phonetic romanizer with 200+ canonical business terms for 8 Indian scripts.
- `src/normalizer.py`: Text cleaning, domain stripping, accent removal, legal suffix normalization.
- `src/blocking.py`: Multi-pass country-partitioned candidate blocker.
- `src/features.py`: 19-dimensional SIMD pairwise feature extractor.
- `src/model.py`: LightGBM classifier with Macro $F_{0.5}$ threshold optimizer.
- `src/train.py`: Model training pipeline with target-guaranteed sampling.
- `src/infer.py`: Streaming test inference with official validator integration.
- `src/run_pipeline.py`: Unified CLI orchestrator.
