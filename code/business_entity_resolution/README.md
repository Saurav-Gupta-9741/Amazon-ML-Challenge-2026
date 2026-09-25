# Business Entity Resolution Pipeline

This repository contains the complete, self-contained solution for the **Amazon ML Challenge 2026: Business Entity Resolution Challenge**.

## Environment Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

All models are open-source with MIT/Apache 2.0 licenses (LightGBM, RapidFuzz) and have zero external API dependencies.

## Architecture

1. **Preprocessing & Transliteration (`src/transliterate.py`, `src/normalizer.py`):**
   - Pure-python phonetic romanization of Hindi Devanagari script records in Source 2 & 3.
   - Unicode NFKD diacritics stripping for French records in test set.
   - Legal suffix canonicalization (`Pvt Ltd`, `LLC`, `SARL`, `SAS`).
   - Postal code and street number extraction.

2. **Candidate Generation / Blocking (`src/blocking.py`):**
   - Country-partitioned multi-pass inverted indexing on core business name tokens.
   - Street number and postal code fallback indexing.
   - Top-25 candidates preserved per Source 1 entity.

3. **Feature Engineering (`src/features.py`):**
   - RapidFuzz SIMD C++ string similarities (Token Sort, Token Set, Levenshtein, Partial).
   - Address overlap, postal code exact/mismatch match, house number match.

4. **Pairwise Classifier & F_0.5 Optimizer (`src/model.py`, `src/train.py`):**
   - LightGBM binary classifier trained on ground truth matches and hard negative candidate pairs.
   - Grid search threshold optimization aligning with precision-heavy Macro F_0.5 metric and singleton pruning.

5. **Inference & Validation (`src/infer.py`):**
   - Streams test Source 1 records and outputs tab-separated `matching_results.tsv` and `candidate_pairs.tsv`.
   - Automatically executes `utils/validate_submission.py`.

## Running the Pipeline

To run the complete pipeline end-to-end:
```bash
python src/run_pipeline.py --mode full
```

To run training only:
```bash
python src/run_pipeline.py --mode train
```

To run inference only (using pre-trained model):
```bash
python src/run_pipeline.py --mode infer
```

To validate outputs against the official competition validator:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
