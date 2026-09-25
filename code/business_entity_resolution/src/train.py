"""
train.py — End-to-End Training Pipeline for Business Entity Resolution.

Workflow:
1. Load ground truth for training S1 entities and identify all true target matches.
2. Load S1 query records.
3. Build multi-pass candidate blocker containing all true targets + challenging background distractors.
4. Generate positive and hard negative pairwise features (19 dimensions).
5. Train LightGBM Classifier.
6. Calibrate singleton and match thresholds via Macro F_0.5 grid search on validation split.
7. Save model artifacts to disk.
"""

import sys
import os
import time
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Relative imports
import config
from normalizer import (
    normalize_business_name,
    normalize_address,
    extract_postal_code,
    extract_street_numbers
)
from blocking import CandidateBlocker
from features import extract_pairwise_features, FEATURE_NAMES
from model import EntityResolutionModel, evaluate_macro_f05


def run_training(n_s1_entities=15000, background_distractors=160000):
    """Executes the full training and threshold optimization pipeline."""
    print("=" * 70)
    print("BUSINESS ENTITY RESOLUTION PIPELINE — MODEL RETRAINING")
    print("=" * 70)

    t_start = time.time()

    # 1. Load Ground Truth for training entities & collect all true targets
    print(f"\n[Step 1/5] Loading ground truth for {n_s1_entities} S1 entities...")
    gt_map = {}
    needed_target_ids = set()
    with open(config.TRAIN_GT, encoding='utf-8') as f:
        f.readline()
        for i, line in enumerate(f):
            if i >= n_s1_entities:
                break
            parts = line.strip().split('\t')
            s1_id = parts[0]
            matches = [m.strip() for m in parts[1].split(',') if m.strip()] if len(parts) > 1 else []
            gt_map[s1_id] = matches
            needed_target_ids.update(matches)

    print(f"  Loaded ground truth for {len(gt_map)} S1 entities.")
    print(f"  Identified {len(needed_target_ids)} true target records to include in candidate pool.")

    # 2. Load S1 Entity Records
    print(f"\n[Step 2/5] Loading S1 query records from train_source1.tsv...")
    s1_dict = {}
    with open(config.TRAIN_S1, encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            s1_id = parts[0]
            if s1_id in gt_map:
                s1_dict[s1_id] = {
                    'eid': s1_id,
                    'name': parts[1] if len(parts) > 1 else '',
                    'address': parts[2] if len(parts) > 2 else '',
                    'country': parts[3] if len(parts) > 3 else 'US',
                    'norm_name': normalize_business_name(parts[1] if len(parts) > 1 else ''),
                    'norm_addr': normalize_address(parts[2] if len(parts) > 2 else ''),
                    'pin': extract_postal_code(parts[2] if len(parts) > 2 else ''),
                    'nums': extract_street_numbers(parts[2] if len(parts) > 2 else '')
                }
                if len(s1_dict) == len(gt_map):
                    break

    print(f"  Loaded {len(s1_dict)} Source 1 records.")

    # 3. Build Candidate Blocker with True Targets + Background Distractors
    print(f"\n[Step 3/5] Building candidate blocker (all {len(needed_target_ids)} true targets + {background_distractors} distractors)...")
    blocker = CandidateBlocker(max_candidates=config.MAX_CANDIDATES_PER_S1)

    t0 = time.time()
    half_distractors = background_distractors // 2
    count_indexed = 0
    targets_found = 0

    for path, prefix in [(config.TRAIN_S2, 'S2'), (config.TRAIN_S3, 'S3')]:
        print(f"  Scanning {os.path.basename(path)}...")
        with open(path, encoding='utf-8') as f:
            f.readline()
            for i, line in enumerate(f):
                parts = line.strip().split('\t')
                eid = parts[0]
                is_target = eid in needed_target_ids
                if is_target:
                    targets_found += 1

                if is_target or i < half_distractors:
                    name = parts[1] if len(parts) > 1 else ''
                    addr = parts[2] if len(parts) > 2 else ''
                    country = parts[3] if len(parts) > 3 else 'US'
                    blocker.add_candidate(eid, name, addr, country)
                    count_indexed += 1

    print(f"  Indexed {count_indexed} total candidate records ({targets_found} true targets) in {time.time() - t0:.2f}s.")

    # 4. Generate Training Pairs & Extract 19-Dimensional Features
    print(f"\n[Step 4/5] Generating pairwise features (Positives + Hard Negatives)...")
    X_list = []
    y_list = []

    # 80/20 train/validation split
    s1_keys = list(s1_dict.keys())
    np.random.seed(config.RANDOM_SEED)
    np.random.shuffle(s1_keys)
    split_idx = int(0.80 * len(s1_keys))
    train_keys = set(s1_keys[:split_idx])
    val_keys = set(s1_keys[split_idx:])

    val_candidates_meta = defaultdict(list)
    val_gt = {k: gt_map[k] for k in val_keys}

    t0 = time.time()
    for s1_id, s1_rec in s1_dict.items():
        true_matches = set(gt_map.get(s1_id, []))
        cands = blocker.retrieve_candidates(s1_id, s1_rec['name'], s1_rec['address'], s1_rec['country'])

        for cand_id in cands:
            cand_meta = blocker.records.get(cand_id)
            if not cand_meta:
                continue
            cand_meta['eid'] = cand_id
            feats = extract_pairwise_features(s1_rec, cand_meta)
            label = 1 if cand_id in true_matches else 0

            if s1_id in train_keys:
                X_list.append(feats)
                y_list.append(label)
            else:
                val_candidates_meta[s1_id].append((cand_id, feats, label))

    X_train = np.array(X_list, dtype=np.float32)
    y_train = np.array(y_list, dtype=np.int32)
    n_pos = int(np.sum(y_train))
    n_neg = len(y_train) - n_pos

    print(f"  Training pairs: {len(X_train)} (Positives: {n_pos}, Negatives: {n_neg}, Ratio: 1:{n_neg//max(n_pos, 1)})")
    print(f"  Pair extraction took {time.time() - t0:.2f}s.")

    # 5. Train LightGBM Classifier
    print(f"\n[Step 5/5] Training LightGBM Classifier ({len(FEATURE_NAMES)} features)...")
    er_model = EntityResolutionModel()
    er_model.fit(X_train, y_train)
    print("  LightGBM model training complete.")

    # 6. Optimize Macro F_0.5 Thresholds on Validation Split
    print("\nOptimizing Macro F_0.5 thresholds on validation split...")
    val_query_candidates = {}
    for s1_id in val_keys:
        cand_tuples = val_candidates_meta.get(s1_id, [])
        if not cand_tuples:
            val_query_candidates[s1_id] = []
            continue
        X_val_s1 = np.array([t[1] for t in cand_tuples], dtype=np.float32)
        probs = er_model.predict_proba(X_val_s1)
        val_query_candidates[s1_id] = [(cand_tuples[idx][0], float(probs[idx])) for idx in range(len(probs))]

    # Expanded threshold grid
    best_score = -1.0
    best_single_th = 0.60
    best_match_th = 0.55

    for single_th in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        for match_th in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
            preds_map = {}
            for s1_id, cand_probs in val_query_candidates.items():
                if not cand_probs:
                    preds_map[s1_id] = []
                    continue

                max_prob = max(prob for _, prob in cand_probs)
                if max_prob < single_th:
                    preds_map[s1_id] = []
                    continue

                selected = [cid for cid, prob in cand_probs if prob >= match_th]
                preds_map[s1_id] = selected

            score = evaluate_macro_f05(val_gt, preds_map)
            if score > best_score:
                best_score = score
                best_single_th = single_th
                best_match_th = match_th

    er_model.singleton_threshold = best_single_th
    er_model.match_threshold = best_match_th
    print(f"  Optimal Calibrated Thresholds: Singleton Th = {best_single_th}, Match Th = {best_match_th}")
    print(f"  Validation Macro F_0.5 Score: {best_score:.4f} ({best_score*100:.2f}%)")

    # 7. Save Model Artifact
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    er_model.save(config.MODEL_PATH)
    print(f"\n  Trained model & calibrated thresholds saved to: {config.MODEL_PATH}")
    print(f"  Total training pipeline completed in {time.time() - t_start:.2f}s.")
    print("=" * 70)
    return best_score


if __name__ == '__main__':
    run_training()
