"""
benchmark_local.py — Rigorous local validation benchmark for Business Entity Resolution.

Evaluates:
1. Exact Macro F_0.5 Score (competition leaderboard metric)
2. Precision and Recall on matching entities
3. Singleton Accuracy (0-match entities correctly identified)
4. Candidate Blocking Recall (ceiling of retrieval stage)
5. Country-wise performance breakdown (US vs India)
6. Error Analysis: Sample False Merges (FP) and Missed Links (FN)
"""

import sys
import os
import time
from collections import defaultdict
import numpy as np

# Add src directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'business_entity_resolution', 'src'))

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import config
from normalizer import (
    normalize_business_name,
    normalize_address,
    extract_postal_code,
    extract_street_numbers
)
from blocking import CandidateBlocker
from features import extract_pairwise_features
from model import EntityResolutionModel, compute_entity_f05


def run_benchmark(num_val_queries=2000, background_candidates=150000):
    print("=" * 75)
    print("LOCAL VALIDATION BENCHMARK — MACRO F_0.5 EVALUATION")
    print("=" * 75)

    # 1. Load Ground Truth for Validation Queries
    print(f"\n[Step 1/5] Sampling {num_val_queries} validation entities from train_ground_truth.tsv...")
    val_gt = {}
    needed_s2_s3_ids = set()

    with open(config.TRAIN_GT, encoding='utf-8') as f:
        f.readline()
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            s1_id = parts[0]
            matches = [m.strip() for m in parts[1].split(',') if m.strip()] if len(parts) > 1 else []
            val_gt[s1_id] = matches
            needed_s2_s3_ids.update(matches)
            if len(val_gt) >= num_val_queries:
                break

    val_s1_ids = set(val_gt.keys())
    num_singletons = sum(1 for m in val_gt.values() if not m)
    total_true_links = sum(len(m) for m in val_gt.values())
    print(f"  Sampled {len(val_gt)} S1 queries:")
    print(f"  - Entities with matches : {len(val_gt) - num_singletons}")
    print(f"  - Singletons (0 matches): {num_singletons} ({num_singletons / len(val_gt) * 100:.1f}%)")
    print(f"  - Total true target links: {total_true_links}")

    # 2. Load S1 Metadata
    print(f"\n[Step 2/5] Loading S1 metadata from train_source1.tsv...")
    val_s1_records = {}
    with open(config.TRAIN_S1, encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            s1_id = parts[0]
            if s1_id in val_s1_ids:
                val_s1_records[s1_id] = {
                    'eid': s1_id,
                    'name': parts[1] if len(parts) > 1 else '',
                    'address': parts[2] if len(parts) > 2 else '',
                    'country': parts[3] if len(parts) > 3 else 'US'
                }
                if len(val_s1_records) == len(val_s1_ids):
                    break

    # 3. Load Candidate Pool (True Matches + Challenging Distractors)
    print(f"\n[Step 3/5] Loading candidate pool ({background_candidates} distractors + all true targets)...")
    blocker = CandidateBlocker(max_candidates=config.MAX_CANDIDATES_PER_S1)

    t0 = time.time()
    loaded_candidates = {}

    for path, prefix in [(config.TRAIN_S2, 'S2'), (config.TRAIN_S3, 'S3')]:
        print(f"  Scanning {os.path.basename(path)}...")
        with open(path, encoding='utf-8') as f:
            f.readline()
            for i, line in enumerate(f):
                parts = line.strip().split('\t')
                eid = parts[0]
                if eid in needed_s2_s3_ids or i < background_candidates // 2:
                    name = parts[1] if len(parts) > 1 else ''
                    addr = parts[2] if len(parts) > 2 else ''
                    country = parts[3] if len(parts) > 3 else 'US'
                    blocker.add_candidate(eid, name, addr, country)
                    loaded_candidates[eid] = {
                        'eid': eid,
                        'name': name,
                        'address': addr,
                        'country': country
                    }

    print(f"  Candidate pool ready: {len(loaded_candidates)} records indexed in {time.time() - t0:.2f}s.")

    # 4. Load Trained Model
    print(f"\n[Step 4/5] Loading trained LightGBM model from {config.MODEL_PATH}...")
    er_model = EntityResolutionModel()
    if os.path.exists(config.MODEL_PATH):
        er_model.load(config.MODEL_PATH)
        print(f"  Model loaded: Match Threshold = {er_model.match_threshold}, Singleton Threshold = {er_model.singleton_threshold}")
    else:
        print("  Model file not found! Falling back to rule-based scoring.")
        er_model = None

    # 5. Run Prediction & Compute Metrics
    print(f"\n[Step 5/5] Running local evaluation across all {num_val_queries} queries...")
    t_start = time.time()

    blocking_hits = 0
    total_evaluable_matches = 0

    predictions_map = {}
    country_f05 = defaultdict(list)
    singleton_results = {'correct': 0, 'false_merges': 0}

    false_merges_examples = []
    missed_matches_examples = []

    for s1_id, true_matches in val_gt.items():
        s1_rec = val_s1_records.get(s1_id)
        if not s1_rec:
            predictions_map[s1_id] = []
            continue

        true_set = set(true_matches)
        total_evaluable_matches += len(true_set)

        # 1. Blocking Stage
        cands = blocker.retrieve_candidates(s1_id, s1_rec['name'], s1_rec['address'], s1_rec['country'])
        cand_set = set(cands)

        # Blocking recall tracking
        blocking_hits += len(true_set.intersection(cand_set))

        # 2. Matching Stage
        s1_meta = {
            'eid': s1_id,
            'norm_name': normalize_business_name(s1_rec['name']),
            'norm_addr': normalize_address(s1_rec['address']),
            'pin': extract_postal_code(s1_rec['address']),
            'nums': extract_street_numbers(s1_rec['address']),
            'country': s1_rec['country']
        }

        selected_matches = []
        if cands and er_model is not None and er_model.model is not None:
            X_batch = []
            cand_id_order = []
            for cid in cands:
                cmeta = blocker.records.get(cid)
                if cmeta:
                    cmeta['eid'] = cid
                    feats = extract_pairwise_features(s1_meta, cmeta)
                    X_batch.append(feats)
                    cand_id_order.append(cid)

            if X_batch:
                probs = er_model.predict_proba(np.array(X_batch, dtype=np.float32))
                max_prob = float(np.max(probs))

                # Singleton gate
                if max_prob >= er_model.singleton_threshold:
                    for idx, p in enumerate(probs):
                        if p >= er_model.match_threshold and p >= (max_prob - config.SECONDARY_MATCH_MARGIN):
                            selected_matches.append(cand_id_order[idx])

        # Deduplicate predictions
        pred_set = set(selected_matches)
        predictions_map[s1_id] = list(selected_matches)

        # Entity F_0.5 score
        entity_score = compute_entity_f05(true_set, pred_set)
        country_f05[s1_rec['country']].append(entity_score)

        # Singleton evaluation
        if not true_set:
            if not pred_set:
                singleton_results['correct'] += 1
            else:
                singleton_results['false_merges'] += 1
                if len(false_merges_examples) < 3:
                    false_merges_examples.append((s1_rec, [loaded_candidates.get(cid) for cid in pred_set]))
        else:
            # Check missed matches
            missed = true_set - pred_set
            if missed and len(missed_matches_examples) < 3:
                missed_matches_examples.append((s1_rec, [loaded_candidates.get(cid) for cid in missed if cid in loaded_candidates]))

    eval_time = time.time() - t_start

    # Macro F_0.5 Overall
    all_scores = [score for scores in country_f05.values() for score in scores]
    macro_f05 = float(np.mean(all_scores))

    # Blocking Recall
    blocking_recall = (blocking_hits / total_evaluable_matches * 100) if total_evaluable_matches > 0 else 0.0

    # Non-singleton Precision & Recall
    non_singleton_tp = 0
    non_singleton_fp = 0
    non_singleton_fn = 0
    for s1_id, true_matches in val_gt.items():
        if true_matches:
            true_s = set(true_matches)
            pred_s = set(predictions_map.get(s1_id, []))
            non_singleton_tp += len(true_s.intersection(pred_s))
            non_singleton_fp += len(pred_s - true_s)
            non_singleton_fn += len(true_s - pred_s)

    micro_prec = (non_singleton_tp / (non_singleton_tp + non_singleton_fp) * 100) if (non_singleton_tp + non_singleton_fp) > 0 else 0.0
    micro_rec = (non_singleton_tp / (non_singleton_tp + non_singleton_fn) * 100) if (non_singleton_tp + non_singleton_fn) > 0 else 0.0

    print("\n" + "=" * 75)
    print("🏆 LOCAL BENCHMARK RESULTS")
    print("=" * 75)
    print(f"  ▶ MACRO F_0.5 SCORE (LEADERBOARD METRIC): {macro_f05:.4f} ({macro_f05 * 100:.2f}%)")
    print("-" * 75)
    print(f"  • Candidate Blocking Recall Ceiling     : {blocking_recall:.2f}%")
    print(f"  • Non-Singleton Precision                : {micro_prec:.2f}%")
    print(f"  • Non-Singleton Recall                   : {micro_rec:.2f}%")
    print(f"  • Singleton Accuracy (Correctly Blank)   : {singleton_results['correct']}/{num_singletons} ({singleton_results['correct']/(num_singletons or 1)*100:.1f}%)")
    print(f"  • Evaluation Throughput                  : {num_val_queries / eval_time:.0f} queries/sec")
    print("-" * 75)
    print("  ▶ COUNTRY-WISE MACRO F_0.5 BREAKDOWN:")
    for c, scs in country_f05.items():
        print(f"    - {c:<8}: {np.mean(scs):.4f} ({np.mean(scs)*100:.2f}%) across {len(scs)} entities")
    print("=" * 75)

    if false_merges_examples:
        print("\n🔍 SAMPLE ERROR ANALYSIS: False Merges (Precision Penalty)")
        for s1_r, cands in false_merges_examples:
            print(f"  [S1 Query]: \"{s1_r['name']}\" | \"{s1_r['address']}\"")
            for c in cands:
                if c:
                    print(f"    -> [Wrongly Merged]: \"{c['name']}\" | \"{c['address']}\"")

    if missed_matches_examples:
        print("\n🔍 SAMPLE ERROR ANALYSIS: Missed Matches (Recall Penalty)")
        for s1_r, cands in missed_matches_examples:
            print(f"  [S1 Query]: \"{s1_r['name']}\" | \"{s1_r['address']}\"")
            for c in cands:
                if c:
                    print(f"    -> [Missed True Match]: \"{c['name']}\" | \"{c['address']}\"")

    print("\n" + "=" * 75)
    return macro_f05


if __name__ == '__main__':
    run_benchmark(num_val_queries=2000, background_candidates=150000)
