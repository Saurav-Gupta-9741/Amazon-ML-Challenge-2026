"""
infer.py — High-Performance, Memory-Bounded Country-Partitioned Inference Pipeline.

Generates:
1. output/matching_results.tsv (scored on leaderboard)
2. output/candidate_pairs.tsv (blocking candidate set)
Runs local validation check via utils/validate_submission.py.
"""

import sys
import os
import time
import gc
import subprocess
from collections import defaultdict
import numpy as np

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from normalizer import (
    normalize_business_name,
    normalize_address,
    extract_postal_code,
    extract_street_numbers
)
from blocking import CandidateBlocker
from features import extract_pairwise_features
from model import EntityResolutionModel


def run_inference(test_s1_path=None, test_s2_path=None, test_s3_path=None, max_test_rows=None):
    if test_s1_path is None:
        test_s1_path = config.TEST_S1
    if test_s2_path is None:
        test_s2_path = config.TEST_S2
    if test_s3_path is None:
        test_s3_path = config.TEST_S3

    print("=" * 70)
    print("INFERENCE PIPELINE — COUNTRY-PARTITIONED HIGH-SPEED ENGINE")
    print("=" * 70)

    # 1. Load Trained Model
    print(f"\n[1/5] Loading trained model from {config.MODEL_PATH}...")
    er_model = EntityResolutionModel()
    if os.path.exists(config.MODEL_PATH):
        er_model.load(config.MODEL_PATH)
        print(f"Model loaded: Match Th = {er_model.match_threshold}, Singleton Th = {er_model.singleton_threshold}")
    else:
        print("WARNING: Model file not found! Falling back to heuristic rule-based scoring.")
        er_model = None

    # 2. Read All S1 Queries & Group by Country (Preserving Exact Original Order)
    print(f"\n[2/5] Reading Source 1 query metadata from {os.path.basename(test_s1_path)}...")
    t0 = time.time()
    all_s1_ids = []
    country_queries = defaultdict(list)

    with open(test_s1_path, encoding='utf-8') as f:
        f.readline()  # skip header
        for i, line in enumerate(f):
            if max_test_rows and i >= max_test_rows:
                break
            parts = line.strip().split('\t')
            s1_id = parts[0]
            name = parts[1] if len(parts) > 1 else ''
            addr = parts[2] if len(parts) > 2 else ''
            country = parts[3] if len(parts) > 3 else 'US'

            all_s1_ids.append(s1_id)
            country_queries[country].append((s1_id, name, addr))

    print(f"  Loaded {len(all_s1_ids)} total S1 queries in {time.time() - t0:.2f}s:")
    for c, q_list in country_queries.items():
        print(f"    - {c}: {len(q_list)} queries")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    temp_match_files = {}
    temp_cand_files = {}

    # 3. Country-by-Country Processing (Zero Memory Leak, Fast RAM-Resident Pools)
    print("\n[3/5] Processing entities Country-by-Country...")
    countries_to_process = ['France', 'US', 'India']
    # Include any unexpected countries if present
    for c in country_queries:
        if c not in countries_to_process:
            countries_to_process.append(c)

    for country in countries_to_process:
        queries = country_queries.get(country, [])
        if not queries:
            continue

        c_match_path = os.path.join(config.OUTPUT_DIR, f"matching_{country}.tsv")
        c_cand_path = os.path.join(config.OUTPUT_DIR, f"candidates_{country}.tsv")
        temp_match_files[country] = c_match_path
        temp_cand_files[country] = c_cand_path

        # Check if already processed (Resume Support)
        if os.path.exists(c_match_path) and os.path.exists(c_cand_path):
            try:
                with open(c_match_path, encoding='utf-8') as f_m, open(c_cand_path, encoding='utf-8') as f_c:
                    m_lines = sum(1 for _ in f_m)
                    c_lines = sum(1 for _ in f_c)
                if m_lines == len(queries) and c_lines == len(queries):
                    print(f"\n--- [{country}] Already completed ({len(queries)} entities). Skipping to next country. ---")
                    continue
            except Exception:
                pass

        print(f"\n--- [{country}] Indexing candidate pool ({len(queries)} S1 queries) ---")
        t_country_start = time.time()
        blocker = CandidateBlocker(max_candidates=config.MAX_CANDIDATES_PER_S1)

        c_cand_count = 0
        for path in [test_s2_path, test_s3_path]:
            if not os.path.exists(path):
                continue
            with open(path, encoding='utf-8') as f:
                f.readline()
                for i, line in enumerate(f):
                    if max_test_rows and i >= max_test_rows * 5:
                        break
                    parts = line.strip().split('\t')
                    c_label = parts[3] if len(parts) > 3 else 'US'
                    if c_label != country:
                        continue
                    eid = parts[0]
                    name = parts[1] if len(parts) > 1 else ''
                    addr = parts[2] if len(parts) > 2 else ''
                    blocker.add_candidate(eid, name, addr, country)
                    c_cand_count += 1

        print(f"  [{country}] Indexed {c_cand_count} candidates in {time.time() - t_country_start:.2f}s.")

        # Stream batched inference for this country
        BATCH_SIZE = 2500
        matched_c = 0
        singletons_c = 0

        with open(c_match_path, 'w', encoding='utf-8') as f_m, \
             open(c_cand_path, 'w', encoding='utf-8') as f_c:

            for start_idx in range(0, len(queries), BATCH_SIZE):
                batch = queries[start_idx:start_idx + BATCH_SIZE]
                X_batch_all = []
                query_meta = []

                for s1_id, name, addr in batch:
                    cands = blocker.retrieve_candidates(s1_id, name, addr, country)
                    if not cands:
                        query_meta.append((s1_id, [], [], 0, 0))
                        continue

                    s1_meta = {
                        'eid': s1_id,
                        'norm_name': normalize_business_name(name),
                        'norm_addr': normalize_address(addr),
                        'pin': extract_postal_code(addr),
                        'nums': extract_street_numbers(addr),
                        'country': country
                    }
                    start = len(X_batch_all)
                    valid_cands = []
                    for cid in cands:
                        cmeta = blocker.records.get(cid)
                        if cmeta:
                            valid_cands.append(cid)
                            feats = extract_pairwise_features(s1_meta, cmeta)
                            X_batch_all.append(feats)
                    end = len(X_batch_all)
                    query_meta.append((s1_id, cands, valid_cands, start, end))

                all_probs = None
                if X_batch_all and er_model is not None and er_model.model is not None:
                    all_probs = er_model.predict_proba(np.array(X_batch_all, dtype=np.float32))

                for s1_id, cands, valid_cands, start, end in query_meta:
                    f_c.write(f"{s1_id}\t{','.join(cands)}\n")

                    selected = []
                    if start < end and all_probs is not None:
                        q_probs = all_probs[start:end]
                        max_prob = float(np.max(q_probs))
                        if max_prob >= er_model.singleton_threshold:
                            for idx, p in enumerate(q_probs):
                                if p >= er_model.match_threshold and p >= (max_prob - config.SECONDARY_MATCH_MARGIN):
                                    selected.append(valid_cands[idx])

                    if selected:
                        seen = set()
                        deduped = []
                        for m in selected:
                            if m not in seen:
                                seen.add(m)
                                deduped.append(m)
                        f_m.write(f"{s1_id}\t{','.join(deduped)}\n")
                        matched_c += 1
                    else:
                        f_m.write(f"{s1_id}\t\n")
                        singletons_c += 1

                processed_so_far = min(start_idx + BATCH_SIZE, len(queries))
                if processed_so_far % 50000 < BATCH_SIZE or processed_so_far == len(queries):
                    elap = time.time() - t_country_start
                    rate = processed_so_far / elap if elap > 0 else 0
                    print(f"  [{country}] Progress: {processed_so_far}/{len(queries)} ({processed_so_far/len(queries)*100:.1f}%) at {rate:.0f} entities/sec...")
                    sys.stdout.flush()

        print(f"  [{country}] Finished in {time.time() - t_country_start:.2f}s. (Matched: {matched_c}, Singletons: {singletons_c})")

        # Explicit cleanup to free all RAM back to the operating system
        del blocker
        gc.collect()

    # 4. Merge All Country Outputs in EXACT Original Source 1 Order
    print("\n[4/5] Merging country predictions into final submission files (exact S1 order)...")
    match_dict = {}
    cand_dict = {}

    for country, m_path in temp_match_files.items():
        c_path = temp_cand_files[country]
        if os.path.exists(m_path):
            with open(m_path, encoding='utf-8') as f:
                for line in f:
                    parts = line.rstrip('\r\n').split('\t')
                    if parts:
                        match_dict[parts[0]] = parts[1] if len(parts) > 1 else ''
        if os.path.exists(c_path):
            with open(c_path, encoding='utf-8') as f:
                for line in f:
                    parts = line.rstrip('\r\n').split('\t')
                    if parts:
                        cand_dict[parts[0]] = parts[1] if len(parts) > 1 else ''

    final_match_path = config.MATCHING_OUTPUT
    final_cand_path = config.CANDIDATE_OUTPUT

    with open(final_match_path, 'w', encoding='utf-8') as f_match, \
         open(final_cand_path, 'w', encoding='utf-8') as f_cand:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for s1_id in all_s1_ids:
            f_match.write(f"{s1_id}\t{match_dict.get(s1_id, '')}\n")
            f_cand.write(f"{s1_id}\t{cand_dict.get(s1_id, '')}\n")

    print(f"  Successfully wrote {len(all_s1_ids)} entities to {final_match_path} and {final_cand_path}")

    # Remove temporary country files
    for p in list(temp_match_files.values()) + list(temp_cand_files.values()):
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    # 5. Run Official Submission Validator
    print("\n[5/5] Running official submission validator (utils/validate_submission.py)...")
    validator_path = os.path.join(config.RESOURCE_DIR, 'utils', 'validate_submission.py')
    if os.path.exists(validator_path):
        cmd = [
            sys.executable,
            validator_path,
            '--matching', final_match_path,
            '--candidate', final_cand_path,
            '--test-dir', config.TEST_DIR
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.returncode == 0:
            print("[PASS] VALIDATOR PASSED (Exit code 0) — Ready for submission!")
        else:
            print("[FAIL] VALIDATOR REPORT:")
            print(res.stdout)
            if res.stderr:
                print(res.stderr)
    else:
        print(f"Validator script not found at {validator_path}")

    return final_match_path, final_cand_path


if __name__ == '__main__':
    run_inference()
