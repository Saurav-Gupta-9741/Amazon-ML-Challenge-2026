"""
run_pipeline.py — Master orchestrator for Business Entity Resolution.

Usage:
    python run_pipeline.py --mode quick_test  # Quick sanity check on 500 rows
    python run_pipeline.py --mode train       # Train model & optimize thresholds
    python run_pipeline.py --mode infer       # Run inference on test dataset
    python run_pipeline.py --mode full        # Train + Infer + Validate
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
from train import run_training
from infer import run_inference


def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution Master Pipeline")
    parser.add_argument('--mode', type=str, default='quick_test',
                        choices=['quick_test', 'train', 'infer', 'full'],
                        help="Execution mode")
    parser.add_argument('--max-test-rows', type=int, default=None,
                        help="Optional limit on test rows for quick debugging")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 70)
    print(f"STARTING BUSINESS ENTITY RESOLUTION PIPELINE [Mode: {args.mode}]")
    print("=" * 70)

    if args.mode == 'quick_test':
        print("\n--- RUNNING QUICK SANITY TEST (500 TEST ROWS) ---")
        run_inference(max_test_rows=500)

    elif args.mode == 'train':
        run_training()

    elif args.mode == 'infer':
        run_inference(max_test_rows=args.max_test_rows)

    elif args.mode == 'full':
        print("\n[Step 1/2] Training Model & Calibrating Thresholds...")
        run_training()
        print("\n[Step 2/2] Running Full Test Inference...")
        run_inference(max_test_rows=args.max_test_rows)

    elapsed = time.time() - t_start
    print(f"\nExecution finished in {elapsed:.1f}s ({elapsed/60:.2f} mins).")


if __name__ == '__main__':
    main()
