"""
config.py — Central configuration for Business Entity Resolution pipeline.
"""

import os

# Root directories
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)  # business_entity_resolution
CODE_DIR = os.path.dirname(PROJECT_DIR)  # code
RESOURCE_DIR = os.path.dirname(CODE_DIR)  # student_resource

# Dataset paths
DATASET_DIR = os.path.join(RESOURCE_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')

TRAIN_S1 = os.path.join(TRAIN_DIR, 'train_source1.tsv')
TRAIN_S2 = os.path.join(TRAIN_DIR, 'train_source2.tsv')
TRAIN_S3 = os.path.join(TRAIN_DIR, 'train_source3.tsv')
TRAIN_GT = os.path.join(TRAIN_DIR, 'train_ground_truth.tsv')

TEST_S1 = os.path.join(TEST_DIR, 'test_source1.tsv')
TEST_S2 = os.path.join(TEST_DIR, 'test_source2.tsv')
TEST_S3 = os.path.join(TEST_DIR, 'test_source3.tsv')

# Output paths
OUTPUT_DIR = os.path.join(RESOURCE_DIR, 'output')
MATCHING_OUTPUT = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
CANDIDATE_OUTPUT = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')

# Model & Artifacts
MODEL_DIR = os.path.join(RESOURCE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'lgb_matcher.joblib')

# Pipeline Hyperparameters
MAX_CANDIDATES_PER_S1 = 25      # Cap on candidate pairs per S1
TOP_K_BLOCKING = 20             # Top nearest candidates from inverted index / TFIDF
MIN_NAME_SIMILARITY = 0.50      # Minimum RapidFuzz token sort ratio for blocking
SINGLETON_THRESHOLD = 0.70      # If max probability < this, predict empty list (singleton)
MATCH_THRESHOLD = 0.65          # Probability threshold to accept a candidate match
SECONDARY_MATCH_MARGIN = 0.15   # Max allowed difference from top prob for multi-matches

# Reproducibility
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
