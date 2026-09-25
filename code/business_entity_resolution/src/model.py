"""
model.py — LightGBM Classifier, Custom Macro F_0.5 Metric, and Threshold Optimizer.
"""

import numpy as np
import lightgbm as lgb
import joblib


def compute_entity_f05(true_set: set, pred_set: set) -> float:
    """
    Computes exact F_0.5 score for a single Source 1 entity:
    - Singletons: If true is empty, returns 1.0 if pred is empty, else 0.0.
    - Matches: Standard F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    if not true_set:
        return 1.0 if not pred_set else 0.0

    if not pred_set:
        return 0.0

    tp = len(true_set.intersection(pred_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0

    return (1.25 * precision * recall) / denom


def evaluate_macro_f05(ground_truth_map: dict, predictions_map: dict) -> float:
    """
    Computes macro-averaged F_0.5 across all Source 1 entities in ground_truth_map.
    """
    scores = []
    for s1_id, true_matches in ground_truth_map.items():
        true_set = set(true_matches)
        pred_set = set(predictions_map.get(s1_id, []))
        scores.append(compute_entity_f05(true_set, pred_set))
    return float(np.mean(scores)) if scores else 0.0


class EntityResolutionModel:
    """LightGBM Pairwise Matcher with F_0.5 Threshold Optimization."""

    def __init__(self, match_threshold=0.65, singleton_threshold=0.70):
        self.match_threshold = match_threshold
        self.singleton_threshold = singleton_threshold
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Trains LightGBM binary classifier on feature matrix."""
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.08,
            'num_leaves': 63,
            'max_depth': 8,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 300,
            'verbose': -1,
            'random_state': 42
        }
        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability of being a true match."""
        if self.model is None:
            raise ValueError("Model is not fitted yet!")
        return self.model.predict_proba(X)[:, 1]

    def optimize_thresholds(self, val_query_candidates: dict, val_gt: dict):
        """
        Grid search on local validation split to find optimal thresholds
        that maximize Macro F_0.5 score.
        val_query_candidates: {s1_id: [(cand_id, prob), ...]}
        """
        best_score = -1.0
        best_match_th = self.match_threshold
        best_single_th = self.singleton_threshold

        # Search over reasonable precision-favoring thresholds
        for single_th in [0.60, 0.65, 0.70, 0.75, 0.80]:
            for match_th in [0.55, 0.60, 0.65, 0.70, 0.75]:
                preds_map = {}
                for s1_id, cand_probs in val_query_candidates.items():
                    if not cand_probs:
                        preds_map[s1_id] = []
                        continue

                    # Singleton check
                    max_prob = max(prob for _, prob in cand_probs)
                    if max_prob < single_th:
                        preds_map[s1_id] = []
                        continue

                    # Select candidates above match threshold
                    selected = [cid for cid, prob in cand_probs if prob >= match_th]
                    preds_map[s1_id] = selected

                score = evaluate_macro_f05(val_gt, preds_map)
                if score > best_score:
                    best_score = score
                    best_single_th = single_th
                    best_match_th = match_th

        self.singleton_threshold = best_single_th
        self.match_threshold = best_match_th
        print(f"Optimal Thresholds Found: Singleton Th = {self.singleton_threshold}, Match Th = {self.match_threshold}")
        print(f"Validation Macro F_0.5 Score: {best_score:.4f}")
        return best_score

    def save(self, path: str):
        """Saves model and calibrated thresholds."""
        artifacts = {
            'model': self.model,
            'match_threshold': self.match_threshold,
            'singleton_threshold': self.singleton_threshold
        }
        joblib.dump(artifacts, path)

    def load(self, path: str):
        """Loads model and calibrated thresholds."""
        artifacts = joblib.load(path)
        self.model = artifacts['model']
        self.match_threshold = artifacts['match_threshold']
        self.singleton_threshold = artifacts['singleton_threshold']
