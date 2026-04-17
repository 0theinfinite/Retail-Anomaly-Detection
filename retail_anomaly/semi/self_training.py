"""
retail_anomaly.semi.self_training
==================================
SelfTrainer: iterative pseudo-labelling using a logistic regression base model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class SelfTrainer:
    """
    Semi-supervised self-training via high-confidence pseudo-labelling.

    At each round:
      - Fit LogisticRegression on current labelled pool.
      - Predict on remaining unlabelled rows.
      - Rows with proba > threshold_pos → pseudo-label 1.
      - Rows with proba < threshold_neg → pseudo-label 0.
      - Add pseudo-labelled rows to training pool; repeat.

    Parameters
    ----------
    threshold_pos   : confidence threshold to assign pseudo-label 1
    threshold_neg   : confidence threshold to assign pseudo-label 0
    max_iter        : maximum self-training rounds
    min_new_samples : stop early if fewer samples added than this
    """

    def __init__(
        self,
        threshold_pos: float = 0.85,
        threshold_neg: float = 0.15,
        max_iter: int = 5,
        min_new_samples: int = 1,
    ) -> None:
        self.threshold_pos   = threshold_pos
        self.threshold_neg   = threshold_neg
        self.max_iter        = max_iter
        self.min_new_samples = min_new_samples

        self.X_train_final_:      np.ndarray | None = None
        self.y_train_final_:      np.ndarray | None = None
        self.iteration_log_:      list[dict]         = []
        self.n_rounds_:           int                = 0
        self.clf_:                LogisticRegression | None = None
        self.scaler_:             StandardScaler | None = None
        self.pseudo_source_indices_: np.ndarray | None = None  # into X_unlabelled

    def fit(
        self,
        X_labelled:   np.ndarray,
        y_labelled:   np.ndarray,
        X_unlabelled: np.ndarray,
    ) -> "SelfTrainer":
        """
        Run self-training.

        Parameters
        ----------
        X_labelled   : (n_lab, n_features)  — original labelled features
        y_labelled   : (n_lab,)             — original labels (0/1)
        X_unlabelled : (n_unlab, n_features) — unlabelled features
        """
        X_train     = X_labelled.copy()
        y_train     = y_labelled.copy()
        X_remaining = X_unlabelled.copy()
        remaining_idx = np.arange(len(X_unlabelled))   # indices into X_unlabelled
        pseudo_idx_list: list[int] = []
        self.iteration_log_ = []

        for round_i in range(1, self.max_iter + 1):
            scaler = StandardScaler()
            X_tr_scaled  = scaler.fit_transform(X_train)
            X_rem_scaled = scaler.transform(X_remaining)

            clf = LogisticRegression(
                C=1.0, class_weight="balanced",
                max_iter=1000, random_state=42,
            )
            clf.fit(X_tr_scaled, y_train)
            proba = clf.predict_proba(X_rem_scaled)[:, 1]

            high_pos  = proba > self.threshold_pos
            high_neg  = proba < self.threshold_neg
            new_mask  = high_pos | high_neg
            n_new     = int(new_mask.sum())

            if n_new < self.min_new_samples:
                break

            pseudo_y = (proba[new_mask] > 0.5).astype(int)
            n_pos_added = int(pseudo_y.sum())
            n_neg_added = n_new - n_pos_added

            # Track which original X_unlabelled indices were consumed
            pseudo_idx_list.extend(remaining_idx[new_mask].tolist())

            X_train     = np.vstack([X_train, X_remaining[new_mask]])
            y_train     = np.hstack([y_train, pseudo_y])
            X_remaining = X_remaining[~new_mask]
            remaining_idx = remaining_idx[~new_mask]

            log_entry = {
                "round":            round_i,
                "n_added":          n_new,
                "n_pos_added":      n_pos_added,
                "n_neg_added":      n_neg_added,
                "total_train_size": len(y_train),
            }
            self.iteration_log_.append(log_entry)
            print(f"[SelfTraining] Round {round_i}: added {n_new} samples "
                  f"(pos={n_pos_added}, neg={n_neg_added}), "
                  f"total train={len(y_train)}")

            self.n_rounds_ = round_i

        # Final model fit on full expanded set
        self.scaler_ = StandardScaler()
        X_final_scaled = self.scaler_.fit_transform(X_train)
        self.clf_ = LogisticRegression(
            C=1.0, class_weight="balanced",
            max_iter=1000, random_state=42,
        )
        self.clf_.fit(X_final_scaled, y_train)

        self.X_train_final_        = X_train
        self.y_train_final_        = y_train
        self.pseudo_source_indices_ = np.array(pseudo_idx_list, dtype=int)

        if self.n_rounds_ == 0:
            print("[SelfTraining] No high-confidence samples found; "
                  "returning original labels only.")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(flagged) for each row in X."""
        X_scaled = self.scaler_.transform(X)
        return self.clf_.predict_proba(X_scaled)[:, 1]

    def iteration_log_df(self) -> pd.DataFrame:
        """Return the per-round log as a DataFrame."""
        return pd.DataFrame(self.iteration_log_)
