"""
retail_anomaly.scorer.final_model
===================================
FinalClassifier: thin sklearn LogisticRegression wrapper with
StandardScaler, serialisation, and a clean predict interface.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class FinalClassifier:
    """
    Standardised logistic regression classifier.

    Parameters
    ----------
    C             : inverse regularisation strength
    class_weight  : passed directly to LogisticRegression
    """

    def __init__(self, C: float = 1.0, class_weight: str = "balanced") -> None:
        self.C            = C
        self.class_weight = class_weight
        self.scaler_: StandardScaler | None = None
        self.clf_:    LogisticRegression | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FinalClassifier":
        """Fit scaler + logistic regression on X, y."""
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos

        self.scaler_ = StandardScaler()
        X_scaled     = self.scaler_.fit_transform(X)

        self.clf_ = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=1000,
            random_state=42,
        )
        self.clf_.fit(X_scaled, y)

        print(f"[FinalModel] Fitted on {len(y)} rows  (pos={n_pos}, neg={n_neg})")
        return self

    def predict_proba_all(self, X: np.ndarray) -> np.ndarray:
        """Return P(flagged=1) for each row in X."""
        X_scaled = self.scaler_.transform(X)
        return self.clf_.predict_proba(X_scaled)[:, 1]

    def save(self, path: str | Path) -> None:
        """Pickle the fitted classifier to *path*."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "FinalClassifier":
        """Load a pickled FinalClassifier from *path*."""
        with open(path, "rb") as f:
            return pickle.load(f)
