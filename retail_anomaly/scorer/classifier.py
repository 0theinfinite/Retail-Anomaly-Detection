"""
retail_anomaly.scorer.classifier
=================================
Phase 1 supervised layer: LightGBM binary classifier trained on LFM factor
scores, with SHAP explainability.

Usage
-----
>>> clf = AnomalyClassifier(cfg)
>>> clf.fit(result_df, labels)
>>> proba = clf.predict_proba(result_df)
>>> clf.shap_summary_plot()
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _LGB_OK = True
except ImportError:
    _LGB_OK = False
    warnings.warn("lightgbm not installed; AnomalyClassifier unavailable.")

try:
    import shap as _shap
    _SHAP_OK = True
except ImportError:
    _SHAP_OK = False

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

from retail_anomaly.utils.config import load_config


class AnomalyClassifier:
    """
    LightGBM binary classifier on LFM factor scores + raw KDE indicator scores.

    Features used (built automatically from result_df)
    ---------------------------------------------------
    * LFM composite score
    * Per-indicator KDE scores  (*_score columns)

    Parameters
    ----------
    config : dict, optional
    """

    _FEATURE_SUFFIX = "_score"

    def __init__(self, config: dict | None = None) -> None:
        if not _LGB_OK:
            raise ImportError("Install lightgbm: pip install lightgbm")
        cfg = config or load_config()
        cc  = cfg["classifier"]
        self._cfg      = cc
        self.model_    = None
        self.feature_names_: list[str] = []
        self.cv_auc_:   float | None   = None
        self._explainer = None
        self._shap_vals = None

    def _build_features(self, result_df: pd.DataFrame) -> pd.DataFrame:
        feat_cols = (
            ["LFM改进得分"]
            + [c for c in result_df.columns if c.endswith(self._FEATURE_SUFFIX)]
        )
        return result_df[feat_cols].copy()

    def fit(
        self,
        result_df: pd.DataFrame,
        labels: pd.Series | np.ndarray,
        verbose: bool = True,
    ) -> "AnomalyClassifier":
        """
        Train with StratifiedKFold CV.  Stores best estimator.

        Parameters
        ----------
        result_df : output of ImprovedLFM.fit_transform()[0]
        labels    : binary series (1 = anomalous)
        """
        X = self._build_features(result_df)
        y = np.asarray(labels)
        self.feature_names_ = X.columns.tolist()

        cc = self._cfg
        base_clf = lgb.LGBMClassifier(
            n_estimators=cc["n_estimators"],
            learning_rate=cc["learning_rate"],
            max_depth=cc["max_depth"],
            random_state=cc["random_state"],
            verbose=-1,
        )

        # CV AUC
        cv = StratifiedKFold(n_splits=cc["cv_folds"], shuffle=True,
                             random_state=cc["random_state"])
        aucs = cross_val_score(base_clf, X, y, cv=cv,
                               scoring="roc_auc", n_jobs=-1)
        self.cv_auc_ = float(aucs.mean())
        if verbose:
            print(f"[Classifier] CV AUC: {self.cv_auc_:.3f} ± {aucs.std():.3f}")

        # Refit on full data
        self.model_ = lgb.LGBMClassifier(
            n_estimators=cc["n_estimators"],
            learning_rate=cc["learning_rate"],
            max_depth=cc["max_depth"],
            random_state=cc["random_state"],
            verbose=-1,
        )
        self.model_.fit(X, y)

        # SHAP explainer (TreeExplainer is fast for GBDT)
        if _SHAP_OK:
            self._explainer = _shap.TreeExplainer(self.model_)
            self._shap_vals = self._explainer.shap_values(X)
            if isinstance(self._shap_vals, list):
                self._shap_vals = self._shap_vals[1]  # class=1

        return self

    def predict_proba(self, result_df: pd.DataFrame) -> np.ndarray:
        """Return P(anomaly) for each row."""
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        X = self._build_features(result_df)
        return self.model_.predict_proba(X)[:, 1]

    def shap_summary_plot(self, max_display: int = 12) -> None:
        """Print SHAP beeswarm summary (requires shap + matplotlib)."""
        if not _SHAP_OK:
            raise ImportError("Install shap: pip install shap")
        if self._shap_vals is None:
            raise RuntimeError("Call fit() first.")
        import matplotlib.pyplot as plt
        _shap.summary_plot(
            self._shap_vals,
            feature_names=self.feature_names_,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.show()

    def feature_importance_df(self) -> pd.DataFrame:
        """Return a tidy DataFrame of mean |SHAP| values."""
        if self._shap_vals is None:
            raise RuntimeError("Call fit() first.")
        mean_abs = np.abs(self._shap_vals).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_names_, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
