"""
retail_anomaly.scorer.lfm
=========================
Improved Latent Factor Model for retail store scan-quality scoring.

Key improvements over the original Ipsos implementation
--------------------------------------------------------
* KDE-based indicator scoring  – no distribution assumption (replaces sigmoid)
* Parallel analysis            – automatic factor-count selection
* Varimax rotation             – interpretable factor loadings
* Regression-method factor scores with variance-proportion weighting
* Sign alignment               – ensures high score = high quality
* Mahalanobis + IsolationForest – multivariate outlier detection
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import svd as scipy_svd
from scipy.spatial.distance import mahalanobis
from scipy.stats import gaussian_kde
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from retail_anomaly.utils.config import load_config
from retail_anomaly.utils.validation import validate_dataframe, clean_numeric


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _varimax(Phi: np.ndarray, gamma: float = 1.0,
             max_iter: int = 1000, tol: float = 1e-8) -> np.ndarray:
    """Varimax rotation of a loading matrix *Phi* (p × k)."""
    p, k = Phi.shape
    if k == 1:
        return Phi
    R = np.eye(k)
    d = 0.0
    for _ in range(max_iter):
        d_old = d
        L = Phi @ R
        u, s, vh = scipy_svd(
            Phi.T @ (L ** 3 - (gamma / p) * L @ np.diag((L ** 2).sum(axis=0)))
        )
        R = u @ vh
        d = s.sum()
        if abs(d - d_old) < tol:
            break
    return Phi @ R


def _parallel_analysis(
    X: np.ndarray, n_iter: int = 200, percentile: float = 95
) -> tuple[int, np.ndarray, np.ndarray]:
    """Return (n_factors, real_eigenvalues, random_threshold)."""
    n, p = X.shape
    X_std = StandardScaler().fit_transform(X)
    real_eigs = np.linalg.eigvalsh(np.corrcoef(X_std.T))[::-1]
    rand_eigs = np.array([
        np.linalg.eigvalsh(np.corrcoef(np.random.normal(0, 1, (n, p)).T))[::-1]
        for _ in range(n_iter)
    ])
    threshold = np.percentile(rand_eigs, percentile, axis=0)
    n_chosen = max(1, int((real_eigs > threshold).sum()))
    return n_chosen, real_eigs, threshold


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class ImprovedLFM:
    """
    Unsupervised store quality scorer.

    Parameters
    ----------
    config : dict, optional
        Merged config dict.  Loads configs/default.yaml if not provided.

    Attributes (post-fit)
    ---------------------
    loadings_         : np.ndarray  (p × k)  varimax-rotated factor loadings
    factor_weights_   : np.ndarray  (k,)     variance-proportion weights
    n_factors_chosen_ : int
    kde_functions_    : dict[str, gaussian_kde]
    scaler_           : StandardScaler  (fitted on clean subset)
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or load_config()
        sc  = cfg["scorer"]
        d   = cfg["data"]

        self.score_cols        = d["score_cols"]
        self.id_col            = d["id_col"]
        self.city_col          = d["city_col"]
        self.label_col         = d["label_col"]

        self.iqr_mult          = sc["iqr_multiplier"]
        self.contamination     = sc["iso_contamination"]
        self.n_factors_cfg     = sc["n_factors"]       # "auto" or int
        self.pa_n_iter         = sc["pa_n_iter"]
        self.pa_percentile     = sc["pa_percentile"]
        self.kde_bw            = sc["kde_bw_method"]

        # fitted state
        self.loadings_:         np.ndarray | None = None
        self.factor_weights_:   np.ndarray | None = None
        self.n_factors_chosen_: int               = 2
        self.kde_functions_:    dict              = {}
        self.iqr_bounds_:       dict              = {}
        self.scaler_:           StandardScaler | None = None
        self._pa_real_eigs:     np.ndarray | None = None
        self._pa_rand_eigs:     np.ndarray | None = None

    # ── private helpers ────────────────────────────────────────────────────

    def _iqr_mask(self, df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        for col in self.score_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lb, ub = q1 - self.iqr_mult * iqr, q3 + self.iqr_mult * iqr
            self.iqr_bounds_[col] = (lb, ub)
            mask &= df[col].between(lb, ub)
        return mask

    def _kde_scores(self, df: pd.DataFrame, mask_clean: pd.Series) -> pd.DataFrame:
        scores = pd.DataFrame(index=df.index)
        for col in self.score_cols:
            clean_vals = df.loc[mask_clean, col].dropna()
            kde = gaussian_kde(clean_vals, bw_method=self.kde_bw)
            self.kde_functions_[col] = kde
            raw = kde.evaluate(df[col].fillna(df[col].median()).values)
            mn, mx = raw.min(), raw.max()
            scores[f"{col}_kde"] = (raw - mn) / (mx - mn + 1e-12) * 100
        return scores

    def _fit_factors(self, X_kde: np.ndarray) -> np.ndarray:
        """Fit varimax factor model; return correlation matrix."""
        scaler = StandardScaler()
        X_std  = scaler.fit_transform(X_kde)
        corr   = np.corrcoef(X_std.T)
        eigvals, eigvecs = np.linalg.eigh(corr)
        idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
        k = self.n_factors_chosen_
        A = eigvecs[:, :k] * np.sqrt(np.maximum(eigvals[:k], 0.0))
        self.loadings_ = _varimax(A)
        self.scaler_   = scaler
        var_per_factor = (self.loadings_ ** 2).sum(axis=0)
        self.factor_weights_ = var_per_factor / var_per_factor.sum()
        return corr

    def _factor_scores(self, X_kde: np.ndarray, corr: np.ndarray) -> np.ndarray:
        X_std      = self.scaler_.transform(X_kde)
        score_coef = np.linalg.pinv(corr) @ self.loadings_
        return X_std @ score_coef

    # ── public API ─────────────────────────────────────────────────────────

    def fit_transform(
        self, df: pd.DataFrame, verbose: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Fit the model and return scored results.

        Returns
        -------
        result_df  : scored DataFrame (includes LFM改进得分, indicator scores)
        score_df   : raw KDE indicator scores only
        mask_iqr   : boolean Series of IQR-clean rows
        """
        validate_dataframe(df, self.score_cols + [self.id_col, self.city_col])
        df_work = clean_numeric(df, self.score_cols)

        # Step 1 – clean mask
        mask_iqr = self._iqr_mask(df_work)
        if verbose:
            print(f"[LFM] IQR clean: {mask_iqr.sum()} / {len(df_work)}")

        # IsolationForest (informational flag, not used in scoring)
        X_tmp   = StandardScaler().fit_transform(df_work[self.score_cols])
        iso     = IsolationForest(contamination=self.contamination, random_state=42)
        iso_flag = iso.fit_predict(X_tmp) == -1
        if verbose:
            print(f"[LFM] IsolationForest anomalies: {iso_flag.sum()}")

        # Step 2 – KDE indicator scores
        score_df = self._kde_scores(df_work, mask_iqr)
        if verbose:
            print(f"[LFM] KDE scores computed ({len(self.score_cols)} indicators)")

        # Step 3 – parallel analysis
        if self.n_factors_cfg == "auto":
            n_chosen, real_eigs, rand_eigs = _parallel_analysis(
                score_df.values, self.pa_n_iter, self.pa_percentile
            )
            self.n_factors_chosen_  = min(n_chosen, 6)
            self._pa_real_eigs      = real_eigs
            self._pa_rand_eigs      = rand_eigs
        else:
            self.n_factors_chosen_ = int(self.n_factors_cfg)
        if verbose:
            print(f"[LFM] n_factors = {self.n_factors_chosen_}")

        # Step 4 – factor analysis + varimax
        corr = self._fit_factors(score_df.values)

        # Step 5 – factor scores with sign alignment
        fscores   = self._factor_scores(score_df.values, corr)
        mean_score = score_df.values.mean(axis=1)
        for fi in range(fscores.shape[1]):
            if np.corrcoef(fscores[:, fi], mean_score)[0, 1] < 0:
                fscores[:, fi] = -fscores[:, fi]

        composite = (fscores * self.factor_weights_).sum(axis=1)
        composite = (composite - composite.min()) / (composite.max() - composite.min() + 1e-12) * 100

        # Assemble result
        keep_cols = [self.id_col, self.city_col]
        if self.label_col in df.columns:
            keep_cols.append(self.label_col)
        result_df = df[keep_cols].copy()
        result_df["LFM改进得分"] = composite
        result_df["iso_anomaly"] = iso_flag
        for col in self.score_cols:
            result_df[f"{col}_score"] = score_df[f"{col}_kde"].values

        if verbose:
            print(f"[LFM] Done. Score range: "
                  f"{composite.min():.1f}–{composite.max():.1f}")

        return result_df, score_df, mask_iqr

    def score_new(self, df: pd.DataFrame) -> np.ndarray:
        """
        Score new rows using a fitted model (no re-fitting).
        Returns composite scores in [0, 100].
        """
        if self.loadings_ is None:
            raise RuntimeError("Call fit_transform() before score_new().")
        df_work  = clean_numeric(df, self.score_cols)
        # KDE score each column
        kde_cols = []
        for col in self.score_cols:
            kde  = self.kde_functions_[col]
            raw  = kde.evaluate(df_work[col].fillna(df_work[col].median()).values)
            mn   = raw.min(); mx = raw.max()
            kde_cols.append((raw - mn) / (mx - mn + 1e-12) * 100)
        X_kde     = np.column_stack(kde_cols)
        corr      = np.corrcoef(self.scaler_.transform(X_kde).T)
        fscores   = self._factor_scores(X_kde, corr)
        mean_score = X_kde.mean(axis=1)
        for fi in range(fscores.shape[1]):
            if np.corrcoef(fscores[:, fi], mean_score)[0, 1] < 0:
                fscores[:, fi] = -fscores[:, fi]
        composite = (fscores * self.factor_weights_).sum(axis=1)
        return (composite - composite.min()) / (composite.max() - composite.min() + 1e-12) * 100
