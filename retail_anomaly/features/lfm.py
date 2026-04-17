"""
retail_anomaly.features.lfm
============================
LFMFeatureExtractor: factor analysis for dimensionality reduction.

Simplified from scorer/lfm.py — no KDE, no IsolationForest, no parallel
analysis. Factor count is fixed by config.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import svd as scipy_svd
from sklearn.preprocessing import StandardScaler


def _varimax(Phi: np.ndarray, gamma: float = 1.0,
             max_iter: int = 1000, tol: float = 1e-8) -> np.ndarray:
    """Varimax rotation of loading matrix Phi (p × k)."""
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


class LFMFeatureExtractor:
    """
    Fit a varimax-rotated factor model on a score matrix and extract
    orthogonal factor scores.

    Parameters
    ----------
    n_factors      : number of factors to retain (default 3)
    rotation       : 'varimax' (only option currently)
    sign_alignment : flip factor sign so it correlates positively with
                     the row mean of the standardised input
    """

    def __init__(
        self,
        n_factors: int = 3,
        rotation: str = "varimax",
        sign_alignment: bool = True,
    ) -> None:
        self.n_factors     = n_factors
        self.rotation      = rotation
        self.sign_alignment = sign_alignment

        self.loadings_:   np.ndarray | None = None
        self.score_coef_: np.ndarray | None = None
        self.scaler_:     StandardScaler | None = None
        self.corr_:       np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "LFMFeatureExtractor":
        """
        Fit the factor model on X (n_samples × n_indicators).
        """
        n, p = X.shape
        print(f"[LFM] Fitting on {n} rows, n_factors={self.n_factors}")

        scaler  = StandardScaler()
        X_std   = scaler.fit_transform(X)
        R       = np.corrcoef(X_std.T)

        eigvals, eigvecs = np.linalg.eigh(R)
        idx     = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        k = self.n_factors
        A = eigvecs[:, :k] * np.sqrt(np.maximum(eigvals[:k], 0.0))
        loadings = _varimax(A) if self.rotation == "varimax" else A

        if self.sign_alignment:
            mean_std = X_std.mean(axis=1)
            for fi in range(k):
                proj = X_std @ loadings[:, fi]
                if np.corrcoef(proj, mean_std)[0, 1] < 0:
                    loadings[:, fi] = -loadings[:, fi]

        self.scaler_     = scaler
        self.corr_       = R
        self.loadings_   = loadings
        self.score_coef_ = np.linalg.pinv(R) @ loadings

        # Diagnostics
        var_per = (loadings ** 2).sum(axis=0)
        prop    = var_per / var_per.sum()
        factor_names = [f"F{i + 1}" for i in range(k)]
        var_str = "  ".join(f"{n}={v:.1%}" for n, v in zip(factor_names, prop))
        total   = var_per.sum() / p
        print(f"[LFM] Variance explained: {var_str}  Total={total:.1%}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted model to X → factor scores (n, n_factors)."""
        X_std = self.scaler_.transform(X)
        return X_std @ self.score_coef_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def loadings_df(self, indicator_names: list[str]) -> pd.DataFrame:
        """Return varimax loadings as a tidy DataFrame."""
        factor_names = [f"F{i + 1}" for i in range(self.n_factors)]
        df = pd.DataFrame(self.loadings_, index=indicator_names,
                          columns=factor_names)
        # Print top indicators per factor
        tops = " | ".join(
            f"{fn}: {','.join(df[fn].abs().nlargest(3).index.tolist())}"
            for fn in factor_names
        )
        print(f"[LFM] Top indicators: {tops}")
        return df
