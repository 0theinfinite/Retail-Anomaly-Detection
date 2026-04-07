"""
retail_anomaly.cvae.validation
================================
Statistical validation of synthetic vs real data (KS test per column).
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


def ks_validate(
    real_df:      pd.DataFrame,
    synthetic_df: pd.DataFrame,
    cols:         list[str],
    alpha:        float = 0.05,
    verbose:      bool  = True,
) -> pd.DataFrame:
    """
    Per-column two-sample KS test.

    Returns a DataFrame with columns: indicator, ks_stat, p_value, pass.
    'pass' = True means p > alpha (cannot reject H₀ that distributions match).
    """
    rows = []
    for col in cols:
        real_col = real_df[col].dropna().values
        syn_col  = synthetic_df[col].dropna().values
        stat, p  = ks_2samp(real_col, syn_col)
        rows.append({"indicator": col, "ks_stat": round(stat, 4),
                     "p_value": round(p, 4), "pass": p > alpha})
    result = pd.DataFrame(rows)
    if verbose:
        n_pass = result["pass"].sum()
        print(f"[KS] {n_pass}/{len(result)} columns pass at α={alpha}")
        fails = result[~result["pass"]]["indicator"].tolist()
        if fails:
            print(f"[KS] Failed: {fails}")
    return result
