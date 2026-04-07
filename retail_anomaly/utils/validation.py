"""Input validation helpers shared across modules."""
from __future__ import annotations

import numpy as np
import pandas as pd


def validate_dataframe(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")


def clean_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Replace inf/nan with 0 for modelling columns (in-place copy)."""
    out = df.copy()
    out[cols] = (
        out[cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype(float)
    )
    return out
