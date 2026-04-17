"""
retail_anomaly.utils.data_loader
=================================
Public function: load_pipeline_data
"""
from __future__ import annotations

import pandas as pd


def load_pipeline_data(
    path: str,
    score_cols: list[str],
    label_col: str,
    city_col: str = "city_code",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and validate a retail CSV/TSV for the pipeline.

    Returns
    -------
    df_full        : all rows
    df_labelled    : rows where label_col is not NaN
    df_unlabelled  : rows where label_col is NaN
    """
    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    df  = pd.read_csv(path, sep=sep)

    # Validate columns
    missing = [c for c in score_cols + [label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"[data] Missing columns: {missing}")

    # Coerce score columns
    df[score_cols] = (
        df[score_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace({float("inf"): float("nan"), float("-inf"): float("nan")})
        .fillna(0)
    )

    labelled_mask   = df[label_col].notna()
    df_labelled     = df[labelled_mask].copy()
    df_unlabelled   = df[~labelled_mask].copy()

    # Validate label counts
    n_lab = labelled_mask.sum()
    if n_lab < 20:
        raise ValueError(
            f"[data] Only {n_lab} labelled rows. Need at least 20."
        )
    n_pos = int((df_labelled[label_col] == 1).sum())
    if n_pos < 10:
        raise ValueError(
            f"[data] Only {n_pos} positive labels. Need at least 10."
        )

    n_neg     = int(n_lab) - n_pos
    pos_rate  = n_pos / n_lab

    print(f"[data] {len(df)} rows total")
    print(f"[data] Labelled: {n_lab}  (pos={n_pos}, neg={n_neg}, "
          f"pos_rate={pos_rate:.1%})")
    print(f"[data] Unlabelled: {len(df_unlabelled)}")

    if city_col in df.columns:
        city_dist = df[city_col].value_counts().to_dict()
        print(f"[data] City distribution: {city_dist}")

    return df, df_labelled, df_unlabelled
