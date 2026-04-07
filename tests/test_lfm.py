"""Tests for retail_anomaly.scorer.lfm"""
import numpy as np
import pandas as pd
import pytest

from retail_anomaly.scorer.lfm import ImprovedLFM, _varimax, _parallel_analysis
from retail_anomaly.utils.config import load_config


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def sample_df(cfg):
    """200-row synthetic store DataFrame."""
    np.random.seed(0)
    n = 200
    cols = cfg["data"]["score_cols"]
    data = {c: np.abs(np.random.normal(1, 0.3, n)) for c in cols}
    data[cfg["data"]["id_col"]]   = [f"cust{i:06d}" for i in range(n)]
    data[cfg["data"]["city_col"]] = np.random.choice(["A", "B", "C"], n)
    data[cfg["data"]["label_col"]] = np.random.randint(0, 2, n)
    return pd.DataFrame(data)


# ── unit tests ────────────────────────────────────────────────────────────────

def test_varimax_shape():
    Phi = np.random.randn(11, 3)
    rotated = _varimax(Phi)
    assert rotated.shape == (11, 3)


def test_parallel_analysis_returns_positive(sample_df, cfg):
    cols = cfg["data"]["score_cols"]
    X = sample_df[cols].values
    n, real_eigs, rand_eigs = _parallel_analysis(X, n_iter=20)
    assert n >= 1
    assert len(real_eigs) == X.shape[1]


def test_fit_transform_output_shape(sample_df, cfg):
    model = ImprovedLFM(cfg)
    result_df, score_df, mask = model.fit_transform(sample_df, verbose=False)

    assert len(result_df) == len(sample_df)
    assert "LFM改进得分" in result_df.columns
    assert score_df.shape[0] == len(sample_df)
    assert mask.dtype == bool


def test_scores_in_range(sample_df, cfg):
    model = ImprovedLFM(cfg)
    result_df, _, _ = model.fit_transform(sample_df, verbose=False)
    scores = result_df["LFM改进得分"]
    assert scores.min() >= 0.0
    assert scores.max() <= 100.0
    assert not scores.isna().any()


def test_score_new_consistent(sample_df, cfg):
    """score_new on same data should correlate strongly with fit_transform."""
    model = ImprovedLFM(cfg)
    result_df, _, _ = model.fit_transform(sample_df, verbose=False)
    new_scores = model.score_new(sample_df)
    corr = np.corrcoef(result_df["LFM改进得分"].values, new_scores)[0, 1]
    assert corr > 0.95


def test_missing_column_raises(sample_df, cfg):
    model = ImprovedLFM(cfg)
    bad_df = sample_df.drop(columns=[cfg["data"]["score_cols"][0]])
    with pytest.raises(ValueError, match="Missing required columns"):
        model.fit_transform(bad_df)
