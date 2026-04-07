"""Tests for retail_anomaly.cvae"""
import numpy as np
import pandas as pd
import pytest

from retail_anomaly.cvae.model import RetailCVAE
from retail_anomaly.cvae.validation import ks_validate
from retail_anomaly.utils.config import load_config, merge_config


@pytest.fixture
def cfg():
    base = load_config()
    # Fast config for tests
    return merge_config(base, {"cvae": {"epochs": 5, "batch_size": 32,
                                        "beta_warmup_epochs": 2}})


@pytest.fixture
def score_df(cfg):
    """Minimal KDE score DataFrame for CVAE training."""
    np.random.seed(1)
    n     = 120
    cols  = cfg["data"]["score_cols"]
    city  = cfg["data"]["city_col"]
    kde_cols = {f"{c}_kde": np.random.uniform(20, 90, n) for c in cols}
    kde_cols[city] = np.random.choice(["A", "B", "C"], n)
    return pd.DataFrame(kde_cols)


def test_cvae_fit_and_generate(score_df, cfg):
    cvae = RetailCVAE(config=cfg, city_map={"A": 0, "B": 1, "C": 2})
    cvae.fit(score_df, verbose=False)

    synthetic = cvae.generate(n=200)
    assert len(synthetic) == 200
    assert cfg["data"]["city_col"] in synthetic.columns


def test_generated_values_in_range(score_df, cfg):
    cvae = RetailCVAE(config=cfg, city_map={"A": 0, "B": 1, "C": 2})
    cvae.fit(score_df, verbose=False)
    syn = cvae.generate(n=100)
    kde_cols = [c for c in syn.columns if c.endswith("_kde")]
    assert (syn[kde_cols].values >= 0).all()
    assert (syn[kde_cols].values <= 100).all()


def test_cvae_generate_before_fit_raises(cfg):
    cvae = RetailCVAE(config=cfg)
    with pytest.raises(RuntimeError, match="fit"):
        cvae.generate(n=10)


def test_ks_validate_returns_dataframe(score_df, cfg):
    cols = cfg["data"]["score_cols"]
    kde_cols = [f"{c}_kde" for c in cols]
    # Compare df with itself — should all pass
    result = ks_validate(score_df, score_df, kde_cols, verbose=False)
    assert "pass" in result.columns
    assert result["pass"].all()
