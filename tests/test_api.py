"""Tests for retail_anomaly.api.app"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from retail_anomaly.api.app import create_app
from retail_anomaly.utils.config import load_config


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return load_config()


def _make_store_payload(n: int = 3) -> list[dict]:
    np.random.seed(42)
    stores = []
    for i in range(n):
        stores.append({
            "客户编码":         f"cust{i:06d}",
            "地市编码":         np.random.choice(["A", "B", "C"]),
            "开机天数":         float(np.random.randint(20, 31)),
            "包销售比例":       round(np.random.uniform(0.7, 1.0), 3),
            "日扫码稳定性指数": round(np.random.uniform(0.2, 0.8), 3),
            "扫码间隔时间":     round(np.random.uniform(8.0, 25.0), 2),
            "日均扫码品牌宽度": round(np.random.uniform(10.0, 25.0), 2),
            "存销比":           round(np.random.uniform(0.3, 1.2), 3),
            "销订比":           round(np.random.uniform(0.7, 1.0), 3),
            "进销存偏移率":     round(np.random.uniform(-0.05, 0.05), 4),
            "单笔量":           round(np.random.uniform(0.1, 0.3), 3),
            "库存变化比":       round(np.random.uniform(0.8, 1.6), 3),
            "经营匹配指数":     round(np.random.uniform(3.0, 6.0), 2),
        })
    return stores


@pytest.fixture
def mock_lfm(cfg):
    """LFM mock that returns plausible scores."""
    m = MagicMock()
    m.score_new.return_value = np.array([75.0, 60.0, 45.0])
    m.fit_transform.return_value = (
        pd.DataFrame({
            "客户编码": ["c0", "c1", "c2"],
            "地市编码": ["A", "B", "C"],
            "LFM改进得分": [75.0, 60.0, 45.0],
        }),
        pd.DataFrame(),
        pd.Series([True, True, True]),
    )
    return m


@pytest.fixture
def mock_cvae():
    m = MagicMock()
    m.generate.return_value = pd.DataFrame({
        "开机天数_kde":         np.random.uniform(50, 90, 5),
        "包销售比例_kde":       np.random.uniform(50, 90, 5),
        "地市编码":             ["A", "B", "C", "A", "B"],
    })
    return m


@pytest.fixture
def client(mock_lfm, mock_cvae, cfg):
    app = create_app(
        lfm_model=mock_lfm,
        classifier=None,
        cvae_model=mock_cvae,
        config=cfg,
    )
    return TestClient(app)


# ── health ────────────────────────────────────────────────────────────────────

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_model_states(client):
    data = r = client.get("/health").json()
    assert data["lfm"]  == "ready"
    assert data["cvae"] == "ready"
    assert data["classifier"] == "not loaded"


# ── /score ────────────────────────────────────────────────────────────────────

def test_score_returns_list(client):
    payload = {"stores": _make_store_payload(3)}
    r = client.post("/score", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 3


def test_score_fields_present(client):
    payload = {"stores": _make_store_payload(1)}
    r = client.post("/score", json=payload)
    item = r.json()[0]
    assert "quality_score" in item
    assert "客户编码" in item
    assert "地市编码" in item


def test_score_range(client):
    payload = {"stores": _make_store_payload(3)}
    r = client.post("/score", json=payload)
    for item in r.json():
        assert 0.0 <= item["quality_score"] <= 100.0


def test_score_no_lfm():
    """503 when LFM not loaded."""
    from retail_anomaly.utils.config import load_config
    app = create_app(lfm_model=None, config=load_config())
    c   = TestClient(app)
    r   = c.post("/score", json={"stores": _make_store_payload(1)})
    assert r.status_code == 503


def test_score_empty_stores(client):
    r = client.post("/score", json={"stores": []})
    assert r.status_code == 422   # Pydantic validation: min_length=1


# ── /generate ─────────────────────────────────────────────────────────────────

def test_generate_returns_n(client, mock_cvae):
    mock_cvae.generate.return_value = pd.DataFrame({
        "开机天数_kde": np.random.uniform(40, 90, 50),
        "地市编码":     ["A"] * 50,
    })
    r = client.post("/generate", json={"n": 50})
    assert r.status_code == 200
    assert r.json()["n_generated"] == 50


def test_generate_no_cvae():
    from retail_anomaly.utils.config import load_config
    app = create_app(cvae_model=None, config=load_config())
    c   = TestClient(app)
    r   = c.post("/generate", json={"n": 10})
    assert r.status_code == 503
