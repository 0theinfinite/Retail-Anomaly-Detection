"""
retail_anomaly.api.app
=======================
FastAPI service exposing three endpoints:

  POST /score     – score a batch of stores
  POST /generate  – generate synthetic rows via CVAE
  GET  /health    – liveness probe
  GET  /report    – city-level summary statistics

The app is stateless at the module level; models are loaded once at startup
via lifespan context manager and stored in app.state.
"""
from __future__ import annotations

import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from retail_anomaly.utils.config import load_config


# ──────────────────────────────────────────────────────────────────────────────
# Request / response schemas
# ──────────────────────────────────────────────────────────────────────────────

class StoreRecord(BaseModel):
    """One store's raw indicator values."""
    客户编码:         str
    地市编码:         str
    开机天数:         float
    包销售比例:       float
    日扫码稳定性指数: float
    扫码间隔时间:     float
    日均扫码品牌宽度: float
    存销比:           float
    销订比:           float
    进销存偏移率:     float
    单笔量:           float
    库存变化比:       float
    经营匹配指数:     float


class ScoreRequest(BaseModel):
    stores: list[StoreRecord] = Field(..., min_length=1)


class ScoreResult(BaseModel):
    客户编码:     str
    地市编码:     str
    quality_score: float = Field(..., description="LFM composite score [0,100]")
    anomaly_proba: float | None = Field(None, description="P(anomaly) from classifier")


class GenerateRequest(BaseModel):
    n:            int             = Field(1000, ge=1, le=100_000)
    city_weights: dict[str, float] | None = None


# ──────────────────────────────────────────────────────────────────────────────
# App factory
# ──────────────────────────────────────────────────────────────────────────────

def create_app(
    lfm_model=None,
    classifier=None,
    cvae_model=None,
    config: dict | None = None,
) -> FastAPI:
    """
    Create the FastAPI application.

    Parameters
    ----------
    lfm_model, classifier, cvae_model
        Pre-fitted model objects.  Pass None to load from disk (not yet
        implemented; placeholder for Phase 3 persistence layer).
    """
    cfg = config or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.lfm        = lfm_model
        app.state.classifier = classifier
        app.state.cvae       = cvae_model
        app.state.cfg        = cfg
        yield

    app = FastAPI(
        title="Retail Anomaly Detection API",
        description=(
            "LFM-based unsupervised scorer + GBDT classifier + "
            "Conditional VAE synthetic data generator"
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # ── /health ─────────────────────────────────────────────────────────────

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        return {
            "status":     "ok",
            "lfm":        "ready" if app.state.lfm        else "not loaded",
            "classifier": "ready" if app.state.classifier else "not loaded",
            "cvae":       "ready" if app.state.cvae       else "not loaded",
        }

    # ── /score ──────────────────────────────────────────────────────────────

    @app.post("/score", response_model=list[ScoreResult], tags=["inference"])
    async def score(req: ScoreRequest) -> list[ScoreResult]:
        if app.state.lfm is None:
            raise HTTPException(503, "LFM model not loaded")

        df = pd.DataFrame([s.model_dump() for s in req.stores])

        try:
            scores = app.state.lfm.score_new(df)
        except Exception as e:
            raise HTTPException(500, f"Scoring failed: {e}") from e

        proba_arr: np.ndarray | None = None
        if app.state.classifier is not None:
            # Classifier needs indicator _score columns; re-run KDE on-the-fly
            result_df, _, _ = app.state.lfm.fit_transform(df, verbose=False)
            proba_arr = app.state.classifier.predict_proba(result_df)

        results = []
        for i, store in enumerate(req.stores):
            results.append(ScoreResult(
                客户编码=store.客户编码,
                地市编码=store.地市编码,
                quality_score=round(float(scores[i]), 2),
                anomaly_proba=(round(float(proba_arr[i]), 4)
                               if proba_arr is not None else None),
            ))
        return results

    # ── /generate ───────────────────────────────────────────────────────────

    @app.post("/generate", tags=["synthetic"])
    async def generate(req: GenerateRequest) -> dict[str, Any]:
        if app.state.cvae is None:
            raise HTTPException(503, "CVAE model not loaded")
        try:
            df = app.state.cvae.generate(n=req.n, city_weights=req.city_weights)
        except Exception as e:
            raise HTTPException(500, f"Generation failed: {e}") from e
        return {
            "n_generated": len(df),
            "columns":     df.columns.tolist(),
            "preview":     df.head(5).to_dict(orient="records"),
        }

    # ── /report ─────────────────────────────────────────────────────────────

    @app.get("/report", tags=["analytics"])
    async def report(result_json: str | None = None) -> dict[str, Any]:
        """
        City-level quality summary.  Pass result_json (JSON records) or
        provide pre-loaded data via app.state.last_result.
        """
        if result_json:
            df = pd.read_json(io.StringIO(result_json))
        elif hasattr(app.state, "last_result") and app.state.last_result is not None:
            df = app.state.last_result
        else:
            raise HTTPException(400, "No result data available. POST /score first.")

        score_col = "quality_score" if "quality_score" in df.columns else "LFM改进得分"
        city_col  = app.state.cfg["data"]["city_col"]

        summary = (
            df.groupby(city_col)[score_col]
            .agg(["mean", "median", "std", "count"])
            .round(2)
            .reset_index()
            .rename(columns={"mean": "avg_score", "median": "median_score",
                              "std": "std_score", "count": "n_stores"})
            .to_dict(orient="records")
        )
        return {"city_summary": summary}

    return app


# ──────────────────────────────────────────────────────────────────────────────
# Entry-point for `uvicorn retail_anomaly.api.app:app`
# ──────────────────────────────────────────────────────────────────────────────
# Models are None here (not fitted).  In production, load from disk / MLflow.
app = create_app()
