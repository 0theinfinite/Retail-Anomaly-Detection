#!/usr/bin/env python3
"""
scripts/train.py
================
End-to-end training pipeline: LFM → GBDT classifier → CVAE.
All experiments tracked in MLflow.

Usage
-----
# With real data:
python scripts/train.py --data data/raw/stores.tsv

# With synthetic seed data (demo):
python scripts/train.py --demo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from retail_anomaly import ImprovedLFM, AnomalyClassifier, RetailCVAE
from retail_anomaly.cvae.validation import ks_validate
from retail_anomaly.utils.config import load_config


# ── helpers ───────────────────────────────────────────────────────────────────

def make_demo_data(cfg: dict, n: int = 3700, seed: int = 42) -> pd.DataFrame:
    """Reproducible synthetic dataset matching the real schema."""
    np.random.seed(seed)
    cols  = cfg["data"]["score_cols"]
    id_c  = cfg["data"]["id_col"]
    city_c = cfg["data"]["city_col"]
    lbl_c = cfg["data"]["label_col"]

    data = {c: np.abs(np.random.normal(1, 0.3, n)) for c in cols}
    data[id_c]    = [f"cust{i:08d}" for i in range(n)]
    data[city_c]  = np.random.choice(["A", "B", "C"], n, p=[0.25, 0.50, 0.25])

    # City A anomaly: low 包销售比例, high 单笔量
    mask_a = np.array(data[city_c]) == "A"
    data["包销售比例"][mask_a] *= 0.5
    data["单笔量"][mask_a]     *= 1.9

    # 15 % labelled anomalous
    flagged = np.random.choice(n, int(n * 0.15), replace=False)
    data[lbl_c] = 0
    for col in ["扫码间隔时间", "进销存偏移率", "存销比"]:
        data[col][flagged] *= np.random.choice([0.25, 3.0], len(flagged))
    labels = np.zeros(n, dtype=int)
    labels[flagged] = 1
    data[lbl_c] = labels
    return pd.DataFrame(data)


# ── main ──────────────────────────────────────────────────────────────────────

def main(data_path: str | None, demo: bool, config_path: str | None) -> None:
    cfg = load_config(config_path)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    # ── Load data ─────────────────────────────────────────────────────────
    if demo:
        print("[train] Generating demo data (n=3700) …")
        df = make_demo_data(cfg)
    else:
        sep = "\t" if str(data_path).endswith(".tsv") else ","
        df  = pd.read_csv(data_path, sep=sep)
        print(f"[train] Loaded {len(df)} rows from {data_path}")

    label_col = cfg["data"]["label_col"]
    has_labels = label_col in df.columns and df[label_col].notna().sum() > 20

    with mlflow.start_run():
        mlflow.log_params({
            "n_rows":        len(df),
            "has_labels":    has_labels,
            "demo_mode":     demo,
            "n_factors_cfg": cfg["scorer"]["n_factors"],
        })

        # ── Phase 0: LFM ─────────────────────────────────────────────────
        print("\n── Phase 0: LFM scorer ──")
        lfm      = ImprovedLFM(cfg)
        result_df, score_df, mask_iqr = lfm.fit_transform(df)

        mlflow.log_params({
            "n_factors_chosen": lfm.n_factors_chosen_,
            "iqr_clean_pct":    round(mask_iqr.mean(), 3),
        })

        # Validate AUC if labels available
        if has_labels:
            from sklearn.metrics import roc_auc_score
            from scipy.stats import mannwhitneyu
            y = df[label_col].fillna(0).astype(int).values
            auc_lfm = roc_auc_score(y, -result_df["LFM改进得分"].values)
            _, mw_p = mannwhitneyu(
                result_df.loc[y == 1, "LFM改进得分"],
                result_df.loc[y == 0, "LFM改进得分"],
                alternative="less",
            )
            print(f"[LFM] AUC={auc_lfm:.3f}  Mann-Whitney p={mw_p:.2e}")
            mlflow.log_metrics({"lfm_auc": auc_lfm, "lfm_mw_p": float(mw_p)})

        # ── Phase 1: Classifier ───────────────────────────────────────────
        if has_labels:
            print("\n── Phase 1: GBDT classifier ──")
            y = df[label_col].fillna(0).astype(int)
            clf = AnomalyClassifier(cfg)
            clf.fit(result_df, y)

            mlflow.log_metrics({"classifier_cv_auc": clf.cv_auc_})

            # Log feature importance as artifact
            fi_df = clf.feature_importance_df()
            fi_path = Path("feature_importance.csv")
            fi_df.to_csv(fi_path, index=False)
            mlflow.log_artifact(str(fi_path))
            fi_path.unlink(missing_ok=True)

            print(f"[Classifier] CV AUC: {clf.cv_auc_:.3f}")
            print(fi_df.head(5).to_string(index=False))
        else:
            clf = None
            print("[train] No labels — skipping classifier.")

        # ── Phase 2: CVAE ─────────────────────────────────────────────────
        print("\n── Phase 2: Conditional VAE ──")
        city_col  = cfg["data"]["city_col"]
        cities    = sorted(score_df.index.map(
            lambda _: None  # dummy; city from df
        ).dropna().unique() if False else df[city_col].unique())
        city_map  = {c: i for i, c in enumerate(sorted(cities))}

        # Add city column to score_df for CVAE
        score_df_with_city = score_df.copy()
        score_df_with_city[city_col] = df[city_col].values

        cvae = RetailCVAE(config=cfg, city_map=city_map)
        cvae.fit(score_df_with_city, loadings=lfm.loadings_)

        # Generate synthetic data
        n_gen     = cfg["cvae"]["n_synthetic"]
        synthetic = cvae.generate(n=n_gen)
        out_path  = Path("data/synthetic/synthetic_100k.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        synthetic.to_csv(out_path, index=False)
        mlflow.log_artifact(str(out_path))
        print(f"[CVAE] Generated {n_gen} rows → {out_path}")

        # KS validation
        kde_cols = [f"{c}_kde" for c in cfg["data"]["score_cols"]
                    if f"{c}_kde" in score_df_with_city.columns]
        if kde_cols:
            ks_result = ks_validate(score_df_with_city, synthetic, kde_cols)
            n_pass = ks_result["pass"].sum()
            mlflow.log_metric("ks_pass_rate", n_pass / len(ks_result))
            ks_path = Path("ks_validation.csv")
            ks_result.to_csv(ks_path, index=False)
            mlflow.log_artifact(str(ks_path))
            ks_path.unlink(missing_ok=True)

        # ── City-level summary ────────────────────────────────────────────
        print("\n── City-level quality summary ──")
        summary = (
            result_df.groupby(city_col)["LFM改进得分"]
            .agg(["mean", "median", "std"])
            .round(2)
        )
        print(summary.to_string())
        for city, row in summary.iterrows():
            mlflow.log_metrics({
                f"city_{city}_mean":   row["mean"],
                f"city_{city}_median": row["median"],
            })

        print("\n[train] Run complete. View results: mlflow ui")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   type=str, default=None,
                        help="Path to TSV/CSV input file")
    parser.add_argument("--demo",   action="store_true",
                        help="Generate and use synthetic demo data")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML (default: configs/default.yaml)")
    args = parser.parse_args()

    if not args.demo and args.data is None:
        parser.error("Provide --data <path> or --demo")

    main(data_path=args.data, demo=args.demo, config_path=args.config)
