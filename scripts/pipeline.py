"""
scripts/pipeline.py
====================
Main pipeline entry point.  Orchestration only — no computation logic.

Steps
-----
  1. Load data
  2. LFM on full data (unsupervised)
  3. Self-Training (semi-supervised pseudo-labelling)
  4. Final classifier (3 LFM factors + brand_wid)
  5. Save results

Usage
-----
  python scripts/pipeline.py --data data/raw/retail_processed.csv
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from retail_anomaly.utils.config import load_config
from retail_anomaly.utils.data_loader import load_pipeline_data
from retail_anomaly.features.lfm import LFMFeatureExtractor
from retail_anomaly.semi.self_training import SelfTrainer
from retail_anomaly.scorer.final_model import FinalClassifier

SCORE_COLS = [
    "active_days_score", "pack_sale_ratio_score", "daily_scan_stability_score",
    "scan_interval_score", "daily_brand_width_score", "inventory_sales_ratio_score",
    "sales_order_ratio_score", "inventory_deviation_score", "avg_transaction_qty_score",
    "inventory_change_ratio_score", "operation_match_index_score",
]
SHORT_NAMES = [
    "active_days", "pack_sale", "scan_stab", "scan_intv", "brand_wid",
    "inv_sales", "sales_ord", "inv_dev", "txn_qty", "inv_chg", "op_match",
]
ID_COL    = "customer_id"
CITY_COL  = "city_code"
LABEL_COL = "quality_flagged"


def main(data_path: str, config_path: str, output_dir: str) -> None:
    reports = Path(output_dir)
    models_dir = reports / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    cfg    = load_config(config_path)
    lfm_cfg = cfg["pipeline"]["lfm"]
    st_cfg  = cfg["pipeline"]["self_training"]
    fm_cfg  = cfg["pipeline"]["final_model"]

    # Step 1 — Load data
    df_full, df_labelled, df_unlabelled = load_pipeline_data(
        data_path, SCORE_COLS, LABEL_COL
    )

    # Step 2 — LFM on all rows
    lfm = LFMFeatureExtractor(
        n_factors=lfm_cfg["n_factors"],
        rotation=lfm_cfg["rotation"],
        sign_alignment=lfm_cfg["sign_alignment"],
    )
    F_full = lfm.fit_transform(df_full[SCORE_COLS].values)
    lfm.loadings_df(SHORT_NAMES).to_csv(reports / "factor_loadings.csv")
    np.save(models_dir / "lfm_loadings.npy", lfm.loadings_)
    with open(models_dir / "lfm_scaler.pkl", "wb") as f:
        pickle.dump(lfm.scaler_, f)

    # Step 3 — Self-Training
    labelled_pos   = np.where(df_full[LABEL_COL].notna().values)[0]
    unlabelled_pos = np.where(df_full[LABEL_COL].isna().values)[0]
    X_lab   = F_full[labelled_pos]
    y_lab   = df_labelled[LABEL_COL].astype(int).values
    X_unlab = F_full[unlabelled_pos]

    trainer = SelfTrainer(
        threshold_pos=st_cfg["threshold_pos"],
        threshold_neg=st_cfg["threshold_neg"],
        max_iter=st_cfg["max_iter"],
        min_new_samples=st_cfg["min_new_samples"],
    )
    trainer.fit(X_lab, y_lab, X_unlab)
    trainer.iteration_log_df().to_csv(reports / "self_training_log.csv", index=False)

    # Step 4 — Final classifier (3 factors + brand_wid)
    extra_col    = fm_cfg["extra_features"][0]
    bw_full      = df_full[extra_col].values
    bw_lab       = bw_full[labelled_pos]
    bw_pseudo    = bw_full[unlabelled_pos[trainer.pseudo_source_indices_]] \
                   if len(trainer.pseudo_source_indices_) else np.array([])
    bw_train     = np.hstack([bw_lab, bw_pseudo])
    X_train_4d   = np.hstack([trainer.X_train_final_, bw_train.reshape(-1, 1)])
    X_full_4d    = np.hstack([F_full, bw_full.reshape(-1, 1)])

    clf = FinalClassifier(C=fm_cfg["C"], class_weight=fm_cfg["class_weight"])
    clf.fit(X_train_4d, trainer.y_train_final_)
    anomaly_proba = clf.predict_proba_all(X_full_4d)
    clf.save(models_dir / "final_clf.pkl")

    # Step 5 — Save results
    factor_names = [f"F{i + 1}" for i in range(lfm_cfg["n_factors"])]
    all_scores = df_full[[ID_COL, CITY_COL]].copy()
    for i, fn in enumerate(factor_names):
        all_scores[fn] = F_full[:, i]
    all_scores[extra_col]     = bw_full
    all_scores["anomaly_proba"] = anomaly_proba
    all_scores[LABEL_COL]     = df_full[LABEL_COL].values
    all_scores.to_csv(reports / "all_scores.csv", index=False)

    flagged_mask = anomaly_proba > 0.5
    city_summary = (
        all_scores.groupby(CITY_COL)
        .agg(n_stores=(ID_COL, "count"),
             mean_proba=("anomaly_proba", "mean"),
             n_predicted_flagged=(CITY_COL, lambda x: flagged_mask[x.index].sum()))
        .round(4).reset_index()
    )
    city_summary.to_csv(reports / "city_summary.csv", index=False)

    n_pseudo = len(trainer.pseudo_source_indices_)
    print(f"\n{'=' * 55}")
    print(f"Self-Training : {trainer.n_rounds_} round(s), "
          f"{n_pseudo} pseudo-labels added")
    print(f"Final train   : {len(trainer.y_train_final_)} rows "
          f"({len(y_lab)} original + {n_pseudo} pseudo)")
    rates = city_summary.set_index(CITY_COL)["mean_proba"].round(3).to_dict()
    print(f"City anomaly rates: "
          + "  ".join(f"{c}={v:.1%}" for c, v in rates.items()))
    print(f"{'=' * 55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retail anomaly pipeline")
    parser.add_argument("--data",       required=True,
                        help="Path to retail_processed.csv")
    parser.add_argument("--config",     default="configs/default.yaml",
                        help="Config YAML (default: configs/default.yaml)")
    parser.add_argument("--output-dir", default="reports",
                        help="Output directory (default: reports)")
    args = parser.parse_args()
    main(data_path=args.data, config_path=args.config, output_dir=args.output_dir)
