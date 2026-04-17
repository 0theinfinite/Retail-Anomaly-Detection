"""
retail_anomaly
==============
Retail store scan-quality anomaly detection.

Quick start
-----------
>>> from retail_anomaly import ImprovedLFM
>>> from retail_anomaly.utils.config import load_config
>>> cfg = load_config()
>>> model = ImprovedLFM(cfg)
>>> result_df, _, _ = model.fit_transform(df)
"""
from retail_anomaly.scorer.lfm import ImprovedLFM
from retail_anomaly.scorer.final_model import FinalClassifier

try:
    from retail_anomaly.cvae.model import RetailCVAE
    __all__ = ["ImprovedLFM", "FinalClassifier", "RetailCVAE"]
except Exception:
    __all__ = ["ImprovedLFM", "FinalClassifier"]

__version__ = "0.1.0"
