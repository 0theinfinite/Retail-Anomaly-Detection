"""Thin wrapper around PyYAML so every module loads config the same way."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CFG = Path(__file__).parents[2] / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config.  Falls back to configs/default.yaml."""
    cfg_path = Path(path) if path else _DEFAULT_CFG
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def merge_config(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins)."""
    out = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_config(out[k], v)
        else:
            out[k] = v
    return out
