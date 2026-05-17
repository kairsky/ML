"""Configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration from project root."""
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
