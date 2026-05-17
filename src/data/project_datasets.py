"""Consistent dataset loading across all project pipelines."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd

from .loaders import load_covertype, load_wine_quality, split_train_val_test


def get_covertype_splits(config: Dict[str, Any]) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
]:
    """
    Load Covertype with the same sample size and split as classical ML / dimensionality.

    Uses config['dataset_b']['max_samples'] and config['split_ratios'] — never subsampled
    differently per script.
    """
    seed = config["random_seed"]
    max_samples = config["dataset_b"]["max_samples"]
    X, y = load_covertype(max_samples=max_samples, random_state=seed)
    return split_train_val_test(
        X,
        y,
        train_ratio=config["split_ratios"]["train"],
        val_ratio=config["split_ratios"]["val"],
        test_ratio=config["split_ratios"]["test"],
        random_state=seed,
    )


def get_wine_quality_splits(config: Dict[str, Any]) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
]:
    """Load Wine Quality with project-standard split ratios."""
    seed = config["random_seed"]
    X, y = load_wine_quality()
    return split_train_val_test(
        X,
        y,
        train_ratio=config["split_ratios"]["train"],
        val_ratio=config["split_ratios"]["val"],
        test_ratio=config["split_ratios"]["test"],
        random_state=seed,
    )
