"""Dataset loading utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split


def load_wine_quality() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load UCI Wine Quality dataset (OpenML id=287).

    Returns
    -------
    X : DataFrame
        Feature matrix.
    y : Series
        Target (wine quality class).
    """
    from sklearn.datasets import fetch_openml

    data = fetch_openml(data_id=287, as_frame=True, parser="auto")
    X = data.data.copy()
    y = data.target.copy()
    y = y.astype(int)
    return X, y


def load_covertype(max_samples: int | None = 15000, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load Covertype multiclass dataset (>=1000 samples).

    Parameters
    ----------
    max_samples : int, optional
        Subsample for tractable training; None uses full dataset.
    """
    data = fetch_covtype(as_frame=True)
    X = data.data.copy()
    y = data.target.copy()
    y = y.astype(int)

    if max_samples is not None and len(X) > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)

    return X, y


def _can_stratify(y: pd.Series, min_count: int = 2) -> bool:
    """Return True if every class has at least min_count samples (required for stratified split)."""
    counts = y.value_counts()
    return len(counts) > 1 and int(counts.min()) >= min_count


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train / validation / test (70/15/15 by default).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    if stratify and not _can_stratify(y):
        stratify = False

    strat = y if stratify else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), random_state=random_state, stratify=strat
    )
    relative_val = val_ratio / (val_ratio + test_ratio)
    strat_temp = y_temp if stratify and _can_stratify(y_temp) else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - relative_val), random_state=random_state, stratify=strat_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
