"""Preprocessing pipelines."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def detect_imbalance(y: pd.Series, threshold: float = 0.3) -> Tuple[bool, dict]:
    """
    Detect class imbalance.

    Returns whether imbalance handling is recommended and class ratios.
    """
    counts = y.value_counts(normalize=True).sort_index()
    ratios = counts.to_dict()
    minority_ratio = counts.min()
    is_imbalanced = minority_ratio < threshold
    return is_imbalanced, ratios


def build_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:
    """
    Build ColumnTransformer with imputation, scaling, and one-hot encoding.
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers: List[tuple] = [("num", numeric_pipeline, numeric_cols)]
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")
