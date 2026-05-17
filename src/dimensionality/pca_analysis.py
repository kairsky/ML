"""PCA-based retraining and comparison."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


def fit_pca_95(X: np.ndarray, random_state: int = 42) -> Tuple[PCA, np.ndarray, int]:
    """Fit PCA retaining 95% variance."""
    pca_full = PCA(random_state=random_state)
    pca_full.fit(X)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumsum, 0.95) + 1)
    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X)
    return pca, X_reduced, n_components


def compare_before_after_pca(
    metrics_before: Dict[str, float],
    metrics_after: Dict[str, float],
    n_components: int,
    original_dim: int,
) -> Dict[str, Any]:
    """Summarize performance trade-offs after PCA."""
    return {
        "original_dimensions": original_dim,
        "pca_components": n_components,
        "variance_retained": 0.95,
        "accuracy_before": metrics_before.get("accuracy"),
        "accuracy_after": metrics_after.get("accuracy"),
        "f1_before": metrics_before.get("f1"),
        "f1_after": metrics_after.get("f1"),
        "accuracy_delta": metrics_after.get("accuracy", 0) - metrics_before.get("accuracy", 0),
        "discussion": (
            "PCA projects data onto orthogonal directions of maximum variance. "
            "Retaining 95% variance reduces dimensionality while discarding noise; "
            "performance may improve (denoising) or degrade (lost discriminative signal)."
        ),
    }
