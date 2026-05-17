"""PCA, t-SNE, UMAP, and LDA utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE

try:
    import umap
except ImportError:
    umap = None  # type: ignore


class DimensionalityReducer:
    """Fit and transform with multiple reduction methods."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.models: Dict[str, object] = {}
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    def fit_pca(self, X: np.ndarray, n_components: Optional[int] = None, variance: float = 0.95) -> np.ndarray:
        """Fit PCA; if n_components is None, retain `variance` fraction."""
        if n_components is None:
            pca_full = PCA(random_state=self.random_state)
            pca_full.fit(X)
            cumsum = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = int(np.searchsorted(cumsum, variance) + 1)

        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_reduced = pca.fit_transform(X)
        self.models["pca"] = pca
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        return X_reduced

    def fit_tsne(self, X: np.ndarray, n_components: int = 2, perplexity: float = 30.0) -> np.ndarray:
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=self.random_state,
            init="pca",
            learning_rate="auto",
        )
        X_emb = tsne.fit_transform(X)
        self.models["tsne"] = tsne
        return X_emb

    def fit_umap(self, X: np.ndarray, n_components: int = 2) -> np.ndarray:
        if umap is None:
            raise ImportError("umap-learn is required. Install with: pip install umap-learn")
        reducer = umap.UMAP(n_components=n_components, random_state=self.random_state)
        X_emb = reducer.fit_transform(X)
        self.models["umap"] = reducer
        return X_emb

    def fit_lda(self, X: np.ndarray, y: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
        n_classes = len(np.unique(y))
        max_comp = min(n_classes - 1, X.shape[1])
        if n_components is None:
            n_components = max_comp
        n_components = min(n_components, max_comp)
        lda = LDA(n_components=n_components)
        X_lda = lda.fit_transform(X, y)
        self.models["lda"] = lda
        return X_lda

    def save(self, path: str | Path) -> None:
        joblib.dump(self.models, path)
