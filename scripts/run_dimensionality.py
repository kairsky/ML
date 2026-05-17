#!/usr/bin/env python
"""Dimensionality reduction and PCA retraining experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classical_ml.evaluator import ModelEvaluator
from src.data.loaders import load_covertype, split_train_val_test
from src.data.preprocessing import build_preprocessor
from src.dimensionality.pca_analysis import compare_before_after_pca, fit_pca_95
from src.dimensionality.reduction import DimensionalityReducer
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.reproducibility import set_global_seed
from src.visualization.plots import PlotManager


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    config = load_config()
    seed = config["random_seed"]
    set_global_seed(seed)
    logger = get_logger("dimensionality", config["paths"]["logs"])

    max_samples = 5000 if args.quick else config["dataset_b"]["max_samples"]
    X, y = load_covertype(max_samples=max_samples)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y, random_state=seed)

    prep = build_preprocessor(X_train)
    X_train_t = prep.fit_transform(X_train)
    X_test_t = prep.transform(X_test)

    reducer = DimensionalityReducer(random_state=seed)
    plot_mgr = PlotManager(ROOT / config["paths"]["plots"] / "dimensionality")

    pca = PCA(random_state=seed)
    pca.fit(X_train_t)
    plot_mgr.plot_pca_variance(pca.explained_variance_ratio_, filename="pca_explained_variance.png")

    pca_model, X_pca, n_comp = fit_pca_95(X_train_t, random_state=seed)
    X_test_pca = pca_model.transform(X_test_t)

    # Embeddings on subsample
    sub = min(2000, len(X_train_t))
    idx = np.random.RandomState(seed).choice(len(X_train_t), sub, replace=False)
    X_sub, y_sub = X_train_t[idx], y_train.values[idx]

    X_tsne = reducer.fit_tsne(X_sub, perplexity=30.0)
    plot_mgr.plot_embedding_2d(X_tsne, y_sub, "t-SNE Embedding", "tsne_2d.png")

    X_umap = reducer.fit_umap(X_sub)
    plot_mgr.plot_embedding_2d(X_umap, y_sub, "UMAP Embedding", "umap_2d.png")

    X_lda = reducer.fit_lda(X_sub, y_sub)
    plot_mgr.plot_embedding_2d(X_lda[:, :2], y_sub, "LDA Embedding", "lda_2d.png")

    X_pca_2d = PCA(n_components=2, random_state=seed).fit_transform(X_sub)
    plot_mgr.plot_embedding_2d(X_pca_2d, y_sub, "PCA 2D Embedding", "pca_2d.png")

    # Retrain best classifier (Random Forest) before vs after PCA
    rf_params = {"n_estimators": 50 if args.quick else 200, "max_depth": 15, "random_state": seed, "n_jobs": -1}
    rf_before = RandomForestClassifier(**rf_params)
    rf_before.fit(X_train_t, y_train)
    rf_after = RandomForestClassifier(**rf_params)
    rf_after.fit(X_pca, y_train)

    evaluator = ModelEvaluator(seed)
    m_before = evaluator.evaluate(
        Pipeline([("clf", rf_before)]), pd.DataFrame(X_test_t), y_test, "rf_full"
    )
    m_after = evaluator.evaluate(
        Pipeline([("clf", rf_after)]), pd.DataFrame(X_test_pca), y_test, "rf_pca95"
    )

    comparison = compare_before_after_pca(m_before, m_after, n_comp, X_train_t.shape[1])
    out_path = ROOT / config["paths"]["metrics"] / "pca_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    logger.info("PCA components: %d / %d | accuracy delta: %.4f", n_comp, X_train_t.shape[1], comparison["accuracy_delta"])
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
