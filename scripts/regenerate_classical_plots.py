#!/usr/bin/env python
"""Regenerate classical ML plots from saved models (quick)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classical_ml.evaluator import ModelEvaluator
from src.data.loaders import load_covertype, load_wine_quality, split_train_val_test
from src.utils.config import load_config
from src.utils.reproducibility import set_global_seed
from src.visualization.plots import PlotManager


def main() -> None:
    config = load_config()
    set_global_seed(config["random_seed"])
    evaluator = ModelEvaluator()
    models_dir = ROOT / config["paths"]["models"] / "classical_ml"

    datasets = {
        "wine_quality": load_wine_quality(),
        "covertype": load_covertype(max_samples=config["dataset_b"]["max_samples"]),
    }

    for name, (X, y) in datasets.items():
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
            X, y, random_state=config["random_seed"]
        )
        plot_mgr = PlotManager(ROOT / config["paths"]["plots"] / "classical_ml" / name)
        rows = []
        for model_path in (models_dir / name).glob("*.joblib"):
            model = joblib.load(model_path)
            mname = model_path.stem
            metrics = evaluator.evaluate(model, X_test, y_test, model_name=mname)
            rows.append(
                {
                    "model": mname,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                }
            )
            plot_mgr.plot_confusion_matrix(
                np.array(metrics["confusion_matrix"]),
                title=f"{mname}",
                filename=f"cm_{mname}.png",
            )
            if metrics.get("fpr"):
                plot_mgr.plot_roc_curves(metrics["fpr"], metrics["tpr"], filename=f"roc_{mname}.png")
        if rows:
            plot_mgr.plot_metric_comparison(pd.DataFrame(rows), filename="model_comparison.png")
    print("Plots regenerated under outputs/plots/classical_ml/")


if __name__ == "__main__":
    main()
