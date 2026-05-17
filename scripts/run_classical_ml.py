#!/usr/bin/env python
"""Run classical ML pipeline on Wine Quality and Covertype datasets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classical_ml.evaluator import ModelEvaluator
from src.classical_ml.trainer import ClassicalMLTrainer
from src.classical_ml.theory import get_comparison_table
from src.data.loaders import load_wine_quality
from src.data.project_datasets import get_covertype_splits
from src.data.preprocessing import build_preprocessor, detect_imbalance
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.reproducibility import set_global_seed
from src.visualization.plots import PlotManager
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def run_dataset(
    dataset_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
    plots_dir: Path,
    models_dir: Path,
    metrics_dir: Path,
    quick: bool = False,
) -> None:
    logger = get_logger(f"classical_{dataset_name}", config["paths"]["logs"])
    seed = config["random_seed"]
    set_global_seed(seed)

    is_imbalanced, ratios = detect_imbalance(y)
    logger.info("Dataset %s | samples=%d | classes=%d | imbalanced=%s", dataset_name, len(X), y.nunique(), is_imbalanced)

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X, y,
        train_ratio=config["split_ratios"]["train"],
        val_ratio=config["split_ratios"]["val"],
        test_ratio=config["split_ratios"]["test"],
        random_state=seed,
    )

    preprocessor = build_preprocessor(X_train)
    trainer = ClassicalMLTrainer(
        preprocessor=preprocessor,
        cv_folds=config["classical_ml"]["cv_folds"],
        random_state=seed,
        use_class_weight=is_imbalanced,
    )

    if quick:
        param_override = {
            "logistic_regression": {"clf__C": [1.0], "clf__solver": ["lbfgs"]},
            "random_forest": {"clf__n_estimators": [50], "clf__max_depth": [10]},
            "svm": {"clf__C": [1.0], "clf__kernel": ["rbf"]},
            "knn": {"clf__n_neighbors": [5]},
        }
        trainer.train_all(X_train, y_train, param_grids=param_override)
    else:
        trainer.train_all(X_train, y_train, use_randomized={"svm": True}, n_iter=15)

    trainer.save_models(models_dir / dataset_name)

    evaluator = ModelEvaluator(random_state=seed)
    plot_mgr = PlotManager(plots_dir / dataset_name)
    all_metrics = []

    for name, model in trainer.fitted_models.items():
        metrics = evaluator.evaluate(model, X_test, y_test, model_name=name)
        cv_scores = evaluator.cross_validate(model, pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        metrics["cv_scores"] = cv_scores
        all_metrics.append(metrics)

        cm = np.array(metrics["confusion_matrix"])
        plot_mgr.plot_confusion_matrix(cm, title=f"{name} - {dataset_name}", filename=f"cm_{name}.png")
        if "fpr" in metrics and metrics["fpr"]:
            plot_mgr.plot_roc_curves(metrics["fpr"], metrics["tpr"], title=f"ROC - {name}", filename=f"roc_{name}.png")

    evaluator.save_metrics(all_metrics, metrics_dir / f"{dataset_name}_test_metrics.json")

    df = pd.DataFrame(
        [{"model": m["model"], "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "roc_auc": m.get("roc_auc")} for m in all_metrics]
    )
    plot_mgr.plot_metric_comparison(df, filename="model_comparison.png")
    df.to_csv(metrics_dir / f"{dataset_name}_comparison.csv", index=False)
    logger.info("Completed %s. Best model: %s", dataset_name, df.loc[df["f1"].idxmax(), "model"])

    # Decision tree visualization (subset of features)
    try:
        prep = build_preprocessor(X_train)
        X_train_prep = prep.fit_transform(X_train)
        feature_names = [f"feat_{i}" for i in range(X_train_prep.shape[1])]
        dt = DecisionTreeClassifier(max_depth=3, random_state=seed)
        dt.fit(X_train_prep, y_train)
        class_names = [str(c) for c in sorted(y.unique())]
        plot_mgr.plot_decision_tree(
            dt, feature_names, class_names, filename="decision_tree_depth3.png", max_depth=3
        )
    except Exception as exc:
        logger.warning("Decision tree visualization skipped: %s", exc)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fast run with reduced hyperparameter grid")
    args = parser.parse_args()

    config = load_config()
    paths = config["paths"]
    plots_dir = ROOT / paths["plots"] / "classical_ml"
    models_dir = ROOT / paths["models"] / "classical_ml"
    metrics_dir = ROOT / paths["metrics"]

    for d in [plots_dir, models_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Save theory comparison table
    theory_df = pd.DataFrame(get_comparison_table())
    theory_df.to_csv(metrics_dir / "algorithm_comparison_table.csv", index=False)

    X_wine, y_wine = load_wine_quality()
    run_dataset("wine_quality", X_wine, y_wine, config, plots_dir, models_dir, metrics_dir, quick=args.quick)

    X_train, X_val, X_test, y_train, y_val, y_test = get_covertype_splits(config)
    import pandas as pd
    X_cov = pd.concat([X_train, X_val, X_test], ignore_index=True)
    y_cov = pd.concat([y_train, y_val, y_test], ignore_index=True)
    run_dataset("covertype", X_cov, y_cov, config, plots_dir, models_dir, metrics_dir, quick=args.quick)

    print("Classical ML pipeline completed.")


if __name__ == "__main__":
    main()
