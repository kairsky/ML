"""Model evaluation metrics and reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize


class ModelEvaluator:
    """Compute comprehensive classification metrics."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def evaluate(
        self,
        model: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """Evaluate model on test set."""
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
        elif hasattr(model.named_steps.get("clf", model), "predict_proba"):
            y_proba = model.predict_proba(X_test)

        metrics: Dict[str, Any] = {
            "model": model_name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, zero_division=0),
        }

        classes = np.unique(y_test)
        if y_proba is not None and len(classes) > 2:
            y_bin = label_binarize(y_test, classes=classes)
            try:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_bin, y_proba, multi_class="ovr", average="weighted")
                )
            except ValueError:
                metrics["roc_auc"] = None
        elif y_proba is not None and len(classes) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba[:, 1]))
        else:
            metrics["roc_auc"] = None

        if y_proba is not None:
            metrics["y_proba"] = y_proba
            metrics["fpr"], metrics["tpr"] = self._roc_curves(y_test, y_proba, classes)

        metrics["y_pred"] = y_pred
        metrics["y_true"] = y_test.values
        return metrics

    def cross_validate(
        self,
        model: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
    ) -> Dict[str, float]:
        """Return mean ± std for key metrics via cross-validation."""
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        scores: Dict[str, List[float]] = {}
        for metric in ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]:
            try:
                vals = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
                scores[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            except Exception:
                scores[metric] = {"mean": float("nan"), "std": float("nan")}
        return scores

    @staticmethod
    def _roc_curves(
        y_true: pd.Series,
        y_proba: np.ndarray,
        classes: np.ndarray,
    ) -> Tuple[Dict, Dict]:
        """Per-class ROC curves for multiclass."""
        y_bin = label_binarize(y_true, classes=classes)
        fpr_dict, tpr_dict = {}, {}
        for i, c in enumerate(classes):
            if y_bin.shape[1] > i:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                fpr_dict[str(c)] = fpr.tolist()
                tpr_dict[str(c)] = tpr.tolist()
        return fpr_dict, tpr_dict

    def save_metrics(self, all_metrics: List[Dict], output_path: str | Path) -> None:
        """Save metrics JSON (excluding large arrays)."""
        serializable = []
        for m in all_metrics:
            entry = {k: v for k, v in m.items() if k not in ("y_proba", "fpr", "tpr", "y_pred", "y_true")}
            serializable.append(entry)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
