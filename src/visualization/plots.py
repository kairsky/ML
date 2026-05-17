"""Visualization utilities for ML/DL experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree


class PlotManager:
    """Centralized plotting for reports and notebooks."""

    def __init__(self, output_dir: str | Path, style: str = "seaborn-v0_8-whitegrid") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("ggplot")
        sns.set_palette("husl")

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        filename: str = "confusion_matrix.png",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", colorbar=True)
        ax.set_title(title)
        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_roc_curves(
        self,
        fpr_dict: Dict[str, List[float]],
        tpr_dict: Dict[str, List[float]],
        title: str = "ROC Curves",
        filename: str = "roc_curves.png",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(8, 6))
        for label in fpr_dict:
            ax.plot(fpr_dict[label], tpr_dict[label], label=f"Class {label}")
        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_metric_comparison(
        self,
        metrics_df: pd.DataFrame,
        filename: str = "model_comparison.png",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_df.set_index("model")[["accuracy", "precision", "recall", "f1"]].plot(
            kind="bar", ax=ax, rot=0
        )
        ax.set_ylabel("Score")
        ax.set_title("Classifier Performance Comparison")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right")
        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_pca_variance(
        self,
        explained_variance_ratio: np.ndarray,
        filename: str = "pca_explained_variance.png",
    ) -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
        axes[0].set_xlabel("Principal Component")
        axes[0].set_ylabel("Explained Variance Ratio")
        axes[0].set_title("Scree Plot")
        cumsum = np.cumsum(explained_variance_ratio)
        axes[1].plot(range(1, len(cumsum) + 1), cumsum, "o-")
        axes[1].axhline(0.95, color="r", linestyle="--", label="95% threshold")
        axes[1].set_xlabel("Number of Components")
        axes[1].set_ylabel("Cumulative Explained Variance")
        axes[1].set_title("Cumulative Explained Variance")
        axes[1].legend()
        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_embedding_2d(
        self,
        X_emb: np.ndarray,
        y: np.ndarray,
        title: str,
        filename: str,
    ) -> Path:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X_emb[:, 0], X_emb[:, 1], c=y, cmap="tab10", s=8, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label="Class")
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: str = "Training History",
        filename: str = "training_history.png",
    ) -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        if "loss" in history:
            axes[0].plot(history["loss"], label="Train Loss")
            if "val_loss" in history:
                axes[0].plot(history["val_loss"], label="Val Loss")
            axes[0].set_title("Loss")
            axes[0].legend()
        if "accuracy" in history:
            axes[1].plot(history["accuracy"], label="Train Acc")
            if "val_accuracy" in history:
                axes[1].plot(history["val_accuracy"], label="Val Acc")
            axes[1].set_title("Accuracy")
            axes[1].legend()
        fig.suptitle(title)
        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_optimizer_comparison(
        self,
        histories: Dict[str, Dict[str, List[float]]],
        filename: str = "optimizer_comparison.png",
    ) -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for name, hist in histories.items():
            axes[0].plot(hist.get("val_loss", hist.get("loss", [])), label=name)
            axes[1].plot(hist.get("val_accuracy", hist.get("accuracy", [])), label=name)
        axes[0].set_title("Validation Loss")
        axes[1].set_title("Validation Accuracy")
        axes[0].legend()
        axes[1].legend()
        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_decision_tree(
        self,
        tree_estimator,
        feature_names: List[str],
        class_names: List[str],
        filename: str = "decision_tree.png",
        max_depth: int = 3,
    ) -> Path:
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            tree_estimator,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            ax=ax,
            max_depth=max_depth,
            fontsize=8,
        )
        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_heatmap(
        self,
        data: np.ndarray,
        xlabels: List[str],
        ylabels: List[str],
        title: str,
        filename: str,
    ) -> Path:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data, xticklabels=xlabels, yticklabels=ylabels, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
        ax.set_title(title)
        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
