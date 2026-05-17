#!/usr/bin/env python
"""Train CNN from scratch on CIFAR-10."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.deep_learning.cnn import CNNTrainer
from src.utils.config import load_config
from src.utils.reproducibility import set_global_seed
from src.visualization.plots import PlotManager


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    config = load_config()
    set_global_seed(config["random_seed"])
    cnn_cfg = config["cnn"]
    epochs = 3 if args.quick else cnn_cfg["epochs"]

    x_train, y_train, x_test, y_test = CNNTrainer.load_cifar10()
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.15, random_state=config["random_seed"], stratify=y_train
    )

    trainer = CNNTrainer()
    model = trainer.build_cnn()
    param_count = trainer.count_parameters(model)
    plots_dir = ROOT / config["paths"]["plots"] / "cnn"
    plots_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_architecture_diagram(model, str(plots_dir / "architecture.png"))

    model, history = trainer.train(
        x_train, y_train, x_val, y_val,
        epochs=epochs,
        batch_size=cnn_cfg["batch_size"],
        learning_rate=cnn_cfg["learning_rate"],
        checkpoint_dir=str(ROOT / cnn_cfg["checkpoint_dir"]),
    )

    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    PlotManager(plots_dir).plot_training_history(hist_dict, title="CIFAR-10 CNN", filename="cnn_training.png")

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    plot_mgr = PlotManager(plots_dir)
    plot_mgr.plot_confusion_matrix(cm, class_names=CNNTrainer.CIFAR10_CLASSES, title="CIFAR-10 CNN", filename="cnn_confusion_matrix.png")

    report = classification_report(y_test, y_pred, target_names=CNNTrainer.CIFAR10_CLASSES)
    metrics = {
        "param_count": param_count,
        "test_accuracy": float((y_pred == y_test).mean()),
        "classification_report": report,
    }
    metrics_path = ROOT / config["paths"]["metrics"] / "cnn_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    model_dir = ROOT / config["paths"]["models"] / "cnn"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "cifar10_cnn_final.keras")
    print(f"CNN completed. Parameters: {param_count:,} | Test accuracy: {metrics['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
