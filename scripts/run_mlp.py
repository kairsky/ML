#!/usr/bin/env python
"""Train MLP on Fashion-MNIST with optimizer and activation comparisons."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.deep_learning.mlp import MLPTrainer
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
    mlp_cfg = config["mlp"]
    epochs = 5 if args.quick else mlp_cfg["epochs"]

    x_train, y_train, x_test, y_test = MLPTrainer.load_fashion_mnist()
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.15, random_state=config["random_seed"], stratify=y_train
    )

    trainer = MLPTrainer(hidden_units=mlp_cfg["hidden_units"], dropout_rate=mlp_cfg["dropout_rate"])
    plots = PlotManager(ROOT / config["paths"]["plots"] / "mlp")
    models_dir = ROOT / config["paths"]["models"] / "mlp"
    models_dir.mkdir(parents=True, exist_ok=True)

    histories = trainer.compare_optimizers(
        x_train, y_train, x_val, y_val,
        optimizers_config=mlp_cfg["learning_rates"],
        epochs=epochs,
        batch_size=mlp_cfg["batch_size"],
    )
    plots.plot_optimizer_comparison(histories, filename="optimizer_comparison.png")

    trainer_act = MLPTrainer(hidden_units=mlp_cfg["hidden_units"], dropout_rate=mlp_cfg["dropout_rate"])
    act_histories = trainer_act.compare_activations(
        x_train, y_train, x_val, y_val,
        activations=["relu", "tanh", "sigmoid"],
        epochs=min(epochs, 10),
    )

    metrics_path = ROOT / config["paths"]["metrics"] / "mlp_histories.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"optimizers": histories, "activations": act_histories}, f, indent=2)

    for key, hist in histories.items():
        plots.plot_training_history(hist, title=f"MLP {key}", filename=f"history_{key}.png")

    print("MLP training completed.")


if __name__ == "__main__":
    main()
