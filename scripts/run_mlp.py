#!/usr/bin/env python
"""Train MLP on Fashion-MNIST: optimizers, activations, L1/L2 + MSE tracking."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.deep_learning.mlp import MLPTrainer
from src.utils.config import load_config
from src.utils.reproducibility import set_global_seed
from src.visualization.plots import PlotManager


def _build_reg_configs(mlp_cfg: dict) -> dict:
    """Map config regularization block to l1/l2 lambdas per experiment."""
    reg = mlp_cfg.get("regularization", {})
    lam = float(reg.get("l2", 1e-4))
    lam_l1 = float(reg.get("l1", 1e-4))
    return {
        "none": {"l1": 0.0, "l2": 0.0},
        "l2": {"l1": 0.0, "l2": lam},
        "l1": {"l1": lam_l1, "l2": 0.0},
        "l1_l2": {"l1": lam_l1, "l2": lam},
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fewer epochs only")
    args = parser.parse_args()

    config = load_config()
    set_global_seed(config["random_seed"])
    mlp_cfg = config["mlp"]
    epochs = 5 if args.quick else mlp_cfg["epochs"]
    reg_epochs = 8 if args.quick else mlp_cfg["epochs"]

    x_train, y_train, x_test, y_test = MLPTrainer.load_fashion_mnist()
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.15, random_state=config["random_seed"], stratify=y_train
    )

    plots = PlotManager(ROOT / config["paths"]["plots"] / "mlp")
    metrics_dir = ROOT / config["paths"]["metrics"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # --- L1 / L2 / L1+L2 vs none (primary regularization experiment) ---
    print("\n=== MLP Regularization experiment (Loss + MSE + Accuracy) ===\n")
    reg_trainer = MLPTrainer(
        hidden_units=mlp_cfg["hidden_units"],
        dropout_rate=mlp_cfg["dropout_rate"],
    )
    reg_configs = _build_reg_configs(mlp_cfg)
    reg_histories = reg_trainer.compare_regularizers(
        x_train,
        y_train,
        x_val,
        y_val,
        reg_configs=reg_configs,
        epochs=reg_epochs,
        batch_size=mlp_cfg["batch_size"],
        optimizer="adam",
        learning_rate=mlp_cfg["learning_rates"]["adam"],
    )
    reg_summary = reg_trainer.summarize_regularization()
    reg_df = pd.DataFrame(reg_summary)
    print(reg_df.to_string(index=False))
    reg_df.to_csv(metrics_dir / "mlp_regularization_summary.csv", index=False)

    plots.plot_regularization_comparison(
        reg_histories, filename="regularization_l1_l2_mse.png"
    )
    for key, hist in reg_histories.items():
        plots.plot_training_history(
            hist,
            title=f"MLP {key} (loss & accuracy)",
            filename=f"history_{key}.png",
        )

    # --- Optimizers (baseline, no extra reg) ---
    trainer = MLPTrainer(hidden_units=mlp_cfg["hidden_units"], dropout_rate=mlp_cfg["dropout_rate"])
    histories = trainer.compare_optimizers(
        x_train, y_train, x_val, y_val,
        optimizers_config=mlp_cfg["learning_rates"],
        epochs=epochs,
        batch_size=mlp_cfg["batch_size"],
    )
    plots.plot_optimizer_comparison(histories, filename="optimizer_comparison.png")

    # --- Activations ---
    trainer_act = MLPTrainer(hidden_units=mlp_cfg["hidden_units"], dropout_rate=mlp_cfg["dropout_rate"])
    act_histories = trainer_act.compare_activations(
        x_train, y_train, x_val, y_val,
        activations=["relu", "tanh", "sigmoid"],
        epochs=min(epochs, 10),
    )

    metrics_path = metrics_dir / "mlp_histories.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "regularization": reg_histories,
                "regularization_summary": reg_summary,
                "optimizers": histories,
                "activations": act_histories,
                "reg_lambda_config": reg_configs,
            },
            f,
            indent=2,
        )

    print("\nSaved:")
    print(f"  {metrics_path}")
    print(f"  {metrics_dir / 'mlp_regularization_summary.csv'}")
    print(f"  {plots.output_dir / 'regularization_l1_l2_mse.png'}")
    print("MLP training completed.")


if __name__ == "__main__":
    main()
