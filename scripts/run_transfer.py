#!/usr/bin/env python
"""Transfer learning with ResNet18 on CIFAR-10."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.deep_learning.transfer_learning import TransferLearningTrainer
from src.utils.config import load_config
from src.utils.reproducibility import set_global_seed


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    config = load_config()
    set_global_seed(config["random_seed"])
    tl_cfg = config["transfer"]

    epochs_head = 2 if args.quick else tl_cfg["epochs_head"]
    epochs_finetune = 2 if args.quick else tl_cfg["epochs_finetune"]
    subset = 2000 if args.quick else 10000

    trainer = TransferLearningTrainer(backbone=tl_cfg["backbone"])
    metrics = trainer.run_full_pipeline(
        epochs_head=epochs_head,
        epochs_finetune=epochs_finetune,
        batch_size=tl_cfg["batch_size"],
        lr_head=tl_cfg["learning_rate_head"],
        lr_finetune=tl_cfg["learning_rate_finetune"],
        subset_train=subset,
    )

    out = ROOT / config["paths"]["metrics"] / "transfer_learning_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    trainer.save_model(str(ROOT / config["paths"]["models"] / "transfer" / "resnet18_cifar10.pt"))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
