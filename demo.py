#!/usr/bin/env python
"""
Interactive demo for project defense / quick verification.

Shows: data loading, saved model inference, metrics, and sample plots.
Run: python demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.reproducibility import set_global_seed


def _banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def demo_classical() -> None:
    import joblib
    import pandas as pd

    from src.classical_ml.evaluator import ModelEvaluator
    from src.data.loaders import load_covertype, split_train_val_test

    config = load_config()
    set_global_seed(config["random_seed"])

    _banner("DEMO 1: Classical ML (Covertype + Random Forest)")
    X, y = load_covertype(max_samples=3000)
    _, _, X_test, _, _, y_test = split_train_val_test(X, y, random_state=config["random_seed"])
    model_path = ROOT / config["paths"]["models"] / "classical_ml" / "covertype" / "random_forest.joblib"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Run: python scripts/run_classical_ml.py --quick")
        return

    model = joblib.load(model_path)
    metrics = ModelEvaluator().evaluate(model, X_test, y_test, "random_forest")
    print(f"Samples: {len(X_test)} | Classes: {y.nunique()}")
    print(f"Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")
    cmp_path = ROOT / config["paths"]["metrics"] / "covertype_comparison.csv"
    if cmp_path.exists():
        print("\nModel comparison (test set):")
        print(pd.read_csv(cmp_path).to_string(index=False))


def demo_dimensionality() -> None:
    _banner("DEMO 2: PCA comparison metrics")
    path = ROOT / "results" / "metrics" / "pca_comparison.json"
    if path.exists():
        print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2, ensure_ascii=False))
    else:
        print("Run: python scripts/run_dimensionality.py --quick")


def demo_deep_learning() -> None:
    _banner("DEMO 3: Deep learning metrics summary")
    for name in ["cnn_metrics.json", "transfer_learning_metrics.json", "mlp_histories.json"]:
        path = ROOT / "results" / "metrics" / name
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            print(f"\n--- {name} ---")
            if isinstance(data, dict) and "test_accuracy" in data:
                print(f"Test accuracy: {data['test_accuracy']:.4f}")
            elif isinstance(data, dict) and "param_count" in data:
                print(f"Parameters: {data['param_count']:,} | Test acc: {data.get('test_accuracy', 'N/A')}")
            else:
                keys = list(data.keys())[:3]
                print(f"Keys: {keys} ...")


def demo_plots() -> None:
    _banner("DEMO 4: Generated figures (paths)")
    plots = ROOT / "results" / "plots"
    if not plots.exists():
        print("No plots folder. Run pipelines first.")
        return
    for p in sorted(plots.rglob("*.png"))[:12]:
        print(f"  {p.relative_to(ROOT)}")
    extra = list(plots.rglob("*.png"))
    if len(extra) > 12:
        print(f"  ... and {len(extra) - 12} more PNG files")


def main() -> None:
    print("University ML/DL Project — LIVE DEMO")
    print(f"Project root: {ROOT}")
    demo_classical()
    demo_dimensionality()
    demo_deep_learning()
    demo_plots()
    _banner("Next steps")
    print("  Full report: report.pdf")
    print("  Notebooks:  notebooks/02 .. 05")
    print("  All pipelines: python scripts/run_all.py")


if __name__ == "__main__":
    main()
