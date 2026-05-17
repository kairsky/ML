#!/usr/bin/env python
"""Generate all Jupyter notebooks for the university project."""

from __future__ import annotations

import json
from pathlib import Path


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


def notebook(cells: list) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }


def save(nb: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)


ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

# 01 - Theory
save(
    notebook(
        [
            md_cell("# 01 — Theoretical Foundations of Classical ML\n\nThis notebook documents algorithm theory, comparison tables, and bias-variance analysis."),
            code_cell(
                """import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))

import pandas as pd
from IPython.display import Markdown, display
from src.classical_ml.theory import ALGORITHM_THEORY, get_comparison_table

for name, info in ALGORITHM_THEORY.items():
    display(Markdown(f\"## {name}\\n- **Intuition:** {info['intuition']}\\n- **Decision rule:** {info['decision_rule']}\\n- **Training:** {info['training_complexity']}\\n- **Inference:** {info['inference_complexity']}\"))

df = pd.DataFrame(get_comparison_table())
display(df)
df.to_csv(ROOT / 'outputs/metrics/algorithm_comparison_notebook.csv', index=False)
"""
            ),
            md_cell("See `docs/theory/logistic_regression_derivation.md` and `docs/theory/manual_knn_example.md` for full mathematical derivations."),
        ]
    ),
    NB / "01_theoretical_foundations.ipynb",
)

# 02 - Classical ML
save(
    notebook(
        [
            md_cell("# 02 — Classical ML Pipeline\n\nDatasets: **Wine Quality (UCI)** and **Covertype (multiclass, ≥1000 samples)**."),
            code_cell(
                """import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))

# Run full pipeline (use --quick in terminal for faster execution)
!python scripts/run_classical_ml.py --quick
"""
            ),
            code_cell(
                """import pandas as pd
from IPython.display import display, Image

metrics_wine = pd.read_csv(ROOT / 'outputs/metrics/wine_quality_comparison.csv')
metrics_cov = pd.read_csv(ROOT / 'outputs/metrics/covertype_comparison.csv')
display(metrics_wine, metrics_cov)
display(Image(filename=str(ROOT / 'outputs/plots/classical_ml/covertype/model_comparison.png')))
"""
            ),
        ]
    ),
    NB / "02_classical_ml_pipeline.ipynb",
)

# 03 - Dimensionality
save(
    notebook(
        [
            md_cell("# 03 — Dimensionality Reduction\n\nPCA, t-SNE, UMAP, LDA with 2D visualizations and PCA 95% retraining."),
            code_cell("!python scripts/run_dimensionality.py --quick"),
            code_cell(
                """import json
from pathlib import Path
from IPython.display import Image, display
with open(ROOT / 'outputs/metrics/pca_comparison.json') as f:
    display(json.load(f))
for img in ['pca_explained_variance.png','tsne_2d.png','umap_2d.png','lda_2d.png','pca_2d.png']:
    p = ROOT / 'outputs/plots/dimensionality' / img
    if p.exists():
        display(Image(filename=str(p)))
"""
            ),
        ]
    ),
    NB / "03_dimensionality_reduction.ipynb",
)

# 04 - MLP
save(
    notebook(
        [
            md_cell(
                """# 04 — MLP on Fashion-MNIST

## Theoretical Notes
- **Vanishing gradients:** Sigmoid/tanh saturate; gradients → 0 in deep nets. ReLU mitigates.
- **Batch normalization:** Normalizes activations; stabilizes training, allows higher LR.
- **Learning rate:** Too high → divergence; too low → slow convergence. Schedulers adapt LR.
"""
            ),
            code_cell("!python scripts/run_mlp.py --quick"),
            code_cell(
                """from IPython.display import Image, display
display(Image(filename=str(ROOT / 'outputs/plots/mlp/optimizer_comparison.png')))
"""
            ),
        ]
    ),
    NB / "04_mlp_experiments.ipynb",
)

# 05 - CNN
save(
    notebook(
        [
            md_cell("# 05 — CNN from Scratch (CIFAR-10)\n\n≥3 conv blocks, augmentation, 30 epochs (use `--quick` for smoke test)."),
            code_cell("!python scripts/run_cnn.py --quick"),
            code_cell(
                """import json
from IPython.display import Image, display
with open(ROOT / 'outputs/metrics/cnn_metrics.json') as f:
    print(json.load(f))
display(Image(filename=str(ROOT / 'outputs/plots/cnn/cnn_training.png')))
display(Image(filename=str(ROOT / 'outputs/plots/cnn/cnn_confusion_matrix.png')))
"""
            ),
        ]
    ),
    NB / "05_cnn_from_scratch.ipynb",
)

# 06 - Transfer
save(
    notebook(
        [
            md_cell("# 06 — Transfer Learning (ResNet18)\n\nFreeze backbone → train head → fine-tune full network."),
            code_cell("!python scripts/run_transfer.py --quick"),
            code_cell(
                """import json
from IPython.display import display
with open(ROOT / 'outputs/metrics/transfer_learning_metrics.json') as f:
    display(json.load(f))
"""
            ),
        ]
    ),
    NB / "06_transfer_learning.ipynb",
)

print("Notebooks generated in", NB)
