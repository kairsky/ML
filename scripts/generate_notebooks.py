#!/usr/bin/env python
"""Generate Jupyter notebooks with full theory text for Tasks 2-5."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"


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
        json.dump(nb, f, indent=1, ensure_ascii=False)


SETUP = """import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
"""

# Task 2 notebook — expanded theory + pipeline
save(
    notebook(
        [
            md_cell(
                """# Задача 2. Классический конвейер машинного обучения

## Цель
Реализовать полный pipeline: предобработка, обучение 4+ классификаторов, GridSearchCV/RandomizedSearchCV, метрики, визуализации.

## Датасеты
- **Wine Quality (UCI)** — табличные признаки, мультиклассовая целевая переменная качества.
- **Covertype** — ≥15000 образцов, 7 классов лесного покрова.

## Предобработка
1. `SimpleImputer` (медиана / мода)
2. `StandardScaler` для числовых признаков
3. `OneHotEncoder` для категориальных
4. Объединение через `ColumnTransformer` и `Pipeline`
5. Стратифицированное разбиение **70% / 15% / 15%**

## Модели
| Модель | Поиск гиперпараметров | Особенности |
|--------|----------------------|-------------|
| Logistic Regression | GridSearchCV | Вероятности, L2 |
| Random Forest | GridSearchCV | Нелинейность, важность признаков |
| SVM | RandomizedSearchCV | RBF/linear kernel |
| k-NN | GridSearchCV | Требует масштабирования |

## Метрики
Accuracy, Precision, Recall, F1 (weighted), ROC-AUC (OvR), confusion matrix, CV mean±std.

## Переобучение vs недообучение
- **Недообучение:** высокие ошибки на train и val (высокое смещение).
- **Переобучение:** низкая train error, высокая val error (высокая дисперсия).
- **Регуляризация:** `C`, `max_depth`, `k`, `class_weight='balanced'`.
"""
            ),
            code_cell(SETUP + "!python scripts/run_classical_ml.py --quick"),
            code_cell(
                SETUP
                + """import pandas as pd
from IPython.display import display, Image

for name in ['wine_quality', 'covertype']:
    p = ROOT / 'results/metrics' / f'{name}_comparison.csv'
    if p.exists():
        print('===', name, '===')
        display(pd.read_csv(p))

img = ROOT / 'results/plots/classical_ml/covertype/model_comparison.png'
if img.exists():
    display(Image(filename=str(img)))
"""
            ),
        ]
    ),
    NB / "02_classical_ml_pipeline.ipynb",
)

save(
    notebook(
        [
            md_cell(
                """# Задача 3. Снижение размерности

## PCA
Линейная проекция на главные компоненты — собственные векторы ковариационной матрицы. Критерий 95% объяснённой дисперсии выбирает размерность сжатия.

## t-SNE
Сохраняет локальную структуру попарных расстояний; хорош для визуализации, плох для production features без переобучения embedding.

## UMAP
Графовый метод на основе многообразий; быстрее t-SNE на больших n, лучше сохраняет глобальную топологию.

## LDA
Супервизорная линейная проекция, максимизирующая межклассовую дисперсию Фишера.

## Эксперимент
Переобучение Random Forest на PCA-признаках (95% variance) и сравнение с полным набором признаков.
"""
            ),
            code_cell(SETUP + "!python scripts/run_dimensionality.py --quick"),
            code_cell(
                SETUP
                + """import json
from IPython.display import Image, display

with open(ROOT / 'results/metrics/pca_comparison.json', encoding='utf-8') as f:
    display(json.load(f))
for img in ['pca_explained_variance.png','tsne_2d.png','umap_2d.png','lda_2d.png']:
    p = ROOT / 'results/plots/dimensionality' / img
    if p.exists():
        display(Image(filename=str(p)))
"""
            ),
        ]
    ),
    NB / "03_dimensionality_reduction.ipynb",
)

save(
    notebook(
        [
            md_cell(
                """# Задача 4. MLP (Fashion-MNIST)

## Архитектура
Вход 784 → скрытые 512-256-128 → BatchNorm → Dropout 0.3 → Softmax(10).

## Оптимизаторы
- **SGD** + momentum — классический, чувствителен к lr
- **Adam** — адаптивные шаги
- **RMSprop** — скользящее среднее квадратов градиентов

## Теория
**Исчезающие градиенты:** насыщение sigmoid/tanh. **BatchNorm:** стабилизация активаций. **LR scheduling:** ReduceLROnPlateau при плато val_loss.

## Регуляризация
Early stopping, dropout, (опционально) L2 на весах.
"""
            ),
            code_cell(SETUP + "!python scripts/run_mlp.py --quick"),
            code_cell(
                SETUP
                + """from IPython.display import Image, display
p = ROOT / 'results/plots/mlp/optimizer_comparison.png'
if p.exists():
    display(Image(filename=str(p)))
"""
            ),
        ]
    ),
    NB / "04_mlp_experiments.ipynb",
)

save(
    notebook(
        [
            md_cell(
                """# Задача 5. CNN (CIFAR-10) + Transfer Learning (ResNet18)

## CNN с нуля
3 conv-блока (Conv2D, BatchNorm, ReLU, MaxPool, Dropout), аугментация, 30 эпох, Adam, checkpoints.

## Transfer learning
1. Заморозить backbone ImageNet
2. Обучить голову классификатора
3. Разморозить и fine-tune с меньшим lr

## Сравнение
CNN scratch — меньше параметров, ниже accuracy при малых эпохах.
ResNet18 — выше accuracy, больше параметров, дольше обучение.
"""
            ),
            code_cell(SETUP + "!python scripts/run_cnn.py --quick"),
            code_cell(SETUP + "!python scripts/run_transfer.py --quick"),
            code_cell(
                SETUP
                + """import json
from IPython.display import Image, display

for f in ['cnn_metrics.json', 'transfer_learning_metrics.json']:
    p = ROOT / 'results/metrics' / f
    if p.exists():
        print(f, json.load(open(p, encoding='utf-8')))
for img in ['cnn/cnn_training.png', 'cnn/cnn_confusion_matrix.png']:
    p = ROOT / 'results/plots' / img
    if p.exists():
        display(Image(filename=str(p)))
"""
            ),
        ]
    ),
    NB / "05_cnn_and_transfer_learning.ipynb",
)

print("Notebooks generated:", NB)
