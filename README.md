# University Machine Learning & Deep Learning Project

A complete, reproducible end-to-end ML/DL portfolio covering classical algorithms, dimensionality reduction, MLP/CNN training, and transfer learning — with academic documentation, Jupyter notebooks, and saved artifacts.

## Features

| Component | Description |
|-----------|-------------|
| **Classical ML** | Logistic Regression, Random Forest, SVM, k-NN with `GridSearchCV` / `RandomizedSearchCV` |
| **Datasets** | Wine Quality (UCI), Covertype (multiclass ≥1000 samples) |
| **Preprocessing** | `Pipeline`, `ColumnTransformer`, imputation, `StandardScaler`, `OneHotEncoder` |
| **Splits** | Stratified 70/15/15 train/val/test |
| **Evaluation** | Accuracy, precision, recall, F1, ROC-AUC, confusion matrices, CV mean±std |
| **Dim. Reduction** | PCA, t-SNE, UMAP, LDA + PCA 95% retraining |
| **Deep Learning** | MLP (Fashion-MNIST), CNN (CIFAR-10), ResNet18 transfer learning |
| **Docs** | Academic report, theory derivations, architecture survey |

## Project Structure

```
ML Assik/
├── config/config.yaml          # Hyperparameters and paths
├── src/                        # Modular Python package
│   ├── data/                   # Loaders and preprocessing
│   ├── classical_ml/           # Training, evaluation, theory
│   ├── dimensionality/         # PCA, t-SNE, UMAP, LDA
│   ├── deep_learning/          # MLP, CNN, transfer, Grad-CAM
│   ├── visualization/          # Plotting utilities
│   └── utils/                  # Config, logging, seeds
├── scripts/                    # Runnable pipelines
├── notebooks/                  # Jupyter notebooks 01–06
├── docs/
│   ├── report/academic_report.md
│   ├── theory/                 # Derivations, manual examples
│   └── survey/                 # RNN, LSTM, GAN, etc.
└── outputs/
    ├── models/                 # Saved .joblib / .keras / .pt
    ├── plots/                  # Figures for report
    └── metrics/                # JSON/CSV metrics
```

## Quick Start

### 1. Create environment

```bash
cd "ML Assik"
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Generate notebooks

```bash
python scripts/generate_notebooks.py
```

### 3. Run full pipeline

```bash
# Full training (may take hours on CPU; GPU recommended for CNN/transfer)
python scripts/run_all.py

# Fast smoke test (reduced grids and epochs)
python scripts/run_all.py --quick
```

### 4. Run individual modules

```bash
python scripts/run_classical_ml.py
python scripts/run_dimensionality.py
python scripts/run_mlp.py
python scripts/run_cnn.py
python scripts/run_transfer.py
```

### 5. Jupyter notebooks

```bash
jupyter notebook notebooks/
```

Execute notebooks `01`–`06` in order, or run the embedded script commands.

## Configuration

Edit `config/config.yaml` for:

- Random seed (`42`)
- Split ratios (70/15/15)
- Hyperparameter grids
- Epoch counts, batch sizes, learning rates
- Output directories

## Outputs

After running pipelines:

| Path | Content |
|------|---------|
| `outputs/models/classical_ml/` | Tuned sklearn pipelines (`.joblib`) |
| `outputs/plots/classical_ml/` | ROC curves, confusion matrices, comparisons |
| `outputs/metrics/` | JSON/CSV metrics and comparison tables |
| `outputs/plots/dimensionality/` | PCA, t-SNE, UMAP, LDA embeddings |
| `outputs/models/cnn/` | CIFAR-10 CNN checkpoints |
| `outputs/models/transfer/` | ResNet18 weights |

## Theory & Report

- **Academic report:** `docs/report/academic_report.md` (IEEE/APA-compatible structure; export to PDF via Pandoc)
- **Logistic regression derivation:** `docs/theory/logistic_regression_derivation.md`
- **Manual k-NN example:** `docs/theory/manual_knn_example.md`
- **Overfitting discussion:** `docs/theory/overfitting_underfitting.md`
- **Advanced architectures survey:** `docs/survey/advanced_architectures.md`

## Algorithms Covered

k-NN, Decision Trees, Random Forest, SVM, Logistic Regression, Naive Bayes (theory in `src/classical_ml/theory.py` and notebook 01).

## Deep Learning Notes

- **MLP:** 3+ hidden layers, ReLU/tanh/sigmoid, BatchNorm, Dropout, SGD/Adam/RMSprop
- **CNN:** 3 conv blocks, augmentation, 30 epochs, LR scheduling, checkpoints
- **Transfer:** ResNet18 — freeze → train head → fine-tune

## Reproducibility

All scripts call `set_global_seed(42)` from `src/utils/reproducibility.py`.

## Hardware

- CPU sufficient for classical ML and MLP
- GPU recommended for CNN (30 epochs) and transfer learning
- TensorFlow GPU and PyTorch CUDA auto-detected

## License

Educational / academic use.

## Citation

If using this project academically, cite the included report and reference standard ML literature (Hastie et al.; Goodfellow et al.).
