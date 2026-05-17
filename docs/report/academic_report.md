# End-to-End Machine Learning and Deep Learning: A Comparative Empirical Study

**Author:** [Student Name]  
**Institution:** [University Name]  
**Course:** Machine Learning & Deep Learning  
**Date:** May 2026  

---

## Abstract

This report presents a comprehensive empirical study spanning classical machine learning, dimensionality reduction, and deep neural architectures. We evaluate logistic regression, random forests, support vector machines, and k-nearest neighbors on UCI Wine Quality and Covertype datasets using rigorous preprocessing (70/15/15 splits, `ColumnTransformer`, hyperparameter search with cross-validation). Dimensionality reduction via PCA, t-SNE, UMAP, and LDA is analyzed, including retraining with 95% variance retention. Deep learning experiments include multi-layer perceptrons on Fashion-MNIST, convolutional networks on CIFAR-10, and ResNet18 transfer learning. Results demonstrate trade-offs among accuracy, training time, model size, and interpretability. Classical ensembles excel on tabular data; CNNs and transfer learning dominate image classification.

**Keywords:** supervised learning, ensemble methods, PCA, convolutional neural networks, transfer learning, hyperparameter optimization

---

## 1. Introduction

### 1.1 Motivation

Machine learning systems must balance predictive performance, computational cost, and interpretability. University curricula traditionally separate classical ML from deep learning; this project unifies both in a reproducible pipeline suitable for portfolio and academic assessment.

### 1.2 Objectives

1. Implement and compare ≥4 classical classifiers with full evaluation metrics.
2. Study manifold learning and PCA-based feature compression.
3. Train MLP and CNN architectures with systematic optimizer and activation analysis.
4. Conduct transfer learning with staged fine-tuning.
5. Survey advanced architectures (RNN, LSTM, GAN, Transformers, GNN, Autoencoders, VAE).

### 1.3 Contributions

- Modular Python package (`src/`) with type hints, logging, and fixed seeds.
- Executable scripts and Jupyter notebooks producing saved models and figures.
- Documented mathematical derivations and manual k-NN worked example.

---

## 2. Background and Related Work

### 2.1 Classical Supervised Learning

Instance-based (k-NN), tree-based (decision trees, random forests), margin-based (SVM), and probabilistic linear (logistic regression, naive Bayes) methods form the foundation of tabular prediction [1,2].

### 2.2 Dimensionality Reduction

PCA finds orthogonal directions of maximum variance [3]. t-SNE and UMAP preserve local neighborhood structure for visualization [4,5]. LDA maximizes between-class separability [6].

### 2.3 Deep Learning

MLPs universalize function approximation; CNNs exploit spatial locality [7]. Transfer learning leverages ImageNet-pretrained features for data-efficient fine-tuning [8].

---

## 3. Methodology

### 3.1 Datasets

| ID | Dataset | Type | Samples | Classes | Source |
|----|---------|------|---------|---------|--------|
| A | Wine Quality | Tabular (UCI) | ~6,497 | Multi (quality) | OpenML #287 |
| B | Covertype | Tabular | 15,000 (subset) | 7 | sklearn |
| C | Fashion-MNIST | Images 28×28 | 70,000 | 10 | Keras |
| D | CIFAR-10 | Images 32×32×3 | 60,000 | 10 | Keras / torchvision |

### 3.2 Preprocessing Pipeline

1. **Missing values:** median (numeric), mode (categorical).
2. **Encoding:** `OneHotEncoder(handle_unknown='ignore')`.
3. **Scaling:** `StandardScaler` on numeric features.
4. **Split:** stratified 70% train / 15% validation / 15% test.
5. **Imbalance:** `class_weight='balanced'` when minority ratio < 0.3.

### 3.3 Classical Model Training

Hyperparameters tuned via `GridSearchCV` (logistic regression, RF, k-NN) and `RandomizedSearchCV` (SVM). Scoring: weighted F1. Five-fold stratified cross-validation.

### 3.4 Evaluation Metrics

Accuracy, precision, recall, F1 (weighted), ROC-AUC (OvR), confusion matrices, CV mean ± std.

### 3.5 Dimensionality Reduction

PCA scree plots; 2D embeddings (PCA, t-SNE, UMAP, LDA). Best classifier (Random Forest) retrained on PCA features retaining 95% variance.

### 3.6 Deep Learning

**MLP:** 3 hidden layers (512-256-128), BatchNorm, Dropout 0.3, softmax output. Optimizers: SGD, Adam, RMSprop. Early stopping + `ReduceLROnPlateau`.

**CNN:** Three conv blocks (32→64→128 filters), augmentation (flip, translation, zoom, contrast), 30 epochs, checkpointing.

**Transfer Learning:** ResNet18 — freeze backbone, train FC head (10 epochs), unfreeze, fine-tune (20 epochs).

### 3.7 Reproducibility

`random_seed=42` across Python, NumPy, TensorFlow, PyTorch. Configuration centralized in `config/config.yaml`.

---

## 4. Theoretical Analysis

### 4.1 Algorithm Comparison

*Table 1: Classical algorithm complexity and use cases (see `outputs/metrics/algorithm_comparison_table.csv`).*

### 4.2 Logistic Regression Derivation

See `docs/theory/logistic_regression_derivation.md` for full softmax gradient derivation.

### 4.3 Manual k-NN Example

See `docs/theory/manual_knn_example.md` for step-by-step distance and majority vote calculations.

### 4.4 Overfitting and Underfitting

See `docs/theory/overfitting_underfitting.md`. Learning curves and validation gaps diagnose bias-variance trade-offs.

### 4.5 Deep Learning Theory

**Vanishing gradients:** Saturating activations shrink backpropagated gradients; ReLU and residual connections alleviate.

**Batch normalization:** $\hat{x} = (x - \mu)/\sigma$ stabilizes internal covariate shift.

**Learning rate:** Controls optimization step size; schedulers reduce LR on plateau.

---

## 5. Experiments

### 5.1 Experimental Setup

- Hardware: CPU/GPU as available (`tf.config` memory growth enabled).
- Software: Python 3.11, scikit-learn ≥1.3, TensorFlow ≥2.14, PyTorch ≥2.1.
- Execution: `python scripts/run_all.py` or per-module scripts.

### 5.2 Classical ML Results

Results saved to `outputs/metrics/{dataset}_comparison.csv` and confusion/ROC plots under `outputs/plots/classical_ml/`.

*Figure 1: Model comparison bar chart (generated per dataset).*  
*Caption: Weighted accuracy, precision, recall, and F1 across tuned classifiers.*

### 5.3 PCA Experiment

*Figure 2: PCA explained variance scree and cumulative plot.*  
*Table 2: Performance before vs after 95% PCA (`outputs/metrics/pca_comparison.json`).*

### 5.4 MLP Results

*Figure 3: Optimizer comparison — validation loss and accuracy.*  
Activations ReLU, tanh, sigmoid compared for convergence behavior.

### 5.5 CNN Results

*Figure 4: CIFAR-10 training/validation loss and accuracy curves.*  
*Figure 5: Confusion matrix with per-class metrics.*

Model summary and parameter count in `outputs/metrics/cnn_metrics.json`.

### 5.6 Transfer Learning

*Table 3: ResNet18 staged training — head-only vs fine-tuned test accuracy, training time, parameter count.*

---

## 6. Results and Discussion

### 6.1 Classical vs Deep Learning

| Criterion | Classical ML (RF/SVM) | Deep Learning (CNN/ResNet) |
|-----------|----------------------|----------------------------|
| Tabular accuracy | Strong | Requires feature engineering |
| Image accuracy | Weak without features | Strong |
| Training time | Minutes | Hours (GPU) |
| Model size | MB | 10–50+ MB |
| Inference speed | Fast | Moderate (GPU faster) |
| Interpretability | Trees, coefficients | Grad-CAM, attention |

### 6.2 Dimensionality Reduction Trade-offs

PCA at 95% variance reduces dimensionality and may denoise, but discards discriminative directions not aligned with maximum variance. t-SNE/UMAP excel for visualization but are non-parametric and not recommended for production feature transformation without careful validation.

### 6.3 Generalization

Regularization (L2, dropout, early stopping), data augmentation, and transfer learning improve generalization. Overfitting observed when validation loss diverges from training loss.

---

## 7. Conclusion

This project delivered a reproducible, production-quality ML/DL codebase covering classical algorithms through transfer learning. Random forests and SVMs remain competitive on tabular Covertype data; CNNs and ResNet18 fine-tuning achieve superior image classification. Future work includes automated ML (AutoML), transformer-based tabular models (TabPFN), and deployment via ONNX/TorchServe.

---

## References

[1] Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.  
[2] Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.  
[3] Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: a review and recent developments. *Phil. Trans. R. Soc. A*.  
[4] van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *JMLR*.  
[5] McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection. *arXiv:1802.03426*.  
[6] Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics*.  
[7] LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition. *Proc. IEEE*.  
[8] Yosinski, J., et al. (2014). How transferable are features in deep neural networks? *NeurIPS*.  
[9] He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.  
[10] Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. *ICLR*.

---

## Appendix A: Project Structure

```
ML Assik/
├── config/config.yaml
├── src/                    # Modular package
├── scripts/                # Executable pipelines
├── notebooks/              # Jupyter notebooks 01-06
├── docs/report/            # This report
├── docs/theory/            # Derivations
├── docs/survey/            # Advanced architectures
└── outputs/                # models, plots, metrics
```

## Appendix B: Figure and Table Index

- Table 1: Algorithm comparison  
- Table 2: PCA before/after metrics  
- Table 3: Transfer learning comparison  
- Figure 1–5: Generated under `outputs/plots/`

*Note: Export this document to PDF with Pandoc for IEEE/APA submission; expand page count by embedding generated figures from `outputs/plots/`.*
