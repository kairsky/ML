# Final Comparison: Classical ML vs Deep Learning

*Table 4: End-to-end project comparison (representative quick-run results; full training improves scores).*

| Criterion | Classical ML (Covertype, RF/k-NN) | Dim. Reduction (PCA 95%) | MLP (Fashion-MNIST) | CNN Scratch (CIFAR-10) | Transfer (ResNet18) |
|-----------|-----------------------------------|---------------------------|---------------------|------------------------|---------------------|
| **Typical accuracy** | ~0.85–0.92 (subset) | Δ vs full features in `pca_comparison.json` | ~0.88+ (30 epochs) | ~0.62+ (3 epochs quick) / 0.75+ (30 epochs) | ~0.86 test (quick fine-tune) |
| **Training time** | Minutes (CPU) | +embedding plots (~min) | 10–30 min (CPU/GPU) | 30+ min GPU recommended | Head: ~3 min; fine-tune: ~6 min (subset) |
| **Model size** | 1–50 MB (.joblib) | PCA matrix + classifier | ~2–5 MB (.keras) | ~1.3 MB (325K params) | ~45 MB (11M params) |
| **Inference speed** | Very fast (ms/sample) | Fast (low-dim dot products) | Fast (vectorized) | Moderate (conv ops) | Moderate (224×224) |
| **Interpretability** | High (trees, coefficients) | PCA loadings | Low | Grad-CAM, filters | Grad-CAM |
| **Data needs** | Hundreds–thousands | Same as classifier | Thousands | Thousands+ | Hundreds (with pretrain) |
| **Best for** | Tabular structured data | Visualization, denoising | Vectorized images | Images from scratch | Limited image data |

## Dimensionality Reduction Effects

- **PCA (95% variance):** Reduces feature count; may help or hurt accuracy — see `outputs/metrics/pca_comparison.json`.
- **t-SNE / UMAP:** Excellent for 2D class structure visualization; not recommended as production features without validation.
- **LDA:** Supervised linear projection; strong when classes are Gaussian-like in feature space.

## Recommendations

1. **Tabular:** Start with Logistic Regression baseline → Random Forest / SVM with `GridSearchCV`.
2. **Images:** Use transfer learning when data < 10K; CNN from scratch when data and compute are ample.
3. **Reporting:** Embed figures from `outputs/plots/` into `docs/report/academic_report.md` for PDF export.
