# Overfitting vs Underfitting

## Definitions

- **Underfitting:** High bias — model too simple to capture data structure. High training and validation error.
- **Overfitting:** High variance — model memorizes training noise. Low training error, high validation error.

## Bias-Variance Decomposition

Expected test error decomposes (squared loss):

$$\mathbb{E}[(y - \hat{f})^2] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

| Phenomenon | Bias | Variance | Train Error | Val Error |
|------------|------|----------|-------------|-----------|
| Underfitting | High | Low | High | High |
| Good fit | Moderate | Moderate | Low | Low |
| Overfitting | Low | High | Very Low | High |

## Mitigation Strategies

| Strategy | Underfitting | Overfitting |
|----------|--------------|-------------|
| Model complexity | Increase | Decrease |
| Regularization (L1/L2) | Reduce λ | Increase λ |
| More data | Helps both | Strongly helps |
| Dropout / early stopping | — | Helps DL |
| Cross-validation | Tune complexity | Select λ, k, depth |
| Ensemble (RF, bagging) | — | Reduces variance |

## Learning Curves

Plot training vs validation metric vs training set size:

- **Underfitting:** Both curves converge high.
- **Overfitting:** Large gap between train and validation.
- **Good model:** Curves converge low with small gap.
