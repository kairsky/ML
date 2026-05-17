# Mathematical Derivation: Multinomial Logistic Regression

## Problem Setup

Given training data $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{n}$ with $x_i \in \mathbb{R}^d$ and $y_i \in \{1,\ldots,K\}$, multinomial logistic regression models:

$$P(y=k \mid x; W) = \frac{\exp(w_k^\top x)}{\sum_{j=1}^{K} \exp(w_j^\top x)} = \text{softmax}(Wx)_k$$

where $W \in \mathbb{R}^{K \times d}$ and we use one-vs-rest or full softmax with identifiability constraint.

## Loss Function (Cross-Entropy)

For one-hot encoded labels $y_i$:

$$\mathcal{L}(W) = -\frac{1}{n}\sum_{i=1}^{n} \sum_{k=1}^{K} \mathbb{1}[y_i = k] \log P(y_i=k \mid x_i)$$

With L2 regularization ($\lambda > 0$):

$$J(W) = \mathcal{L}(W) + \frac{\lambda}{2}\|W\|_F^2$$

## Gradient Derivation

Let $z_{ik} = w_k^\top x_i$ and $p_{ik} = P(y=k|x_i)$. For softmax:

$$\frac{\partial p_{ik}}{\partial z_{ij}} = p_{ik}(\mathbb{1}[k=j] - p_{ij})$$

The gradient of negative log-likelihood for sample $i$:

$$\frac{\partial \mathcal{L}_i}{\partial z_{ij}} = p_{ij} - \mathbb{1}[y_i = j]$$

Therefore:

$$\nabla_{w_j} \mathcal{L} = \frac{1}{n}\sum_{i=1}^{n} (p_{ij} - \mathbb{1}[y_i = j]) x_i$$

With regularization: $\nabla_{w_j} J = \nabla_{w_j}\mathcal{L} + \lambda w_j$.

## Decision Rule

$$\hat{y} = \arg\max_k P(y=k \mid x) = \arg\max_k \, w_k^\top x$$

## Optimization

Convex in $W$ for fixed $\lambda$. Solved via L-BFGS, SAGA, or stochastic gradient descent with learning rate $\eta_t$:

$$w_j \leftarrow w_j - \eta_t \nabla_{w_j} J$$
