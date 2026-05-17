# Survey: Advanced Deep Learning Architectures

## Summary Comparison Table

| Architecture | Core Innovation | Advantages | Limitations | Applications |
|--------------|-----------------|------------|-------------|--------------|
| RNN | Sequential hidden state $h_t = f(h_{t-1}, x_t)$ | Variable-length sequences | Vanishing gradients, slow parallelization | Time series, early NLP |
| LSTM | Gated memory cell (forget/input/output gates) | Long-range dependencies | Complex, slower than Transformers | Speech, machine translation |
| GAN | Adversarial min-max: Generator vs Discriminator | Realistic synthetic data | Training instability, mode collapse | Image generation, augmentation |
| Transformers | Self-attention $\text{softmax}(QK^T/\sqrt{d})V$ | Parallelizable, SOTA NLP/Vision | Quadratic memory in sequence length | LLMs, ViT, multimodal |
| GNN | Message passing on graph structure | Relational inductive bias | Scalability on large graphs | Molecules, social networks, recommendations |
| Autoencoder | Unsupervised reconstruction $\|x - \hat{x}\|^2$ | Dimensionality reduction, denoising | Latent space may be non-smooth | Anomaly detection, compression |
| VAE | Probabilistic latent $z \sim \mathcal{N}(\mu, \sigma)$ + KL regularization | Generative, smooth latent space | Blurry reconstructions vs GANs | Generative modeling, representation learning |

---

## RNN (Recurrent Neural Networks)

**Architecture:** Hidden state propagates across time steps; output depends on history.

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b), \quad y_t = W_{hy} h_t$$

**Innovation:** Parameter sharing across time for sequential modeling.

**Advantages:** Natural for sequences; any input length.

**Limitations:** Vanishing/exploding gradients; limited long-term memory.

**Applications:** Stock prediction, sensor streams, character-level language models.

---

## LSTM (Long Short-Term Memory)

**Architecture:** Cell state $C_t$ with gates controlling information flow.

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t]), \quad i_t = \sigma(W_i \cdot [h_{t-1}, x_t])$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tanh(W_C \cdot [h_{t-1}, x_t])$$

**Innovation:** Explicit memory highway mitigating vanishing gradients.

**Advantages:** Captures long dependencies better than vanilla RNN.

**Limitations:** More parameters; largely superseded by Transformers for NLP.

**Applications:** Speech recognition, video captioning, time-series forecasting.

---

## GAN (Generative Adversarial Networks)

**Architecture:** Generator $G(z)$ vs Discriminator $D(x)$ in adversarial game.

$$\min_G \max_D \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1 - D(G(z)))]$$

**Innovation:** Implicit generative modeling without explicit likelihood.

**Advantages:** Sharp, realistic samples for images.

**Limitations:** Mode collapse, training instability, no likelihood estimate.

**Applications:** Style transfer, super-resolution, synthetic data for training.

---

## Transformers

**Architecture:** Multi-head self-attention + feed-forward blocks.

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Innovation:** Global receptive field; parallel training.

**Advantages:** State-of-the-art across modalities; scalable with pretraining.

**Limitations:** $O(n^2)$ attention cost; needs large data for training from scratch.

**Applications:** ChatGPT, BERT, Vision Transformers (ViT), protein folding (AlphaFold uses attention variants).

---

## GNN (Graph Neural Networks)

**Architecture:** Node embeddings updated via neighborhood aggregation.

$$h_v^{(l+1)} = \text{UPDATE}\left(h_v^{(l)}, \text{AGG}\left(\{h_u^{(l)} : u \in \mathcal{N}(v)\}\right)\right)$$

**Innovation:** Operates on irregular relational structure.

**Advantages:** Incorporates topology; permutation invariant aggregations.

**Limitations:** Depth = oversmoothing; large graphs need sampling (GraphSAGE).

**Applications:** Drug discovery, fraud detection, traffic networks.

---

## Autoencoders

**Architecture:** Encoder $z = f(x)$, decoder $\hat{x} = g(z)$; minimize reconstruction loss.

**Innovation:** Unsupervised learned representations.

**Advantages:** Simple; effective for denoising and compression.

**Limitations:** Latent space not necessarily continuous or generative.

**Applications:** Anomaly detection, dimensionality reduction, pretraining.

---

## Variational Autoencoders (VAE)

**Architecture:** Encoder outputs $\mu(x), \sigma(x)$; sample $z \sim \mathcal{N}(\mu, \sigma)$; KL divergence regularizes latent.

$$\mathcal{L} = \|x - g(z)\|^2 + D_{KL}(q(z|x) \| p(z))$$

**Innovation:** Principled probabilistic latent variable model.

**Advantages:** Smooth latent space; principled generation.

**Limitations:** Blurry images compared to GANs; ELBO gap.

**Applications:** Molecular generation, representation learning, semi-supervised learning.
