"""Grad-CAM, activation maps, and filter visualization."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    """Grad-CAM for PyTorch models."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    @staticmethod
    def overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Overlay heatmap on RGB image [H,W,3] in [0,1]."""
        import matplotlib.cm as cm

        heatmap = cm.jet(cam)[:, :, :3]
        if image.max() > 1.0:
            image = image / 255.0
        return alpha * heatmap + (1 - alpha) * image


def plot_predictions_grid(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    output_path: str,
    n: int = 16,
) -> None:
    """Plot correct vs incorrect predictions with confidence."""
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    indices = np.random.choice(len(images), size=min(n, len(images)), replace=False)
    for ax, idx in zip(axes.flat, indices):
        img = images[idx]
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.shape[-1] == 1:
            img = img.squeeze(-1)
        ax.imshow(img.squeeze() if img.ndim == 3 and img.shape[-1] == 1 else img, cmap="gray" if img.ndim == 2 else None)
        correct = y_true[idx] == y_pred[idx]
        conf = probs[idx, y_pred[idx]]
        color = "green" if correct else "red"
        ax.set_title(
            f"T:{class_names[y_true[idx]]}\nP:{class_names[y_pred[idx]]} ({conf:.2f})",
            fontsize=7,
            color=color,
        )
        ax.axis("off")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
