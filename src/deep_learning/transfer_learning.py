"""Transfer learning with ResNet18 (PyTorch) on CIFAR-10."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms


class TransferLearningTrainer:
    """
    Transfer learning pipeline:
    1. Freeze backbone, train head
    2. Unfreeze, fine-tune full network
    """

    def __init__(self, backbone: str = "resnet18", num_classes: int = 10, device: Optional[str] = None) -> None:
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.train_times: Dict[str, float] = {}
        self.metrics: Dict[str, float] = {}

    def _build_model(self) -> nn.Module:
        if self.backbone_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, self.num_classes)
        elif self.backbone_name == "mobilenet":
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            model = models.mobilenet_v2(weights=weights)
            model.classifier[1] = nn.Linear(model.last_channel, self.num_classes)
        elif self.backbone_name == "vgg16":
            weights = models.VGG16_Weights.IMAGENET1K_V1
            model = models.vgg16(weights=weights)
            model.classifier[6] = nn.Linear(4096, self.num_classes)
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")
        return model.to(self.device)

    def _freeze_backbone(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False

    def _unfreeze_all(self, model: nn.Module) -> None:
        for param in model.parameters():
            param.requires_grad = True

    @staticmethod
    def _prepare_cifar10_loaders(
        batch_size: int = 32,
        subset_train: int = 10000,
        subset_val: int = 2000,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load CIFAR-10 with ImageNet normalization and resize to 224."""
        from torchvision.datasets import CIFAR10

        transform_train = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224, padding=16),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        train_set = CIFAR10(root="data/cifar10", train=True, download=True, transform=transform_train)
        val_set = CIFAR10(root="data/cifar10", train=True, download=True, transform=transform_test)
        test_set = CIFAR10(root="data/cifar10", train=False, download=True, transform=transform_test)

        # Subsample for tractable university runtime
        train_idx = list(range(subset_train))
        val_idx = list(range(subset_train, subset_train + subset_val))
        train_subset = torch.utils.data.Subset(train_set, train_idx)
        val_subset = torch.utils.data.Subset(val_set, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
        return train_loader, val_loader, test_loader

    def _train_epoch(self, model: nn.Module, loader: DataLoader, criterion, optimizer) -> Tuple[float, float]:
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        return total_loss / total, correct / total

    @torch.no_grad()
    def _evaluate(self, model: nn.Module, loader: DataLoader, criterion) -> Tuple[float, float]:
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        return total_loss / total, correct / total

    def run_full_pipeline(
        self,
        epochs_head: int = 10,
        epochs_finetune: int = 20,
        batch_size: int = 32,
        lr_head: float = 1e-3,
        lr_finetune: float = 1e-4,
        subset_train: int = 10000,
    ) -> Dict[str, Any]:
        """Freeze backbone -> train head -> unfreeze -> fine-tune."""
        train_loader, val_loader, test_loader = self._prepare_cifar10_loaders(
            batch_size=batch_size, subset_train=subset_train
        )
        self.model = self._build_model()
        criterion = nn.CrossEntropyLoss()

        # Stage 1: frozen backbone
        self._freeze_backbone(self.model)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr_head)
        t0 = time.time()
        for epoch in range(epochs_head):
            train_loss, train_acc = self._train_epoch(self.model, train_loader, criterion, optimizer)
            val_loss, val_acc = self._evaluate(self.model, val_loader, criterion)
        self.train_times["head_only"] = time.time() - t0
        _, head_val_acc = self._evaluate(self.model, val_loader, criterion)

        # Stage 2: fine-tune all
        self._unfreeze_all(self.model)
        optimizer = optim.Adam(self.model.parameters(), lr=lr_finetune)
        t0 = time.time()
        for epoch in range(epochs_finetune):
            train_loss, train_acc = self._train_epoch(self.model, train_loader, criterion, optimizer)
            val_loss, val_acc = self._evaluate(self.model, val_loader, criterion)
        self.train_times["finetune"] = time.time() - t0
        test_loss, test_acc = self._evaluate(self.model, test_loader, criterion)

        param_count = sum(p.numel() for p in self.model.parameters())
        self.metrics = {
            "head_val_accuracy": head_val_acc,
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "param_count": param_count,
            "train_time_head_sec": self.train_times["head_only"],
            "train_time_finetune_sec": self.train_times["finetune"],
        }
        return self.metrics

    def save_model(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
