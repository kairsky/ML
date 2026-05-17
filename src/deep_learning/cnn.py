"""CNN from scratch for CIFAR-10."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers


class CNNTrainer:
    """Custom CNN with 3+ conv blocks for CIFAR-10."""

    CIFAR10_CLASSES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model: Optional[keras.Model] = None
        self.history: Optional[keras.callbacks.History] = None

    @staticmethod
    def load_cifar10() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        return x_train, y_train, x_test, y_test

    def build_augmentation(self) -> keras.Sequential:
        """Data augmentation: flip, crop, color jitter."""
        return keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
                layers.RandomZoom(0.1),
                layers.RandomContrast(0.1),
            ],
            name="augmentation",
        )

    def _conv_block(self, x, filters: int, name: str):
        x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv1")(x)
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)
        x = layers.Activation("relu", name=f"{name}_relu1")(x)
        x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv2")(x)
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)
        x = layers.Activation("relu", name=f"{name}_relu2")(x)
        x = layers.MaxPooling2D(2, name=f"{name}_pool")(x)
        x = layers.Dropout(0.25, name=f"{name}_drop")(x)
        return x

    def build_cnn(self, num_classes: int = 10) -> keras.Model:
        """CNN with 3 convolution blocks + FC head."""
        inputs = keras.Input(shape=(32, 32, 3))
        x = self.build_augmentation()(inputs)
        x = self._conv_block(x, 32, "block1")
        x = self._conv_block(x, 64, "block2")
        x = self._conv_block(x, 128, "block3")
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        x = layers.Dense(256, activation="relu", name="fc1")(x)
        x = layers.BatchNormalization(name="fc_bn")(x)
        x = layers.Dropout(0.5, name="fc_drop")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10_cnn")
        return model

    @staticmethod
    def count_parameters(model: keras.Model) -> int:
        return int(model.count_params())

    @staticmethod
    def save_architecture_diagram(model: keras.Model, path: str) -> None:
        """Save model architecture diagram."""
        try:
            keras.utils.plot_model(model, to_file=path, show_shapes=True, show_layer_names=True)
        except Exception:
            with open(path.replace(".png", ".txt"), "w", encoding="utf-8") as f:
                model.summary(print_fn=lambda x: f.write(x + "\n"))

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 30,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        checkpoint_dir: Optional[str] = None,
    ) -> Tuple[keras.Model, keras.callbacks.History]:
        """Train with LR scheduler, checkpoints, mixed precision optional."""
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass

        self.model = self.build_cnn()
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        cbs: List[callbacks.Callback] = [
            callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        ]
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            cbs.append(
                callbacks.ModelCheckpoint(
                    filepath=str(Path(checkpoint_dir) / "cnn_best.keras"),
                    monitor="val_accuracy",
                    save_best_only=True,
                )
            )

        self.history = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cbs,
            verbose=1,
        )
        return self.model, self.history
