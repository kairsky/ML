"""Multi-Layer Perceptron with TensorFlow/Keras."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers


class MLPTrainer:
    """Train MLP on Fashion-MNIST with multiple optimizers and activations."""

    FASHION_MNIST_CLASSES = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]

    def __init__(
        self,
        hidden_units: List[int] = None,
        dropout_rate: float = 0.3,
        random_state: int = 42,
    ) -> None:
        self.hidden_units = hidden_units or [512, 256, 128]
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.models: Dict[str, keras.Model] = {}
        self.histories: Dict[str, Dict[str, List[float]]] = {}

    @staticmethod
    def load_fashion_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and flatten Fashion-MNIST."""
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
        x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
        return x_train, y_train, x_test, y_test

    def build_model(
        self,
        input_dim: int = 784,
        num_classes: int = 10,
        activation: str = "relu",
        optimizer_name: str = "adam",
        learning_rate: float = 0.001,
    ) -> keras.Model:
        """Build MLP with >=3 hidden layers, dropout, batch norm, softmax output."""
        act_layer = layers.Activation(activation)
        inputs = keras.Input(shape=(input_dim,))
        x = layers.Dense(self.hidden_units[0])(inputs)
        x = layers.BatchNormalization()(x)
        x = act_layer(x)
        x = layers.Dropout(self.dropout_rate)(x)

        for units in self.hidden_units[1:]:
            x = layers.Dense(units)(x)
            x = layers.BatchNormalization()(x)
            x = act_layer(x)
            x = layers.Dropout(self.dropout_rate)(x)

        outputs = layers.Dense(num_classes, activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name=f"mlp_{activation}_{optimizer_name}")

        opt_map = {
            "sgd": optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
            "adam": optimizers.Adam(learning_rate=learning_rate),
            "rmsprop": optimizers.RMSprop(learning_rate=learning_rate),
        }
        optimizer = opt_map.get(optimizer_name.lower(), optimizers.Adam(learning_rate=learning_rate))

        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        optimizer: str = "adam",
        activation: str = "relu",
        epochs: int = 30,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        checkpoint_dir: Optional[str] = None,
        early_stopping_patience: int = 5,
    ) -> Tuple[keras.Model, keras.callbacks.History]:
        """Train with LR scheduling and early stopping."""
        model = self.build_model(activation=activation, optimizer_name=optimizer, learning_rate=learning_rate)
        key = f"{optimizer}_{activation}"
        cbs: List[keras.callbacks.Callback] = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
            ),
        ]
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            cbs.append(
                callbacks.ModelCheckpoint(
                    filepath=str(Path(checkpoint_dir) / f"{key}_best.keras"),
                    monitor="val_accuracy",
                    save_best_only=True,
                )
            )

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cbs,
            verbose=1,
        )
        self.models[key] = model
        self.histories[key] = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        return model, history

    def compare_optimizers(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        optimizers_config: Dict[str, float],
        epochs: int = 30,
        batch_size: int = 128,
        activation: str = "relu",
    ) -> Dict[str, Dict[str, List[float]]]:
        """Train with SGD, Adam, RMSprop."""
        for opt_name, lr in optimizers_config.items():
            self.train(
                x_train, y_train, x_val, y_val,
                optimizer=opt_name,
                activation=activation,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=lr,
            )
        return self.histories

    def compare_activations(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        activations: List[str] = None,
        epochs: int = 15,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
    ) -> Dict[str, Dict[str, List[float]]]:
        activations = activations or ["relu", "tanh", "sigmoid"]
        for act in activations:
            self.train(
                x_train, y_train, x_val, y_val,
                optimizer=optimizer,
                activation=act,
                epochs=epochs,
                learning_rate=learning_rate,
            )
        return self.histories
