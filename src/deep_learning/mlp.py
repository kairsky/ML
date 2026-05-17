"""Multi-Layer Perceptron with TensorFlow/Keras."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers, regularizers


def mse_one_hot(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
  MSE between predicted class probabilities and one-hot encoded labels.

  Useful to compare calibration alongside cross-entropy (classification loss).
  """
    y_true = tf.cast(y_true, tf.int32)
    num_classes = tf.shape(y_pred)[-1]
    y_true_oh = tf.one_hot(y_true, depth=num_classes)
    return tf.reduce_mean(tf.square(y_true_oh - y_pred))


class MLPTrainer:
    """Train MLP on Fashion-MNIST with optimizers, activations, and L1/L2 regularization."""

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

    @staticmethod
    def _kernel_regularizer(
        reg_type: str,
        l1_lambda: float,
        l2_lambda: float,
    ) -> Optional[regularizers.Regularizer]:
        """Build Keras kernel regularizer for Dense layers."""
        reg_type = reg_type.lower()
        if reg_type == "none" or (l1_lambda == 0 and l2_lambda == 0):
            return None
        if reg_type == "l1":
            return regularizers.L1(l1_lambda)
        if reg_type == "l2":
            return regularizers.L2(l2_lambda)
        if reg_type == "l1_l2":
            return regularizers.L1L2(l1=l1_lambda, l2=l2_lambda)
        raise ValueError(f"Unknown regularization type: {reg_type}")

    def build_model(
        self,
        input_dim: int = 784,
        num_classes: int = 10,
        activation: str = "relu",
        optimizer_name: str = "adam",
        learning_rate: float = 0.001,
        reg_type: str = "none",
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
    ) -> keras.Model:
        """Build MLP with >=3 hidden layers, dropout, batch norm, softmax output."""
        kernel_reg = self._kernel_regularizer(reg_type, l1_lambda, l2_lambda)
        act_layer = layers.Activation(activation)
        inputs = keras.Input(shape=(input_dim,))
        x = layers.Dense(self.hidden_units[0], kernel_regularizer=kernel_reg)(inputs)
        x = layers.BatchNormalization()(x)
        x = act_layer(x)
        x = layers.Dropout(self.dropout_rate)(x)

        for units in self.hidden_units[1:]:
            x = layers.Dense(units, kernel_regularizer=kernel_reg)(x)
            x = layers.BatchNormalization()(x)
            x = act_layer(x)
            x = layers.Dropout(self.dropout_rate)(x)

        outputs = layers.Dense(num_classes, activation="softmax")(x)
        name = f"mlp_{activation}_{optimizer_name}_reg{reg_type}"
        model = keras.Model(inputs=inputs, outputs=outputs, name=name)

        opt_map = {
            "sgd": optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
            "adam": optimizers.Adam(learning_rate=learning_rate),
            "rmsprop": optimizers.RMSprop(learning_rate=learning_rate),
        }
        optimizer = opt_map.get(optimizer_name.lower(), optimizers.Adam(learning_rate=learning_rate))

        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy", mse_one_hot],
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
        reg_type: str = "none",
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
        checkpoint_dir: Optional[str] = None,
        early_stopping_patience: int = 5,
        run_key: Optional[str] = None,
    ) -> Tuple[keras.Model, keras.callbacks.History]:
        """Train with LR scheduling and early stopping."""
        model = self.build_model(
            activation=activation,
            optimizer_name=optimizer,
            learning_rate=learning_rate,
            reg_type=reg_type,
            l1_lambda=l1_lambda,
            l2_lambda=l2_lambda,
        )
        key = run_key or f"{optimizer}_{activation}_reg_{reg_type}"
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

    def compare_regularizers(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        reg_configs: Dict[str, Dict[str, float]],
        epochs: int = 30,
        batch_size: int = 128,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        activation: str = "relu",
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Compare L1, L2, L1+L2 and no regularization.

        reg_configs example:
            {"none": {"l1": 0, "l2": 0}, "l2": {"l1": 0, "l2": 1e-4}, ...}
        """
        self.histories = {}
        for reg_name, lambdas in reg_configs.items():
            l1 = float(lambdas.get("l1", 0.0))
            l2 = float(lambdas.get("l2", 0.0))
            self.train(
                x_train,
                y_train,
                x_val,
                y_val,
                optimizer=optimizer,
                activation=activation,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                reg_type=reg_name,
                l1_lambda=l1,
                l2_lambda=l2,
                run_key=f"reg_{reg_name}",
            )
        return self.histories

    def summarize_regularization(self) -> List[Dict[str, Any]]:
        """Final-epoch metrics table for regularization runs."""
        rows = []
        for key, hist in self.histories.items():
            rows.append(
                {
                    "model": key,
                    "final_train_loss": hist["loss"][-1],
                    "final_val_loss": hist["val_loss"][-1],
                    "final_train_mse": hist["mse_one_hot"][-1],
                    "final_val_mse": hist["val_mse_one_hot"][-1],
                    "final_val_accuracy": hist["val_accuracy"][-1],
                    "best_val_accuracy": max(hist["val_accuracy"]),
                    "epochs_run": len(hist["loss"]),
                }
            )
        return rows

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
