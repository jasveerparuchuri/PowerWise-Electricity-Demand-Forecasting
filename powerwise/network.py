"""Small vectorized MLP with explicit NumPy feedforward/backpropagation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .activations import relu, relu_derivative


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Flatten both arrays so (n,) targets and (n, 1) predictions do not broadcast to (n, n).
    return float(np.mean((np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()) ** 2))


@dataclass
class TrainingHistory:
    train_loss: list[float]
    validation_loss: list[float]


class NumpyMLP:
    """One ReLU hidden layer and one linear output, trained with mini-batch SGD."""
    def __init__(self, input_size: int, hidden_size: int = 16, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.normal(0, np.sqrt(2 / hidden_size), (hidden_size, 1))
        self.b2 = np.zeros((1, 1))

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        X = np.asarray(X, dtype=float)
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        prediction = a1 @ self.W2 + self.b2
        return prediction, (X, z1, a1)

    def gradients(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        pred, (X, z1, a1) = self.forward(X)
        n = len(X)
        dz2 = 2.0 * (pred - y) / n
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        dz1 = (dz2 @ self.W2.T) * relu_derivative(z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        return dW1, db1, dW2, db2

    def train_batch(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> None:
        grads = self.gradients(X, y)
        for param, grad in zip((self.W1, self.b1, self.W2, self.b2), grads):
            param -= learning_rate * grad

    def fit(self, X: np.ndarray, y: np.ndarray, *, epochs: int = 300,
            learning_rate: float = 0.01, batch_size: int = 32,
            seed: int = 7, validation: tuple[np.ndarray, np.ndarray] | None = None) -> TrainingHistory:
        X, y = np.asarray(X, float), np.asarray(y, float).reshape(-1, 1)
        if len(X) != len(y) or not len(X):
            raise ValueError("X and y must contain the same non-zero number of rows")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        rng = np.random.default_rng(seed)
        history = TrainingHistory([], [])
        for _ in range(epochs):
            order = rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                idx = order[start:start + batch_size]  # includes final partial batch
                self.train_batch(X[idx], y[idx], learning_rate)
            history.train_loss.append(mse(y, self.forward(X)[0]))
            if validation is not None:
                vx, vy = validation
                history.validation_loss.append(mse(vy, self.forward(vx)[0]))
        return history


def sigmoid_neuron_step(x: float, weight: float, bias: float, target: float,
                        learning_rate: float = 0.1) -> dict[str, float]:
    """One explicit sigmoid/MSE chain-rule update for the syllabus."""
    z = weight * x + bias
    a = float(1 / (1 + np.exp(-z)))
    dloss_da = 2 * (a - target)
    da_dz = a * (1 - a)
    dloss_dw = dloss_da * da_dz * x
    return {"z": z, "activation": a, "gradient_weight": dloss_dw,
            "updated_weight": weight - learning_rate * dloss_dw}
