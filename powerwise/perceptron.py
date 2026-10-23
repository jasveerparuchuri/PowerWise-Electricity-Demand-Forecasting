"""A single-layer binary perceptron for learning logic gates."""
from __future__ import annotations

import numpy as np


class Perceptron:
    def __init__(self, learning_rate: float = 0.1, epochs: int = 30) -> None:
        self.learning_rate, self.epochs = learning_rate, epochs
        self.weights: np.ndarray | None = None
        self.bias = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        X, y = np.asarray(X, float), np.asarray(y, int).reshape(-1)
        self.weights = np.zeros(X.shape[1], dtype=float)
        self.bias = 0.0
        for _ in range(self.epochs):
            for row, target in zip(X, y):
                prediction = int(row @ self.weights + self.bias >= 0)
                error = target - prediction
                self.weights += self.learning_rate * error * row
                self.bias += self.learning_rate * error
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Call fit before predict")
        return (np.asarray(X, float) @ self.weights + self.bias >= 0).astype(int)
