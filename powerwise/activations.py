"""Elementary activation functions used by the teaching examples."""
from __future__ import annotations

import numpy as np


def step(z: np.ndarray | float) -> np.ndarray:
    return (np.asarray(z) >= 0).astype(float)


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    # Avoid overflow while preserving the same mathematical function.
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    out[~positive] = ez / (1.0 + ez)
    return out


def relu(z: np.ndarray | float) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (np.asarray(z) > 0).astype(float)


def tanh(z: np.ndarray | float) -> np.ndarray:
    return np.tanh(z)


def linear(z: np.ndarray | float) -> np.ndarray:
    return np.asarray(z, dtype=float)
