"""Short runnable classroom demonstrations independent of the forecaster."""
from __future__ import annotations

import numpy as np

from .activations import linear, relu, sigmoid, step, tanh
from .network import sigmoid_neuron_step
from .perceptron import Perceptron


def run() -> None:
    print("Activation examples at [-2, 0, 2]")
    values = np.array([-2., 0., 2.])
    for name, fn in (("step", step), ("sigmoid", sigmoid), ("ReLU", relu), ("tanh", tanh), ("linear", linear)):
        print(f"{name:8s}: {fn(values)}")

    # A neuron calculation: z = 1.5*2 + (-1)*1 + 0.5 = 2.5; ReLU(z)=2.5.
    print("Neuron example: inputs [2, 1], weights [1.5, -1], bias 0.5 => z=2.5, ReLU=2.5")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    for gate, labels in (("AND", [0, 0, 0, 1]), ("OR", [0, 1, 1, 1])):
        p = Perceptron().fit(X, np.array(labels))
        print(f"{gate} predictions: {p.predict(X).tolist()}")

    # Multiclass argmax: three output scores correspond to three classes.
    scores = np.array([[.1, .8, .1], [.7, .2, .1], [.1, .2, .7]])
    print(f"Multiclass predictions from three output scores: {np.argmax(scores, axis=1).tolist()}")
    print(f"Sigmoid backprop single update: {sigmoid_neuron_step(x=1, weight=.5, bias=0, target=1)}")
    image = np.arange(64).reshape(8, 8)
    print(f"An 8x8 grayscale image has shape {image.shape}; flattening gives {image.ravel().shape[0]} input features.")
    # Shape illustration: (batch,6) @ (6,8) -> (batch,8) @ (8,1) -> (batch,1).
    print("Illustrative matrix shapes: X (batch,6), W1 (6,8), b1 (1,8), W2 (8,1), b2 (1,1).")


if __name__ == "__main__":
    run()
