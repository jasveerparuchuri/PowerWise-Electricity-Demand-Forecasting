"""Run the end-to-end, time-aware forecasting demonstration."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import (Standardizer, chronological_split, make_features,
                   make_synthetic_demand, regression_metrics)
from .network import NumpyMLP


def run(output: str = "outputs", epochs: int = 300, seed: int = 7) -> dict[str, dict[str, float]]:
    frame = make_synthetic_demand()
    X, y = make_features(frame)
    xt, yt, xv, yv, xe, ye = chronological_split(X, y)
    scaler = Standardizer().fit(xt.to_numpy())
    train_x, val_x, test_x = (scaler.transform(part.to_numpy()) for part in (xt, xv, xe))
    model = NumpyMLP(train_x.shape[1], hidden_size=16, seed=seed)
    history = model.fit(train_x, yt.to_numpy(), epochs=epochs, learning_rate=.005,
                        batch_size=32, seed=seed, validation=(val_x, yv.to_numpy()))
    pred = model.forward(test_x)[0].ravel()
    baseline = xe["previous_demand"].to_numpy()
    scores = {"persistence_baseline": regression_metrics(ye, baseline),
              "numpy_mlp": regression_metrics(ye, pred)}
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(history.train_loss, label="Train"); ax.plot(history.validation_loss, label="Validation")
    ax.set(title="Training and validation MSE", xlabel="Epoch", ylabel="MSE (kWh²)"); ax.legend(); fig.tight_layout(); fig.savefig(out / "loss.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4)); ax.plot(ye.to_numpy()[:168], label="Actual"); ax.plot(pred[:168], label="NumPy MLP"); ax.plot(baseline[:168], label="Persistence", alpha=.75)
    ax.set(title="Test-set demand: first seven days", xlabel="Test hour", ylabel="Demand (kWh)"); ax.legend(); fig.tight_layout(); fig.savefig(out / "predictions.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3)); ax.plot(ye.to_numpy() - pred); ax.axhline(0, color="black", linewidth=.8)
    ax.set(title="Test prediction errors", xlabel="Test hour", ylabel="Actual − predicted (kWh)"); fig.tight_layout(); fig.savefig(out / "errors.png"); plt.close(fig)
    with (out / "metrics.txt").open("w") as f:
        f.write("Illustrative synthetic dataset; these are not real utility results.\n")
        for name, metrics in scores.items(): f.write(f"{name}: {metrics}\n")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="outputs"); parser.add_argument("--epochs", type=int, default=300); parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(); print(run(args.output, args.epochs, args.seed))
