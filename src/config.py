"""Default settings shared by the command-line and dashboard entry points."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ForecastConfig:
    epochs: int = 300
    learning_rate: float = 0.005
    hidden_size: int = 16
    batch_size: int = 32
    seed: int = 7
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
