"""Leakage-aware utilities and a clearly labeled synthetic hourly dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_demand(hours: int = 24 * 90, seed: int = 11) -> pd.DataFrame:
    """Create illustrative hourly kWh; this is not measured utility data."""
    rng = np.random.default_rng(seed)
    timestamp = pd.date_range("2024-01-01", periods=hours, freq="h")
    hour = timestamp.hour.to_numpy()
    dow = timestamp.day_of_week.to_numpy()
    temperature = 20 + 8 * np.sin(2 * np.pi * np.arange(hours) / (24 * 30))
    daily = 12 * np.sin(2 * np.pi * (hour - 7) / 24)
    weekend = np.where(dow >= 5, -5, 0)
    demand = 100 + daily + weekend + 0.7 * temperature + rng.normal(0, 2, hours)
    return pd.DataFrame({"timestamp": timestamp, "demand": demand, "temperature": temperature})


def make_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Predict current demand from its previous hour and known calendar/weather inputs."""
    data = frame.copy().sort_values("timestamp")
    ts = pd.to_datetime(data["timestamp"])
    X = pd.DataFrame({
        "previous_demand": data["demand"].shift(1),
        "hour_sin": np.sin(2 * np.pi * ts.dt.hour / 24),
        "hour_cos": np.cos(2 * np.pi * ts.dt.hour / 24),
        "weekday_sin": np.sin(2 * np.pi * ts.dt.day_of_week / 7),
        "weekday_cos": np.cos(2 * np.pi * ts.dt.day_of_week / 7),
        "temperature": data["temperature"],
    }, index=data.index)
    y = data["demand"].rename("demand")
    valid = X.notna().all(axis=1)
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)


def chronological_split(X: pd.DataFrame, y: pd.Series, train: float = .70,
                        validation: float = .15):
    n = len(X)
    a, b = int(n * train), int(n * (train + validation))
    if not (0 < a < b < n):
        raise ValueError("split fractions must leave non-empty train, validation, and test sets")
    return (X.iloc[:a], y.iloc[:a], X.iloc[a:b], y.iloc[a:b], X.iloc[b:], y.iloc[b:])


class Standardizer:
    """Small training-only standardizer; fit only on the training partition."""
    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, float) - self.mean_) / self.scale_


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual, predicted = np.asarray(actual).ravel(), np.asarray(predicted).ravel()
    error = actual - predicted
    denom = np.sum((actual - np.mean(actual)) ** 2)
    return {"MAE": float(np.mean(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "R2": float(1 - np.sum(error ** 2) / denom) if denom else float("nan")}
