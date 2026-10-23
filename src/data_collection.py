"""Load a user CSV or create the explicitly synthetic offline demonstration."""
from pathlib import Path

import pandas as pd

from .data import make_synthetic_demand


REQUIRED_COLUMNS = {"timestamp", "demand", "temperature"}


def load_demand_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    if csv_path is None:
        return make_synthetic_demand()
    frame = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["demand"] = pd.to_numeric(frame["demand"], errors="coerce")
    frame["temperature"] = pd.to_numeric(frame["temperature"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "demand", "temperature"])
    return frame.sort_values("timestamp").reset_index(drop=True)
