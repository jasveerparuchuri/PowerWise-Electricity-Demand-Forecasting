"""Small orchestration layer for reproducible train/evaluate/report runs."""
from .forecast import run


def run_forecast(output: str = "reports/latest", epochs: int = 300, seed: int = 7,
                 csv_path: str | None = None):
    return run(output=output, epochs=epochs, seed=seed, csv_path=csv_path)
