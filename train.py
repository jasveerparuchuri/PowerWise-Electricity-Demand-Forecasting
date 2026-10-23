"""Command-line entry point for the PowerWise time-aware forecast demo."""
import argparse

from src.pipeline import run_forecast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="reports/latest")
    parser.add_argument("--csv", help="CSV with timestamp,demand,temperature columns")
    args = parser.parse_args()
    print(run_forecast(output=args.output, epochs=args.epochs, seed=args.seed, csv_path=args.csv))


if __name__ == "__main__":
    main()
