"""Generate all synthetic data for local development."""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synthetic_generator import generate_all_synthetic_data
from src.logging_setup import setup_logging


def main():
    setup_logging()
    print("=" * 60)
    print("  Generating Synthetic Data for GIC Plan-to-Perform")
    print("=" * 60)

    datasets = generate_all_synthetic_data(seed=42)

    print("\nGenerated datasets:")
    for name, df in datasets.items():
        print(f"  {name:30s} → {df.shape[0]:>6} rows × {df.shape[1]:>3} columns")

    print("\nData saved to: data/synthetic/")
    print("Done.")


if __name__ == "__main__":
    main()
