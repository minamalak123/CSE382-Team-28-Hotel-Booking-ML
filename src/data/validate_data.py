
from pathlib import Path

import pandas as pd


def main():
    # Find the project root from this script's location.
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "raw" / "H2.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {data_path}"
        )

    df = pd.read_csv(data_path)

    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    print(f"\nFile: {data_path.name}")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())

    print(f"\nDuplicate rows: {df.duplicated().sum():,}")

    for target in ["ADR", "IsCanceled"]:
        if target in df.columns:
            print(f"\nTarget summary: {target}")
            print(df[target].describe().to_string())
        else:
            print(f"\nWARNING: Target '{target}' not found.")

    print("\nValidation finished.")


if __name__ == "__main__":
    main()
