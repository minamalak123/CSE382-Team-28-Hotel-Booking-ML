
from pathlib import Path
import pandas as pd



def main():
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

    duplicate_mask = df.duplicated(keep=False)
    duplicate_rows = df[duplicate_mask]

    print("\n" + "=" * 60)
    print("DUPLICATE ROW INVESTIGATION")
    print("=" * 60)

    print(f"Rows belonging to duplicate groups: {len(duplicate_rows):,}")
    print(
        "Rows remaining after hypothetical exact-duplicate removal: "
        f"{len(df.drop_duplicates()):,}"
    )

    print("\nNumber of duplicate groups by identical row content:")
    print(
        duplicate_rows.value_counts()
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nSample duplicate rows:")
    if not duplicate_rows.empty:
        print(duplicate_rows.head(10).to_string(index=False))
    else:
        print("No duplicate rows found.")


    for target in ["ADR", "IsCanceled"]:
        if target in df.columns:
            print(f"\nTarget summary: {target}")
            print(df[target].describe().to_string())
        else:
            print(f"\nWARNING: Target '{target}' not found.")

    print("\n" + "=" * 60)
    print("ADR INVESTIGATION")
    print("=" * 60)

    print(f"Rows with ADR = 0: {(df['ADR'] == 0).sum():,}")
    print(f"Rows with ADR > 500: {(df['ADR'] > 500).sum():,}")

    print("\nHighest ADR values:")
    print(
        df.nlargest(10, "ADR")[
            ["ADR", "IsCanceled", "LeadTime", "Adults",
             "Children", "ArrivalDateYear", "ArrivalDateMonth"]
        ].to_string(index=False)
    )

    zero_adr = df[df["ADR"] == 0]

    print("\nZero-ADR investigation:")
    print(f"Total rows: {len(zero_adr)}")

    print("\nCancellation status:")
    print(zero_adr["IsCanceled"].value_counts().to_string())

    print("\nGuest counts:")
    print(
        zero_adr[["Adults", "Children", "Babies"]]
        .value_counts()
        .head(10)
        .to_string()
    )

    print("\nRecords by arrival year:")
    print(zero_adr["ArrivalDateYear"].value_counts().sort_index().to_string())
    
    print("\n" + "=" * 60)
    print("MISSING VALUES INVESTIGATION")
    print("=" * 60)

    missing_rows = df[df.isnull().any(axis=1)]

    print(f"Rows containing missing values: {len(missing_rows)}")

    print("\nMissing values by column:")
    print(df.isnull().sum()[df.isnull().sum() > 0].to_string())

    print("\nRows with missing values:")
    print(
        missing_rows[
            ["IsCanceled", "LeadTime", "Adults", "Children",
             "Country", "ADR", "ReservationStatus"]
        ].to_string(index=False)
    )

    print("\nValidation finished.")


if __name__ == "__main__":
    main()
