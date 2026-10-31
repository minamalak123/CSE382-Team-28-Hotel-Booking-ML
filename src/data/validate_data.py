
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

    print("\n" + "=" * 60)
    print("TARGET VALIDITY CHECKS")
    print("=" * 60)

    print("\nUnique IsCanceled values:")
    print(sorted(df["IsCanceled"].dropna().unique()))

    invalid_cancellation = df[
    ~df["IsCanceled"].isin([0, 1])
]

    print(
    "Rows with invalid IsCanceled values:",
    len(invalid_cancellation)
)

    invalid_adr = df[
    (df["ADR"] < 0) |
    (~df["ADR"].between(float("-inf"), float("inf")))
]

    print("Rows with negative or non-finite ADR:", len(invalid_adr))

    print("\nTarget validity checks finished.")
    print("\n" + "=" * 60)
    print("NON-NEGATIVE COUNT CHECKS")
    print("=" * 60)

    count_columns = [
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "BookingChanges",
    "DaysInWaitingList",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests"
]

    for column in count_columns:
     negative_count = (df[column] < 0).sum()
     print(f"{column}: {negative_count} negative values")

    print("\nNon-negative count checks finished.")

    
    print("\n" + "=" * 60)
    print("ARRIVAL DATE VALIDITY CHECKS")
    print("=" * 60)

    valid_months = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

    invalid_months = df[
    ~df["ArrivalDateMonth"].isin(valid_months)
]

    print("Invalid arrival month values:", len(invalid_months))

    invalid_weeks = df[
    ~df["ArrivalDateWeekNumber"].between(1, 53)
]

    print("Invalid arrival week numbers:", len(invalid_weeks))
    invalid_days = df[
    ~df["ArrivalDateDayOfMonth"].between(1, 31)
]

    print("Invalid arrival day numbers:", len(invalid_days))

    print("\nArrival date validity checks finished.")


    print("\n" + "=" * 60)
    print("CANCELLATION STATUS CONSISTENCY CHECKS")
    print("=" * 60)

    print("\nReservation status counts:")
    print(df["ReservationStatus"].value_counts(dropna=False).to_string())

    print("\nIsCanceled vs ReservationStatus:")
    print(
        pd.crosstab(
            df["IsCanceled"],
            df["ReservationStatus"],
            dropna=False
        ).to_string()
    )

    print("\nReservation status by cancellation target:")
    print(
        df.groupby(
            ["ReservationStatus", "IsCanceled"],
            dropna=False
        ).size().to_string()
    )

    print("\nCancellation status consistency investigation finished.")


    print("\n" + "=" * 60)
    print("FULL CALENDAR DATE VALIDITY CHECKS")
    print("=" * 60)

    # Convert month names into month numbers.
    month_numbers = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12
    }

    # Build temporary date components.
    date_parts = pd.DataFrame({
        "year": df["ArrivalDateYear"],
        "month": df["ArrivalDateMonth"].map(month_numbers),
        "day": df["ArrivalDateDayOfMonth"]
    })

    # Invalid or missing components cannot form a valid date.
    valid_components = date_parts.notna().all(axis=1)

    # Check whether each complete date is a real calendar date.
    parsed_dates = pd.to_datetime(
        date_parts,
        errors="coerce"
    )

    invalid_dates = parsed_dates.isna() | ~valid_components

    print(f"Invalid or incomplete arrival dates: {invalid_dates.sum()}")

    if invalid_dates.any():
        print("\nSample invalid arrival-date records:")
        print(
            df.loc[
                invalid_dates,
                [
                    "ArrivalDateYear",
                    "ArrivalDateMonth",
                    "ArrivalDateDayOfMonth"
                ]
            ].head(10).to_string(index=False)
        )

    print("\nFull calendar date validity checks finished.")


    
    print("\n" + "=" * 60)
    print("BOOKING LOGICAL CONSISTENCY CHECKS")
    print("=" * 60)

    # Check bookings with zero adults.
    zero_adults = df["Adults"].eq(0)

    print(f"\nBookings with zero adults: {zero_adults.sum()}")

    # Check bookings with zero total nights.
    total_nights = (
        df["StaysInWeekendNights"]
        + df["StaysInWeekNights"]
    )

    zero_nights = total_nights.eq(0)

    print(f"Bookings with zero total nights: {zero_nights.sum()}")

    # Check records with both zero adults and zero nights.
    zero_adults_and_nights = zero_adults & zero_nights

    print(
        "Bookings with both zero adults and zero nights: "
        f"{zero_adults_and_nights.sum()}"
    )

    # Display examples for investigation only.
    suspicious_bookings = zero_adults | zero_nights

    if suspicious_bookings.any():
        print("\nSample potentially unusual bookings:")
        print(
            df.loc[
                suspicious_bookings,
                [
                    "IsCanceled",
                    "Adults",
                    "Children",
                    "Babies",
                    "StaysInWeekendNights",
                    "StaysInWeekNights",
                    "ReservationStatus"
                ]
            ].head(10).to_string(index=False)
        )

    print("\nBooking logical consistency checks finished.")

    print("\nValidation finished.")


if __name__ == "__main__":
    main()
