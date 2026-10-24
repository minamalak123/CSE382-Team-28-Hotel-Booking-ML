"""Loading and record-level cleaning of H2.csv (City Hotel).

Everything here is a fixed, row-level rule (no statistics learned from the data), so it
can be applied to the full file BEFORE splitting without leaking information.
"""
import pandas as pd

SOURCE_LABELS = [
    "IsCanceled", "LeadTime", "ArrivalDateYear", "ArrivalDateMonth",
    "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth", "StaysInWeekendNights",
    "StaysInWeekNights", "Adults", "Children", "Babies", "Meal", "Country",
    "MarketSegment", "DistributionChannel", "IsRepeatedGuest",
    "PreviousCancellations", "PreviousBookingsNotCanceled", "ReservedRoomType",
    "AssignedRoomType", "BookingChanges", "DepositType", "Agent", "Company",
    "DaysInWaitingList", "CustomerType", "ADR", "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests", "ReservationStatus", "ReservationStatusDate",
]
# "is_canceled" / "IsCanceled" / "iscanceled" all normalise to the same source label.
_KEY = {lbl.lower(): lbl for lbl in SOURCE_LABELS}


def load_h2(path) -> pd.DataFrame:
    """Read H2.csv, map headers to the spec's source labels, strip padded strings,
    turn the literal 'NULL' (Agent / Company) into real missing values and parse the date."""
    df = pd.read_csv(path)
    df.columns = [_KEY.get(c.replace("_", "").lower(), c) for c in df.columns]
    missing = set(SOURCE_LABELS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV does not match the H2 schema, missing: {sorted(missing)}")
    df = df[SOURCE_LABELS].copy()
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].astype(str).str.strip().where(df[c].notna())   # "HB       " -> "HB"
            df[c] = s.mask(s.str.upper() == "NULL")                   # "NULL"      -> NaN
    df["ReservationStatusDate"] = pd.to_datetime(df["ReservationStatusDate"])
    return df


def clean_records(df: pd.DataFrame, max_valid_adr: float):
    """Remove records that are duplicates or impossible. Returns (clean_df, report).

    1. Exact duplicate rows are dropped (keep first). The file has no booking id, so a
       repeated row cannot be told apart from a copy; keeping them would put identical
       rows in train AND test and inflate every score (indirect leakage).
    2. Bookings with zero guests (Adults + Children + Babies == 0) are impossible stays.
    3. ADR < 0 or ADR > max_valid_adr is a data error (the only such row is ADR = 5400,
       ten times the next-highest value).
    """
    report = {"rows_raw": len(df)}
    d = df.drop_duplicates()
    report["exact_duplicates_removed"] = len(df) - len(d)

    guests = d["Adults"] + d["Children"].fillna(0) + d["Babies"]
    bad = (guests == 0) | (d["ADR"] < 0) | (d["ADR"] > max_valid_adr)
    report["zero_guest_rows_removed"] = int((guests == 0).sum())
    report["invalid_adr_rows_removed"] = int(((d["ADR"] < 0) | (d["ADR"] > max_valid_adr)).sum())
    report["rows_removed_invalid_total"] = int(bad.sum())
    d = d.loc[~bad].reset_index(drop=True)
    report["rows_clean"] = len(d)
    report["canceled_rate_raw"] = round(float(df["IsCanceled"].mean()), 4)
    report["canceled_rate_clean"] = round(float(d["IsCanceled"].mean()), 4)
    return d, report
