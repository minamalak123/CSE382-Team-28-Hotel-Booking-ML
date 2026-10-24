"""Dataset validation: verifies the file against the H2 specification and writes a text report.

Run from the project root:  python -m src.data.validate_data
It uses the shared loader, so padded strings and the literal 'NULL' are handled exactly as in
the rest of the project (otherwise Agent / Company would wrongly look complete).
"""
import numpy as np

from src.config import load_config, path_for
from src.data.loader import load_h2

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
COUNT_COLUMNS = ["LeadTime", "StaysInWeekendNights", "StaysInWeekNights", "Adults", "Children",
                 "Babies", "PreviousCancellations", "PreviousBookingsNotCanceled",
                 "BookingChanges", "DaysInWaitingList", "RequiredCarParkingSpaces",
                 "TotalOfSpecialRequests"]


def run_validation(cfg=None) -> list:
    cfg = cfg or load_config()
    df = load_h2(path_for(cfg, "raw_data"))
    lines, problems = [], []

    def log(msg=""):
        lines.append(msg); print(msg)

    def check(ok, msg):
        log(("PASS  " if ok else "FAIL  ") + msg)
        if not ok:
            problems.append(msg)

    log("=" * 60 + "\nDATASET VALIDATION\n" + "=" * 60)
    check(df.shape == (cfg["data"]["expected_rows"], cfg["data"]["expected_columns"]),
          f"shape is {df.shape} (expected {cfg['data']['expected_rows']} x {cfg['data']['expected_columns']})")
    check(df["IsCanceled"].isin([0, 1]).all(), "IsCanceled contains only 0/1")
    check(df["ADR"].notna().all() and np.isfinite(df["ADR"]).all() and (df["ADR"] >= 0).all(),
          "ADR is finite and non-negative")
    for c in COUNT_COLUMNS:
        check(not (df[c] < 0).any(), f"{c} has no negative values")
    check(df["ArrivalDateMonth"].isin(MONTHS).all(), "ArrivalDateMonth holds valid month names")
    check(df["ArrivalDateWeekNumber"].between(1, 53).all(), "ArrivalDateWeekNumber in 1..53")
    check(df["ArrivalDateDayOfMonth"].between(1, 31).all(), "ArrivalDateDayOfMonth in 1..31")
    check(df["ReservationStatus"].isin(["Check-Out", "Canceled", "No-Show"]).all(),
          "ReservationStatus has the 3 documented values")

    log("\nMissing values after cleaning padded strings and 'NULL':")
    log(df.isna().sum()[lambda s: s > 0].to_string())
    log(f"\nExact duplicate rows: {df.duplicated().sum():,} "
        f"({df.duplicated().mean():.1%}) -> removed by clean_records()")
    log(f"ADR == 0 rows: {(df['ADR'] == 0).sum():,};  ADR > 1000 rows: {(df['ADR'] > 1000).sum()}")
    guests = df["Adults"] + df["Children"].fillna(0) + df["Babies"]
    log(f"Zero-guest rows: {(guests == 0).sum()}")
    log(f"\nValidation finished: {len(problems)} failed check(s).")

    out = path_for(cfg, "outputs") / "eda"
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_report.txt").write_text("\n".join(lines))
    return problems


if __name__ == "__main__":
    run_validation()
