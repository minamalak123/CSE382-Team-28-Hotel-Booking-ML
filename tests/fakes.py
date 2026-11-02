"""Fake H2-shaped data for fast tests (all 31 source columns, no real data needed)."""
import numpy as np
import pandas as pd

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def fake_raw_bookings(n_unique: int = 400, seed: int = 0) -> pd.DataFrame:
    """n_unique distinct bookings in the raw 31-column schema. Some are copied
    2-5 times (like group bookings), and each copy gets a DIFFERENT
    ReservationStatusDate, so copies are 'twins': identical to a model,
    different in the raw file. ~40% canceled, like the real data."""
    rng = np.random.default_rng(seed)
    pick = lambda opts: rng.choice(opts, n_unique)
    base = pd.DataFrame({
        "IsCanceled": (rng.random(n_unique) < 0.4).astype(int),
        "LeadTime": np.arange(n_unique),                      # makes base rows unique
        "ArrivalDateYear": pick([2015, 2016, 2017]),
        "ArrivalDateMonth": pick(MONTHS),
        "ArrivalDateWeekNumber": rng.integers(1, 53, n_unique),
        "ArrivalDateDayOfMonth": rng.integers(1, 29, n_unique),
        "StaysInWeekendNights": rng.integers(0, 3, n_unique),
        "StaysInWeekNights": rng.integers(0, 6, n_unique),
        "Adults": rng.integers(1, 4, n_unique),
        "Children": rng.integers(0, 2, n_unique).astype(float),
        "Babies": 0,
        "Meal": pick(["BB", "HB", "SC"]),
        "Country": pick(["PRT", "GBR", "FRA"]),
        "MarketSegment": pick(["Online TA", "Groups", "Direct"]),
        "DistributionChannel": pick(["TA/TO", "Direct"]),
        "IsRepeatedGuest": rng.integers(0, 2, n_unique),
        "PreviousCancellations": 0,
        "PreviousBookingsNotCanceled": 0,
        "ReservedRoomType": pick(["A", "D"]),
        "AssignedRoomType": pick(["A", "D"]),
        "BookingChanges": 0,
        "DepositType": pick(["No Deposit", "Non Refund"]),
        "Agent": pick(["9", "1"]).astype(object),
        "Company": pd.Series([None] * n_unique, dtype=object),
        "DaysInWaitingList": 0,
        "CustomerType": pick(["Transient", "Group"]),
        "ADR": rng.uniform(50, 200, n_unique).round(2),
        "RequiredCarParkingSpaces": 0,
        "TotalOfSpecialRequests": rng.integers(0, 3, n_unique),
        "ReservationStatus": "Check-Out",
        "ReservationStatusDate": "2016-01-01",
    })
    base["ReservationStatus"] = np.where(base["IsCanceled"] == 1, "Canceled", "Check-Out")
    copies = rng.choice([1, 1, 1, 2, 5], size=n_unique)
    df = base.loc[base.index.repeat(copies)].reset_index(drop=True)
    # copies differ ONLY in a column no model sees -> raw-different, model-identical
    df["ReservationStatusDate"] = pd.date_range("2016-01-01", periods=len(df)).strftime("%Y-%m-%d")
    return df
