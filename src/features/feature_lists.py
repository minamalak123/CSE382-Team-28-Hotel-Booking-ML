"""
PREDICTION POINT


Tiers:
  safe       known at confirmation -> used where relevant
  uncertain  plausibly known at confirmation, but the stored value can be
             updated later; kept by default, dropped when strict=True
             (run both settings as an ablation in the report)
  leak       written after the prediction point -> never a feature
  target     a task's label
  dropped    available but not useful (redundant / ~all missing)

"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PREDICTION_POINT = (
    "Booking confirmation: the moment the hotel confirms the reservation to the "
    "customer. Only information final at that moment may be used."
)

TASKS = ("regression", "classification", "clustering")
TARGETS = {"regression": "ADR", "classification": "IsCanceled", "clustering": None}

COLUMN_DECISIONS: dict[str, tuple[str, str]] = {
    "IsCanceled": ("target", "Classification target. Happens after confirmation, so it is "
                             "excluded from regression features; used only to profile clusters."),
    "ADR": ("target", "Regression target. The agreed nightly price is known at confirmation, "
                      "so it is a valid feature for classification and clustering."),
    "LeadTime": ("safe", "Days between booking entry and arrival, fixed at booking."),
    "ArrivalDateYear": ("safe", "Known. Used for ADR only (price trend 86.1->103.8->116.9 by year); "
                                "no cancel signal and only partial years, so not used elsewhere."),
    "ArrivalDateMonth": ("safe", "Planned arrival month, known."),
    "ArrivalDateDayOfMonth": ("safe", "Planned arrival day, known."),
    "StaysInWeekendNights": ("safe", "Planned stay length, known."),
    "StaysInWeekNights": ("safe", "Planned stay length, known."),
    "Adults": ("safe", "Party size, known."),
    "Children": ("safe", "Party size, known (3 missing -> impute 0)."),
    "Babies": ("safe", "Party size, known."),
    "Meal": ("safe", "Board type chosen at booking."),
    "MarketSegment": ("safe", "Sales segment, known."),
    "DistributionChannel": ("safe", "Booking channel, known."),
    "IsRepeatedGuest": ("safe", "Guest history before this booking."),
    "PreviousCancellations": ("safe", "Guest history before this booking."),
    "PreviousBookingsNotCanceled": ("safe", "Guest history before this booking."),
    "ReservedRoomType": ("safe", "Room type requested at booking."),
    "DepositType": ("safe", "Deposit policy agreed at booking. Note: 'Non Refund' cancels 99.8% "
                            "of the time, a known dataset quirk, not a timing leak."),
    "Agent": ("safe", "Travel agent ID, known (NULL = direct booking -> 'None')."),
    "DaysInWaitingList": ("safe", "Days on the waiting list BEFORE confirmation, so final at "
                                  "the prediction point."),
    "CustomerType": ("safe", "Contract / Transient / Group, known."),
    "TotalOfSpecialRequests": ("uncertain", "Requests can be added after booking (cancel rate "
                                            "21.4% when >0 vs 55.1% when 0). Kept; ablate."),
    "Country": ("uncertain", "Often only confirmed at check-in (PRT cancels 65.4% vs 26.8% for "
                             "others). Kept by default; dropped in the strict ablation."),
    "ReservationStatus": ("leak", "Final outcome (Canceled / Check-Out / No-Show): a 1:1 copy "
                                  "of IsCanceled."),
    "ReservationStatusDate": ("leak", "Date the final status was set: after the outcome."),
    "AssignedRoomType": ("leak", "Assigned at check-in. Cancel rate 6.4% when it differs from "
                                 "the reserved room vs 45.3% when equal."),
    "BookingChanges": ("leak", "Counts changes up to check-in or cancellation (future)."),
    "RequiredCarParkingSpaces": ("leak", "Recorded at arrival: 0.0% of the 1,141 train bookings "
                                         "with parking > 0 were canceled (a perfect leak)."),
    "ArrivalDateWeekNumber": ("dropped", "Redundant with month + day of month."),
    "Company": ("dropped", "95.2% missing. Replaced by the engineered flag HasCompany."),
}

ENGINEERED = {
    "TotalNights": "StaysInWeekendNights + StaysInWeekNights",
    "TotalGuests": "Adults + Children + Babies",
    "ArrivalDayOfWeek": "Weekday name from arrival year/month/day",
    "HasAgent": "1 if Agent is not missing",
    "HasCompany": "1 if Company is not missing",
}

_BASE_NUMERIC = [
    "LeadTime", "ArrivalDateDayOfMonth", "StaysInWeekendNights", "StaysInWeekNights",
    "Adults", "Children", "Babies", "IsRepeatedGuest", "PreviousCancellations",
    "PreviousBookingsNotCanceled", "DaysInWaitingList",
    "TotalNights", "TotalGuests", "HasAgent", "HasCompany",
]
_BASE_CATEGORICAL = [
    "ArrivalDateMonth", "ArrivalDayOfWeek", "Meal", "MarketSegment",
    "DistributionChannel", "ReservedRoomType", "DepositType", "CustomerType", "Agent",
]

_SPEC = {
    "regression": {
        "numeric": _BASE_NUMERIC + ["ArrivalDateYear", "TotalOfSpecialRequests"],
        "categorical": _BASE_CATEGORICAL + ["Country"],
    },
    "classification": {
        "numeric": _BASE_NUMERIC + ["ADR", "TotalOfSpecialRequests"],
        "categorical": _BASE_CATEGORICAL + ["Country"],
    },
    "clustering": {
        "numeric": ["LeadTime", "TotalNights", "TotalGuests", "ADR", "IsRepeatedGuest",
                    "PreviousBookingsNotCanceled", "TotalOfSpecialRequests"],
        "categorical": ["MarketSegment", "CustomerType", "DepositType"],
    },
}
PROFILE_ONLY = {"clustering": ["IsCanceled"]} 


def get_feature_lists(task: str, strict: bool = False) -> dict:
    """Return {'target', 'numeric', 'categorical'} for a task.
    strict=True also removes the 'uncertain' tier (for the ablation run)."""
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}', expected one of {TASKS}")
    drop = {c for c, (tier, _) in COLUMN_DECISIONS.items() if tier == "uncertain"} if strict else set()
    return {
        "target": TARGETS[task],
        "numeric": [c for c in _SPEC[task]["numeric"] if c not in drop],
        "categorical": [c for c in _SPEC[task]["categorical"] if c not in drop],
    }


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ENGINEERED columns. Row-wise only (no statistics learned), so it is safe
    to apply before the split without leaking information between splits."""
    out = df.copy()
    out["TotalNights"] = out["StaysInWeekendNights"] + out["StaysInWeekNights"]
    out["TotalGuests"] = out["Adults"] + out["Children"].fillna(0) + out["Babies"]
    dates = pd.to_datetime(
        out["ArrivalDateYear"].astype(str) + "-" + out["ArrivalDateMonth"].astype(str)
        + "-" + out["ArrivalDateDayOfMonth"].astype(str),
        format="%Y-%B-%d", errors="raise")
    out["ArrivalDayOfWeek"] = dates.dt.day_name()
    out["HasAgent"] = out["Agent"].notna().astype(int)
    out["HasCompany"] = out["Company"].notna().astype(int)
    return out


def select_features(df: pd.DataFrame, task: str, strict: bool = False):
    """Return (X, y) for a task. y is None for clustering."""
    fl = get_feature_lists(task, strict)
    X = df[fl["numeric"] + fl["categorical"]].copy()
    y = df[fl["target"]].copy() if fl["target"] else None
    return X, y


def validate() -> None:
    """Fail hard if the lists break the audit rules."""
    tiers = {"safe", "uncertain", "leak", "target", "dropped"}
    bad = {c: t for c, (t, _) in COLUMN_DECISIONS.items() if t not in tiers}
    if bad:
        raise ValueError(f"Unknown tiers: {bad}")
    if len(COLUMN_DECISIONS) != 31:
        raise ValueError(f"Expected 31 audited columns, got {len(COLUMN_DECISIONS)}")
    leaks = {c for c, (t, _) in COLUMN_DECISIONS.items() if t in ("leak", "dropped")}
    for task in TASKS:
        fl = get_feature_lists(task)
        feats = fl["numeric"] + fl["categorical"]
        if len(feats) != len(set(feats)):
            raise ValueError(f"{task}: duplicate feature names")
        if set(feats) & leaks:
            raise ValueError(f"{task}: uses leaky/dropped columns {set(feats) & leaks}")
        if fl["target"] in feats:
            raise ValueError(f"{task}: target used as a feature")
        if task == "regression" and "IsCanceled" in feats:
            raise ValueError("regression: IsCanceled is a future outcome")
        unknown = set(feats) - set(COLUMN_DECISIONS) - set(ENGINEERED)
        if unknown:
            raise ValueError(f"{task}: undocumented features {unknown}")


def audit_table() -> pd.DataFrame:
    """One row per raw column: tier, reason and which tasks use it."""
    used = {}
    for t in TASKS:
        fl = get_feature_lists(t)
        used[t] = set(fl["numeric"]) | set(fl["categorical"])
    rows = []
    for col, (tier, reason) in COLUMN_DECISIONS.items():
        row = {"column": col, "tier": tier, "reason": reason}
        for t in TASKS:
            row[t] = "target" if TARGETS[t] == col else ("yes" if col in used[t] else "-")
        rows.append(row)
    return pd.DataFrame(rows)


def leakage_evidence(train: pd.DataFrame) -> pd.DataFrame:
    """Cancel rate inside vs outside suspicious groups, measured on TRAIN only.
    A group near 0% or 100% that is set after confirmation is a leak."""
    checks = {
        "RequiredCarParkingSpaces > 0": train["RequiredCarParkingSpaces"] > 0,
        "AssignedRoomType != ReservedRoomType": train["AssignedRoomType"] != train["ReservedRoomType"],
        "BookingChanges > 0": train["BookingChanges"] > 0,
        "TotalOfSpecialRequests > 0": train["TotalOfSpecialRequests"] > 0,
        "Country == PRT": train["Country"] == "PRT",
        "DepositType == Non Refund": train["DepositType"] == "Non Refund",
        "DaysInWaitingList > 0": train["DaysInWaitingList"] > 0,
    }
    y = train["IsCanceled"]
    rows = [{"check": name, "n_in_group": int(m.sum()),
             "cancel_rate_in_group": round(float(y[m].mean()), 4) if m.any() else np.nan,
             "cancel_rate_rest": round(float(y[~m].mean()), 4) if (~m).any() else np.nan}
            for name, m in checks.items()]
    status = pd.crosstab(train["ReservationStatus"], y)
    rows.append({"check": "ReservationStatus vs IsCanceled (rows mixing labels)",
                 "n_in_group": int((status > 0).sum(axis=1).gt(1).sum()),
                 "cancel_rate_in_group": np.nan, "cancel_rate_rest": np.nan})
    return pd.DataFrame(rows)


def main(out_dir: Path = Path("outputs/features")) -> None:
    validate()
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_table().to_csv(out_dir / "column_audit.csv", index=False)
    payload = {"prediction_point": PREDICTION_POINT, "engineered": ENGINEERED,
               "tasks": {t: {"default": get_feature_lists(t),
                             "strict": get_feature_lists(t, strict=True),
                             "profile_only": PROFILE_ONLY.get(t, [])} for t in TASKS}}
    (out_dir / "feature_lists.json").write_text(json.dumps(payload, indent=2))

    # Evidence is measured on the SAME cleaned train rows the models use (saved split).
    # Imported here, not at the top: split.py -> eda.py would otherwise load at import time.
    from src.data.eda import CONFIG_PATH, load_config, load_h2
    from src.data.split import SPLIT_FILE, clean_rows, load_split
    cfg = load_config(CONFIG_PATH)
    raw = Path(cfg["paths"]["raw_data"])
    if raw.exists() and SPLIT_FILE.exists():
        df = clean_rows(load_h2(raw), cfg["data"]["max_valid_adr"])
        train, _, _ = load_split(df)
        leakage_evidence(train).to_csv(out_dir / "leakage_evidence.csv", index=False)
        add_engineered_features(train)  # smoke test on real data
    else:
        print(f"[skip] needs {raw} and {SPLIT_FILE} (run: python -m src.data.split): "
              "leakage_evidence.csv not written")
    for t in TASKS:
        fl = get_feature_lists(t)
        print(f"{t:15s} target={fl['target']!s:10s} "
              f"{len(fl['numeric'])} numeric + {len(fl['categorical'])} categorical")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
