from pathlib import Path

import pandas as pd
import pytest

from src.data.eda import SOURCE_LABELS
from src.features.feature_lists import (
    COLUMN_DECISIONS, TASKS, add_engineered_features, get_feature_lists, validate)

LEAKY = {"ReservationStatus", "ReservationStatusDate", "AssignedRoomType",
         "BookingChanges", "RequiredCarParkingSpaces"}


def test_every_source_column_is_audited():
    assert set(COLUMN_DECISIONS) == set(SOURCE_LABELS)


def test_validate_passes():
    validate()


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("strict", [False, True])
def test_no_leaky_columns(task, strict):
    fl = get_feature_lists(task, strict)
    feats = set(fl["numeric"]) | set(fl["categorical"])
    assert not feats & LEAKY
    assert fl["target"] not in feats


def test_regression_never_sees_cancellation():
    fl = get_feature_lists("regression")
    assert "IsCanceled" not in fl["numeric"] + fl["categorical"]


def test_strict_drops_uncertain():
    for task in TASKS:
        fl = get_feature_lists(task, strict=True)
        assert "Country" not in fl["categorical"]
        assert "TotalOfSpecialRequests" not in fl["numeric"]


def test_engineered_features():
    df = pd.DataFrame({
        "StaysInWeekendNights": [1], "StaysInWeekNights": [2], "Adults": [2],
        "Children": [float("nan")], "Babies": [1], "ArrivalDateYear": [2016],
        "ArrivalDateMonth": ["July"], "ArrivalDateDayOfMonth": [1],
        "Agent": [None], "Company": ["40"]})
    out = add_engineered_features(df)
    assert out.loc[0, "TotalNights"] == 3
    assert out.loc[0, "TotalGuests"] == 3
    assert out.loc[0, "ArrivalDayOfWeek"] == "Friday"   
    assert out.loc[0, "HasAgent"] == 0 and out.loc[0, "HasCompany"] == 1


def test_main_runs_end_to_end(tmp_path):
    """Catches broken imports inside main() (e.g. a renamed split function).
    Without data/split it still runs and must skip the evidence file cleanly."""
    from src.features.feature_lists import main
    main(out_dir=tmp_path)
    assert (tmp_path / "column_audit.csv").exists()
    assert (tmp_path / "feature_lists.json").exists()
    from src.data.split import SPLIT_FILE
    if SPLIT_FILE.exists() and Path("data/raw/H2.csv").exists():
        assert (tmp_path / "leakage_evidence.csv").exists()
