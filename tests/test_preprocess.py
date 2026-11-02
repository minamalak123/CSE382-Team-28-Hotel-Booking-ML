
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.features.feature_lists import (TASKS, add_engineered_features,
                                        get_feature_lists, select_features)
from src.features.preprocess import build_preprocessor, make_model_pipeline

RAW = Path("data/raw/H2.csv")
SPLIT = Path("artifacts/split_indices.csv")
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def fake_bookings(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Fake rows with every raw column the feature lists need, plus a few
    missing values and one rare category, like the real data."""
    rng = np.random.default_rng(seed)
    pick = lambda opts: rng.choice(opts, n)
    df = pd.DataFrame({
        "IsCanceled": (rng.random(n) < 0.4).astype(int),
        "LeadTime": rng.integers(0, 400, n),
        "ArrivalDateYear": pick([2015, 2016, 2017]),
        "ArrivalDateMonth": pick(MONTHS),
        "ArrivalDateDayOfMonth": rng.integers(1, 29, n),
        "StaysInWeekendNights": rng.integers(0, 3, n),
        "StaysInWeekNights": rng.integers(0, 6, n),
        "Adults": rng.integers(1, 4, n),
        "Children": rng.integers(0, 2, n).astype(float),
        "Babies": rng.integers(0, 2, n),
        "Meal": pick(["BB", "HB", "SC"]),
        "Country": pick(["PRT", "GBR", "FRA", "DEU"]),
        "MarketSegment": pick(["Online TA", "Offline TA/TO", "Groups", "Direct"]),
        "DistributionChannel": pick(["TA/TO", "Direct", "Corporate"]),
        "IsRepeatedGuest": rng.integers(0, 2, n),
        "PreviousCancellations": rng.integers(0, 3, n),
        "PreviousBookingsNotCanceled": rng.integers(0, 3, n),
        "ReservedRoomType": pick(["A", "D", "E"]),
        "DepositType": pick(["No Deposit", "Non Refund"]),
        "Agent": pick(["9", "1", "14"]).astype(object),
        "Company": pd.Series([None] * n, dtype=object),
        "DaysInWaitingList": rng.integers(0, 5, n),
        "CustomerType": pick(["Transient", "Contract", "Group"]),
        "ADR": rng.uniform(40, 250, n).round(2),
        "TotalOfSpecialRequests": rng.integers(0, 3, n),
    })
    df.loc[:9, "Children"] = np.nan          # missing numeric values
    df.loc[10:59, "Agent"] = None            # missing categorical values (direct bookings)
    df.loc[60:61, "Country"] = "MDV"         # rare: 2 of 1000 rows = 0.2% < 0.5%
    return add_engineered_features(df)


@pytest.fixture(scope="module")
def train_val():
    df = fake_bookings()
    return df.iloc[:800].copy(), df.iloc[800:].copy()


@pytest.mark.parametrize("task", TASKS)
def test_no_nan_after_transform(task, train_val):
    train, val = train_val
    X_tr, _ = select_features(train, task)
    assert X_tr.isna().any().any() or task == "clustering"   # input really has gaps
    prep = build_preprocessor(task).fit(X_tr)
    assert not np.isnan(prep.transform(X_tr)).any()
    assert not np.isnan(prep.transform(select_features(val, task)[0])).any()


@pytest.mark.parametrize("task", TASKS)
def test_column_counts(task, train_val):
    train, _ = train_val
    X_tr, _ = select_features(train, task)
    prep = build_preprocessor(task).fit(X_tr)
    names = prep.get_feature_names_out()
    fl = get_feature_lists(task)
    assert prep.transform(X_tr).shape[1] == len(names)
    assert sum(n.startswith("num__") for n in names) == len(fl["numeric"])  # 1 col each
    assert sum(n.startswith("cat__") for n in names) >= len(fl["categorical"])


def test_unseen_category_does_not_crash(train_val):
    train, val = train_val
    val = val.copy()
    val["Country"] = "XYZ"                    # never seen in train
    prep = build_preprocessor("classification").fit(select_features(train, "classification")[0])
    Z = prep.transform(select_features(val, "classification")[0])
    assert Z.shape[0] == len(val) and not np.isnan(Z).any()


def test_rare_category_is_pooled(train_val):
    train, _ = train_val
    prep = build_preprocessor("regression").fit(select_features(train, "regression")[0])
    names = list(prep.get_feature_names_out())
    assert "cat__Country_MDV" not in names                 # no own column for 0.2%
    assert "cat__Country_infrequent_sklearn" in names       # pooled instead


def test_fit_on_train_only(train_val):
    """Scaler mean must equal the TRAIN mean, not train+val."""
    train, val = train_val
    val = val.copy()
    val["LeadTime"] = 10_000                  
    X_tr, _ = select_features(train, "classification")
    prep = build_preprocessor("classification").fit(X_tr)
    scaler = prep.named_transformers_["num"].named_steps["scale"]
    i = get_feature_lists("classification")["numeric"].index("LeadTime")
    assert scaler.mean_[i] == pytest.approx(X_tr["LeadTime"].mean())
    prep.transform(select_features(val, "classification")[0])
    assert scaler.mean_[i] == pytest.approx(X_tr["LeadTime"].mean())  


def test_strict_mode_drops_uncertain_columns(train_val):
    train, _ = train_val
    X_tr, _ = select_features(train, "classification", strict=True)
    names = build_preprocessor("classification", strict=True).fit(X_tr).get_feature_names_out()
    assert not any("Country" in n or "TotalOfSpecialRequests" in n for n in names)


def test_model_pipeline_fits_and_predicts(train_val):
    train, val = train_val
    X_tr, y_tr = select_features(train, "classification")
    X_va, _ = select_features(val, "classification")
    pipe = make_model_pipeline("classification", LogisticRegression(max_iter=1000))
    proba = pipe.fit(X_tr, y_tr).predict_proba(X_va)[:, 1]
    assert proba.shape == (len(val),) and ((proba >= 0) & (proba <= 1)).all()


@pytest.fixture(scope="module")
def real_train_val():
    if not (RAW.exists() and SPLIT.exists()):
        pytest.skip("needs data/raw/H2.csv and artifacts/split_indices.csv "
                    "(run: python -m src.data.split)")
    from src.data.eda import load_config, load_h2
    from src.data.split import clean_rows, load_split
    cfg = load_config()
    df = add_engineered_features(clean_rows(load_h2(RAW), cfg["data"]["max_valid_adr"]))
    train, val, _ = load_split(df)
    return train, val


@pytest.mark.parametrize("task,low,high",
                         [("regression", 95, 120), ("classification", 95, 120),
                          ("clustering", 18, 25)])
def test_real_data_shapes(task, low, high, real_train_val):
    """The exact column count depends on which categories pass the 0.5% rare
    threshold in train (e.g. one agent can drop to 'infrequent' when the split
    changes), so check a sensible band and internal consistency, not one number."""
    train, val = real_train_val
    prep = build_preprocessor(task).fit(select_features(train, task)[0])
    Z = prep.transform(select_features(val, task)[0])
    assert Z.shape[0] == len(val)
    assert Z.shape[1] == len(prep.get_feature_names_out())
    assert low <= Z.shape[1] <= high
    assert not np.isnan(Z).any()
