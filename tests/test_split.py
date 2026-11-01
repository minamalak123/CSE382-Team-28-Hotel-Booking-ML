"""
Tests for src/data/split.py.   Run from project root:  python -m pytest tests/test_split.py -q

Part A uses a small fake dataset (always runs, < 1 s).
Part B uses the real data/raw/H2.csv (skipped automatically if the file is missing).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.split import check_no_overlap, clean_rows, make_groups, make_split

CFG = {"project": {"random_seed": 42},
       "validation": {"test_size": 0.20, "validation_size": 0.20}}
RAW = Path("data/raw/H2.csv")


def fake_bookings(n_unique: int = 400, seed: int = 0) -> pd.DataFrame:
    """n_unique distinct bookings, some copied 2-5 times (like group bookings).
    ~40% canceled, mirroring the real data."""
    rng = np.random.default_rng(seed)
    base = pd.DataFrame({
        "LeadTime": np.arange(n_unique),                    # makes every base row unique
        "Adults": rng.integers(1, 4, n_unique),
        "Children": rng.integers(0, 2, n_unique).astype(float),
        "Babies": 0,
        "ADR": rng.uniform(50, 200, n_unique).round(2),
        "IsCanceled": (rng.random(n_unique) < 0.4).astype(int),
    })
    copies = rng.choice([1, 1, 1, 2, 5], size=n_unique)     # some rows get duplicates
    return base.loc[base.index.repeat(copies)].reset_index(drop=True)


# ----------------------------------------------------------- Part A: fake data
def test_clean_rows_removes_impossible_rows():
    df = pd.DataFrame({
        "Adults":   [2, 0, 1, 1],
        "Children": [0.0, 0.0, np.nan, 0.0],
        "Babies":   [0, 0, 0, 0],
        "ADR":      [100.0, 90.0, 5400.0, -5.0],
    })
    out = clean_rows(df, max_adr=1000)
    # row 1 = zero guests, row 2 = ADR 5400, row 3 = negative ADR
    assert list(out.index) == [0]


def test_clean_rows_keeps_original_row_ids():
    df = pd.DataFrame({"Adults": [0, 2, 2], "Children": [0.0] * 3,
                       "Babies": [0] * 3, "ADR": [10.0] * 3}, index=[10, 11, 12])
    assert list(clean_rows(df, 1000).index) == [11, 12]   # IDs not renumbered


def test_identical_rows_get_same_group():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "x"]})
    g = make_groups(df)
    assert g[0] == g[1] and g[0] != g[2]


def test_every_row_gets_exactly_one_label():
    df = fake_bookings()
    labels = make_split(df, CFG)
    assert labels.index.equals(df.index)
    assert set(labels) == {"train", "val", "test"}


def test_no_duplicate_group_crosses_splits():
    df = fake_bookings()
    check_no_overlap(df, make_split(df, CFG))             # raises if a leak exists


def test_overlap_check_catches_a_leak():
    df = pd.DataFrame({"a": [1, 1, 2, 3]})
    leaky = pd.Series(["train", "test", "val", "train"])   # rows 0 and 1 are identical
    with pytest.raises(RuntimeError):
        check_no_overlap(df, leaky)


def test_split_is_reproducible():
    df = fake_bookings()
    pd.testing.assert_series_equal(make_split(df, CFG), make_split(df, CFG))


def test_different_seed_gives_different_split():
    df = fake_bookings()
    other = {**CFG, "project": {"random_seed": 7}}
    assert not make_split(df, CFG).equals(make_split(df, other))


# ----------------------------------------------------------- Part B: real data
@pytest.fixture(scope="module")
def real_split():
    if not RAW.exists():
        pytest.skip(f"{RAW} not found")
    from src.data.eda import load_config, load_h2
    cfg = load_config()
    df = clean_rows(load_h2(RAW), cfg["data"]["max_valid_adr"])
    return df, make_split(df, cfg)


def test_real_rows_after_cleaning(real_split):
    df, _ = real_split
    assert len(df) == 79_162          # 79,330 - 167 zero-guest - 1 ADR outlier


def test_real_shares_are_60_20_20(real_split):
    _, labels = real_split
    shares = labels.value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.60, abs=0.01)
    assert shares["val"] == pytest.approx(0.20, abs=0.01)
    assert shares["test"] == pytest.approx(0.20, abs=0.01)


def test_real_split_is_stratified(real_split):
    df, labels = real_split
    rates = df["IsCanceled"].groupby(labels).mean()
    assert rates.max() - rates.min() < 0.01           #  ~41.8%


def test_real_no_leak(real_split):
    df, labels = real_split
    check_no_overlap(df, labels)
