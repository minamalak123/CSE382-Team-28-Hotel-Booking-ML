"""
eda.py - Exploratory Data Analysis for H2 (City Hotel) booking data.

Run from the project root:  python -m src.data.eda

Design rules (CSE382 spec):
  * Uses ONLY the training split -> no peeking at val/test (indirect leakage).
  * Seed and split sizes come from config.yaml (single source of truth).
  * Uses the spec's source labels (IsCanceled, LeadTime, ...) unchanged (no renaming).
  * The split comes from src/data/split.py: cleaned + de-duplicated once, then saved.
  * Cleans the raw file's quirks: padded strings ("HB       ") and the literal
    string "NULL" (Agent / Company) are converted to real missing values.
  * Saves every table/figure to outputs/eda/ so the app and report can reuse them.

Complexity (n rows, d columns):
  Time : O(n*d) summaries/plots/hashing, O(n*d_num^2) correlation matrix,
         O(n log n) stratified split.
  Space: O(n*d)  (79,330 x 31 is only a few MB).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: save files, never open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import load_config, path_for
from src.data.loader import load_h2
from src.data.split import load_splits

OUT_DIR = path_for(load_config(), "outputs") / "eda"
MONTH_ORDER = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


# ---------------------------------------------------------------- helpers
def _save(fig, name: str):
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=150)
    plt.close(fig)


def _row_hashes(df: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(df, index=False)


# ---------------------------------------------------------------- tables
def overview_tables(full, train, val, test):
    """Dimensions, dtypes, missing, duplicates, cardinality, summary stats."""
    info = pd.DataFrame({
        "dtype": train.dtypes.astype(str),
        "n_missing": train.isna().sum(),
        "pct_missing": (train.isna().mean() * 100).round(2),
        "n_unique": train.nunique(),
    })
    info.to_csv(OUT_DIR / "column_overview.csv")
    train.select_dtypes(include="number").describe().T.to_csv(OUT_DIR / "numeric_summary.csv")
    train.select_dtypes(exclude="number").describe().T.to_csv(OUT_DIR / "categorical_summary.csv")

    guests = train["Adults"] + train["Children"].fillna(0) + train["Babies"]
    nights = train["StaysInWeekendNights"] + train["StaysInWeekNights"]

    # Indirect-leakage evidence: identical rows living in BOTH train and val/test.
    train_h = set(_row_hashes(train))
    dup_val = int(_row_hashes(val).isin(train_h).sum())
    dup_test = int(_row_hashes(test).isin(train_h).sum())

    checks = {
        "rows_full": len(full),
        "rows_train": len(train),
        "rows_val": len(val),
        "rows_test": len(test),
        "duplicate_rows_full": int(full.duplicated().sum()),
        "duplicate_rows_train": int(train.duplicated().sum()),
        "val_rows_identical_to_a_train_row": dup_val,
        "test_rows_identical_to_a_train_row": dup_test,
        "ADR_negative": int((train["ADR"] < 0).sum()),
        "ADR_zero": int((train["ADR"] == 0).sum()),
        "ADR_above_1000": int((train["ADR"] > 1000).sum()),
        "zero_guests": int((guests == 0).sum()),
        "zero_nights": int((nights == 0).sum()),
        "LeadTime_above_500": int((train["LeadTime"] > 500).sum()),
        "canceled_rate": round(float(train["IsCanceled"].mean()), 4),
    }
    out = pd.Series(checks, name="value")
    out.to_csv(OUT_DIR / "data_quality_checks.csv")
    print(out.to_string())


# ---------------------------------------------------------------- plots
def plot_missing(train):
    miss = (train.isna().mean() * 100).loc[lambda s: s > 0].sort_values()
    fig, ax = plt.subplots(figsize=(7, 2.5 + 0.4 * len(miss)))
    miss.plot.barh(ax=ax, color="tab:red")
    for i, v in enumerate(miss.values):
        ax.text(v, i, f" {v:.1f}%", va="center")
    ax.set_xlabel("% missing"); ax.set_title("Missing values (train)")
    _save(fig, "01_missing_values.png")


def plot_class_balance(train):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    counts = train["IsCanceled"].value_counts().sort_index()
    ax.bar(["Kept (0)", "Canceled (1)"], counts.values, color=["tab:green", "tab:red"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n({v / counts.sum():.1%})", ha="center", va="bottom")
    ax.set_title("Class balance: IsCanceled (train)")
    _save(fig, "02_class_balance.png")


def plot_adr(train):
    adr = train["ADR"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(adr, bins=80, color="tab:blue"); axes[0].set_title("ADR (raw)")
    axes[1].hist(np.log1p(adr.clip(lower=0)), bins=80, color="tab:blue")
    axes[1].set_title("log(1 + ADR) - skew check")
    sns.boxplot(x=adr, ax=axes[2], color="lightsteelblue")
    axes[2].set_title("ADR outliers (boxplot)")
    _save(fig, "03_adr_distribution.png")


def plot_numeric_distributions(train):
    cols = ["LeadTime", "StaysInWeekendNights", "StaysInWeekNights", "Adults",
            "Children", "BookingChanges", "DaysInWaitingList",
            "PreviousCancellations", "TotalOfSpecialRequests"]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, c in zip(axes.ravel(), cols):
        ax.hist(train[c].dropna(), bins=50, color="tab:purple")
        ax.set_title(f"{c} (skew={train[c].skew():.2f})")
        ax.set_yscale("log")
    fig.suptitle("Numeric distributions (log y-axis exposes the long tails)")
    _save(fig, "04_numeric_distributions.png")


def plot_cancel_rate_by(train, col, name, order=None, top=None):
    rate = train.groupby(col)["IsCanceled"].agg(["mean", "size"])
    if top:
        rate = rate.sort_values("size", ascending=False).head(top)
    rate = rate.reindex(order).dropna() if order else rate.sort_values("mean")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(rate.index.astype(str), rate["mean"], color="tab:orange")
    ax.axhline(train["IsCanceled"].mean(), ls="--", c="k", label="overall rate")
    ax.set_ylabel("cancellation rate"); ax.set_title(f"Cancellation rate by {col}")
    ax.tick_params(axis="x", rotation=45); ax.legend()
    _save(fig, name)


def plot_adr_by_month(train):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.boxplot(data=train[train["ADR"].between(0, 1000)], x="ArrivalDateMonth",
                y="ADR", order=MONTH_ORDER, ax=ax, color="lightsteelblue")
    ax.tick_params(axis="x", rotation=45); ax.set_title("ADR by arrival month")
    _save(fig, "08_adr_by_month.png")


def plot_adr_by_category(train):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    data = train[train["ADR"].between(0, 1000)]
    for ax, c in zip(axes, ["ReservedRoomType", "CustomerType", "MarketSegment"]):
        sns.boxplot(data=data, x=c, y="ADR", ax=ax, color="lightsteelblue")
        ax.tick_params(axis="x", rotation=45); ax.set_title(f"ADR by {c}")
    _save(fig, "09_adr_by_category.png")


def plot_correlation(train):
    corr = train.select_dtypes(include="number").corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, square=True,
                cbar_kws={"shrink": .7})
    ax.set_title("Correlation matrix (numeric, train)")
    _save(fig, "10_correlation_heatmap.png")
    corr[["ADR", "IsCanceled"]].sort_values("ADR").to_csv(
        OUT_DIR / "correlation_with_targets.csv")


def plot_cardinality(train):
    card = train.select_dtypes(exclude="number").nunique().sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    card.plot.barh(ax=ax); ax.set_xscale("log")
    ax.set_title("Category cardinality (log scale)")
    _save(fig, "11_cardinality.png")


# ---------------------------------------------------------------- main
def run_eda(config_path: Path = None):
    cfg = load_config(config_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    raw = load_h2(path_for(cfg, "raw_data"))      # only used to REPORT what cleaning removed
    train, val, test = load_splits(cfg)           # cleaned + de-duplicated, saved once
    print(f"split sizes -> train {len(train)}, val {len(val)}, test {len(test)}")

    overview_tables(raw, train, val, test)
    plot_missing(train)
    plot_class_balance(train)
    plot_adr(train)
    plot_numeric_distributions(train)
    plot_cancel_rate_by(train, "ArrivalDateMonth", "05_cancel_by_month.png", MONTH_ORDER)
    plot_cancel_rate_by(train, "DepositType", "06_cancel_by_deposit.png")
    plot_cancel_rate_by(train, "MarketSegment", "07_cancel_by_segment.png")
    plot_adr_by_month(train)
    plot_adr_by_category(train)
    plot_correlation(train)
    plot_cardinality(train)
    print(f"EDA done -> {OUT_DIR.resolve()}")


if __name__ == "__main__":
    run_eda()