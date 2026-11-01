from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data.eda import CONFIG_PATH, load_config, load_h2

SPLIT_FILE = Path("artifacts/split_indices.csv")
SUMMARY_FILE = Path("outputs/split/split_summary.csv")


def clean_rows(df: pd.DataFrame, max_adr: float) -> pd.DataFrame:
    """Drop impossible rows. Judged row by row, so it is safe before splitting.
    The original index (raw row number) is KEPT as the row ID."""
    guests = df["Adults"] + df["Children"].fillna(0) + df["Babies"]
    bad = (guests == 0) | (df["ADR"] < 0) | (df["ADR"] > max_adr)
    return df[~bad]


def make_groups(df: pd.DataFrame) -> pd.Series:
    """Same group ID for identical rows (hash of all 31 columns)."""
    row_hash = pd.util.hash_pandas_object(df, index=False)
    return pd.Series(pd.factorize(row_hash)[0], index=df.index)


def make_split(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Return 'train'/'val'/'test' for each row ID.
    Step 1: 5 folds -> fold 0 = test (20%).
    Step 2: remaining 80% into 4 folds -> fold 0 = val (20% of total)."""
    seed = cfg["project"]["random_seed"]
    y, groups = df["IsCanceled"], make_groups(df)

    n_test = round(1 / cfg["validation"]["test_size"])                     # 5
    rest_share = 1 - cfg["validation"]["test_size"]                        # 0.8
    n_val = round(rest_share / cfg["validation"]["validation_size"])       # 4

    outer = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=seed)
    trval_pos, test_pos = next(outer.split(df, y, groups))

    inner = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=seed)
    tr_rel, va_rel = next(inner.split(df.iloc[trval_pos], y.iloc[trval_pos],
                                      groups.iloc[trval_pos]))

    labels = pd.Series("train", index=df.index, name="split")
    labels.iloc[test_pos] = "test"
    labels.iloc[trval_pos[va_rel]] = "val"
    return labels


def check_no_overlap(df: pd.DataFrame, labels: pd.Series) -> None:
    """Fail hard if any identical row appears in two splits."""
    groups = make_groups(df)
    sets = {s: set(groups[labels == s]) for s in ("train", "val", "test")}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        shared = sets[a] & sets[b]
        if shared:
            raise RuntimeError(f"{len(shared)} duplicate groups shared by {a} and {b}")


def load_split(df: pd.DataFrame):
    """Return (train, val, test) using the SAVED split, so every script uses the same rows."""
    labels = pd.read_csv(SPLIT_FILE, index_col="row_id")["split"]
    df = df.loc[labels.index]
    return tuple(df[labels == s] for s in ("train", "val", "test"))


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    df = clean_rows(load_h2(Path(cfg["paths"]["raw_data"])), cfg["data"]["max_valid_adr"])
    labels = make_split(df, cfg)
    check_no_overlap(df, labels)

    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    labels.rename_axis("row_id").to_csv(SPLIT_FILE)

    summary = (df.assign(split=labels).groupby("split")
                 .agg(rows=("IsCanceled", "size"), cancel_rate=("IsCanceled", "mean")))
    summary["share"] = summary["rows"] / len(df)
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(SUMMARY_FILE)
    print(summary.round(4))


if __name__ == "__main__":
    main()