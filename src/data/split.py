"""One reproducible train / validation / test split, computed once and saved.

All tasks (regression, classification, clustering, EDA, tuning) load the SAME rows from
data/processed/, so a config change can never silently give each module a different test set.
The test set is only read by the final-evaluation code (Milestone 2).
"""
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config, path_for
from src.data.loader import clean_records, load_h2


def make_splits(df: pd.DataFrame, cfg: dict):
    """Stratified (on IsCanceled) split. Fractions are of the FULL clean data: 60/20/20."""
    seed = cfg["project"]["random_seed"]
    test_size = cfg["validation"]["test_size"]
    val_size = cfg["validation"]["validation_size"]
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["IsCanceled"], random_state=seed)
    train, val = train_test_split(
        train_val, test_size=val_size / (1.0 - test_size),
        stratify=train_val["IsCanceled"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def build_and_save(cfg: dict | None = None):
    cfg = cfg or load_config()
    raw = load_h2(path_for(cfg, "raw_data"))
    exp = (cfg["data"]["expected_rows"], cfg["data"]["expected_columns"])
    if raw.shape != exp:
        raise ValueError(f"Expected {exp}, got {raw.shape}")
    clean, report = clean_records(raw, cfg["data"]["max_valid_adr"])
    train, val, test = make_splits(clean, cfg)

    out = path_for(cfg, "processed")
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_pickle(out / f"{name}.pkl")
        report[f"rows_{name}"] = len(part)
        report[f"canceled_rate_{name}"] = round(float(part["IsCanceled"].mean()), 4)
    (out / "split_summary.json").write_text(json.dumps(report, indent=2))
    return train, val, test, report


def load_splits(cfg: dict | None = None):
    """Return (train, val, test); builds and saves them on first use."""
    cfg = cfg or load_config()
    out = path_for(cfg, "processed")
    if not all((out / f"{n}.pkl").exists() for n in ("train", "val", "test")):
        train, val, test, _ = build_and_save(cfg)
        return train, val, test
    return tuple(pd.read_pickle(out / f"{n}.pkl") for n in ("train", "val", "test"))


if __name__ == "__main__":
    _, _, _, rep = build_and_save()
    print(json.dumps(rep, indent=2))
