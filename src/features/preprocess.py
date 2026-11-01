"""
preprocess.py - Reusable impute/encode/scale pipelines per task.

Run from project root:  python -m src.features.preprocess
Everything is FITTED ON TRAIN ONLY; val/test are only transformed.

Complexity (n rows, p output columns):
  fit/transform Time O(n*p), Space O(n*p)  (~47k x 108 floats ≈ 41 MB)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.eda import CONFIG_PATH, load_config, load_h2
from src.data.split import clean_rows, load_split
from src.features.feature_lists import (TASKS, add_engineered_features,
                                        get_feature_lists, select_features)

MIN_FREQ = 0.005          # categories in <0.5% of train rows -> "infrequent"
ART_DIR = Path("artifacts")
OUT_DIR = Path("outputs/preprocess")


def build_preprocessor(task: str, strict: bool = False) -> ColumnTransformer:
    fl = get_feature_lists(task, strict)
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=MIN_FREQ, sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, fl["numeric"]),
        ("cat", categorical, fl["categorical"]),
    ])


def make_model_pipeline(task: str, model, strict: bool = False) -> Pipeline:
    """Preprocessor + model as ONE object: fit(train) learns both, CV stays leak-free."""
    return Pipeline([("prep", build_preprocessor(task, strict)), ("model", model)])


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    df = clean_rows(load_h2(Path(cfg["paths"]["raw_data"])), cfg["data"]["max_valid_adr"])
    train, val, _ = load_split(add_engineered_features(df))   # test stays locked away

    ART_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        X_tr, _ = select_features(train, task)
        X_va, _ = select_features(val, task)
        prep = build_preprocessor(task).fit(X_tr)              # FIT ON TRAIN ONLY
        Z_va = prep.transform(X_va)
        if np.isnan(Z_va).any():
            raise RuntimeError(f"{task}: NaN after preprocessing")
        joblib.dump(prep, ART_DIR / f"preprocessor_{task}.joblib")
        pd.Series(prep.get_feature_names_out(), name="feature").to_csv(
            OUT_DIR / f"feature_names_{task}.csv", index=False)
        print(f"{task:15s} {X_tr.shape[1]:>3} raw cols -> {Z_va.shape[1]:>3} model cols")


if __name__ == "__main__":
    main()