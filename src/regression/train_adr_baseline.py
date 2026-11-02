import time
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.eda import CONFIG_PATH, load_config, load_h2
from src.data.split import clean_rows, load_split
from src.features.feature_lists import add_engineered_features, select_features
from src.features.preprocess import make_model_pipeline

OUT = Path("outputs/regression")


def evaluate(name, model, X_tr, y_tr, X_va, y_va):
    t0 = time.perf_counter()
    pipe = make_model_pipeline("regression", model).fit(X_tr, y_tr)
    row = {"model": name, "fit_seconds": round(time.perf_counter() - t0, 2)}
    for split, X, y in (("train", X_tr, y_tr), ("val", X_va, y_va)):
        p = pipe.predict(X)
        row[f"{split}_MAE"] = mean_absolute_error(y, p)
        row[f"{split}_RMSE"] = mean_squared_error(y, p) ** 0.5
        row[f"{split}_R2"] = r2_score(y, p)
    return row


def main():
    cfg = load_config(CONFIG_PATH)
    df = add_engineered_features(
        clean_rows(load_h2(Path(cfg["paths"]["raw_data"])), cfg["data"]["max_valid_adr"]))
    train_df, val_df, _ = load_split(df)            # test stays locked
    X_tr, y_tr = select_features(train_df, "regression")
    X_va, y_va = select_features(val_df, "regression")

    models = {"dummy_mean": DummyRegressor(strategy="mean"),
              "linear_regression": LinearRegression()}
    res = pd.DataFrame([evaluate(n, m, X_tr, y_tr, X_va, y_va) for n, m in models.items()])

    OUT.mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(OUT / "baseline_val.csv", index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()