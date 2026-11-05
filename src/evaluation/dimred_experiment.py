from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from src.classification.train_cancel_baseline import score_split
from src.data.eda import CONFIG_PATH, load_config, load_h2
from src.data.split import clean_rows, load_split
from src.features.dim_reduction import (components_for, explained_variance_table,
                                        make_reduced_pipeline)
from src.features.feature_lists import add_engineered_features, select_features

OUT = Path("outputs/dimred")


def _models(task: str, seed: int):
    if task == "classification":
        return {"logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=seed),
                "knn": lambda: KNeighborsClassifier(n_neighbors=15, n_jobs=-1)}
    return {"linear_regression": lambda: LinearRegression(),
            "knn": lambda: KNeighborsRegressor(n_neighbors=15, n_jobs=-1)}


def _score(task, pipe, X, y) -> dict:
    if task == "classification":
        return score_split(pipe, X, y)
    p = pipe.predict(X)
    return {"MAE": mean_absolute_error(y, p), "RMSE": mean_squared_error(y, p) ** 0.5,
            "R2": r2_score(y, p)}


def run_one(task, name, pipe, X_tr, y_tr, X_va, y_va, kind) -> dict:
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    row = {"model": name, "features": kind, "fit_seconds": round(time.perf_counter() - t0, 2),
           "n_model_inputs": pipe[:-1].transform(X_va.head(2)).shape[1]}
    for split, X, y in (("train", X_tr, y_tr), ("val", X_va, y_va)):
        row.update({f"{split}_{k}": v for k, v in _score(task, pipe, X, y).items()})
    return row


def plot_variance(table: pd.DataFrame, k90: int, k95: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(table["component"], table["cumulative"], marker=".")
    for share, k in ((0.90, k90), (0.95, k95)):
        ax.axhline(share, ls="--", c="grey", lw=0.8)
        ax.axvline(k, ls=":", c="grey", lw=0.8)
        ax.annotate(f"{share:.0%} -> {k} PCs", (k, share), textcoords="offset points", xytext=(6, -12))
    ax.set(xlabel="Principal components", ylabel="Cumulative explained variance",
           title="PCA on prepared features (fit on TRAIN only)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["classification", "regression"], default="classification")
    task = ap.parse_args().task

    cfg = load_config(CONFIG_PATH)
    seed = cfg["project"]["random_seed"]
    df = add_engineered_features(clean_rows(load_h2(Path(cfg["paths"]["raw_data"])),
                                            cfg["data"]["max_valid_adr"]))
    train, val, _ = load_split(df)
    X_tr, y_tr = select_features(train, task)
    X_va, y_va = select_features(val, task)

    OUT.mkdir(parents=True, exist_ok=True)
    table = explained_variance_table(task, X_tr, seed)
    k90, k95 = components_for(table, 0.90), components_for(table, 0.95)
    table.round(6).to_csv(OUT / f"pca_variance_{task}.csv", index=False)
    plot_variance(table, k90, k95, OUT / f"pca_variance_{task}.png")
    print(f"[{task}] PCs for 90% / 95% train variance: {k90} / {k95} of {len(table)}")

    kinds = ["none", "pca"] + (["lda"] if task == "classification" else [])
    rows = []
    for name, factory in _models(task, seed).items():
        for kind in kinds:
            pipe = make_reduced_pipeline(task, factory(), kind, seed)
            rows.append(run_one(task, name, pipe, X_tr, y_tr, X_va, y_va, kind))
    if task == "classification":
        pipe = make_reduced_pipeline(task, LinearDiscriminantAnalysis(solver="svd"), "none", seed)
        rows.append(run_one(task, "lda_as_classifier", pipe, X_tr, y_tr, X_va, y_va, "none"))

    res = pd.DataFrame(rows).round(4)
    res.to_csv(OUT / f"dimred_comparison_{task}.csv", index=False)
    cols = ["model", "features", "n_model_inputs", "fit_seconds"] + \
           [c for c in res.columns if c.startswith("val_")]
    print(res[cols].to_string(index=False))


if __name__ == "__main__":
    main()