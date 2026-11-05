from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from src.features.preprocess import build_preprocessor

REDUCERS = ("none", "pca", "lda")
PCA_VARIANCE = 0.95


def make_reducer(kind: str, seed: int, n_components=PCA_VARIANCE, **lda_kwargs):
    if kind == "none":
        return None
    if kind == "pca":
        return PCA(n_components=n_components, svd_solver="full", random_state=seed)
    if kind == "lda":
        return LinearDiscriminantAnalysis(n_components=1, **({"solver": "svd"} | lda_kwargs))
    raise ValueError(f"Unknown reducer '{kind}', expected one of {REDUCERS}")


def make_reduced_pipeline(task: str, model, kind: str, seed: int,
                          strict: bool = False, **reducer_kwargs) -> Pipeline:
    if kind == "lda" and task != "classification":
        raise ValueError("LDA needs a class label: use it for classification only")
    steps = [("prep", build_preprocessor(task, strict))]
    reducer = make_reducer(kind, seed, **reducer_kwargs)
    if reducer is not None:
        steps.append((kind, reducer))
    steps.append(("model", model))
    return Pipeline(steps)


def explained_variance_table(task: str, X_train: pd.DataFrame, seed: int,
                             strict: bool = False) -> pd.DataFrame:
    Z = build_preprocessor(task, strict).fit_transform(X_train)
    pca = PCA(svd_solver="full", random_state=seed).fit(Z)
    cum = np.cumsum(pca.explained_variance_ratio_)
    return pd.DataFrame({"component": np.arange(1, len(cum) + 1),
                         "explained_variance_ratio": pca.explained_variance_ratio_,
                         "cumulative": cum})


def components_for(table: pd.DataFrame, share: float) -> int:
    return int(np.searchsorted(table["cumulative"].to_numpy(), share) + 1)