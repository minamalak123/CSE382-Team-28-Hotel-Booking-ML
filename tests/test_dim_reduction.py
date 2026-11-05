import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.features.dim_reduction import (components_for, explained_variance_table,
                                        make_reduced_pipeline)
from src.features.feature_lists import select_features
from tests.test_preprocess import fake_bookings


@pytest.fixture(scope="module")
def data():
    df = fake_bookings(1200)
    return df.iloc[:900].copy(), df.iloc[900:].copy()


@pytest.mark.parametrize("kind,step", [("none", None), ("pca", "pca"), ("lda", "lda")])
def test_step_order(kind, step, data):
    pipe = make_reduced_pipeline("classification", LogisticRegression(max_iter=500), kind, 42)
    names = [n for n, _ in pipe.steps]
    assert names == ["prep"] + ([step] if step else []) + ["model"]


def test_lda_gives_one_component_and_scores(data):
    train, val = data
    X_tr, y_tr = select_features(train, "classification")
    X_va, _ = select_features(val, "classification")
    pipe = make_reduced_pipeline("classification", LogisticRegression(max_iter=500), "lda", 42)
    pipe.fit(X_tr, y_tr)
    assert pipe[:-1].transform(X_va).shape == (len(val), 1)
    proba = pipe.predict_proba(X_va)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_lda_rejected_for_regression():
    with pytest.raises(ValueError):
        make_reduced_pipeline("regression", LinearRegression(), "lda", 42)


def test_lda_is_fitted_on_train_only(data):
    train, val = data
    X_tr, y_tr = select_features(train, "classification")
    pipe = make_reduced_pipeline("classification", LogisticRegression(max_iter=500), "lda", 42)
    pipe.fit(X_tr, y_tr)
    Z = pipe.named_steps["prep"].transform(X_tr)
    expected = np.vstack([Z[y_tr.values == c].mean(axis=0) for c in (0, 1)])
    assert np.allclose(pipe.named_steps["lda"].means_, expected)
    before = pipe.named_steps["lda"].scalings_.copy()
    pipe.predict(select_features(val, "classification")[0])
    assert np.array_equal(before, pipe.named_steps["lda"].scalings_)


def test_pca_keeps_requested_variance_on_train(data):
    train, _ = data
    X_tr, y_tr = select_features(train, "regression")
    pipe = make_reduced_pipeline("regression", LinearRegression(), "pca", 42, n_components=0.95)
    pipe.fit(X_tr, y_tr)
    assert pipe.named_steps["pca"].explained_variance_ratio_.sum() >= 0.95


def test_variance_table_and_components_for(data):
    train, _ = data
    X_tr, _ = select_features(train, "regression")
    t = explained_variance_table("regression", X_tr, 42)
    assert t["cumulative"].is_monotonic_increasing and t["cumulative"].iloc[-1] == pytest.approx(1.0)
    k = components_for(t, 0.90)
    assert t["cumulative"].iloc[k - 1] >= 0.90 and (k == 1 or t["cumulative"].iloc[k - 2] < 0.90)


def test_unseen_category_through_lda_pipeline(data):
    train, val = data
    val = val.copy()
    val["Country"] = "XYZ"
    X_tr, y_tr = select_features(train, "classification")
    pipe = make_reduced_pipeline("classification", LogisticRegression(max_iter=500), "lda", 42).fit(X_tr, y_tr)
    assert np.isfinite(pipe.predict_proba(select_features(val, "classification")[0])).all()