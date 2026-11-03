from pathlib import Path
import pandas as pd
import pytest
import src.classification.train_cancel_baseline as mod
from src.data.split import SPLIT_FILE, load_split

RAW = Path("data/raw/H2.csv")
MODELS = ["dummy_prior", "logistic_regression"]


class LockedTestSplit:
    """Stand in for the test split: any access raises an error"""

    def _boom(self, *args, **kwargs):
        raise AssertionError("The test split was used by the baseline script")

    __getitem__ = __len__ = __iter__ = __getattr__ = _boom


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    """Run the baseline once with a locked test split, writing to a temp folder."""
    if not RAW.exists() or not SPLIT_FILE.exists():
        pytest.skip("needs data/raw/H2.csv and artifacts/split_indices.csv")

    out_dir = tmp_path_factory.mktemp("classification")

    def load_split_without_test(df):
        train, val, _ = load_split(df)
        return train, val, LockedTestSplit()

    original = mod.load_split
    mod.load_split = load_split_without_test
    try:
        res = mod.run(out_dir)
    finally:
        mod.load_split = original

    return res, out_dir


def test_outputs_exist(baseline):
    _, out_dir = baseline

    assert (out_dir / "baseline_val.csv").exists()
    for name in MODELS:
        assert (out_dir / f"confusion_matrix_{name}_val.png").exists()


def test_results_table_has_both_models_and_all_metrics(baseline):
    res, out_dir = baseline
    saved = pd.read_csv(out_dir / "baseline_val.csv")

    assert list(saved["model"]) == MODELS
    for split in ("train", "val"):
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
            assert f"{split}_{metric}" in saved.columns
    assert "fit_seconds" in saved.columns


def test_test_split_is_not_used(baseline):
    res, _ = baseline

    assert len(res) == 2


def test_metrics_are_between_0_and_1(baseline):
    res, _ = baseline
    metrics = res.drop(columns=["model", "fit_seconds"])

    assert ((metrics >= 0) & (metrics <= 1)).all().all()


def test_logistic_beats_dummy_on_roc_auc(baseline):
    res, _ = baseline
    auc = res.set_index("model")["val_roc_auc"]

    assert auc["logistic_regression"] > auc["dummy_prior"]