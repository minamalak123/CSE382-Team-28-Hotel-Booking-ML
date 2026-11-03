"""Untuned classification baselines (dummy and logistic regression) for IsCanceled
Both models are scored on train and validation only; the test split is never used"""

import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.eda import CONFIG_PATH, load_config, load_h2
from src.data.split import clean_rows, load_split
from src.features.feature_lists import add_engineered_features, select_features
from src.features.preprocess import make_model_pipeline

OUT = Path("outputs/classification")
CLASS_NAMES = ["Kept (0)", "Canceled (1)"]


def score_split(pipe, X, y) -> dict:
    """Return the six metrics for one split (positive class = 1 = canceled)."""
    y_pred = pipe.predict(X)
    y_prob = pipe.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_prob),
        "pr_auc": average_precision_score(y, y_prob),
    }


def save_confusion_matrix(pipe, X, y, name: str, out_dir: Path) -> Path:
    """Save the validation confusion matrix as a PNG and return its path."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y,
        pipe.predict(X),
        display_labels=CLASS_NAMES,
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(f"{name} - validation")
    fig.tight_layout()

    path = out_dir / f"confusion_matrix_{name}_val.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def evaluate(name, model, X_tr, y_tr, X_va, y_va, out_dir: Path) -> dict:
    """Fit preprocessing + model on train, then score train and validation."""
    t0 = time.perf_counter()
    pipe = make_model_pipeline("classification", model).fit(X_tr, y_tr)
    row = {"model": name, "fit_seconds": round(time.perf_counter() - t0, 2)}

    for split, X, y in (("train", X_tr, y_tr), ("val", X_va, y_va)):
        for metric, value in score_split(pipe, X, y).items():
            row[f"{split}_{metric}"] = value

    save_confusion_matrix(pipe, X_va, y_va, name, out_dir)
    return row


def run(out_dir: Path = OUT) -> pd.DataFrame:
    """Run the full baseline; write the results CSV and confusion-matrix PNGs."""
    cfg = load_config(CONFIG_PATH)
    seed = cfg["project"]["random_seed"]

    df = add_engineered_features(
        clean_rows(
            load_h2(Path(cfg["paths"]["raw_data"])),
            cfg["data"]["max_valid_adr"],
        )
    )
    train_df, val_df, _ = load_split(df)

    X_tr, y_tr = select_features(train_df, "classification")
    X_va, y_va = select_features(val_df, "classification")

    models = {
        "dummy_prior": DummyClassifier(strategy="prior", random_state=seed),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(
        [
            evaluate(name, model, X_tr, y_tr, X_va, y_va, out_dir)
            for name, model in models.items()
        ]
    ).round(4)
    res.to_csv(out_dir / "baseline_val.csv", index=False)
    return res


def main() -> None:
    res = run()
    print(res.to_string(index=False))
    print(f"\nSaved to {OUT}/")


if __name__ == "__main__":
    main()