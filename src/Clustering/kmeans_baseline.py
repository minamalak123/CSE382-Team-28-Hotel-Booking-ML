"""
kmeans_baseline.py - Milestone 1 clustering baseline (K-Means) on H2.

Run from the project root (AFTER python -m src.data.split):
    python -m src.clustering.kmeans_baseline

"""
from pathlib import Path
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data.eda import CONFIG_PATH, load_config, load_h2
from src.data.split import clean_rows, load_split
from src.features.feature_lists import add_engineered_features, get_feature_lists

OUT_DIR = Path("outputs/clustering")
ART_DIR = Path("artifacts")
SKEWED = ["LeadTime", "TotalNights", "PreviousBookingsNotCanceled", "TotalOfSpecialRequests"]
K_RANGE = range(2, 11)
SIL_SAMPLE = 10_000      # silhouette costs O(n^2); a fixed-seed sample keeps it fast
MIN_CLUSTER_PCT = 2.0    # reject k whose smallest cluster is a tiny niche


def clustering_filter(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Rows that make no sense as a 'stay profile' (0 nights, free/zero price)."""
    bad = (df["TotalNights"] <= 0) | (df["ADR"] <= 0)
    print(f"  {name:5s}: dropped {int(bad.sum()):5d} rows (0 nights or ADR <= 0), "
          f"kept {int((~bad).sum()):,}")
    return df[~bad]


def build_preprocessor(numeric, categorical) -> ColumnTransformer:
    skewed = [c for c in numeric if c in SKEWED]
    plain = [c for c in numeric if c not in SKEWED]
    return ColumnTransformer([
        ("log_scale", Pipeline([("log", FunctionTransformer(np.log1p)),
                                ("scale", StandardScaler())]), skewed),
        ("scale", StandardScaler(), plain),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])


def _sil(Z, labels, seed):
    idx = np.random.RandomState(seed).choice(len(Z), min(SIL_SAMPLE, len(Z)), replace=False)
    return silhouette_score(Z[idx], labels[idx])


def search_k(Z_tr, Z_va, seed) -> pd.DataFrame:
    rows = []
    for k in K_RANGE:
        t0 = time.time()
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z_tr)
        lab_va = km.predict(Z_va)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette_train": _sil(Z_tr, km.labels_, seed),
            "silhouette_val": _sil(Z_va, lab_va, seed),
            "davies_bouldin_train": davies_bouldin_score(Z_tr, km.labels_),
            "davies_bouldin_val": davies_bouldin_score(Z_va, lab_va),
            "smallest_cluster_pct": np.bincount(km.labels_).min() / len(Z_tr) * 100,
            "fit_seconds": time.time() - t0,
        })
        r = rows[-1]
        print(f"  k={k:2d}  sil tr/va={r['silhouette_train']:.3f}/{r['silhouette_val']:.3f}  "
              f"DB tr/va={r['davies_bouldin_train']:.3f}/{r['davies_bouldin_val']:.3f}  "
              f"smallest={r['smallest_cluster_pct']:.1f}%")
    return pd.DataFrame(rows).round(4)


def pick_k(res: pd.DataFrame) -> int:
    """Best train silhouette among k >= 3 with no cluster under MIN_CLUSTER_PCT.
    (k = 2 only splits repeat guests from everyone else - true but useless.)"""
    ok = res[(res["k"] >= 3) & (res["smallest_cluster_pct"] >= MIN_CLUSTER_PCT)]
    return int(ok.loc[ok["silhouette_train"].idxmax(), "k"]) if len(ok) else 4


def plot_k_search(res: pd.DataFrame, k_final: int):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(res["k"], res["inertia"], marker="o")
    axes[0].set_title("Elbow: inertia (lower = tighter)")
    for split in ("train", "val"):
        axes[1].plot(res["k"], res[f"silhouette_{split}"], marker="o", label=split)
        axes[2].plot(res["k"], res[f"davies_bouldin_{split}"], marker="o", label=split)
    axes[1].set_title("Silhouette (higher = better)")
    axes[2].set_title("Davies-Bouldin (lower = better)")
    for ax in axes:
        ax.axvline(k_final, ls="--", c="grey"); ax.set_xlabel("k")
    axes[1].legend(); axes[2].legend()
    fig.tight_layout(); fig.savefig(OUT_DIR / "01_k_search.png", dpi=150); plt.close(fig)


def plot_pca(Z, labels, seed):
    pca = PCA(n_components=2, random_state=seed).fit(Z)
    P = pca.transform(Z)
    idx = np.random.RandomState(seed).choice(len(P), min(15_000, len(P)), replace=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(P[idx, 0], P[idx, 1], c=labels[idx], cmap="tab10", s=4, alpha=0.6)
    ax.legend(*sc.legend_elements(), title="cluster", loc="best")
    ev = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.set_title("K-Means clusters (train) in PCA 2-D space")
    fig.tight_layout(); fig.savefig(OUT_DIR / "02_pca_clusters.png", dpi=150); plt.close(fig)


def profile(df, labels, numeric, categorical) -> pd.DataFrame:
    """Describe each cluster. IsCanceled and DepositType are added here only,
    AFTER clustering, to describe the groups - never to build them."""
    d = df.assign(cluster=labels)
    g = d.groupby("cluster")
    prof = g[numeric].median().add_prefix("median_")
    prof.insert(0, "size", g.size())
    prof.insert(1, "size_pct", (prof["size"] / len(d) * 100).round(1))
    for c in categorical:
        prof[f"top_{c}"] = g[c].agg(lambda s: s.value_counts().index[0])
    prof["non_refund_share_profile_only"] = g["DepositType"].agg(
        lambda s: (s == "Non Refund").mean()).round(3)
    prof["cancel_rate_profile_only"] = g["IsCanceled"].mean().round(3)
    return prof


def main(k_final: int | None = None):
    cfg = load_config(CONFIG_PATH)
    seed = cfg["project"]["random_seed"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(exist_ok=True)

    print("1) load -> clean -> engineer -> team split")
    df = add_engineered_features(
        clean_rows(load_h2(Path(cfg["paths"]["raw_data"])), cfg["data"]["max_valid_adr"]))
    train, val, _ = load_split(df)          # test is not used by clustering
    train = clustering_filter(train, "train")
    val = clustering_filter(val, "val")

    fl = get_feature_lists("clustering")
    numeric, categorical = fl["numeric"], fl["categorical"]
    assert "IsCanceled" not in numeric + categorical
    print(f"2) features: {numeric} + {categorical}")

    pre = build_preprocessor(numeric, categorical).fit(train[numeric + categorical])
    Z_tr = pre.transform(train[numeric + categorical])
    Z_va = pre.transform(val[numeric + categorical])

    print("3) search k (unsupervised metrics only)")
    res = search_k(Z_tr, Z_va, seed)
    if k_final is None:
        k_final = pick_k(res)
    res.to_csv(OUT_DIR / "k_search.csv", index=False)
    plot_k_search(res, k_final)

    print(f"4) final K-Means, k={k_final}")
    model = Pipeline([("pre", build_preprocessor(numeric, categorical)),
                      ("kmeans", KMeans(n_clusters=k_final, n_init=10, random_state=seed))])
    model.fit(train[numeric + categorical])
    joblib.dump(model, ART_DIR / "kmeans_baseline.joblib")

    labels = model.named_steps["kmeans"].labels_
    plot_pca(Z_tr, labels, seed)
    prof = profile(train, labels, numeric, categorical)
    prof.to_csv(OUT_DIR / "cluster_profiles.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(prof)
    print(f"Done -> {OUT_DIR}/ and {ART_DIR}/kmeans_baseline.joblib")


if __name__ == "__main__":
    main()
