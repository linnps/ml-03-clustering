"""
Cluster the synthetic dataset with K-means / DBSCAN / Agglomerative,
score against ground-truth labels, and produce the dashboard figures.

Palette (shared across the portfolio):
    background  : white
    grid / axes : light gray  (#E5E5E5)
    primary     : muted blue  (#3B6EA8)
    accent      : muted red   (#C04040)
    neutral     : medium gray (#7A7A7A)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from generate_data import DataConfig, LABEL_NOISE, generate

# ---------------------------------------------------------------- style ----
COLOR_BG = "#FFFFFF"
COLOR_GRID = "#E5E5E5"
COLOR_TEXT = "#333333"
COLOR_BLUE = "#3B6EA8"
COLOR_RED = "#C04040"
COLOR_GRAY = "#7A7A7A"
COLOR_LIGHT_GRAY = "#CCCCCC"
COLOR_LIGHT_BLUE = "#9EB7D6"

# Cluster-id → color. -1 (noise / outlier) gets a faded gray.
CLUSTER_COLORS = [COLOR_BLUE, COLOR_RED, COLOR_GRAY, "#5A8FCC", "#D88080"]
NOISE_COLOR = "#BFBFBF"

mpl.rcParams.update({
    "figure.facecolor": COLOR_BG,
    "axes.facecolor": COLOR_BG,
    "axes.edgecolor": COLOR_LIGHT_GRAY,
    "axes.labelcolor": COLOR_TEXT,
    "axes.titlecolor": COLOR_TEXT,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": COLOR_TEXT,
    "ytick.color": COLOR_TEXT,
    "grid.color": COLOR_GRID,
    "grid.linewidth": 0.6,
    "axes.grid": True,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
    "font.size": 11,
})


# ------------------------------------------------------------- modelling ---
@dataclass
class ClusterResult:
    name: str
    labels: np.ndarray
    n_clusters_found: int
    n_noise_found: int
    ari: float
    nmi: float
    silhouette: float | None  # None when only one cluster found


def score(y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray) -> tuple[float, float, float | None]:
    ari = float(adjusted_rand_score(y_true, y_pred))
    nmi = float(normalized_mutual_info_score(y_true, y_pred))
    # Silhouette needs ≥2 distinct labels and ignores noise (-1).
    mask = y_pred != LABEL_NOISE
    sil: float | None
    if mask.sum() >= 2 and len(set(y_pred[mask])) >= 2:
        sil = float(silhouette_score(X[mask], y_pred[mask]))
    else:
        sil = None
    return ari, nmi, sil


def fit_clusterers(X: np.ndarray, y_true: np.ndarray) -> list[ClusterResult]:
    cfgs = [
        ("K-means (k=3)", KMeans(n_clusters=3, n_init=10, random_state=42)),
        ("DBSCAN (ε=0.7)", DBSCAN(eps=0.7, min_samples=10)),
        ("Agglomerative (ward, k=3)", AgglomerativeClustering(n_clusters=3, linkage="ward")),
    ]
    out: list[ClusterResult] = []
    for name, m in cfgs:
        labels = m.fit_predict(X)
        ari, nmi, sil = score(y_true, labels, X)
        n_clusters = int(len(set(labels)) - (1 if LABEL_NOISE in labels else 0))
        n_noise = int(np.sum(labels == LABEL_NOISE))
        out.append(ClusterResult(
            name=name, labels=labels,
            n_clusters_found=n_clusters, n_noise_found=n_noise,
            ari=ari, nmi=nmi, silhouette=sil,
        ))
    return out


# ---------------------------------------------------------------- figures --
def _scatter_clusters(ax, X: np.ndarray, labels: np.ndarray, title: str) -> None:
    unique = sorted(set(labels))
    for i, lab in enumerate(unique):
        mask = labels == lab
        if lab == LABEL_NOISE:
            ax.scatter(X[mask, 0], X[mask, 1], s=18, c=NOISE_COLOR,
                       marker="x", linewidth=0.9, alpha=0.8, label="noise / outlier")
        else:
            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            ax.scatter(X[mask, 0], X[mask, 1], s=18, c=color, alpha=0.85,
                       edgecolor="white", linewidth=0.4,
                       label=f"cluster {lab}")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)


def fig_panels(X: np.ndarray, y_true: np.ndarray,
               results: list[ClusterResult], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), constrained_layout=True)

    _scatter_clusters(axes[0, 0], X, y_true, "Ground truth")

    for ax, r in zip(axes.ravel()[1:], results):
        subtitle = (f"{r.name}\n"
                    f"ARI {r.ari:.3f} · "
                    f"silhouette {'—' if r.silhouette is None else f'{r.silhouette:.3f}'}")
        _scatter_clusters(ax, X, r.labels, subtitle)

    fig.suptitle("Clustering — three algorithms vs. ground truth",
                 fontsize=14, fontweight="bold", color=COLOR_TEXT, y=1.02)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_metrics_bar(results: list[ClusterResult], out_path: Path) -> None:
    metric_names = ["Adjusted Rand", "NMI", "Silhouette"]
    raw = [
        [r.ari for r in results],
        [r.nmi for r in results],
        [(r.silhouette if r.silhouette is not None else 0.0) for r in results],
    ]
    names = [r.name for r in results]
    palette = [COLOR_BLUE, COLOR_RED, COLOR_GRAY]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    for ax, mname, vals in zip(axes, metric_names, raw):
        bars = ax.bar(names, vals, color=palette,
                      edgecolor=COLOR_LIGHT_GRAY, linewidth=0.8)
        ax.set_title(mname); ax.set_ylim(-0.15, 1.05)
        ax.tick_params(axis="x", labelrotation=15)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.3f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=9, color=COLOR_TEXT)
    fig.suptitle("Cluster-quality metrics (higher = better)",
                 fontsize=14, fontweight="bold", color=COLOR_TEXT, y=1.05)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_dendrogram(X: np.ndarray, out_path: Path) -> None:
    # Subsample for legibility — full dendrogram would be a wall of leaves.
    rng = np.random.default_rng(0)
    idx = rng.choice(X.shape[0], size=min(120, X.shape[0]), replace=False)
    Z = linkage(X[idx], method="ward")

    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    dendrogram(
        Z, ax=ax, color_threshold=0.7 * max(Z[:, 2]),
        above_threshold_color=COLOR_GRAY,
        leaf_font_size=6,
    )
    ax.set_xlabel("Subsampled points (index)")
    ax.set_ylabel("Ward distance")
    ax.set_title("Hierarchical (ward) linkage — three high-level branches "
                 "show as expected")
    ax.grid(False)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- main ----
def main() -> None:
    cfg = DataConfig()
    X_df, y_series = generate(cfg)
    X = X_df.values
    y_true = y_series.values

    results = fit_clusterers(X, y_true)

    print(f"\nDataset: {len(X)} points, "
          f"{cfg.n_per_cluster} per true cluster + {cfg.n_noise} noise")
    print(f"\n{'algorithm':<28} {'k_found':>7} {'noise':>6} {'ARI':>7} {'NMI':>7} {'silhouette':>11}")
    for r in results:
        sil = "—" if r.silhouette is None else f"{r.silhouette:.3f}"
        print(f"{r.name:<28} {r.n_clusters_found:>7} {r.n_noise_found:>6} "
              f"{r.ari:>7.3f} {r.nmi:>7.3f} {sil:>11}")

    Path("results").mkdir(exist_ok=True)
    summary = {
        "config": cfg.__dict__,
        "metrics": [{
            "model": r.name,
            "k_found": r.n_clusters_found,
            "noise_found": r.n_noise_found,
            "ari": r.ari, "nmi": r.nmi,
            "silhouette": r.silhouette,
        } for r in results],
    }
    with open("results/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    assets = Path("assets"); assets.mkdir(exist_ok=True)
    fig_panels(X, y_true, results, assets / "01_panels.png")
    fig_metrics_bar(results, assets / "02_metrics.png")
    fig_dendrogram(X, assets / "03_dendrogram.png")

    print(f"\nFigures saved to: {assets.resolve()}")


if __name__ == "__main__":
    main()
