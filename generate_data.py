"""
Synthetic clustering data designed to break each algorithm at a different
point.

We mix three structurally distinct clusters plus a noise band:

    cluster 0 — round Gaussian blob          (K-means happy)
    cluster 1 — round Gaussian blob          (K-means happy)
    cluster 2 — elongated anisotropic blob   (K-means unhappy: equal-axis assumption)
    noise     — uniform points away from all centers (DBSCAN should isolate)

Ground-truth labels are kept aside so we can score Adjusted Rand Index
post-clustering — the kind of evaluation only synthetic data allows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_NOISE = -1


@dataclass
class DataConfig:
    n_per_cluster: int = 250
    n_noise: int = 60
    seed: int = 42


def generate(cfg: DataConfig) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(cfg.seed)

    # Cluster 0: round Gaussian at (-4, -3)
    c0 = rng.normal(loc=[-4.0, -3.0], scale=0.55, size=(cfg.n_per_cluster, 2))
    y0 = np.full(cfg.n_per_cluster, 0)

    # Cluster 1: round Gaussian at (4, -3)
    c1 = rng.normal(loc=[4.0, -3.0], scale=0.55, size=(cfg.n_per_cluster, 2))
    y1 = np.full(cfg.n_per_cluster, 1)

    # Cluster 2: anisotropic Gaussian at (0, 3), elongated ~7:1 along a tilted axis.
    base = rng.normal(0, 1, size=(cfg.n_per_cluster, 2))
    base[:, 0] *= 3.5    # stretch x
    base[:, 1] *= 0.45   # squash y
    theta = np.deg2rad(20)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
    c2 = base @ rot.T + np.array([0.0, 3.5])
    y2 = np.full(cfg.n_per_cluster, 2)

    # Uniform noise points across a wide bounding box.
    noise = rng.uniform(low=[-9, -7], high=[9, 7], size=(cfg.n_noise, 2))
    yn = np.full(cfg.n_noise, LABEL_NOISE)

    X = np.vstack([c0, c1, c2, noise])
    y = np.concatenate([y0, y1, y2, yn])

    perm = rng.permutation(len(y))
    X = X[perm]
    y = y[perm]

    X_df = pd.DataFrame(X, columns=["x1", "x2"])
    y_series = pd.Series(y, name="cluster")
    return X_df, y_series


def save(out_dir: Path, X: pd.DataFrame, y: pd.Series) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    X.to_csv(out_dir / "X.csv", index=False)
    y.to_csv(out_dir / "y_true.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic clustering data.")
    p.add_argument("--n-per-cluster", type=int, default=250)
    p.add_argument("--n-noise", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    args = p.parse_args()

    cfg = DataConfig(n_per_cluster=args.n_per_cluster,
                     n_noise=args.n_noise, seed=args.seed)
    X, y = generate(cfg)
    save(args.out_dir, X, y)
    counts = y.value_counts().sort_index().to_dict()
    print(f"Generated {len(X)} points: {counts}")
    print(f"Saved to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
