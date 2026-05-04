<div align="center">

# Clustering — Three Algorithms vs. Synthetic Ground Truth

**K-means · DBSCAN · Agglomerative · on a deliberately tricky synthetic dataset**

![status](https://img.shields.io/badge/status-complete-3B6EA8?style=flat-square)
![python](https://img.shields.io/badge/python-3.10%2B-3B6EA8?style=flat-square)
![data](https://img.shields.io/badge/data-self--generated-7A7A7A?style=flat-square)
![license](https://img.shields.io/badge/license-MIT-7A7A7A?style=flat-square)

</div>

---

## At a glance

> Cluster a 2-D dataset that mixes two round Gaussian blobs, one **elongated anisotropic blob** that breaks K-means' equal-axis assumption, and a sparse band of **uniform-noise outliers**. Score against the ground-truth labels with Adjusted Rand Index — a luxury real unsupervised tasks don't give you.

<table>
<tr>
<td align="center" width="33%">
<sub>K-means (k=3)</sub><br>
<b style="font-size:1.5em; color:#7A7A7A;">ARI 0.764</b><br>
<sub>shears the elongated cluster</sub>
</td>
<td align="center" width="33%">
<sub>DBSCAN (ε=0.7)</sub><br>
<b style="font-size:1.5em; color:#3B6EA8;">ARI 0.893</b><br>
<sub>+ correctly flags 89 outliers</sub>
</td>
<td align="center" width="33%">
<sub>Agglomerative (ward)</sub><br>
<b style="font-size:1.5em; color:#3B6EA8;">ARI 0.885</b><br>
<sub>but absorbs noise into clusters</sub>
</td>
</tr>
</table>

| Algorithm | Clusters found | Noise flagged | ARI | NMI | Silhouette |
|---|:---:|:---:|---:|---:|---:|
| K-means (k=3) | 3 | 0 | 0.764 | 0.743 | **0.693** |
| **DBSCAN (ε=0.7, min_samples=10)** | **3** | **89** | **0.893** | **0.857** | **0.774** |
| Agglomerative (ward, k=3) | 3 | 0 | 0.885 | 0.856 | 0.678 |

<sub>**Headline finding:** the elongated cluster reveals each algorithm's bias. K-means commits to spherical clusters and slices the long blob in two; DBSCAN follows the actual density and additionally separates the uniform-noise band as outliers; Agglomerative gets the partition right but, like K-means, has no concept of "noise" so it forces the outliers into one of the three clusters.</sub>

---

## Dashboard

### 1. Side-by-side panels — ground truth vs. each algorithm

![panels](assets/01_panels.png)

The top-left panel is what an oracle would output. The other three are what each algorithm comes up with from the same input points.

- **K-means** treats every cluster as if it were a sphere. It fits the two round blobs perfectly, then cuts the elongated cluster down its short axis to balance variance — visible as the abrupt color change in the upper-right panel.
- **DBSCAN** doesn't impose a shape; it follows local density. It recovers all three clusters and additionally tags the sparse points (`x` markers) as noise. Note how the noise count of **89** *exceeds* the 60 we injected — a few low-density edges of the elongated blob also look noise-like to DBSCAN at this ε, which is a true-to-life behavior of the algorithm.
- **Agglomerative (ward linkage)** matches DBSCAN's structural recovery but without the noise concept; the outliers get absorbed into whichever cluster they're closest to, which is why its silhouette is lower despite a comparable ARI.

### 2. Cluster-quality metrics

![metrics](assets/02_metrics.png)

Three different lenses on "cluster quality":

- **Adjusted Rand Index (ARI)** — agreement with ground truth, corrected for chance. Range [-1, 1]. Highest for DBSCAN.
- **Normalized Mutual Information (NMI)** — information shared between predicted and true labels. Highest for DBSCAN, but Agglomerative is essentially tied.
- **Silhouette** — purely intrinsic, doesn't use ground truth. *Highest for K-means*. That's a paradox worth noticing: K-means scores best on the metric that asks "how compact are my clusters," because it actively *optimizes for* compact spherical clusters — even when the ground truth has a non-spherical one. That's the textbook lesson on why intrinsic metrics alone can be misleading.

### 3. Hierarchical (ward) dendrogram

![dendrogram](assets/03_dendrogram.png)

A subsample of points (120) plotted as a dendrogram with ward linkage. Three high-level branches dominate at the top of the tree — the same three clusters Agglomerative recovers in panel 1. Cutting horizontally at any height in the gray region produces a 3-cluster partition.

---

## What's actually happening

### K-means

Pick `k` centroids; assign each point to the nearest one; recompute centroids as the mean of their assigned points; repeat until stable. Minimizes within-cluster sum of squares.

- **Bias**: every cluster is implicitly an isotropic Gaussian (equal axes). Rotated, elongated, or non-convex clusters get split.
- **Knobs**: `k` (you have to pick it), random initialization (matters — `n_init=10` averages it out).

### DBSCAN

Define a "core point" as one with at least `min_samples` neighbors within radius `ε`. Connect core points that are within ε of each other; sweep up reachable non-core neighbors. Anything left is noise.

- **Bias**: assumes clusters are *connected dense regions* with similar density. Breaks if cluster densities vary a lot.
- **Knobs**: `ε` (the only crucial one — too small and everything is noise, too large and clusters merge), `min_samples`.
- **Killer feature**: built-in noise / outlier detection. The other two have to be modeled with a separate stage.

### Agglomerative (Ward linkage)

Bottom-up: every point starts as its own cluster; repeatedly merge the two clusters whose union increases within-cluster variance the least; cut the resulting tree at a chosen number of clusters.

- **Bias**: ward linkage prefers compact, equal-size clusters — like K-means but globally optimized. Single-linkage variants would prefer chains.
- **Knobs**: `n_clusters` (or a distance threshold), `linkage` strategy.
- **Bonus**: produces a full hierarchy, not just a partition.

### Choosing one

| If you... | Use |
|---|---|
| Know `k` and clusters look spherical | K-means |
| Don't know `k`, expect outliers, density-based clusters | DBSCAN |
| Want a hierarchy / a tree to inspect | Agglomerative |

---

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py    # synthetic clustering data (deterministic)
python train.py            # fit, score, render dashboard figures
```

### Tweak the difficulty

`DataConfig` in [`generate_data.py`](generate_data.py):

```python
DataConfig(
    n_per_cluster=250,   # more points → all algorithms generally improve
    n_noise=60,          # uniform-noise outliers — DBSCAN should flag them
    seed=42,
)
```

Or change the cluster shapes in `generate_data.py` itself: rotate the elongated blob, increase its aspect ratio to 10:1, or add a fourth cluster of concentric rings — only DBSCAN will survive that one.

---

## Project layout

```
03-clustering/
├── README.md              ← this dashboard
├── requirements.txt
├── generate_data.py       ← synthetic 3-cluster + noise dataset
├── train.py               ← K-means / DBSCAN / Agglomerative + figures
├── assets/                ← rendered dashboard figures (3 PNGs)
└── results/metrics.json
```

---

## What I learned

- **The silhouette score has no idea what "right" looks like.** It rewards compactness, which is why K-means can win silhouette while losing ARI. Lesson: when ground truth exists (or you can synthesize it), use ARI; intrinsic metrics are a fall-back, not a default.
- **DBSCAN's noise count exceeds the truth here on purpose, not by accident.** A handful of points at the low-density tails of the elongated blob look like outliers to a density-based method. That's not a bug — it's an honest reflection of "what counts as a cluster" being algorithm-dependent.
- **K-means + Agglomerative have an *implicit shape prior*; DBSCAN has an *implicit density prior*.** Switch datasets and the right answer flips. The skill is recognizing which prior matches the structure you actually have.
- **Synthetic data with built-in outliers is the only reasonable way to evaluate noise detection.** With real data you don't know which points are "real" outliers vs. genuine signal in a long-tail cluster — so the question becomes unanswerable. With a generator, the answer is one comparison away.

---

<div align="center">
<sub>Part of a hands-on machine-learning portfolio. Data is fully synthetic and self-generated.</sub>
</div>
