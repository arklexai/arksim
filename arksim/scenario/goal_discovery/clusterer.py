# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def hdbscan_cluster(
    embeddings: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int | None = None,
) -> np.ndarray:
    """Cluster embeddings with HDBSCAN. Returns label array (-1 = noise).

    Args:
        embeddings: (N, D) float32 array of L2-normalised embeddings.
        min_cluster_size: Smallest group considered a cluster.
        min_samples: Controls how conservative clustering is.
                     Defaults to min_cluster_size if not set.

    Returns:
        Integer label array of shape (N,). Noise points get label -1.
    """
    try:
        import hdbscan
    except ImportError as e:
        raise ImportError(
            "hdbscan is required for HDBSCAN clustering. "
            "Install it with: pip install hdbscan"
        ) from e

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples or min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels: np.ndarray = clusterer.fit_predict(embeddings)
    return labels


def kmeans_cluster(
    embeddings: np.ndarray,
    k: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster embeddings with K-Means.

    Args:
        embeddings: (N, D) float32 array.
        k: Number of clusters.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (labels array of shape (N,), centroids array of shape (k, D)).
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for K-Means clustering. "
            "Install it with: pip install scikit-learn"
        ) from e

    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    labels = km.fit_predict(embeddings)
    centroids: np.ndarray = km.cluster_centers_.astype(np.float32)
    return labels.astype(np.int32), centroids


def best_k(
    embeddings: np.ndarray,
    k_range: tuple[int, int] = (5, 30),
    seed: int = 42,
) -> int:
    """Find the K that maximises the silhouette score in k_range.

    Args:
        embeddings: (N, D) float32 array.
        k_range: Inclusive (min_k, max_k) range to search.
        seed: Random seed.

    Returns:
        The best K found.
    """
    try:
        from sklearn.metrics import silhouette_score
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for K selection. "
            "Install it with: pip install scikit-learn"
        ) from e

    min_k, max_k = k_range
    max_k = min(max_k, len(embeddings) - 1)
    min_k = max(2, min_k)
    if max_k < min_k:
        # Corpus smaller than k_range[0]; clamp to the largest feasible k.
        return max(2, max_k)

    best_score = -1.0
    best_k_val = min_k
    for k in range(min_k, max_k + 1):
        labels, _ = kmeans_cluster(embeddings, k=k, seed=seed)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(embeddings, labels, metric="euclidean"))
        if score > best_score:
            best_score = score
            best_k_val = k
    return best_k_val


def select_exemplars(
    embeddings: np.ndarray,
    labels: np.ndarray,
    texts: list[str],
    cluster_id: int,
    n: int = 6,
    text_filter: Callable[[str], bool] | None = None,
) -> list[str]:
    """Pick the n texts closest to the centroid of the given cluster.

    Args:
        embeddings: (N, D) array of all embeddings.
        labels: (N,) label array.
        texts: Texts corresponding to each embedding row.
        cluster_id: Which cluster to select exemplars from.
        n: How many exemplars to return.
        text_filter: Optional callable that returns True for texts that should
            be excluded from exemplar candidates (e.g. contains_profanity).
            The centroid is computed from all cluster members regardless.
            Falls back to the full candidate pool if filtering removes everyone.

    Returns:
        Up to n representative texts from the cluster.
    """
    mask = labels == cluster_id
    if not np.any(mask):
        return []

    cluster_embeddings = embeddings[mask]
    cluster_texts = [texts[i] for i, m in enumerate(mask) if m]

    centroid = cluster_embeddings.mean(axis=0)
    similarities = cluster_embeddings @ centroid
    ranked = np.argsort(similarities)[::-1]

    if text_filter is not None:
        allowed = [i for i in ranked if not text_filter(cluster_texts[i])]
        # Fall back to full pool so we always return something
        ranked = np.array(allowed if allowed else list(ranked))

    top_indices = ranked[: min(n, len(cluster_texts))]
    return [cluster_texts[i] for i in top_indices]


def align_centroids(
    new_centroids: np.ndarray,
    prev_centroids: np.ndarray,
) -> np.ndarray:
    """Reorder new_centroids to best match prev_centroids via Hungarian algorithm.

    Useful for stabilising cluster IDs across iterative runs.

    Args:
        new_centroids: (K, D) array from the current run.
        prev_centroids: (K, D) array from a previous run.

    Returns:
        new_centroids reordered to minimise total distance to prev_centroids.
    """
    try:
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.distance import cdist
    except ImportError as e:
        raise ImportError(
            "scipy is required for centroid alignment. "
            "Install it with: pip install scipy"
        ) from e

    cost_matrix = cdist(prev_centroids, new_centroids, metric="euclidean")
    _, col_ind = linear_sum_assignment(cost_matrix)
    return new_centroids[col_ind]
