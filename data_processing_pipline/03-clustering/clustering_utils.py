"""Shared utilities for the service-element clustering pipeline.

This module contains functions reused by the final HDBSCAN clustering step and
clustering-method comparison experiments. It deliberately contains no file-system
side effects or model loading at import time.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity


def safe_name(text: str) -> str:
    """Convert a category/model label into a file-system-safe name."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    return text.strip("_") or "unnamed"


def build_cluster_text(df: pd.DataFrame, include_dimension: bool = True) -> pd.DataFrame:
    """Build the semantic text used for final clustering.

    Parameters
    ----------
    df:
        Input AOCS/AOP-level data containing ``a`` (aspect), ``o`` (opinion),
        and, when ``include_dimension=True``, ``D_Final``.
    include_dimension:
        If True, use ``A O (D_Final)``; otherwise use ``A O``. The latter is
        useful for comparison/baseline experiments.
    """
    out = df.copy()
    for col in ["a", "o"]:
        out[col] = out[col].fillna("").astype(str)

    if include_dimension:
        out["D_Final"] = out["D_Final"].fillna("").astype(str)

        def _compose(row: pd.Series) -> str:
            d = row["D_Final"].strip()
            return f"{row['a']} {row['o']} ({d})" if d else f"{row['a']} {row['o']}"

        out["cluster_text"] = out.apply(_compose, axis=1)
    else:
        out["cluster_text"] = (out["a"] + " " + out["o"])

    out["cluster_text"] = out["cluster_text"].str.lower().str.strip()
    return out[out["cluster_text"] != ""].copy()


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization with numerical protection."""
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def load_or_encode_embeddings(
    texts,
    model,
    cache_dir: str | Path,
    category_name: str,
    cache_tag: str,
) -> np.ndarray:
    """Load cached embeddings or encode and cache them.

    The cache filename includes an explicit ``cache_tag`` so embeddings from
    different representation models/strategies cannot silently overwrite each
    other during comparison experiments.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{safe_name(category_name)}__{safe_name(cache_tag)}.npy"

    if cache_path.exists():
        embeddings = np.load(cache_path)
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Cached embeddings ({len(embeddings)}) do not match current data "
                f"({len(texts)}): {cache_path}. Remove the stale cache file."
            )
        print(f"Using cached embeddings: {cache_path}")
        return embeddings

    print(f"Encoding {category_name} with cache tag '{cache_tag}'...")
    embeddings = model.encode(list(texts), show_progress_bar=True)
    np.save(cache_path, embeddings)
    return embeddings


def identify_approx_medoid(
    df_cluster: pd.DataFrame,
    vectors: np.ndarray,
    candidate_size: int = 1024,
    random_state: int = 42,
    metric: str = "euclidean",
) -> Tuple[str, int]:
    """Find an approximate medoid from at most ``candidate_size`` candidates."""
    n = len(df_cluster)
    if n == 0:
        return "N/A", 0
    if n == 1:
        return df_cluster.iloc[0]["cluster_text"], 0

    rng = np.random.RandomState(random_state)
    candidates = np.arange(n) if n <= candidate_size else rng.choice(n, candidate_size, replace=False)
    candidate_vectors = vectors[candidates]
    distances = cdist(
        vectors.astype(np.float32),
        candidate_vectors.astype(np.float32),
        metric=metric,
    )
    best_candidate = int(np.argmin(distances.sum(axis=0)))
    local_index = int(candidates[best_candidate])
    return df_cluster.iloc[local_index]["cluster_text"], local_index


def reassign_noise_to_medoids(
    x_norm: np.ndarray,
    labels_orig: np.ndarray,
    medoid_vecs_by_label: Dict[int, np.ndarray],
    sim_threshold: float = 0.72,
    keep_unassigned: int = -1,
):
    """Reassign HDBSCAN noise points to a medoid when cosine similarity is high enough."""
    assigned_labels = labels_orig.copy()
    if not medoid_vecs_by_label:
        best_sim = np.full(len(labels_orig), np.nan, dtype=float)
        best_label = np.full(len(labels_orig), keep_unassigned, dtype=int)
        return assigned_labels, best_sim, best_label

    cluster_labels = np.array(sorted(medoid_vecs_by_label), dtype=int)
    medoid_matrix = np.stack([medoid_vecs_by_label[k] for k in cluster_labels], axis=0)
    similarities = x_norm @ medoid_matrix.T

    best_idx = np.argmax(similarities, axis=1)
    best_sim = similarities[np.arange(len(labels_orig)), best_idx]
    best_label = cluster_labels[best_idx]

    noise_mask = labels_orig == -1
    accept_mask = noise_mask & (best_sim >= sim_threshold)
    assigned_labels[accept_mask] = best_label[accept_mask]
    assigned_labels[noise_mask & ~accept_mask] = keep_unassigned
    return assigned_labels, best_sim, best_label


def evaluate_clustering_performance(
    df: pd.DataFrame,
    embedding_matrix: np.ndarray,
    cluster_col: str,
    d_final_col: str,
    sbert_encoder,
    noise_col: str = "is_noise",
    random_seed: int = 42,
) -> dict:
    """Evaluate geometric and business-oriented clustering quality.

    Metrics follow the original experiments: noise percentage, number of
    clusters, silhouette, Calinski-Harabasz, Davies-Bouldin, semantic purity
    based on D_Final, and embedding-space semantic coherence.
    """
    total_samples = len(df)
    noise_mask = df[noise_col].to_numpy(dtype=bool)
    clean_df = df.loc[~noise_mask].copy()
    clean_embeddings = embedding_matrix[~noise_mask]
    labels = clean_df[cluster_col].tolist()
    unique_clusters = list(set(labels))

    results = {
        "Noise Percentage": round(float(noise_mask.sum()) / max(total_samples, 1), 4),
        "Num Clusters": len(unique_clusters),
    }
    if len(unique_clusters) < 2:
        results["Error"] = "Not enough clusters"
        return results

    try:
        sample_size = min(5000, len(clean_embeddings))
        results["Silhouette Score"] = round(
            silhouette_score(
                clean_embeddings,
                labels,
                metric="cosine",
                sample_size=sample_size,
                random_state=random_seed,
            ),
            4,
        )
    except Exception:
        results["Silhouette Score"] = "N/A"

    results["Calinski-Harabasz"] = round(
        calinski_harabasz_score(clean_embeddings, labels), 4
    )
    results["Davies-Bouldin"] = round(
        davies_bouldin_score(clean_embeddings, labels), 4
    )

    if sbert_encoder is None:
        raise ValueError("sbert_encoder is required for Semantic Purity.")

    semantic_purities = []
    for label in unique_clusters:
        cluster_data = clean_df[clean_df[cluster_col] == label]
        d_texts = cluster_data[d_final_col].astype(str).tolist()
        if len(d_texts) < 2:
            continue
        d_vecs = sbert_encoder.encode(d_texts, show_progress_bar=False)
        sim_matrix = cosine_similarity(d_vecs)
        semantic_purities.append(np.mean(sim_matrix[np.triu_indices(len(sim_matrix), 1)]))

    results["Semantic Purity"] = (
        round(float(np.mean(semantic_purities)), 4) if semantic_purities else "N/A"
    )

    rng = np.random.RandomState(random_seed)
    sample_labels = rng.choice(
        unique_clusters,
        size=min(20, len(unique_clusters)),
        replace=False,
    )
    cluster_sims = []
    labels_array = np.asarray(labels)
    for label in sample_labels:
        idx = np.where(labels_array == label)[0]
        if len(idx) < 2:
            continue
        sim = cosine_similarity(clean_embeddings[idx])
        cluster_sims.append(np.mean(sim[np.triu_indices(len(sim), 1)]))

    results["Semantic Coherence"] = (
        round(float(np.mean(cluster_sims)), 4) if cluster_sims else "N/A"
    )
    return results
