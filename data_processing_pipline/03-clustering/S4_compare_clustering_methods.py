"""Clustering-method comparisons used alongside Step 4.

The original supplemental script contained three alternative methods in one
file, but each redefined the same function name. Here they are exposed as
separate implementations:

- K-means
- Gaussian Mixture Model (GMM)
- kNN-graph pivot correlation clustering

The final HDBSCAN implementation remains in ``S4_final_clustering.py``. Shared
preprocessing, medoid identification, caching, and evaluation are imported from
``clustering_utils.py`` so all methods use the same evaluation interface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from clustering_utils import (
    build_cluster_text,
    evaluate_clustering_performance,
    identify_approx_medoid,
    l2_normalize,
    load_or_encode_embeddings,
)

def load_hdbscan_cluster_counts(reference_csv: str | Path) -> dict[str, int]:
    """Derive the per-category K used by K-means/GMM from the Step-4 HDBSCAN output.

    This implements the paper's comparison rule directly: K-means and GMM use
    the number of non-noise clusters identified by HDBSCAN for each attribute
    category.
    """
    reference = pd.read_csv(reference_csv)
    required = {"c", "local_cluster_id"}
    missing = required - set(reference.columns)
    if missing:
        raise ValueError(
            f"HDBSCAN reference is missing columns: {sorted(missing)}. "
            "Use S4_final_clusters.csv produced by S4_final_clustering.py."
        )

    k_map = {}
    for category, group in reference.groupby("c"):
        labels = set(pd.to_numeric(group["local_cluster_id"], errors="coerce").dropna().astype(int))
        labels.discard(-1)
        if labels:
            k_map[str(category)] = len(labels)
    return k_map


def add_cluster_names(df, x_norm, labels, category, seed=42):
    df = df.copy()
    df["local_cluster_id"] = labels
    df["distance_to_medoid"] = np.nan
    df["medoid_text"] = "N/A"
    df["final_topic_name"] = None

    for label in tqdm(sorted(set(labels)), desc=f"Medoids: {category}"):
        mask = df["local_cluster_id"] == label
        if label == -1:
            df.loc[mask, "final_topic_name"] = f"【{category}】 Noise/Outliers"
            continue

        cluster_df = df[mask].copy()
        cluster_x = x_norm[mask.to_numpy()]
        if len(cluster_df) == 1:
            medoid_text = cluster_df.iloc[0]["cluster_text"]
            medoid_idx = 0
        else:
            medoid_text, medoid_idx = identify_approx_medoid(
                cluster_df,
                cluster_x,
                candidate_size=1024,
                random_state=seed,
                metric="cosine",
            )
        medoid = cluster_x[medoid_idx].reshape(1, -1)
        distances = cdist(cluster_x, medoid, metric="cosine").ravel()
        df.loc[mask, "distance_to_medoid"] = distances
        df.loc[mask, "medoid_text"] = medoid_text
        df.loc[mask, "final_topic_name"] = f"【{category}】 {medoid_text}"
    return df


def cluster_kmeans(x_norm, k, seed):
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(x_norm)


def cluster_gmm(x_norm, k, seed):
    model = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        random_state=seed,
        reg_covar=1e-6,
        n_init=3,
        max_iter=300,
    )
    return model.fit_predict(x_norm)


def build_knn_positive_graph(x, knn_k=50, sim_threshold=0.65, block_size=2000):
    n = x.shape[0]
    neighbors = [set() for _ in range(n)]
    x_norm = l2_normalize(x)
    for start in range(0, n, block_size):
        end = min(n, start + block_size)
        similarities = x_norm[start:end] @ x_norm.T
        for local_i, i in enumerate(range(start, end)):
            row = similarities[local_i]
            row[i] = -1.0
            if knn_k < n:
                idx = np.argpartition(-row, knn_k)[:knn_k]
            else:
                idx = np.arange(n)
            for j in idx:
                if row[j] >= sim_threshold:
                    neighbors[i].add(int(j))
                    neighbors[int(j)].add(i)
    return neighbors


def pivot_correlation_clustering(pos_neighbors, seed=42):
    rng = np.random.RandomState(seed)
    unassigned = set(range(len(pos_neighbors)))
    labels = -np.ones(len(pos_neighbors), dtype=int)
    cluster_id = 0
    while unassigned:
        pivot = rng.choice(list(unassigned))
        cluster = {pivot} | {n for n in pos_neighbors[pivot] if n in unassigned}
        for idx in cluster:
            labels[idx] = cluster_id
        unassigned -= cluster
        cluster_id += 1
    return labels


def cluster_correlation(embeddings, seed, knn_k, sim_threshold, min_size):
    graph = build_knn_positive_graph(
        embeddings,
        knn_k=knn_k,
        sim_threshold=sim_threshold,
    )
    labels = pivot_correlation_clustering(graph, seed=seed)
    counts = pd.Series(labels).value_counts()
    small = set(counts[counts < min_size].index.tolist())
    labels = labels.copy()
    labels[np.isin(labels, list(small))] = -1
    return labels


def run_method(df, model, method, cache_dir, cache_tag, seed, args, k_map=None):
    results, all_embeddings = [], []
    categories = df["c"].dropna().astype(str).str.strip()
    categories = categories[categories != ""].unique()

    for category in categories:
        subset = df[df["c"] == category].copy()
        embeddings = load_or_encode_embeddings(
            subset["cluster_text"].tolist(),
            model,
            cache_dir,
            category,
            cache_tag,
        )
        x_norm = l2_normalize(embeddings)

        if method in {"kmeans", "gmm"}:
            if k_map is None or str(category) not in k_map:
                raise KeyError(f"No HDBSCAN-derived cluster count for category: {category}")
            k = k_map[str(category)]
        if method == "kmeans":
            labels = cluster_kmeans(x_norm, k, seed)
        elif method == "gmm":
            labels = cluster_gmm(x_norm, k, seed)
        elif method == "correlation":
            labels = cluster_correlation(
                embeddings,
                seed,
                args.knn_k,
                args.correlation_sim_threshold,
                args.correlation_min_size,
            )
        else:
            raise ValueError(method)

        results.append(add_cluster_names(subset, x_norm, labels, category, seed=seed))
        all_embeddings.append(embeddings)

    final_df = pd.concat(results, ignore_index=True)
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    final_df["cluster_id_eval"], _ = pd.factorize(final_df["final_topic_name"], sort=True)
    final_df["is_noise"] = final_df["final_topic_name"].str.contains(
        "Noise/Outliers|Noise/Too Few Data", regex=True, na=False
    )
    metrics = evaluate_clustering_performance(
        final_df,
        l2_normalize(final_embeddings),
        cluster_col="cluster_id_eval",
        d_final_col="D_Final",
        sbert_encoder=model,
        random_seed=seed,
    )
    return final_df, final_embeddings, metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="S2 output .xlsx")
    parser.add_argument("--model", required=True, help="Embedding model/strategy under evaluation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", choices=["kmeans", "gmm", "correlation", "all"], default="all")
    parser.add_argument("--hdbscan-reference", default=None, help="S4_final_clusters.csv; required for K-means/GMM")
    parser.add_argument("--text-mode", choices=["ao", "aod"], default="aod")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--cache-tag", default="comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--knn-k", type=int, default=50)
    parser.add_argument("--correlation-sim-threshold", type=float, default=0.65)
    parser.add_argument("--correlation-min-size", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or str(output_dir / "embedding_cache")

    model = SentenceTransformer(args.model)
    raw = pd.read_excel(args.input)
    df = build_cluster_text(raw, include_dimension=(args.text_mode == "aod"))

    methods = ["kmeans", "gmm", "correlation"] if args.method == "all" else [args.method]
    needs_k = any(method in {"kmeans", "gmm"} for method in methods)
    if needs_k and not args.hdbscan_reference:
        raise ValueError("--hdbscan-reference is required for K-means/GMM comparisons.")
    k_map = load_hdbscan_cluster_counts(args.hdbscan_reference) if needs_k else None
    if k_map is not None:
        print("HDBSCAN-derived cluster counts:", k_map)

    summary = {}
    for method in methods:
        print(f"\n=== {method.upper()} ===")
        result, embeddings, metrics = run_method(
            df,
            model,
            method,
            cache_dir,
            f"{args.cache_tag}_{args.text_mode}",
            args.seed,
            args,
            k_map=k_map,
        )
        result.to_csv(output_dir / f"{method}_clusters.csv", index=False)
        np.save(output_dir / f"{method}_embeddings.npy", embeddings)
        with (output_dir / f"{method}_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        summary[method] = metrics

    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
