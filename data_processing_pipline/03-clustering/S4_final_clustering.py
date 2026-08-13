"""Step 4: Final category-stratified HDBSCAN clustering of service elements.

The script uses the BD-CRL encoder produced by Step 3. Within each attribute
category C, A O (D_Final) representations are L2-normalized and clustered with
HDBSCAN. Noise points can be reassigned to existing medoids when their cosine
similarity exceeds a fixed threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from clustering_utils import (
    build_cluster_text,
    evaluate_clustering_performance,
    identify_approx_medoid,
    l2_normalize,
    load_or_encode_embeddings,
    reassign_noise_to_medoids,
)


def cluster_category_hdbscan(
    df: pd.DataFrame,
    category_name: str,
    model,
    cache_dir: str,
    cache_tag: str,
    min_cluster_size: int = 400,
    min_samples: int = 5,
    sim_threshold: float = 0.72,
    random_seed: int = 42,
):
    """Cluster one attribute category and return result + original embeddings."""
    df = df.copy()
    if len(df) < 50:
        df["local_cluster_id"] = -1
        df["assigned_cluster_id"] = -1
        df["final_topic_name"] = f"【{category_name}】 Noise/Too Few Data"
        df["final_topic_name_assigned"] = df["final_topic_name"]
        df["medoid_text"] = "N/A"
        df["medoid_text_assigned"] = "N/A"
        df["distance_to_medoid"] = np.nan
        df["distance_to_medoid_assigned"] = np.nan
        df["best_medoid_sim"] = np.nan
        df["best_medoid_label"] = -1
        zeros = np.zeros((len(df), model.get_sentence_embedding_dimension()))
        return df, zeros

    embeddings = load_or_encode_embeddings(
        df["cluster_text"].tolist(),
        model,
        cache_dir,
        category_name,
        cache_tag,
    )
    x_norm = l2_normalize(embeddings)

    effective_min_cluster_size = max(2, min(min_cluster_size, len(df) // 2))
    effective_min_samples = max(1, min(min_samples, len(df) // 4))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=effective_min_cluster_size,
        min_samples=effective_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        cluster_selection_epsilon=0.0,
    )
    labels = clusterer.fit_predict(x_norm)

    df["local_cluster_id"] = labels
    df["distance_to_medoid"] = np.nan
    df["medoid_text"] = "N/A"
    df["final_topic_name"] = None

    medoid_vecs_by_label = {}
    medoid_text_by_label = {}

    for label in tqdm(sorted(set(labels)), desc=f"Medoids: {category_name}"):
        mask = df["local_cluster_id"] == label
        if label == -1:
            df.loc[mask, "final_topic_name"] = f"【{category_name}】 Noise/Outliers"
            continue

        cluster_df = df[mask].copy()
        cluster_x = x_norm[mask.to_numpy()]
        medoid_text, medoid_idx = identify_approx_medoid(
            cluster_df,
            cluster_x,
            candidate_size=1024,
            random_state=random_seed,
            metric="euclidean",
        )
        medoid = cluster_x[medoid_idx].reshape(1, -1)
        distances = cdist(cluster_x, medoid, metric="euclidean").ravel()

        df.loc[mask, "distance_to_medoid"] = distances
        df.loc[mask, "medoid_text"] = medoid_text
        df.loc[mask, "final_topic_name"] = f"【{category_name}】 {medoid_text}"
        medoid_vecs_by_label[int(label)] = medoid.ravel()
        medoid_text_by_label[int(label)] = medoid_text

    assigned, best_sim, best_label = reassign_noise_to_medoids(
        x_norm,
        df["local_cluster_id"].to_numpy(),
        medoid_vecs_by_label,
        sim_threshold=sim_threshold,
    )
    df["assigned_cluster_id"] = assigned
    df["best_medoid_sim"] = best_sim
    df["best_medoid_label"] = best_label
    df["distance_to_medoid_assigned"] = np.nan
    df["medoid_text_assigned"] = "N/A"
    df["final_topic_name_assigned"] = None

    non_noise = df["local_cluster_id"] != -1
    df.loc[non_noise, "distance_to_medoid_assigned"] = df.loc[non_noise, "distance_to_medoid"]
    df.loc[non_noise, "medoid_text_assigned"] = df.loc[non_noise, "medoid_text"]
    df.loc[non_noise, "final_topic_name_assigned"] = df.loc[non_noise, "final_topic_name"]

    accepted = (df["local_cluster_id"] == -1) & (df["assigned_cluster_id"] != -1)
    if accepted.any():
        positions = np.where(accepted.to_numpy())[0]
        target_labels = df.loc[accepted, "assigned_cluster_id"].to_numpy(dtype=int)
        medoids = np.stack([medoid_vecs_by_label[label] for label in target_labels], axis=0)
        points = x_norm[positions]
        df.loc[accepted, "distance_to_medoid_assigned"] = np.linalg.norm(points - medoids, axis=1)
        df.loc[accepted, "medoid_text_assigned"] = [medoid_text_by_label[label] for label in target_labels]
        df.loc[accepted, "final_topic_name_assigned"] = [
            f"【{category_name}】 {medoid_text_by_label[label]}" for label in target_labels
        ]

    remaining_noise = df["assigned_cluster_id"] == -1
    df.loc[remaining_noise, "medoid_text_assigned"] = "N/A"
    df.loc[remaining_noise, "final_topic_name_assigned"] = f"【{category_name}】 Noise/Outliers"
    return df, embeddings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="S2 output .xlsx")
    parser.add_argument("--model", required=True, help="S3 best_model path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--cache-tag", default="bdcrl")
    parser.add_argument("--min-cluster-size", type=int, default=400)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--noise-sim-threshold", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or str(output_dir / "embedding_cache")

    model = SentenceTransformer(args.model)
    raw = pd.read_excel(args.input)
    df = build_cluster_text(raw, include_dimension=True)

    results, all_embeddings = [], []
    categories = df["c"].dropna().astype(str).str.strip()
    categories = categories[categories != ""].unique()

    for category in categories:
        subset = df[df["c"] == category].copy()
        clustered, embeddings = cluster_category_hdbscan(
            subset,
            category,
            model,
            cache_dir,
            args.cache_tag,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            sim_threshold=args.noise_sim_threshold,
            random_seed=args.seed,
        )
        results.append(clustered)
        all_embeddings.append(embeddings)

    final_df = pd.concat(results, ignore_index=True)
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    x_norm = l2_normalize(final_embeddings)

    final_df["cluster_id_eval"], _ = pd.factorize(
        final_df["final_topic_name_assigned"], sort=True
    )
    final_df["is_noise"] = final_df["final_topic_name_assigned"].str.contains(
        "Noise/Outliers|Noise/Too Few Data", regex=True, na=False
    )

    metrics = evaluate_clustering_performance(
        final_df,
        x_norm,
        cluster_col="cluster_id_eval",
        d_final_col="D_Final",
        sbert_encoder=model,
        random_seed=args.seed,
    )

    final_df.to_csv(output_dir / "S4_final_clusters.csv", index=False)
    np.save(output_dir / "S4_final_embeddings.npy", final_embeddings)
    with (output_dir / "S4_clustering_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
