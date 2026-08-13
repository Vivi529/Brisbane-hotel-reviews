"""Step 1: Initial standardization of evaluation dimensions.

Pipeline
--------
1. Encode unique ``evaluation_dimension`` labels with SentenceTransformer.
2. Reduce embeddings with UMAP.
3. Cluster dimensions with HDBSCAN.
4. Map each cluster to a representative high-frequency/complete label.

The scientific settings follow the original experiment; only file handling and
reporting have been cleaned for reproducibility.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import torch
import umap
from sentence_transformers import SentenceTransformer


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_dimension_embeddings(df, model, cache_dir: Path, force: bool = False):
    cache_dir.mkdir(parents=True, exist_ok=True)
    labels_path = cache_dir / "unique_dimensions.npy"
    emb_path = cache_dir / "dimension_embeddings.npy"

    unique_dimensions = [
        d.strip()
        for d in df["evaluation_dimension"].dropna().astype(str).unique().tolist()
        if d.strip()
    ]
    if not unique_dimensions:
        raise ValueError("No valid evaluation_dimension text was found.")

    if labels_path.exists() and emb_path.exists() and not force:
        cached_labels = np.load(labels_path, allow_pickle=True).tolist()
        cached_embeddings = np.load(emb_path)
        if cached_labels == unique_dimensions and len(cached_embeddings) == len(unique_dimensions):
            print(f"Loaded {len(unique_dimensions)} cached dimension embeddings.")
            return cached_labels, cached_embeddings
        raise ValueError(
            "Dimension cache does not match the current input. Re-run with --force-embeddings."
        )

    print(f"Encoding {len(unique_dimensions)} unique evaluation dimensions...")
    embeddings = model.encode(
        unique_dimensions,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    np.save(labels_path, np.asarray(unique_dimensions, dtype=object))
    np.save(emb_path, embeddings)
    return unique_dimensions, embeddings


def standardize_dimensions(df, unique_dimensions, embeddings, seed: int = 42):
    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=10,
        metric="cosine",
        random_state=seed,
    )
    reduced = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=3,
        min_samples=1,
        cluster_selection_epsilon=0.1,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(reduced)

    unique_arr = np.asarray(unique_dimensions)
    dimension_counts = Counter(df["evaluation_dimension"].dropna().astype(str).tolist())
    dimension_map = {}

    for label in sorted(set(labels)):
        indices = np.where(labels == label)[0]
        cluster_dimensions = unique_arr[indices].tolist()

        if label == -1:
            for text in cluster_dimensions:
                dimension_map[text] = "Noise/Outliers"
            continue

        # Preserve the original representative-label rule:
        # prioritize frequency, then prefer the longer expression.
        representative = max(
            cluster_dimensions,
            key=lambda text: dimension_counts.get(text, 0) * 100 + len(text),
        )
        for text in cluster_dimensions:
            dimension_map[text] = representative

    out = df.copy()
    out["D_Std"] = out["evaluation_dimension"].map(dimension_map).fillna(out["evaluation_dimension"])

    label_set = set(labels)
    noise_count = int(np.sum(labels == -1))
    cluster_count = len(label_set) - (1 if -1 in label_set else 0)
    noise_ratio = noise_count / max(len(unique_dimensions), 1)

    print(f"Unique raw dimensions: {len(unique_dimensions)}")
    print(f"Non-noise clusters: {cluster_count}")
    print(f"Noise dimensions: {noise_count}")
    print(f"Noise ratio (unique-dimension level): {noise_ratio:.4f}")
    print(f"Standardized labels: {len(set(dimension_map.values()))}")
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="AOP/AOCS extraction.xlsx file")
    parser.add_argument("--output", required=True, help="Output Excel file containing the standardized evaluation dimensions")
    parser.add_argument("--model", required=True, help="SentenceTransformer model path/name")
    parser.add_argument("--cache-dir", default="outputs/S1_cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-embeddings", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)

    df = pd.read_excel(args.input)
    model = SentenceTransformer(args.model)
    labels, embeddings = generate_dimension_embeddings(
        df,
        model,
        Path(args.cache_dir),
        force=args.force_embeddings,
    )
    result = standardize_dimensions(df, labels, embeddings, seed=args.seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(output, index=False)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
