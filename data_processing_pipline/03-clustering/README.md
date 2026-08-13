# Service-element clustering pipeline

This folder contains the four sequential clustering/representation-learning steps used in the hotel-review study, plus the clustering-method comparison script.

## Files

1. `S1_standardize_dimensions.py` — initial standardization of `evaluation_dimension` labels using SentenceTransformer + UMAP + HDBSCAN.
2. `S2_llm_assisted_merge.py` — rule-based retrieval plus LLM-assisted merging of low-frequency dimension labels.
3. `S3_train_bd_crl.py` — BD-CRL fine-tuning with MNR/InfoNCE loss using semantic groups `C || D_Final`.
4. `S4_final_clustering.py` — category-stratified final HDBSCAN clustering using the fine-tuned encoder and optional medoid-based noise reassignment.
5. `S4_compare_clustering_methods.py` — alternative clustering algorithms used with the Step-4 representation/evaluation interface: K-means, GMM, and kNN-graph pivot correlation clustering.
6. `clustering_utils.py` — shared preprocessing, embedding cache, medoid, noise-reassignment, and evaluation utilities.

## Data flow

```text
AOP/AOCS extraction result
        |
        v
S1_standardize_dimensions.py
        |  adds D_Std
        v
S2_llm_assisted_merge.py
        |  adds D_Final
        v
S3_train_bd_crl.py
        |  saves best_model/
        v
S4_final_clustering.py
        |  final clusters + metrics
        +---------------------> S4_compare_clustering_methods.py
                                alternative clustering experiments
```

## Example commands

Replace model paths with the corresponding local or Hugging Face model identifiers.

```bash
python S1_standardize_dimensions.py \
  --input data/AOP_extraction_results.xlsx \
  --output outputs/S1_dimension_standardized.xlsx \
  --model /path/to/all-mpnet-base-v2

python S2_llm_assisted_merge.py \
  --input outputs/S1_dimension_standardized.xlsx \
  --output outputs/S2_dimension_merged.xlsx \
  --sbert-model /path/to/all-mpnet-base-v2 \
  --llm-model /path/to/Qwen2-7B-Instruct-GPTQ-Int4

python S3_train_bd_crl.py \
  --input outputs/S2_dimension_merged.xlsx \
  --base-model /path/to/all-MiniLM-L6-v2 \
  --output-dir outputs/bd_crl

python S4_final_clustering.py \
  --input outputs/S2_dimension_merged.xlsx \
  --model outputs/bd_crl/best_model \
  --output-dir outputs/final_clustering

python S4_compare_clustering_methods.py \
  --input outputs/S2_dimension_merged.xlsx \
  --model outputs/bd_crl/best_model \
  --hdbscan-reference outputs/final_clustering/S4_final_clusters.csv \
  --output-dir outputs/clustering_comparison \
  --method all \
  --text-mode aod
```


## Comparison experiments

`S4_compare_clustering_methods.py` retains all three alternative algorithms present in the research script. For the paper's HDBSCAN/K-means/GMM comparison, first run the final HDBSCAN script, then pass its `S4_final_clusters.csv` through `--hdbscan-reference`; K-means and GMM will automatically use the HDBSCAN-derived cluster count for each attribute category. The correlation-clustering implementation is retained as an additional experimental method and can be run separately if needed.

For comparisons among representation strategies (e.g., CJR, AC-CRL, BD-CRL), run the same clustering method with the corresponding model path and text representation. Use a different `--cache-tag` for each representation strategy.
