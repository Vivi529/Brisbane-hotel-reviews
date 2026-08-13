"""Step 3: Business dimension-guided contrastive representation learning (BD-CRL).

The final experiment is MNR/InfoNCE-only. Each semantic group is defined as
``C || D_Final``. A training item samples an anchor-positive pair from the same
group; other pairs in the batch act as negatives.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_group_map(df: pd.DataFrame) -> dict[str, list[str]]:
    grouped = df.groupby("group_key")["ft_text"].apply(list).to_dict()
    return {key: texts for key, texts in grouped.items() if len(texts) >= 2}


class ContrastiveMNRDataset(Dataset):
    def __init__(self, groups_dict):
        self.groups_dict = groups_dict
        self.keys = list(groups_dict)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        texts = self.groups_dict[self.keys[idx]]
        anchor, positive = random.sample(texts, 2)
        return {"anchor": anchor, "positive": positive}


def collate_mnr_batch(samples):
    return {
        "anchors": [s["anchor"] for s in samples],
        "positives": [s["positive"] for s in samples],
    }


def get_embeddings_with_grad(texts, encoder, device):
    encoder.train()
    tokenizer = encoder._first_module().tokenizer
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)
    outputs = encoder(batch)
    return outputs["sentence_embedding"]


def evaluate_mean_diff(encoder, val_loader, device, neg_per_pos: int = 1):
    """Retain the original validation criterion used for early stopping."""
    encoder.eval()
    all_pos, all_neg = [], []

    neg_pool = []
    for batch in val_loader:
        neg_pool.extend(batch["positives"])

    with torch.no_grad():
        for batch in val_loader:
            anchors = batch["anchors"]
            positives = batch["positives"]
            emb_a = encoder.encode(anchors, convert_to_tensor=True, show_progress_bar=False, device=device)
            emb_p = encoder.encode(positives, convert_to_tensor=True, show_progress_bar=False, device=device)
            all_pos.append(F.cosine_similarity(emb_a, emb_p, dim=1).cpu().numpy())

            negatives = []
            for _ in anchors:
                negatives.extend(random.sample(neg_pool, neg_per_pos))
            emb_n = encoder.encode(negatives, convert_to_tensor=True, show_progress_bar=False, device=device)

            batch_size, dim = emb_a.shape
            emb_a_expanded = (
                emb_a.unsqueeze(1)
                .expand(batch_size, neg_per_pos, dim)
                .reshape(-1, dim)
            )
            all_neg.append(F.cosine_similarity(emb_a_expanded, emb_n, dim=1).cpu().numpy())

    pos = np.concatenate(all_pos) if all_pos else np.array([])
    neg = np.concatenate(all_neg) if all_neg else np.array([])
    mean_pos = float(np.mean(pos)) if pos.size else None
    mean_neg = float(np.mean(neg)) if neg.size else None
    mean_diff = mean_pos - mean_neg if mean_pos is not None and mean_neg is not None else None
    encoder.train()
    return {"mean_pos": mean_pos, "mean_neg": mean_neg, "mean_diff": mean_diff}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="S2 output .xlsx")
    parser.add_argument("--base-model", required=True, help="Base all-MiniLM-L6-v2 path/name")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.input).fillna("")
    df = df[df["c"].astype(str).str.strip() != ""].copy()

    # Structured semantic injection: [C] A O (D)
    df["ft_text"] = df.apply(
        lambda r: f"[{r['c']}] {r['a']} {r['o']} ({r['D_Final']})".strip().lower(),
        axis=1,
    )
    df["group_key"] = df["c"].astype(str) + "||" + df["D_Final"].astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_split, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(df, groups=df["group_key"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_groups = build_group_map(train_df)
    val_groups = build_group_map(val_df)
    print(f"Rows: {len(df)}; train groups: {len(train_groups)}; val groups: {len(val_groups)}")

    train_loader = DataLoader(
        ContrastiveMNRDataset(train_groups),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_mnr_batch,
    )
    val_loader = DataLoader(
        ContrastiveMNRDataset(val_groups),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_mnr_batch,
    )

    encoder = SentenceTransformer(args.base_model, device=device)
    encoder.max_seq_length = 128
    encoder.to(device)
    for param in encoder.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = math.ceil(len(train_loader) / args.accum_steps) * args.epochs
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=total_steps)

    best_val = -1e9
    counter_no_improve = 0
    history = {"train_loss": [], "val_mean_diff": []}

    for epoch in range(args.epochs):
        encoder.train()
        optimizer.zero_grad()
        running_loss = 0.0
        step = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in pbar:
            anchors = batch["anchors"]
            positives = batch["positives"]
            emb_all = get_embeddings_with_grad(anchors + positives, encoder, device)
            batch_size = len(anchors)
            emb_a = F.normalize(emb_all[:batch_size], p=2, dim=1)
            emb_p = F.normalize(emb_all[batch_size:], p=2, dim=1)

            logits = (emb_a @ emb_p.T) / args.temperature
            labels = torch.arange(batch_size, device=logits.device)
            loss = F.cross_entropy(logits, labels) / args.accum_steps
            loss.backward()

            running_loss += loss.item() * args.accum_steps
            if (step + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1
            pbar.set_postfix(loss=running_loss / step)

        avg_loss = running_loss / max(step, 1)
        history["train_loss"].append(avg_loss)
        metrics = evaluate_mean_diff(encoder, val_loader, device)
        history["val_mean_diff"].append(metrics["mean_diff"])
        print(
            f"Epoch {epoch + 1}: train_loss={avg_loss:.4f}, "
            f"val_mean_diff={metrics['mean_diff']:.4f}"
        )

        if metrics["mean_diff"] is not None and metrics["mean_diff"] > best_val + 1e-6:
            best_val = metrics["mean_diff"]
            counter_no_improve = 0
            encoder.save(str(output_dir / "best_model"))
            print("Saved new best model.")
        else:
            counter_no_improve += 1
            if counter_no_improve >= args.patience:
                print("Early stopping triggered.")
                break

    with (output_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], marker="o")
    axes[0].set_title("Train loss")
    axes[0].set_xlabel("Epoch")
    axes[1].plot(history["val_mean_diff"], marker="o")
    axes[1].set_title("Validation mean_diff")
    axes[1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=200)
    plt.close(fig)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
