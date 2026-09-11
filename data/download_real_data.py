"""
download_real_data.py
======================
Downloads and prepares the real e-commerce fashion dataset from Hugging Face:
`ashraq/fashion-product-images-small` (44,072 products from Kaggle/Myntra).

Extracts a reproducible subset of 10,000 real fashion products:
  - Saves product images to data/real_product_images/
  - Saves metadata CSV to data/real_products.csv
  - Creates a held-out evaluation query split (1,000 queries) with random seed 42

Usage:
    python data/download_real_data.py --num_samples 10000 --seed 42
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# Fixed random seed for reproducibility
DEFAULT_SEED = 42
NUM_SAMPLES = 10000
NUM_EVAL_QUERIES = 1000


def prepare_real_dataset(num_samples: int = NUM_SAMPLES,
                         num_eval_queries: int = NUM_EVAL_QUERIES,
                         seed: int = DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    images_dir = data_dir / "real_product_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "real_products.csv"
    eval_csv_path = data_dir / "eval_queries.csv"

    print("=" * 65)
    print("  Downloading & Preparing Real Fashion Dataset")
    print("  Source: ashraq/fashion-product-images-small (Hugging Face)")
    print(f"  Target sample size: {num_samples} products")
    print(f"  Evaluation query set: {num_eval_queries} queries (seed={seed})")
    print("=" * 65)

    # Load dataset from HuggingFace
    print("\n[1/4] Loading dataset from local cache / Hugging Face...")
    ds = load_dataset("ashraq/fashion-product-images-small", split="train")
    total_available = len(ds)
    print(f"  Total records available in dataset: {total_available:,}")

    # Deterministic sampling of indices
    all_indices = list(range(total_available))
    random.seed(seed)
    selected_indices = random.sample(all_indices, min(num_samples, total_available))

    print(f"\n[2/4] Extracting {len(selected_indices):,} product images and metadata...")
    records = []
    
    for idx in tqdm(selected_indices, desc="Extracting products"):
        item = ds[idx]
        pid = int(item["id"])
        img = item["image"]
        title = item["productDisplayName"]

        # Validate title and image
        if not title or not isinstance(title, str) or len(title.strip()) < 2:
            continue
        if img is None:
            continue

        filename = f"{pid}.jpg"
        filepath = images_dir / filename

        # Save image if not already present
        if not filepath.exists():
            img.convert("RGB").save(filepath, "JPEG", quality=90)

        records.append({
            "product_id": pid,
            "gender": item.get("gender", "Unisex") or "Unisex",
            "masterCategory": item.get("masterCategory", "Apparel") or "Apparel",
            "subCategory": item.get("subCategory", "Topwear") or "Topwear",
            "articleType": item.get("articleType", "Tshirts") or "Tshirts",
            "baseColour": item.get("baseColour", "Unknown") or "Unknown",
            "season": item.get("season", "Fall") or "Fall",
            "year": int(item["year"]) if item.get("year") is not None and not np.isnan(item["year"]) else 2016,
            "usage": item.get("usage", "Casual") or "Casual",
            "title": title.strip(),
            "image_filename": filename,
        })

    df = pd.DataFrame(records)
    # Deduplicate product_id if any
    df = df.drop_duplicates(subset=["product_id"]).reset_index(drop=True)
    actual_count = len(df)
    print(f"\n[3/4] Successfully prepared {actual_count:,} valid product records.")

    # Save catalog metadata
    df.to_csv(csv_path, index=False)
    print(f"  Catalog metadata saved to: {csv_path}")

    # [4/4] Create held-out evaluation query set (deterministic split)
    print(f"\n[4/4] Sampling {num_eval_queries} held-out evaluation queries...")
    eval_df = df.sample(n=min(num_eval_queries, actual_count), random_state=seed).reset_index(drop=True)
    eval_df.to_csv(eval_csv_path, index=False)
    print(f"  Evaluation query set saved to: {eval_csv_path}")

    print("\n" + "=" * 65)
    print("  Category Distribution (Top 10):")
    for cat, count in df["subCategory"].value_counts().head(10).items():
        print(f"    {cat:<25s}: {count:>5d}")
    print("=" * 65)
    return actual_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare real fashion dataset")
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--num_eval_queries", type=int, default=NUM_EVAL_QUERIES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    prepare_real_dataset(args.num_samples, args.num_eval_queries, args.seed)
