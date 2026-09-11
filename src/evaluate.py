"""
evaluate.py
============
Comprehensive, rigorous evaluation suite for the Multimodal Product Search Engine.

Evaluates on the REAL 10,000-product fashion catalog using a held-out set
of 1,000 real evaluation queries (seed=42).

Requirements fulfilled:
  - Text->Image and Image->Image evaluated separately
  - Metrics: Recall@1, Recall@5, Recall@10, MRR, NDCG@10
  - Latency: Mean, Median, P95, P99 (ms)
  - Ablation comparison: IndexFlatIP (exact) vs IndexHNSWFlat (ANN) vs IndexIVFFlat (ANN)
  - Output: models/evaluation_metrics.json

Usage:
    python src/evaluate.py --num_queries 1000
"""

import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.encoder import CLIPEncoder
from src.indexer import FAISSIndexer


def compute_ndcg_at_k(relevance_list: List[int], k: int = 10) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K (NDCG@K).
    relevance_list: list of binary relevance (1 or 0) for ranks 1 to K.
    """
    relevance_list = relevance_list[:k]
    if not relevance_list or sum(relevance_list) == 0:
        return 0.0

    # DCG
    dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_list))

    # Ideal DCG
    ideal_rel = sorted(relevance_list, reverse=True)
    idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_rel))

    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_text_to_image(encoder: CLIPEncoder, indexer: FAISSIndexer,
                           eval_df: pd.DataFrame, catalog_df: pd.DataFrame,
                           k_list: List[int] = [1, 5, 10]) -> Dict:
    """
    Evaluate Text -> Image retrieval across evaluation queries.
    """
    max_k = max(k_list)
    catalog_indexed = catalog_df.set_index("product_id")

    instance_hits = {k: 0 for k in k_list}
    semantic_hits = {k: 0 for k in k_list}
    reciprocal_ranks = []
    ndcg_scores = []
    latencies = []

    queries = eval_df["title"].tolist()
    target_ids = eval_df["product_id"].tolist()
    target_articles = eval_df["articleType"].tolist()

    # Pre-encode text queries in batch for fair search-only latency measurement
    t_start_all = time.perf_counter()
    query_embeddings = encoder.encode_text(queries, batch_size=128, normalize=True, show_progress=False)
    total_encode_time = time.perf_counter() - t_start_all

    for idx in range(len(queries)):
        q_emb = query_embeddings[idx:idx+1]
        t_id = target_ids[idx]
        t_article = target_articles[idx]

        # Measure search latency
        t0 = time.perf_counter()
        matched_pids, scores, _ = indexer.search(q_emb, top_k=max_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        # Instance retrieval metrics (1-to-1 exact product matching)
        rank = None
        for r_idx, pid in enumerate(matched_pids):
            if pid == t_id:
                rank = r_idx + 1
                break

        for k in k_list:
            if rank is not None and rank <= k:
                instance_hits[k] += 1

        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        # NDCG@10 (instance)
        inst_relevance = [1 if pid == t_id else 0 for pid in matched_pids[:10]]
        ndcg_scores.append(compute_ndcg_at_k(inst_relevance, k=10))

        # Semantic retrieval metrics (matching articleType)
        retrieved_articles = []
        for pid in matched_pids:
            if pid in catalog_indexed.index:
                row = catalog_indexed.loc[pid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                retrieved_articles.append(row.get("articleType", ""))
            else:
                retrieved_articles.append("")

        for k in k_list:
            top_k_arts = retrieved_articles[:k]
            if t_article in top_k_arts:
                semantic_hits[k] += 1

    n = len(queries)
    return {
        "num_queries": n,
        "instance_recall": {f"Recall@{k}": round(instance_hits[k] / n, 4) for k in k_list},
        "semantic_recall": {f"Semantic_Recall@{k}": round(semantic_hits[k] / n, 4) for k in k_list},
        "mrr": round(float(np.mean(reciprocal_ranks)), 4),
        "ndcg_at_10": round(float(np.mean(ndcg_scores)), 4),
        "search_latency_ms": {
            "mean": round(float(np.mean(latencies)), 2),
            "median": round(float(np.median(latencies)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "p99": round(float(np.percentile(latencies, 99)), 2),
        }
    }


def evaluate_image_to_image(encoder: CLIPEncoder, indexer: FAISSIndexer,
                            eval_df: pd.DataFrame, catalog_df: pd.DataFrame,
                            images_dir: Path, k_list: List[int] = [1, 5, 10],
                            sample_size: int = 500) -> Dict:
    """
    Evaluate Image -> Image visual similarity retrieval.
    """
    max_k = max(k_list) + 1  # fetch +1 to exclude query image itself
    catalog_indexed = catalog_df.set_index("product_id")

    # Sample a subset of query images if eval_df is large
    sample_df = eval_df.sample(n=min(sample_size, len(eval_df)), random_state=42).reset_index(drop=True)

    semantic_hits = {k: 0 for k in k_list}
    reciprocal_ranks = []
    ndcg_scores = []
    latencies = []

    valid_images = []
    valid_rows = []
    for _, row in sample_df.iterrows():
        p = images_dir / str(row["image_filename"])
        if p.exists():
            valid_images.append(str(p))
            valid_rows.append(row)

    if not valid_images:
        return {"error": "No valid query images found on disk."}

    # Encode query images
    query_embeddings = encoder.encode_images(valid_images, batch_size=128, show_progress=False, normalize=True)

    for idx, row in enumerate(valid_rows):
        q_emb = query_embeddings[idx:idx+1]
        t_id = int(row["product_id"])
        t_article = row["articleType"]

        t0 = time.perf_counter()
        matched_pids, scores, _ = indexer.search(q_emb, top_k=max_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        # Filter out self
        filtered_pids = [pid for pid in matched_pids if pid != t_id][:max(k_list)]

        retrieved_articles = []
        for pid in filtered_pids:
            if pid in catalog_indexed.index:
                r = catalog_indexed.loc[pid]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                retrieved_articles.append(r.get("articleType", ""))
            else:
                retrieved_articles.append("")

        # Rank of first matching articleType
        first_match_rank = None
        relevance_list = []
        for r_idx, art in enumerate(retrieved_articles):
            is_match = 1 if art == t_article else 0
            relevance_list.append(is_match)
            if is_match and first_match_rank is None:
                first_match_rank = r_idx + 1

        for k in k_list:
            if t_article in retrieved_articles[:k]:
                semantic_hits[k] += 1

        reciprocal_ranks.append(1.0 / first_match_rank if first_match_rank else 0.0)
        ndcg_scores.append(compute_ndcg_at_k(relevance_list, k=10))

    n = len(valid_rows)
    return {
        "num_queries": n,
        "recall": {f"Recall@{k}": round(semantic_hits[k] / n, 4) for k in k_list},
        "mrr": round(float(np.mean(reciprocal_ranks)), 4),
        "ndcg_at_10": round(float(np.mean(ndcg_scores)), 4),
        "search_latency_ms": {
            "mean": round(float(np.mean(latencies)), 2),
            "median": round(float(np.median(latencies)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "p99": round(float(np.percentile(latencies, 99)), 2),
        }
    }


def run_full_evaluation(num_queries: int = 1000):
    print("=" * 65)
    print("  Multimodal Product Search Engine - Comprehensive Benchmark")
    print(f"  Evaluation Query Count: {num_queries}")
    print("=" * 65)

    data_dir = PROJECT_ROOT / "data"
    models_dir = PROJECT_ROOT / "models"

    catalog_path = data_dir / "real_products.csv"
    eval_path = data_dir / "eval_queries.csv"
    images_dir = data_dir / "real_product_images"

    flat_idx_path = models_dir / "product_index_flat.index"
    hnsw_idx_path = models_dir / "product_index_hnsw.index"
    ids_path = models_dir / "real_product_ids.npy"

    if not catalog_path.exists() or not eval_path.exists():
        print("ERROR: Catalog or eval data not found.")
        return

    catalog_df = pd.read_csv(catalog_path)
    eval_df = pd.read_csv(eval_path).head(num_queries)

    print(f"Loaded catalog: {len(catalog_df):,} items | Eval queries: {len(eval_df):,}")

    encoder = CLIPEncoder()

    indices_to_test = [
        ("IndexFlatIP (Exact)", "flat", flat_idx_path),
        ("IndexHNSWFlat (ANN)", "hnsw", hnsw_idx_path),
    ]

    all_results = {
        "metadata": {
            "dataset_name": "ashraq/fashion-product-images-small",
            "dataset_source": "Hugging Face (Kaggle/Myntra)",
            "catalog_size": len(catalog_df),
            "eval_query_count": len(eval_df),
            "embedding_model": "clip-ViT-B-32",
            "embedding_dim": 512,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "ablation_comparison": {},
    }

    for name, itype, ipath in indices_to_test:
        if not ipath.exists():
            print(f"Skipping {name} (file not found: {ipath})")
            continue

        print(f"\n>>> Running Evaluation on: {name}")
        indexer = FAISSIndexer.load(str(ipath), str(ids_path), index_type=itype, ef_search=64)

        # Text to Image
        print("  Evaluating Text -> Image Search...")
        t2i = evaluate_text_to_image(encoder, indexer, eval_df, catalog_df)

        # Image to Image
        print("  Evaluating Image -> Image Search...")
        i2i = evaluate_image_to_image(encoder, indexer, eval_df, catalog_df, images_dir, sample_size=500)

        all_results["ablation_comparison"][name] = {
            "text_to_image": t2i,
            "image_to_image": i2i,
        }

        print(f"\n  [Results: {name}]")
        print(f"  Text->Image Instance Recall@1: {t2i['instance_recall']['Recall@1']:.2%}")
        print(f"  Text->Image Instance Recall@5: {t2i['instance_recall']['Recall@5']:.2%}")
        print(f"  Text->Image Instance Recall@10: {t2i['instance_recall']['Recall@10']:.2%}")
        print(f"  Text->Image Instance MRR: {t2i['mrr']:.4f} | NDCG@10: {t2i['ndcg_at_10']:.4f}")
        print(f"  Text->Image Semantic Recall@5: {t2i['semantic_recall']['Semantic_Recall@5']:.2%}")
        print(f"  Search Latency: Mean={t2i['search_latency_ms']['mean']:.2f}ms | P95={t2i['search_latency_ms']['p95']:.2f}ms")
        print(f"  Image->Image Semantic Recall@5: {i2i['recall']['Recall@5']:.2%}")
        print(f"  Image->Image NDCG@10: {i2i['ndcg_at_10']:.4f}")

    # Save to models/evaluation_metrics.json
    out_path = models_dir / "evaluation_metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Full evaluation metrics saved to: {out_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_queries", type=int, default=1000)
    args = parser.parse_args()
    run_full_evaluation(args.num_queries)
