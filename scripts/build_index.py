"""
build_index.py
===============
Offline pipeline to encode 10,000 real fashion product images and build:
  1. IndexFlatIP: Exact inner product search baseline
  2. IndexHNSWFlat: Fast graph-based ANN index (M=32, efConstruction=64)
  3. IndexIVFFlat: Inverted file index (nlist=100)

Usage:
    python scripts/build_index.py --batch_size 128 --threads 16
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.encoder import CLIPEncoder
from src.indexer import FAISSIndexer


def main():
    parser = argparse.ArgumentParser(description="Build FAISS indices for real fashion catalog")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--force_reencode", action="store_true")
    args = parser.parse_args()

    print("=" * 65)
    print("  Offline Index Pipeline for Real Fashion Products")
    print(f"  Batch size: {args.batch_size} | CPU Threads: {args.threads}")
    print("=" * 65)

    data_dir = PROJECT_ROOT / "data"
    catalog_path = data_dir / "real_products.csv"
    images_dir = data_dir / "real_product_images"

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = models_dir / "real_product_embeddings.npy"
    ids_path = models_dir / "real_product_ids.npy"

    flat_index_path = models_dir / "product_index_flat.index"
    hnsw_index_path = models_dir / "product_index_hnsw.index"
    ivf_index_path = models_dir / "product_index_ivf.index"

    # 1. Load catalog
    if not catalog_path.exists():
        print(f"ERROR: Real catalog not found at {catalog_path}")
        print("Please run 'python data/download_real_data.py' first.")
        sys.exit(1)

    catalog = pd.read_csv(catalog_path)
    total_products = len(catalog)
    print(f"\n[1/4] Loaded catalog metadata: {total_products:,} products.")

    # 2. Check if embeddings are already cached
    embeddings = None
    product_ids = None

    if embeddings_path.exists() and ids_path.exists() and not args.force_reencode:
        print(f"\n[2/4] Found pre-computed embeddings at {embeddings_path}")
        embeddings = np.load(embeddings_path)
        product_ids = np.load(ids_path)
        print(f"  Loaded embedding matrix: {embeddings.shape} float32 (norm={np.linalg.norm(embeddings[0]):.4f})")
    else:
        print(f"\n[2/4] Encoding {total_products:,} real images with CLIP (ViT-B/32)...")
        encoder = CLIPEncoder(num_threads=args.threads)

        img_filenames = catalog["image_filename"].tolist()
        pids = catalog["product_id"].tolist()
        valid_paths = [str(images_dir / fn) for fn in img_filenames if (images_dir / fn).exists()]
        valid_ids = [pid for pid, fn in zip(pids, img_filenames) if (images_dir / fn).exists()]
        total_valid = len(valid_paths)
        print(f"  Valid image files found on disk: {total_valid:,}", flush=True)

        chunk_size = 1000
        num_chunks = (total_valid + chunk_size - 1) // chunk_size
        chunk_files = []
        t0 = time.perf_counter()

        for c_idx in range(num_chunks):
            c_start = c_idx * chunk_size
            c_end = min(c_start + chunk_size, total_valid)
            chk_path = models_dir / f"chk_embs_part_{c_idx}.npy"
            chunk_files.append(chk_path)

            if chk_path.exists():
                print(f"  [Chunk {c_idx+1}/{num_chunks}] Found cached checkpoint ({c_start:,} to {c_end:,})", flush=True)
                continue

            print(f"  [Chunk {c_idx+1}/{num_chunks}] Encoding images {c_start:,} to {c_end:,}...", flush=True)
            c_paths = valid_paths[c_start:c_end]
            c_embs = encoder.encode_images(
                c_paths,
                batch_size=args.batch_size,
                show_progress=True,
                normalize=True
            )
            np.save(chk_path, c_embs)
            print(f"  [Chunk {c_idx+1}/{num_chunks}] Saved {len(c_embs):,} embeddings to {chk_path.name}", flush=True)

        # Merge all chunks
        print(f"\n  Merging {num_chunks} embedding chunks...", flush=True)
        all_parts = [np.load(f) for f in chunk_files]
        embeddings = np.vstack(all_parts).astype(np.float32)
        product_ids = np.array(valid_ids)
        encode_duration = time.perf_counter() - t0

        print(f"  [OK] All {len(embeddings):,} images encoded in {encode_duration:.1f}s ({len(embeddings)/encode_duration:.1f} imgs/sec)", flush=True)

        # Save final merged embeddings & IDs
        np.save(embeddings_path, embeddings)
        np.save(ids_path, product_ids)
        print(f"  Saved final embeddings to: {embeddings_path}", flush=True)
        print(f"  Saved product IDs to: {ids_path}", flush=True)

        # Clean up chunk files
        for f in chunk_files:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass

    # 3. Build IndexFlatIP (Exact Baseline)
    dim = embeddings.shape[1]
    print(f"\n[3/4] Building exact baseline index: IndexFlatIP (dim={dim})...")
    t0 = time.perf_counter()
    indexer_flat = FAISSIndexer(embedding_dim=dim, index_type="flat")
    indexer_flat.build_index(embeddings, product_ids)
    indexer_flat.save(str(flat_index_path), str(ids_path))
    flat_build_time = time.perf_counter() - t0
    print(f"  [OK] IndexFlatIP built and saved in {flat_build_time:.2f}s ({indexer_flat.index.ntotal:,} items).")

    # 4. Build IndexHNSWFlat (Graph ANN)
    print(f"\n[4/4] Building ANN index: IndexHNSWFlat (M=32, efConstruction=64)...")
    t0 = time.perf_counter()
    indexer_hnsw = FAISSIndexer(embedding_dim=dim, index_type="hnsw", hnsw_m=32, ef_construction=64, ef_search=64)
    indexer_hnsw.build_index(embeddings, product_ids)
    indexer_hnsw.save(str(hnsw_index_path), str(ids_path))
    hnsw_build_time = time.perf_counter() - t0
    print(f"  [OK] IndexHNSWFlat built and saved in {hnsw_build_time:.2f}s ({indexer_hnsw.index.ntotal:,} items).")

    # Optional: Build IndexIVFFlat
    print(f"\n[Bonus] Building IndexIVFFlat (nlist=100, nprobe=10)...")
    t0 = time.perf_counter()
    indexer_ivf = FAISSIndexer(embedding_dim=dim, index_type="ivf", nlist=100, nprobe=10)
    indexer_ivf.build_index(embeddings, product_ids)
    indexer_ivf.save(str(ivf_index_path), str(ids_path))
    ivf_build_time = time.perf_counter() - t0
    print(f"  [OK] IndexIVFFlat built and saved in {ivf_build_time:.2f}s.")

    print("\n" + "=" * 65)
    print("  [OK] All FAISS Indices Built Successfully!")
    print(f"  Catalog Size       : {len(product_ids):,} products")
    print(f"  IndexFlatIP file   : {flat_index_path}")
    print(f"  IndexHNSWFlat file : {hnsw_index_path}")
    print(f"  IndexIVFFlat file  : {ivf_index_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
