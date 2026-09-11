"""
search.py
==========
Multimodal Product Search Engine combining CLIP encoder + FAISS index.
Supports:
  - Text-to-Image search
  - Image-to-Image search
  - Index selection (FlatIP exact vs HNSW ANN)
  - Real e-commerce product catalog metadata
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union, Optional
from PIL import Image

from src.encoder import CLIPEncoder
from src.indexer import FAISSIndexer


class ProductSearchEngine:
    """
    End-to-end multimodal product search engine for real e-commerce catalogs.
    """

    def __init__(self, encoder: CLIPEncoder, indexer: FAISSIndexer,
                 product_catalog: pd.DataFrame, images_dir: Union[str, Path]):
        self.encoder = encoder
        self.indexer = indexer
        self.catalog = product_catalog.copy()
        if "product_id" in self.catalog.columns:
            self.catalog = self.catalog.set_index("product_id")
        self.images_dir = Path(images_dir)

    def switch_indexer(self, new_indexer: FAISSIndexer):
        """Switch the underlying indexer (e.g. from Flat to HNSW)."""
        self.indexer = new_indexer

    def search_by_text(self, query: str, top_k: int = 10) -> Dict:
        """
        Text-to-Image search: user types text -> retrieves matching product images.
        """
        t0 = time.perf_counter()
        query_emb = self.encoder.encode_text(query, normalize=True)
        t_encode = (time.perf_counter() - t0) * 1000

        t_search_start = time.perf_counter()
        product_ids, scores, _ = self.indexer.search(query_emb, top_k=top_k)
        t_search = (time.perf_counter() - t_search_start) * 1000

        results = self._build_results(product_ids, scores)
        t_total = (time.perf_counter() - t0) * 1000

        return {
            "query": query,
            "search_mode": "text_to_image",
            "index_type": self.indexer.index_type,
            "num_results": len(results),
            "encoding_latency_ms": round(t_encode, 2),
            "search_latency_ms": round(t_search, 2),
            "total_latency_ms": round(t_total, 2),
            "results": results,
        }

    def search_by_image(self, image: Union[str, Path, Image.Image], top_k: int = 10,
                        exclude_id: Optional[int] = None) -> Dict:
        """
        Image-to-Image search: user provides an image -> retrieves visually similar products.
        """
        t0 = time.perf_counter()
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        query_emb = self.encoder.encode_single_image(image, normalize=True)
        t_encode = (time.perf_counter() - t0) * 1000

        t_search_start = time.perf_counter()
        # Retrieve extra if we need to filter out the query item itself
        fetch_k = top_k + 1 if exclude_id is not None else top_k
        product_ids, scores, _ = self.indexer.search(query_emb, top_k=fetch_k)
        t_search = (time.perf_counter() - t_search_start) * 1000

        if exclude_id is not None:
            mask = product_ids != exclude_id
            product_ids = product_ids[mask][:top_k]
            scores = scores[mask][:top_k]

        results = self._build_results(product_ids, scores)
        t_total = (time.perf_counter() - t0) * 1000

        return {
            "search_mode": "image_to_image",
            "index_type": self.indexer.index_type,
            "num_results": len(results),
            "encoding_latency_ms": round(t_encode, 2),
            "search_latency_ms": round(t_search, 2),
            "total_latency_ms": round(t_total, 2),
            "results": results,
        }

    def _build_results(self, product_ids: np.ndarray, scores: np.ndarray) -> List[Dict]:
        results = []
        for pid, score in zip(product_ids, scores):
            pid = int(pid)
            if pid in self.catalog.index:
                row = self.catalog.loc[pid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]

                img_file = row.get("image_filename", f"{pid}.jpg")
                img_path = self.images_dir / img_file

                results.append({
                    "product_id": pid,
                    "title": str(row.get("title", row.get("productDisplayName", "N/A"))),
                    "masterCategory": str(row.get("masterCategory", "N/A")),
                    "subCategory": str(row.get("subCategory", "N/A")),
                    "articleType": str(row.get("articleType", "N/A")),
                    "baseColour": str(row.get("baseColour", "N/A")),
                    "gender": str(row.get("gender", "N/A")),
                    "usage": str(row.get("usage", "N/A")),
                    "image_filename": img_file,
                    "image_path": str(img_path) if img_path.exists() else "",
                    "similarity_score": round(float(score), 4),
                })
        return results

    @classmethod
    def load_engine(cls, models_dir: str = "models", data_dir: str = "data",
                    index_type: str = "flat") -> "ProductSearchEngine":
        models_dir = Path(models_dir)
        data_dir = Path(data_dir)

        encoder = CLIPEncoder()

        # Check real product catalog first, fall back to sample
        catalog_path = data_dir / "real_products.csv"
        if not catalog_path.exists():
            catalog_path = data_dir / "sample_products.csv"

        images_dir = data_dir / "real_product_images"
        if not images_dir.exists():
            images_dir = data_dir / "product_images"

        catalog = pd.read_csv(catalog_path)

        # Choose index file based on index_type
        if index_type == "hnsw":
            idx_file = models_dir / "product_index_hnsw.index"
            ids_file = models_dir / "real_product_ids.npy"
        elif index_type == "ivf":
            idx_file = models_dir / "product_index_ivf.index"
            ids_file = models_dir / "real_product_ids.npy"
        else:
            idx_file = models_dir / "product_index_flat.index"
            if not idx_file.exists():
                idx_file = models_dir / "product_index.index"
            ids_file = models_dir / "real_product_ids.npy"
            if not ids_file.exists():
                ids_file = models_dir / "product_ids.npy"

        indexer = FAISSIndexer.load(
            index_path=str(idx_file),
            ids_path=str(ids_file),
            index_type=index_type,
        )

        return cls(encoder, indexer, catalog, images_dir)
