"""
indexer.py
===========
FAISS Vector Index Manager supporting both exact and approximate search:
    1. IndexFlatIP: Exact brute-force inner product (cosine similarity on unit vectors). Baseline.
    2. IndexHNSWFlat: Hierarchical Navigable Small World graph index for fast sub-millisecond ANN.
    3. IndexIVFFlat: Inverted file index with Voronoi cells.
"""

import numpy as np
import faiss
from pathlib import Path
from typing import Tuple, Optional


class FAISSIndexer:
    """
    FAISS index manager supporting multiple index types: 'flat', 'hnsw', 'ivf'.
    """

    def __init__(self, embedding_dim: int = 512, index_type: str = "flat",
                 hnsw_m: int = 32, ef_construction: int = 64, ef_search: int = 64,
                 nlist: int = 100, nprobe: int = 10):
        """
        Initialize index.

        Args:
            embedding_dim: 512 for CLIP ViT-B/32.
            index_type: 'flat' (exact), 'hnsw' (graph ANN), or 'ivf' (inverted file ANN).
            hnsw_m: Number of bidirectional links per node in HNSW graph (default: 32).
            ef_construction: Search depth during HNSW index construction (default: 64).
            ef_search: Search depth during HNSW query time (default: 64).
            nlist: Number of cluster centroids for IVF index.
            nprobe: Number of clusters probed at query time for IVF.
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type.lower()
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.nlist = nlist
        self.nprobe = nprobe
        self.product_ids = None

        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(embedding_dim)
        elif self.index_type == "hnsw":
            # HNSW with inner product metric
            self.index = faiss.IndexHNSWFlat(embedding_dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = ef_search
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown index_type '{index_type}'. Must be 'flat', 'hnsw', or 'ivf'.")

        print(f"[FAISSIndexer] Initialized index type='{self.index_type}', dim={embedding_dim}")

    def set_search_params(self, ef_search: Optional[int] = None, nprobe: Optional[int] = None):
        """Configure runtime search accuracy vs latency parameters."""
        if ef_search is not None and hasattr(self.index, "hnsw"):
            self.ef_search = ef_search
            self.index.hnsw.efSearch = ef_search
        if nprobe is not None and hasattr(self.index, "nprobe"):
            self.nprobe = nprobe
            self.index.nprobe = nprobe

    def build_index(self, embeddings: np.ndarray, product_ids: np.ndarray) -> None:
        """
        Add normalized embeddings to index.
        """
        assert embeddings.shape[1] == self.embedding_dim
        assert len(embeddings) == len(product_ids)

        embeddings = embeddings.astype(np.float32)
        self.product_ids = np.array(product_ids)

        if self.index_type == "ivf":
            print(f"[FAISSIndexer] Training IVF index on {len(embeddings)} vectors...")
            self.index.train(embeddings)

        print(f"[FAISSIndexer] Adding {len(embeddings)} vectors to {self.index_type} index...")
        self.index.add(embeddings)
        print(f"[FAISSIndexer] Index built successfully with {self.index.ntotal} items.")

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Search for top-K nearest neighbors.

        Returns:
            (product_ids, similarity_scores, internal_indices)
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)

        if self.index_type == "ivf":
            self.index.nprobe = self.nprobe
        elif self.index_type == "hnsw":
            self.index.hnsw.efSearch = self.ef_search

        scores, indices = self.index.search(query_embedding, top_k)
        scores = scores[0]
        indices = indices[0]

        valid_mask = indices >= 0
        valid_indices = indices[valid_mask]
        valid_scores = scores[valid_mask]
        matched_pids = self.product_ids[valid_indices]

        return matched_pids, valid_scores, valid_indices

    def save(self, index_path: str, ids_path: str) -> None:
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        np.save(ids_path, self.product_ids)
        print(f"[FAISSIndexer] Index saved: {index_path}")

    @classmethod
    def load(cls, index_path: str, ids_path: str, index_type: str = "flat",
             ef_search: int = 64, nprobe: int = 10) -> "FAISSIndexer":
        index = faiss.read_index(str(index_path))
        product_ids = np.load(ids_path)

        indexer = cls.__new__(cls)
        indexer.embedding_dim = index.d
        indexer.index = index
        indexer.index_type = index_type.lower()
        indexer.product_ids = product_ids
        indexer.ef_search = ef_search
        indexer.nprobe = nprobe

        if indexer.index_type == "hnsw" and hasattr(index, "hnsw"):
            indexer.index.hnsw.efSearch = ef_search
        elif indexer.index_type == "ivf" and hasattr(index, "nprobe"):
            indexer.index.nprobe = nprobe

        print(f"[FAISSIndexer] Loaded {indexer.index_type} index with {index.ntotal} vectors (dim={index.d})")
        return indexer
