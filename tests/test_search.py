"""
test_search.py
===============
Unit tests for the FAISS indexer and search engine supporting:
  - IndexFlatIP, IndexHNSWFlat, IndexIVFFlat
  - Exact and approximate nearest neighbor retrieval
  - Serialization and loading
  - Text-to-image and image-to-image search
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.indexer import FAISSIndexer
from src.encoder import CLIPEncoder
from src.search import ProductSearchEngine


@pytest.fixture(scope="module")
def encoder():
    return CLIPEncoder()


class TestFAISSIndexer:
    """Tests for FAISS index building, searching, and saving across index types."""

    def test_build_flat_index(self):
        """Test building a Flat index with synthetic embeddings."""
        dim = 512
        n = 100
        embeddings = np.random.randn(n, dim).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        product_ids = np.arange(1, n + 1)

        indexer = FAISSIndexer(embedding_dim=dim, index_type="flat")
        indexer.build_index(embeddings, product_ids)

        assert indexer.index.ntotal == n
        assert indexer.index_type == "flat"

    def test_build_hnsw_index(self):
        """Test building an HNSW graph index."""
        dim = 512
        n = 100
        embeddings = np.random.randn(n, dim).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        product_ids = np.arange(1, n + 1)

        indexer = FAISSIndexer(embedding_dim=dim, index_type="hnsw", hnsw_m=16, ef_construction=32, ef_search=32)
        indexer.build_index(embeddings, product_ids)

        assert indexer.index.ntotal == n
        assert indexer.index_type == "hnsw"

    def test_search_returns_results(self):
        """Test that search returns correct number of results and self at rank 1."""
        dim = 512
        n = 50
        embeddings = np.random.randn(n, dim).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        product_ids = np.arange(1, n + 1)

        indexer = FAISSIndexer(embedding_dim=dim, index_type="flat")
        indexer.build_index(embeddings, product_ids)

        query = embeddings[0:1]
        pids, scores, indices = indexer.search(query, top_k=5)

        assert len(pids) == 5
        assert len(scores) == 5
        assert pids[0] == 1
        assert scores[0] > 0.99

    def test_hnsw_accuracy_high(self):
        """Test that HNSW search closely matches FlatIP top-1."""
        dim = 512
        n = 100
        embeddings = np.random.randn(n, dim).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        product_ids = np.arange(1, n + 1)

        indexer_flat = FAISSIndexer(embedding_dim=dim, index_type="flat")
        indexer_flat.build_index(embeddings, product_ids)

        indexer_hnsw = FAISSIndexer(embedding_dim=dim, index_type="hnsw", hnsw_m=32, ef_construction=64, ef_search=64)
        indexer_hnsw.build_index(embeddings, product_ids)

        query = embeddings[10:11]
        pids_flat, _, _ = indexer_flat.search(query, top_k=1)
        pids_hnsw, _, _ = indexer_hnsw.search(query, top_k=1)
        assert pids_flat[0] == pids_hnsw[0]

    def test_save_and_load_flat(self, tmp_path):
        """Test saving and loading Flat index."""
        dim = 512
        n = 30
        embeddings = np.random.randn(n, dim).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        product_ids = np.arange(1, n + 1)

        indexer = FAISSIndexer(embedding_dim=dim, index_type="flat")
        indexer.build_index(embeddings, product_ids)

        index_path = str(tmp_path / "test_flat.index")
        ids_path = str(tmp_path / "test_flat_ids.npy")
        indexer.save(index_path, ids_path)

        loaded = FAISSIndexer.load(index_path, ids_path, index_type="flat")
        assert loaded.index.ntotal == n
        assert len(loaded.product_ids) == n


class TestProductSearchEngine:
    """Tests for the end-to-end search engine."""

    @pytest.fixture
    def mini_engine(self, encoder, tmp_path):
        """Create a mini search engine with 10 products for testing."""
        categories = ["T-Shirts", "Jeans", "Sneakers", "Watches", "Backpacks"]
        products = []
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        for i in range(10):
            cat = categories[i % len(categories)]
            title = f"Test {cat} Product {i}"
            filename = f"test_{i}.jpg"

            color = (i * 25, 100, 200 - i * 15)
            img = Image.new("RGB", (224, 224), color=color)
            img.save(images_dir / filename)

            products.append({
                "product_id": i + 1,
                "title": title,
                "masterCategory": "Apparel",
                "subCategory": cat,
                "articleType": cat,
                "baseColour": "Blue",
                "gender": "Men",
                "usage": "Casual",
                "image_filename": filename,
            })

        catalog = pd.DataFrame(products)

        image_paths = [str(images_dir / p["image_filename"]) for p in products]
        embeddings = encoder.encode_images(image_paths, show_progress=False)

        indexer = FAISSIndexer(embedding_dim=encoder.embedding_dim, index_type="flat")
        indexer.build_index(embeddings, np.array([p["product_id"] for p in products]))

        return ProductSearchEngine(encoder, indexer, catalog, images_dir)

    def test_text_search_returns_results(self, mini_engine):
        results = mini_engine.search_by_text("blue sneakers", top_k=5)
        assert results["search_mode"] == "text_to_image"
        assert results["num_results"] > 0
        assert results["total_latency_ms"] > 0
        assert len(results["results"]) <= 5

    def test_text_search_result_structure(self, mini_engine):
        results = mini_engine.search_by_text("test product", top_k=3)
        for r in results["results"]:
            assert "product_id" in r
            assert "title" in r
            assert "subCategory" in r
            assert "articleType" in r
            assert "similarity_score" in r
            assert isinstance(r["similarity_score"], float)

    def test_image_search_returns_results(self, mini_engine):
        query_image = Image.new("RGB", (224, 224), color=(100, 100, 200))
        results = mini_engine.search_by_image(query_image, top_k=5)
        assert results["search_mode"] == "image_to_image"
        assert results["num_results"] > 0

    def test_switch_indexer(self, mini_engine):
        dim = mini_engine.encoder.embedding_dim
        new_indexer = FAISSIndexer(embedding_dim=dim, index_type="hnsw")
        dummy_embs = np.random.randn(5, dim).astype(np.float32)
        norms = np.linalg.norm(dummy_embs, axis=1, keepdims=True)
        new_indexer.build_index(dummy_embs / norms, np.arange(1, 6))

        mini_engine.switch_indexer(new_indexer)
        assert mini_engine.indexer.index_type == "hnsw"
