"""
test_encoder.py
================
Unit tests for the CLIP encoder module.
"""

import sys
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.encoder import CLIPEncoder


@pytest.fixture(scope="module")
def encoder():
    """Load the CLIP encoder once for all tests in this module."""
    return CLIPEncoder()


@pytest.fixture
def sample_image(tmp_path):
    """Create a simple test image."""
    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    path = tmp_path / "test_image.jpg"
    img.save(path)
    return path


class TestCLIPEncoder:
    """Tests for CLIPEncoder functionality."""

    def test_encoder_loads(self, encoder):
        """Test that the CLIP model loads successfully."""
        assert encoder.model is not None
        assert encoder.embedding_dim == 512

    def test_encode_single_text(self, encoder):
        """Test encoding a single text string."""
        embedding = encoder.encode_text("a red leather jacket")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, 512)
        assert embedding.dtype == np.float32

    def test_encode_batch_text(self, encoder):
        """Test encoding a batch of text strings."""
        texts = ["black sneakers", "blue jeans", "white t-shirt"]
        embeddings = encoder.encode_text(texts)
        assert embeddings.shape == (3, 512)

    def test_text_embeddings_normalized(self, encoder):
        """Test that text embeddings are L2-normalized (unit norm)."""
        embedding = encoder.encode_text("red shoes", normalize=True)
        norm = np.linalg.norm(embedding[0])
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_encode_image(self, encoder, sample_image):
        """Test encoding a single image file."""
        embedding = encoder.encode_images(sample_image)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, 512)

    def test_encode_pil_image(self, encoder):
        """Test encoding a PIL Image object directly."""
        img = Image.new("RGB", (224, 224), color=(200, 100, 50))
        embedding = encoder.encode_single_image(img)
        assert embedding.shape == (1, 512)

    def test_image_embeddings_normalized(self, encoder, sample_image):
        """Test that image embeddings are L2-normalized."""
        embedding = encoder.encode_images(sample_image, normalize=True)
        norm = np.linalg.norm(embedding[0])
        assert abs(norm - 1.0) < 1e-5

    def test_cross_modal_similarity(self, encoder, sample_image):
        """Test that text and image embeddings can be compared via dot product."""
        text_emb = encoder.encode_text("a solid brown colored image")
        img_emb = encoder.encode_images(sample_image)

        # Dot product should return a scalar similarity score
        similarity = np.dot(text_emb[0], img_emb[0])
        assert isinstance(float(similarity), float)
        assert -1.0 <= similarity <= 1.0, f"Similarity out of range: {similarity}"

    def test_similar_texts_closer(self, encoder):
        """Test that semantically similar texts have higher similarity than dissimilar ones."""
        emb_jacket = encoder.encode_text("black leather jacket")
        emb_coat = encoder.encode_text("dark leather coat")
        emb_pizza = encoder.encode_text("pepperoni pizza with cheese")

        sim_similar = np.dot(emb_jacket[0], emb_coat[0])
        sim_dissimilar = np.dot(emb_jacket[0], emb_pizza[0])

        assert sim_similar > sim_dissimilar, \
            f"Expected similar texts to be closer: jacket-coat={sim_similar:.4f} vs jacket-pizza={sim_dissimilar:.4f}"

    def test_empty_image_list(self, encoder):
        """Test encoding with no valid images returns empty array."""
        embedding = encoder.encode_images(["/nonexistent/path/image.jpg"])
        assert embedding.shape == (0, 512)
