"""
encoder.py
===========
Optimized CLIP Encoder wrapper using SentenceTransformers and PyTorch.

Encodes both images and text into a shared 512-dimensional vector space
using the pre-trained CLIP (ViT-B/32) model.

Optimizations:
    - Multi-threaded CPU execution (torch.set_num_threads)
    - Batch encoding for images and text with configurable batch size
    - L2 normalization for direct inner product cosine similarity
    - Automatic device detection (CUDA if available, else optimized CPU)
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Union
from PIL import Image
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "clip-ViT-B-32"


class CLIPEncoder:
    """
    High-performance wrapper for SentenceTransformer CLIP ViT-B/32 model.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, num_threads: int = None):
        """
        Initialize the CLIP encoder.

        Args:
            model_name: SentenceTransformer CLIP model identifier.
            num_threads: Number of CPU threads for PyTorch operations (default: all logical cores).
        """
        if num_threads is None:
            num_threads = os.cpu_count() or 8
        try:
            torch.set_num_threads(num_threads)
        except Exception:
            pass

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        print(f"[CLIPEncoder] Loading {model_name} on {self.device.upper()} (threads={num_threads})...")

        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_embedding_dimension()
        print(f"[CLIPEncoder] Loaded successfully. Embedding dimension: {self.embedding_dim}")

    def encode_text(self, texts: Union[str, List[str]], batch_size: int = 128,
                    show_progress: bool = False, normalize: bool = True) -> np.ndarray:
        """
        Encode text queries into normalized 512-d CLIP embeddings.

        Args:
            texts: A single text string or list of text strings.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.
            normalize: L2 normalize embeddings (unit norm).

        Returns:
            np.ndarray of shape (N, 512) float32.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embeddings.astype(np.float32)

    def encode_images(self, images_or_paths: List[Union[str, Path, Image.Image]],
                      batch_size: int = 128, show_progress: bool = True,
                      normalize: bool = True) -> np.ndarray:
        """
        Encode a list of PIL Images or image file paths into normalized 512-d embeddings.
        Streams images in chunks of batch_size to eliminate memory and disk I/O bottlenecks.
        """
        if isinstance(images_or_paths, (str, Path, Image.Image)):
            images_or_paths = [images_or_paths]

        if not images_or_paths:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        total = len(images_or_paths)
        all_embeddings = []
        num_batches = (total + batch_size - 1) // batch_size

        for b_idx in range(num_batches):
            start_idx = b_idx * batch_size
            end_idx = min(start_idx + batch_size, total)
            chunk = images_or_paths[start_idx:end_idx]

            batch_images = []
            for item in chunk:
                if isinstance(item, (str, Path)):
                    try:
                        img = Image.open(item).convert("RGB")
                        batch_images.append(img)
                    except Exception:
                        pass
                elif isinstance(item, Image.Image):
                    batch_images.append(item.convert("RGB"))

            if batch_images:
                chunk_emb = self.model.encode(
                    batch_images,
                    batch_size=len(batch_images),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=normalize,
                )
                all_embeddings.append(chunk_emb)

            if show_progress and ((b_idx + 1) % 5 == 0 or b_idx == num_batches - 1):
                pct = (end_idx / total) * 100
                print(f"  [Progress] {end_idx:,}/{total:,} images encoded ({pct:.1f}%)", flush=True)

        if all_embeddings:
            return np.vstack(all_embeddings).astype(np.float32)
        return np.zeros((0, self.embedding_dim), dtype=np.float32)

    def encode_single_image(self, image: Union[str, Path, Image.Image],
                            normalize: bool = True) -> np.ndarray:
        """
        Encode a single image into a (1, 512) embedding.
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        emb = self.model.encode(
            [image],
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)
