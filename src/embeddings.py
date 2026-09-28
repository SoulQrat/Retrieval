"""A thin, disk-cached wrapper around a local multilingual-E5 model.

E5 models (https://huggingface.co/intfloat/multilingual-e5-large) expect
every input to be prefixed with `"query: "` or `"passage: "` depending on
which side of a retrieval pair it plays, and embeddings to be L2-normalized
before taking a dot product as cosine similarity. This module hides both
details behind `E5Embedder.encode`.

Encoding is the most expensive step in the pipeline, so results are cached
to disk keyed by a caller-provided `cache_key` plus a content hash of the
input texts: unrelated calls never collide, but rerunning a notebook cell
with unchanged inputs is instant.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Optional, Sequence

import numpy as np


def pick_device(preferred: Optional[str] = None) -> str:
    """Selects the best available torch device.

    Args:
        preferred: If given, returned as-is (lets callers override
            detection, e.g. to force `"cpu"`).

    Returns:
        `"mps"` on Apple Silicon with GPU acceleration available, `"cuda"`
        if an NVIDIA GPU is available, otherwise `"cpu"`.
    """
    if preferred is not None:
        return preferred
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class E5Embedder:
    """Encodes text with a local multilingual-E5 SentenceTransformer model.

    Attributes:
        model_dir: Path to a local E5 model directory (as produced by
            `huggingface_hub.snapshot_download`).
        device: Torch device used for inference.
        cache_dir: Directory where encoded arrays are cached as `.npy`
            files. `None` disables caching.
        batch_size: Batch size passed to `SentenceTransformer.encode`.
        show_progress: Whether to display a progress bar while encoding.
    """

    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        batch_size: int = 256,
        show_progress: bool = True,
    ) -> None:
        self.model_dir = model_dir
        self.device = pick_device(device)
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else None
        self.batch_size = batch_size
        self.show_progress = show_progress
        self._model = None

    @property
    def model(self):
        """Lazily loads the SentenceTransformer model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_dir, device=self.device)
        return self._model

    def encode(
        self,
        texts: Sequence[str],
        prefix: str,
        cache_key: Optional[str] = None,
    ) -> np.ndarray:
        """Encodes `texts` into L2-normalized E5 embeddings.

        Args:
            texts: Raw (unprefixed) input strings.
            prefix: Either `"query"` or `"passage"`, per the E5 convention.
            cache_key: A short, human-readable name for this call site
                (e.g. `"item_titles"`). When set together with `cache_dir`,
                results are cached and reused across notebook reruns.

        Returns:
            A `(len(texts), hidden_size)` float32 array of embeddings.

        Raises:
            ValueError: If `prefix` is not `"query"` or `"passage"`.
        """
        if prefix not in ("query", "passage"):
            raise ValueError(f"prefix must be 'query' or 'passage', got {prefix!r}")

        texts = list(texts)
        cache_path = None
        if cache_key is not None and self.cache_dir is not None:
            cache_path = self._cache_path(cache_key, prefix, texts)
            if cache_path.exists():
                return np.load(cache_path)

        prefixed = [f"{prefix}: {text}" for text in texts]
        embeddings = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        if cache_path is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embeddings)
        return embeddings

    def _cache_path(
        self, cache_key: str, prefix: str, texts: Sequence[str]
    ) -> pathlib.Path:
        """Builds a content-addressed cache file path for `texts`."""
        hasher = hashlib.sha1(f"{self.model_dir}|{prefix}|{len(texts)}".encode())
        for text in texts:
            hasher.update((text or "").encode("utf-8", errors="ignore"))
            hasher.update(b"\x1f")
        digest = hasher.hexdigest()[:20]
        return self.cache_dir / f"{cache_key}_{prefix}_{digest}.npy"
