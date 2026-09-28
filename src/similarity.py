"""Dense embedding similarity between a query and every item.

This is the "propose something of your own" signal for query/title-
description relevance mentioned in the task brief: `coverage` (see
`src/coverage.py`) only rewards exact keyword overlap, so it misses
paraphrases and synonyms (e.g. query "уборка квартиры" vs title "клининг
жилых помещений"). A dense E5 sentence embedding captures that semantic
relatedness and complements lexical coverage in the final fusion.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class DenseSimilarityIndex:
    """Cosine similarity between a query embedding and a corpus of items.

    Attributes:
        item_embeddings: `(n_documents, hidden_size)` L2-normalized array,
            typically item titles encoded with `E5Embedder`.
        n_documents: Number of documents in the corpus.
    """

    def __init__(self, item_embeddings: np.ndarray) -> None:
        """Stores the (already L2-normalized) item embedding matrix.

        Args:
            item_embeddings: Output of `E5Embedder.encode(..., prefix="passage")`
                for the corpus, in corpus order.
        """
        self.item_embeddings = item_embeddings
        self.n_documents = item_embeddings.shape[0]

    def score(
        self,
        query_embedding: np.ndarray,
        row_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Computes cosine similarity between one query and every document.

        Args:
            query_embedding: A single `(hidden_size,)` L2-normalized query
                embedding, typically from `E5Embedder.encode(..., prefix="query")`.
            row_mask: Optional boolean array of length `n_documents`
                selecting which documents to score.

        Returns:
            A float32 array of cosine similarities, one per selected
            document.
        """
        embeddings = (
            self.item_embeddings if row_mask is None else self.item_embeddings[row_mask]
        )
        return (embeddings @ query_embedding).astype(np.float32)
