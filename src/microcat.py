"""A soft signal for how well a query matches an item's microcategory.

There is no explicit "intended microcategory" field on a query, but
`item_microcat_id` subdivides Avito's coarse `item_category_id` far more
finely — up to ~200 microcategories inside category 114 alone
(`feature_analysis.ipynb`), which the hard category filter cannot tell
apart at all. Training a separate query -> microcategory classifier would
need its own labelled pipeline; instead, `MicrocatMatchIndex` estimates
query/microcategory affinity from embeddings we already compute: each
microcategory's centroid is the mean of its items' dense (title +
description) E5 embeddings, and a query's affinity for an item's
microcategory is the cosine similarity between the query embedding and
that microcategory's centroid. Reusing the existing dense embeddings keeps
this signal essentially free — no new model, no new encoding pass.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class MicrocatMatchIndex:
    """Query/item-microcategory affinity via centroid cosine similarity.

    Attributes:
        n_documents: Number of documents in the corpus.
    """

    def __init__(
        self,
        items: pd.DataFrame,
        dense_embeddings: np.ndarray,
        microcat_column: str = "item_microcat_id",
    ) -> None:
        """Builds one centroid per microcategory from the corpus's own items.

        Args:
            items: Corpus DataFrame, in the same row order as
                `dense_embeddings`.
            dense_embeddings: `(len(items), hidden_size)` L2-normalized E5
                "passage:" embeddings of title + description (the same
                array used by `DenseSimilarityIndex`).
            microcat_column: Column holding each item's microcategory id.
        """
        if dense_embeddings.shape[0] != len(items):
            raise ValueError("dense_embeddings must have one row per item")

        microcats = items[microcat_column].to_numpy()
        unique_microcats, item_microcat_positions = np.unique(microcats, return_inverse=True)

        hidden_size = dense_embeddings.shape[1]
        centroid_sums = np.zeros((len(unique_microcats), hidden_size), dtype=np.float64)
        counts = np.zeros(len(unique_microcats), dtype=np.int64)
        np.add.at(centroid_sums, item_microcat_positions, dense_embeddings)
        np.add.at(counts, item_microcat_positions, 1)
        centroids = centroid_sums / counts[:, None]

        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # guards a (never-observed) all-zero centroid
        self._centroids = (centroids / norms).astype(np.float32)
        self._item_microcat_positions = item_microcat_positions
        self.n_documents = len(items)

    def score(
        self,
        query_embedding: np.ndarray,
        row_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Scores every document by its microcategory's affinity to the query.

        Args:
            query_embedding: A single `(hidden_size,)` L2-normalized E5
                "query:" embedding.
            row_mask: Optional boolean array of length `n_documents`
                selecting which documents to score.

        Returns:
            A float32 array of cosine similarities between the query and
            each (selected) document's microcategory centroid.
        """
        similarity_per_microcat = self._centroids @ query_embedding
        item_scores = similarity_per_microcat[self._item_microcat_positions]
        return item_scores if row_mask is None else item_scores[row_mask]
