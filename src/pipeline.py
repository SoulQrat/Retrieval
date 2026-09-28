"""Assembles a `CandidateGenerator` from a corpus and its precomputed embeddings.

Both `candidate_generation.ipynb` and `reranking.ipynb` need the same five scoring
indices built over (possibly different) item corpora; this module is the
single place that wiring lives, so the notebooks cannot drift apart on how
a `CandidateGenerator` is put together.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import CandidateGeneratorConfig
from src.coverage import CoverageIndex, SemanticCoverageIndex
from src.embeddings import E5Embedder
from src.location import LocationMatchIndex
from src.microcat import MicrocatMatchIndex
from src.similarity import DenseSimilarityIndex


def build_indices(
    items: pd.DataFrame,
    params_embeddings: np.ndarray,
    dense_embeddings: np.ndarray,
    query_vocabulary: Sequence[str],
    embedder: E5Embedder,
    config: CandidateGeneratorConfig,
    query_cache_key: str = "query_vocab_params",
) -> Tuple[
    SemanticCoverageIndex, CoverageIndex, DenseSimilarityIndex, LocationMatchIndex, MicrocatMatchIndex
]:
    """Builds the five scoring indices for a corpus.

    Args:
        items: Corpus DataFrame with `item_infm_params_text`,
            `item_title_raw`, `item_description_raw`, `item_location_id`,
            `item_latitude`, `item_longitude` and `item_microcat_id`.
        params_embeddings: `(len(items), hidden_size)` E5 "passage:"
            embeddings of `item_infm_params_text` (truncated — see
            `candidate_generation.ipynb` section 7).
        dense_embeddings: `(len(items), hidden_size)` E5 "passage:"
            embeddings of title + truncated description; also reused as
            each microcategory's centroid input.
        query_vocabulary: Closed set of tokens queries' params text may
            contain, for `SemanticCoverageIndex`'s semantic match matrix.
        embedder: E5 model used to embed `query_vocabulary`.
        config: Pipeline hyperparameters (similarity threshold, token
            length, location decay distance).
        query_cache_key: Cache key for embedding `query_vocabulary` (see
            `E5Embedder.encode`); pass the same key across calls that share
            a vocabulary so the (cheap) embedding is only computed once.

    Returns:
        `(params_index, text_index, dense_index, location_index,
        microcat_index)`, ready to pass to `CandidateGenerator`.
    """
    params_index = SemanticCoverageIndex(
        documents=items["item_infm_params_text"],
        query_vocabulary=query_vocabulary,
        embedder=embedder,
        document_embeddings=params_embeddings,
        similarity_threshold=config.params_similarity_threshold,
        min_len=config.min_token_len,
        query_cache_key=query_cache_key,
    )
    text_docs = items["item_title_raw"].fillna("") + " " + items["item_description_raw"].fillna("")
    text_index = CoverageIndex(text_docs, min_len=config.min_token_len)
    dense_index = DenseSimilarityIndex(dense_embeddings)
    location_index = LocationMatchIndex(items, decay_km=config.location_decay_km)
    microcat_index = MicrocatMatchIndex(items, dense_embeddings)
    return params_index, text_index, dense_index, location_index, microcat_index
