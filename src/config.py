"""Configuration objects for the candidate generation pipeline."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class CandidateGeneratorConfig:
    """Hyperparameters for `CandidateGenerator` and its building blocks.

    Attributes:
        top_k: Number of candidates to keep per query after fusion.
        unspecified_category_value: Value of `search_category` that means
            "no category filter was applied by the user". Items are not
            filtered by category for such queries.
        min_token_len: Tokens shorter than this (in characters) are dropped
            by the tokenizer.
        params_similarity_threshold: Minimum cosine similarity between an
            E5 embedding of a query-params token and an item's params-text
            embedding for the item to count as a semantic match in
            `coverage_params`. Calibrated empirically in the notebook on
            true vs. random (query, item) pairs from `train.parquet`.
        rrf_k: The `k` constant of Reciprocal Rank Fusion, controlling how
            quickly the contribution of a signal decays with rank. Higher
            values flatten the fusion (ranks matter less).
        signal_weights: Weight of each ranking signal in the RRF fusion.
            Keys must be a subset of `{"params_coverage", "text_coverage",
            "text_similarity", "location_match", "microcat_match"}`.
        location_decay_km: Distance (km) at which `LocationMatchIndex`'s
            non-exact-match score decays to `1/e`. See `src/location.py`.
        embedding_batch_size: Batch size used when encoding texts with the
            E5 model.
        random_seed: Seed used for validation-query sampling.
    """

    top_k: int = 500
    unspecified_category_value: int = 0
    min_token_len: int = 2
    params_similarity_threshold: float = 0.82
    rrf_k: int = 60
    signal_weights: dict = dataclasses.field(
        default_factory=lambda: {
            "params_coverage": 1.0,
            "text_coverage": 1.0,
            "text_similarity": 1.0,
            "location_match": 1.0,
            "microcat_match": 1.0,
        }
    )
    location_decay_km: float = 5.0
    embedding_batch_size: int = 256
    random_seed: int = 42
