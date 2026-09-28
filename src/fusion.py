"""Reciprocal Rank Fusion for combining heterogeneous ranking signals.

`coverage_params`, `coverage_text` and `dense_similarity` live on different,
not-directly-comparable scales (a coverage ratio in `[0, 1]` vs. a cosine
similarity that is rarely below `0.7` in practice). Rather than hand-tune
weights to make raw scores commensurable, we fuse the *ranks* each signal
induces: Reciprocal Rank Fusion (Cormack et al., 2009) is a standard,
weight-tuning-free way to combine ranked lists from independent retrieval
signals.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import rankdata


def reciprocal_rank_fusion(
    score_arrays: Sequence[np.ndarray],
    weights: Sequence[float],
    k: int = 60,
) -> np.ndarray:
    """Fuses several per-document score arrays via weighted RRF.

    Each score array is converted to ranks (1 = highest score, ties get the
    average rank) and contributes `weight / (k + rank)` to the fused score.
    A signal that is constant across all documents (e.g. a coverage score
    that is 0 everywhere because the query had no usable tokens for it) is
    skipped, since it carries no ranking information and would otherwise
    inject rank noise from arbitrary tie-breaking.

    Args:
        score_arrays: Score arrays to fuse, all of the same length and in
            the same document order. Higher is better.
        weights: One weight per array in `score_arrays`.
        k: RRF's smoothing constant; higher values flatten the influence of
            rank (top-1 vs. top-50 matters less).

    Returns:
        A float64 array of fused scores, in the same document order. Higher
        is still better.

    Raises:
        ValueError: If `score_arrays` and `weights` have different lengths,
            or `score_arrays` is empty.
    """
    if not score_arrays:
        raise ValueError("score_arrays must not be empty")
    if len(score_arrays) != len(weights):
        raise ValueError("score_arrays and weights must have the same length")

    n_documents = len(score_arrays[0])
    fused = np.zeros(n_documents, dtype=np.float64)
    for scores, weight in zip(score_arrays, weights):
        if weight == 0 or np.ptp(scores) == 0:
            continue
        ranks = rankdata(-np.asarray(scores), method="average")
        fused += weight / (k + ranks)
    return fused
