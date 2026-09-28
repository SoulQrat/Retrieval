"""Recall@k evaluation, matching the competition's scoring definition.

    Recall@k = mean over queries of |top_k ∩ relevant| / |relevant|

(see the task README). `recall_at_k` computes this for several `k` at once
by truncating the same ranked candidate list, which is cheap and exactly
matches how a single top-500 candidate generation run should be evaluated
at `k = 50, 100, 200, 250, 500`.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Sequence, Tuple

import pandas as pd


def recall_at_k(
    predictions: Dict[str, List[str]],
    relevance: Dict[str, FrozenSet[str]],
    ks: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Computes mean Recall@k for each `k`, over the queries in `relevance`.

    Args:
        predictions: Mapping from `query_id` to a ranked list of predicted
            `item_id`s (best first). Missing query ids are treated as an
            empty prediction.
        relevance: Mapping from `query_id` to its set of relevant `item_id`s.
            Queries with an empty relevant set are skipped (recall is
            undefined for them).
        ks: The cutoffs to evaluate, e.g. `(50, 100, 200, 250, 500)`.

    Returns:
        A tuple `(summary, per_query)`:
            summary: One row per `k`, with columns `k` and `recall_at_k`
                (the mean over queries).
            per_query: One row per `(query_id, k)`, with columns
                `query_id`, `k`, `n_relevant` and `recall` — useful for
                slicing failures (e.g. by query length or category).
    """
    rows = []
    for query_id, relevant in relevance.items():
        if not relevant:
            continue
        predicted = predictions.get(query_id, [])
        for k in ks:
            hits = len(set(predicted[:k]) & relevant)
            rows.append(
                {
                    "query_id": query_id,
                    "k": k,
                    "n_relevant": len(relevant),
                    "recall": hits / len(relevant),
                }
            )

    per_query = pd.DataFrame(rows)
    summary = (
        per_query.groupby("k")["recall"]
        .mean()
        .reset_index()
        .rename(columns={"recall": "recall_at_k"})
        .sort_values("k")
        .reset_index(drop=True)
    )
    return summary, per_query


def relevant_item_coverage(
    relevance: Dict[str, FrozenSet[str]], corpus_item_ids: Sequence[str]
) -> float:
    """Checks what share of relevant items are present in the corpus.

    This is a sanity check, not a retrieval metric: if it is below 1.0, some
    queries have an unreachable relevant item and Recall@k is capped below
    1.0 for them regardless of how good the candidate generator is.

    Args:
        relevance: Mapping from `query_id` to its set of relevant `item_id`s.
        corpus_item_ids: `item_id`s available in the retrieval corpus.

    Returns:
        The fraction of all (query, relevant item) pairs whose item is in
        `corpus_item_ids`.
    """
    corpus_set = set(corpus_item_ids)
    total = 0
    present = 0
    for relevant in relevance.values():
        total += len(relevant)
        present += len(relevant & corpus_set)
    return present / total if total else float("nan")
