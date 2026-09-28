"""Builds the labeled, item-content-enriched table CatBoost trains on.

Three ingredients go into a reranker training row:

- The five raw candidate-generation signal scores plus the fused rank
  (`CandidateGenerator.generate_with_features`) — these already carry most
  of the query/item relevance signal, but as a *fused rank* rather than
  the original magnitudes a tree model can split on directly. Raw
  `location_distance_km` is included alongside the already-decayed
  `location_match_score`, so the model can learn its own relationship with
  distance instead of inheriting `LocationMatchIndex`'s hand-picked decay.
- Item content features (rating, review count, price, contact flags) from
  `feature_loading` — cheap, structural signals `feature_analysis.ipynb`
  found correlate with what users actually pick.
- `search_is_delivery_search`, the query's own delivery flag — free (no
  extra computation), though `feature_analysis.ipynb` found it's `1` for a
  tiny fraction of queries, so don't expect much from it alone.

`category_match` (query/item category equality) was considered and
dropped: it's fully determined by which branch of the hard category filter
applied (see `CandidateGenerator._category_mask`) — always `1` for a
filtered query, always `0` for an unfiltered one (`item_category_id` is
never `0`) — so it carries no *per-candidate* information within a query's
own candidate pool.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Optional, Sequence

import numpy as np
import pandas as pd

#: Columns fed to CatBoost. Kept as an explicit list (not "everything in the
#: merged frame") so `query_id`/`item_id`/`label` never leak in as features.
FEATURE_COLUMNS: Sequence[str] = (
    "params_coverage_score",
    "text_coverage_score",
    "text_similarity_score",
    "location_match_score",
    "location_distance_km",
    "microcat_match_score",
    "rrf_score",
    "rrf_rank",
    "item_rating",
    "item_rating_reviews_count_log1p",
    "item_price_log1p",
    "item_is_phone_hidden",
    "item_is_message_forbidden",
    "search_is_delivery_search",
)


def prepare_item_content_features(item_features: pd.DataFrame) -> pd.DataFrame:
    """Turns raw item features into the numeric columns the reranker uses.

    Args:
        item_features: Output of `feature_loading.load_item_features` /
            `load_train_item_feature_lookup` (or their concatenation),
            with at least `item_id`, `item_rating`,
            `item_rating_reviews_count`, `item_price`,
            `item_is_phone_hidden`, `item_is_message_forbidden`.

    Returns:
        A DataFrame with `item_id`, `item_rating`,
        `item_rating_reviews_count_log1p`, `item_price_log1p`,
        `item_is_phone_hidden`, `item_is_message_forbidden` (the last two
        as `int`). `item_price` non-positive placeholder values (see
        `feature_analysis.ipynb`) are treated as missing before the log.
    """
    price = item_features["item_price"].where(item_features["item_price"] > 0)
    return pd.DataFrame(
        {
            "item_id": item_features["item_id"],
            "item_rating": item_features["item_rating"],
            "item_rating_reviews_count_log1p": np.log1p(
                item_features["item_rating_reviews_count"]
            ),
            "item_price_log1p": np.log1p(price),
            "item_is_phone_hidden": item_features["item_is_phone_hidden"].astype(int),
            "item_is_message_forbidden": item_features["item_is_message_forbidden"].astype(int),
        }
    )


def build_feature_table(
    candidates: pd.DataFrame,
    item_content_features: pd.DataFrame,
    relevance: Optional[Dict[str, FrozenSet[str]]] = None,
    query_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Joins candidate signal scores with item content and query features.

    Args:
        candidates: Output of `CandidateGenerator.generate_with_features`
            (columns `query_id`, `item_id`, the five raw signal scores,
            `location_distance_km`, `rrf_score`, `rrf_rank`).
        item_content_features: Output of `prepare_item_content_features`.
        relevance: Optional mapping from `query_id` to its relevant
            `item_id`s. When given, a binary `label` column is added (`1`
            if the row's item is relevant to its query, else `0`) — pass
            this for train/validation tables. Omit for inference-only
            tables (e.g. the real, unlabeled benchmark).
        query_features: Optional per-query columns to attach (currently
            just `search_is_delivery_search`) — a DataFrame with `query_id`
            plus one row per query in `candidates`, e.g. `val_queries[
            ["query_id", "search_is_delivery_search"]]`. Omit to skip
            (`search_is_delivery_search` will be missing/NaN for CatBoost,
            which handles it natively).

    Returns:
        The merged, feature-complete table, sorted by `query_id` (CatBoost
        requires rows of the same ranking group to be contiguous).
    """
    merged = candidates.merge(item_content_features, on="item_id", how="left")
    if query_features is not None:
        merged = merged.merge(query_features, on="query_id", how="left")
    if relevance is not None:
        empty: FrozenSet[str] = frozenset()
        merged["label"] = [
            int(item_id in relevance.get(query_id, empty))
            for query_id, item_id in zip(merged["query_id"], merged["item_id"])
        ]
    return merged.sort_values("query_id", kind="stable").reset_index(drop=True)
