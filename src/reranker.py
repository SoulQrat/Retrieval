"""A CatBoost `YetiRank` reranker: turns Recall@500 into Recall@50.

Candidate generation (`notebook.ipynb`) already gets most relevant items
into its top 500 (Recall@500 ≈ 0.88 on local validation) — the ceiling on
Recall@50 is set by how well those 500 are *ordered*, which is exactly what
a learning-to-rank model optimizes for. `YetiRank` (CatBoost's default
ranking loss) directly targets a NDCG-like objective over each query's
candidate group, rather than treating every (query, item) pair as an
independent classification example the way a plain classifier would.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import pandas as pd
from catboost import CatBoostRanker, Pool


def to_pool(
    features_df: pd.DataFrame,
    feature_columns: Sequence[str],
    label_column: Optional[str] = None,
) -> Pool:
    """Builds a CatBoost `Pool` grouped by `query_id`.

    Args:
        features_df: A table from `reranking_features.build_feature_table`
            — must already be sorted by `query_id` (that function does
            this), since CatBoost requires a ranking group's rows to be
            contiguous.
        feature_columns: Columns to use as model input.
        label_column: Column holding relevance labels; omit for an
            inference-only pool (no `label` column required).

    Returns:
        A `catboost.Pool` ready for `train_reranker` or `model.predict`.
    """
    return Pool(
        data=features_df[list(feature_columns)],
        label=features_df[label_column] if label_column else None,
        group_id=features_df["query_id"],
    )


def train_reranker(
    train_pool: Pool,
    eval_pool: Optional[Pool] = None,
    **catboost_params,
) -> CatBoostRanker:
    """Trains a `CatBoostRanker` with a `YetiRank` loss.

    Args:
        train_pool: Training data, from `to_pool` with `label_column` set.
        eval_pool: Optional validation data (same shape) for early
            stopping / best-iteration selection.
        **catboost_params: Overrides for the default hyperparameters
            (`iterations=1000, learning_rate=0.05, depth=6,
            loss_function="YetiRank"`).

    Returns:
        The fitted model.
    """
    params = dict(
        loss_function="YetiRank",
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=100,
        eval_metric="RecallAt:top=50",
        early_stopping_rounds=100, 
    )
    params.update(catboost_params)
    model = CatBoostRanker(**params)
    model.fit(train_pool, eval_set=eval_pool, use_best_model=eval_pool is not None)
    return model


def rerank(
    model: CatBoostRanker,
    features_df: pd.DataFrame,
    feature_columns: Sequence[str],
    top_k: int = 50,
) -> Dict[str, List[str]]:
    """Scores every candidate and keeps the top `top_k` per query.

    Args:
        model: A fitted `CatBoostRanker` (from `train_reranker`).
        features_df: A table from `reranking_features.build_feature_table`
            (no `label` column required).
        feature_columns: Same columns used to train `model`, in the same
            order.
        top_k: Number of item ids to keep per query, best first.

    Returns:
        A dict mapping `query_id` to its reranked, truncated `item_id` list.
    """
    scores = model.predict(features_df[list(feature_columns)])
    ranked = (
        features_df[["query_id", "item_id"]]
        .assign(_score=scores)
        .sort_values(["query_id", "_score"], ascending=[True, False], kind="stable")
    )
    return {
        query_id: group["item_id"].head(top_k).tolist()
        for query_id, group in ranked.groupby("query_id", sort=False)
    }
