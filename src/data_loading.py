"""Loading the competition parquet files and building a local validation set.

`benchmark_queries.parquet` has no relevance labels (it's the file we must
submit predictions for), so `Recall@k` cannot be measured on it directly.
Instead we build a labelled validation set out of `train.parquet`: every row
there is a `(query, chosen item)` pair, so grouping rows by their query
fields and collecting the chosen `item_id`s gives, for each distinct query,
an (approximate, click-based) set of relevant items.

To score retrieval realistically, the validation corpus should look like
the real one: mostly `benchmark_items`, with the small number of true
positive items missing from it patched in from `train.parquet` (which
carries the same `item_*` columns), so recall is always computable and not
capped by an accidental corpus gap.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

QUERY_COLUMNS: List[str] = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]

ITEM_COLUMNS: List[str] = [
    "item_id",
    "item_title_raw",
    "item_description_raw",
    "item_infm_params_text",
    "item_category_id",
    "item_microcat_id",
    "item_location_id",
    "item_latitude",
    "item_longitude",
]

# Fields that define a query's *intent* for grouping train rows into
# distinct labelled queries. `search_location_id` and
# `search_is_delivery_search` are deliberately excluded: they vary almost
# per-session (see `build_relevance_labels`) and fragment otherwise-identical
# queries into mostly single-item groups, which would make the local
# relevance sets much sparser than what "relevant items" plausibly means for
# the benchmark. The candidate generator itself never reads location or
# delivery, so this only affects how validation labels are built.
QUERY_GROUP_COLUMNS: List[str] = [
    "search_query",
    "search_infm_params_text",
    "search_category",
]


def load_dataset(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Reads the three competition parquet files.

    Only the columns used by the candidate generation pipeline are loaded,
    to keep memory usage down (`train.parquet` has ~500k rows with a long
    `item_description_raw` field).

    Args:
        data_dir: Directory containing `train.parquet`,
            `benchmark_queries.parquet` and `benchmark_items.parquet`.

    Returns:
        A dict with keys `"train"`, `"benchmark_queries"`, `"benchmark_items"`.
    """
    train = pd.read_parquet(
        f"{data_dir}/train.parquet", columns=QUERY_COLUMNS + ITEM_COLUMNS
    )
    benchmark_queries = pd.read_parquet(
        f"{data_dir}/benchmark_queries.parquet", columns=["query_id"] + QUERY_COLUMNS
    )
    benchmark_items = pd.read_parquet(
        f"{data_dir}/benchmark_items.parquet", columns=ITEM_COLUMNS
    )
    return {
        "train": train,
        "benchmark_queries": benchmark_queries,
        "benchmark_items": benchmark_items,
    }


def build_relevance_labels(train_df: pd.DataFrame) -> pd.DataFrame:
    """Groups train rows into distinct queries with their relevant items.

    Rows are grouped by `QUERY_GROUP_COLUMNS` (query text, params filter and
    category), so two rows merge whenever a user issued the same query
    intent, regardless of search location or delivery flag. Each group's
    relevant set is the union of items chosen by anyone who issued that
    query; `search_location_id` and `search_is_delivery_search` are kept as
    the first value seen per group, for reference only.

    Args:
        train_df: The `"train"` DataFrame from `load_dataset`.

    Returns:
        A DataFrame with `QUERY_COLUMNS`, a synthetic `query_id`
        (`"train_000000"`, ...), `relevant_item_ids` (a `frozenset` of
        `item_id`) and `n_interactions` (how many train rows fed the group,
        i.e. popularity of that exact query).
    """
    grouped = (
        train_df.fillna({"search_infm_params_text": ""})
        .groupby(QUERY_GROUP_COLUMNS, dropna=False)
        .agg(
            search_location_id=("search_location_id", "first"),
            search_is_delivery_search=("search_is_delivery_search", "first"),
            relevant_item_ids=("item_id", lambda ids: frozenset(ids)),
            n_interactions=("item_id", "count"),
        )
        .reset_index()[QUERY_COLUMNS + ["relevant_item_ids", "n_interactions"]]
    )
    grouped.insert(0, "query_id", [f"train_{i:06d}" for i in range(len(grouped))])
    return grouped


def sample_validation_queries(
    labels_df: pd.DataFrame,
    n: int,
    seed: int = 42,
    max_relevant: Optional[int] = 200,
) -> pd.DataFrame:
    """Samples a validation subset of labelled queries.

    Args:
        labels_df: Output of `build_relevance_labels`.
        n: Number of queries to sample (or all of them if fewer are left
            after filtering).
        seed: Random seed for reproducibility.
        max_relevant: Drop queries with more than this many relevant items
            before sampling. Extremely generic queries (e.g. a single word
            matched by thousands of items) are not representative of the
            benchmark and would dominate the recall average with noise.

    Returns:
        A row-sampled subset of `labels_df`, in random order.
    """
    candidates = labels_df
    if max_relevant is not None:
        candidates = candidates[
            candidates["relevant_item_ids"].map(len) <= max_relevant
        ]
    n = min(n, len(candidates))
    return candidates.sample(n=n, random_state=seed).reset_index(drop=True)


def build_local_eval_corpus(
    benchmark_items: pd.DataFrame,
    train_df: pd.DataFrame,
    val_queries: pd.DataFrame,
) -> pd.DataFrame:
    """Builds the item corpus used to score the local validation queries.

    The corpus is `benchmark_items` plus whichever validation-query true
    positives are missing from it, pulled from `train_df`. This keeps the
    local evaluation corpus close in size and composition to the real
    benchmark corpus while guaranteeing every relevant item is retrievable.

    Args:
        benchmark_items: The `"benchmark_items"` DataFrame from `load_dataset`.
        train_df: The `"train"` DataFrame from `load_dataset` (source of the
            missing items' `item_*` fields).
        val_queries: Output of `sample_validation_queries`.

    Returns:
        A DataFrame with `ITEM_COLUMNS`, reset to a fresh `RangeIndex`.
    """
    needed_ids = set().union(*val_queries["relevant_item_ids"]) if len(val_queries) else set()
    present_ids = set(benchmark_items["item_id"])
    missing_ids = needed_ids - present_ids

    corpus_parts = [benchmark_items[ITEM_COLUMNS]]
    if missing_ids:
        extra_items = (
            train_df[train_df["item_id"].isin(missing_ids)][ITEM_COLUMNS]
            .drop_duplicates(subset="item_id")
        )
        corpus_parts.append(extra_items)

    return pd.concat(corpus_parts, ignore_index=True)
