"""Loading the raw numeric/binary/categorical features for EDA.

`data_loading.load_dataset` only reads the text fields the candidate
generation pipeline needs. This module reads the remaining fields —
location, price, rating, review count, coordinates and the boolean flags —
and cleans their dtypes: `item_price`/`item_latitude`/`item_longitude` come
out of `pandas.read_parquet` as `object` columns of Python `Decimal` (the
parquet `decimal128` type), which cannot be plotted or correlated directly.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

# Query-side fields analyzed here; a superset of what the candidate
# generator uses (adds nothing text-related — that's already covered in
# notebook.ipynb).
QUERY_FEATURE_COLUMNS: List[str] = [
    "search_location_id",
    "search_is_delivery_search",
    "search_category",
]

# Item-side numeric/binary/categorical fields.
ITEM_FEATURE_COLUMNS: List[str] = [
    "item_id",
    "item_location_id",
    "item_latitude",
    "item_longitude",
    "item_category_id",
    "item_microcat_id",
    "item_price",
    "item_rating",
    "item_rating_reviews_count",
    "item_is_phone_hidden",
    "item_is_message_forbidden",
]

_DECIMAL_COLUMNS: List[str] = ["item_price", "item_latitude", "item_longitude"]


def _clean_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Casts parquet `decimal128` columns (read as `Decimal` objects) to float."""
    df = df.copy()
    for column in _DECIMAL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype(float)
    return df


def load_train_features(data_dir: str) -> pd.DataFrame:
    """Reads `train.parquet` with query- and item-side feature columns.

    Each row is still a `(query, chosen item)` pair, exactly as in
    `data_loading.load_dataset`'s `"train"` — this loader just adds the
    numeric/binary/categorical columns needed for feature analysis.

    Args:
        data_dir: Directory containing `train.parquet`.

    Returns:
        A DataFrame with `QUERY_FEATURE_COLUMNS + ITEM_FEATURE_COLUMNS`,
        decimal columns cast to `float64`.
    """
    df = pd.read_parquet(
        f"{data_dir}/train.parquet",
        columns=QUERY_FEATURE_COLUMNS + ITEM_FEATURE_COLUMNS,
    )
    return _clean_dtypes(df)


def load_item_features(data_dir: str) -> pd.DataFrame:
    """Reads `benchmark_items.parquet` with its feature columns.

    Used as the "population" baseline in propensity comparisons: how do
    features of items users actually chose (`train.parquet`) differ from
    the distribution of features across the whole retrieval corpus.

    Args:
        data_dir: Directory containing `benchmark_items.parquet`.

    Returns:
        A DataFrame with `ITEM_FEATURE_COLUMNS`, decimal columns cast to
        `float64`.
    """
    df = pd.read_parquet(
        f"{data_dir}/benchmark_items.parquet", columns=ITEM_FEATURE_COLUMNS
    )
    return _clean_dtypes(df)


def load_train_item_feature_lookup(data_dir: str) -> pd.DataFrame:
    """Reads item-side features from `train.parquet`, deduped by `item_id`.

    `reranking.ipynb`'s local evaluation corpus (like `notebook.ipynb`'s)
    adds a handful of items that are only in `train.parquet`, not
    `benchmark_items.parquet` (see `data_loading.build_local_eval_corpus`).
    This backfills their content features (rating, price, ...) from the
    same rows that supplied their text fields.

    Args:
        data_dir: Directory containing `train.parquet`.

    Returns:
        A DataFrame with `ITEM_FEATURE_COLUMNS`, one row per distinct
        `item_id`, decimal columns cast to `float64`.
    """
    df = pd.read_parquet(f"{data_dir}/train.parquet", columns=ITEM_FEATURE_COLUMNS)
    return _clean_dtypes(df).drop_duplicates(subset="item_id")
