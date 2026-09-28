"""Helpers for correlating query-side and item-side features.

`train.parquet` only tells us which item a user picked for a query — there
is no negative signal. So "correlation" here mostly means two things:

- **Match/propensity analysis**: how does a feature's value (or its match
  with the query's own value, e.g. location) among *chosen* items compare
  to its distribution across the whole retrieval corpus (`benchmark_items`,
  the "population")? A gap suggests the feature carries a preference signal
  worth adding to candidate generation or reranking.
- **A single correlation matrix** across all numeric/binary/categorical-
  match features of a chosen `(query, item)` pair, via `encode_for_correlation`
  + `DataFrame.corr()`, to see which features move together.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def location_match_rate(
    df: pd.DataFrame,
    group_by: Optional[str] = None,
    query_column: str = "search_location_id",
    item_column: str = "item_location_id",
) -> pd.Series:
    """Computes the share of rows where the query and item location match.

    Args:
        df: A DataFrame with `query_column` and `item_column`.
        group_by: Optional column to compute the match rate within (e.g.
            `"search_category"`, to see if the effect differs by category).
        query_column: Query-side location id column.
        item_column: Item-side location id column.

    Returns:
        A `pandas.Series` of match rates. A single float (wrapped in a
        length-1 Series named `"overall"`) if `group_by` is `None`, else one
        value per group.
    """
    is_match = df[query_column] == df[item_column]
    if group_by is None:
        return pd.Series({"overall": is_match.mean()})
    return is_match.groupby(df[group_by]).mean().sort_values(ascending=False)


def location_id_dispersion(
    items_df: pd.DataFrame,
    location_column: str = "item_location_id",
    lat_column: str = "item_latitude",
    lon_column: str = "item_longitude",
    min_items: int = 5,
) -> pd.DataFrame:
    """Checks how geographically tight each `location_id` actually is.

    `search_location_id`/`item_location_id` are opaque ids; this sanity
    check confirms they correspond to a small geographic area (low
    coordinate spread) rather than something coarse like a whole region,
    which is what makes an exact-match location signal meaningful at all.

    Args:
        items_df: Item features, e.g. from `feature_loading.load_item_features`.
        location_column: Column identifying the location.
        lat_column: Latitude column (already `float64`).
        lon_column: Longitude column (already `float64`).
        min_items: Drop locations with fewer items than this (a single-item
            location has zero spread by construction and is not informative).

    Returns:
        A DataFrame indexed by `location_column` with `lat_std`, `lon_std`
        (standard deviation of coordinates, in degrees) and `n_items`.
    """
    grouped = (
        items_df.dropna(subset=[lat_column, lon_column])
        .groupby(location_column)
        .agg(
            lat_std=(lat_column, "std"),
            lon_std=(lon_column, "std"),
            n_items=(lat_column, "size"),
        )
    )
    return grouped[grouped["n_items"] >= min_items].sort_values("n_items", ascending=False)


def propensity_table(
    chosen_df: pd.DataFrame,
    population_df: pd.DataFrame,
    binary_columns: Sequence[str] = (),
    numeric_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Compares chosen-item feature values against the corpus population.

    For each binary column this is `P(feature=1 | chosen)` vs.
    `P(feature=1 | population)`; for each numeric column it's the median
    (robust to the heavy outliers in `item_price`). A negative `delta` means
    chosen items lean lower on that feature than the population, positive
    means higher.

    Args:
        chosen_df: Feature values of items users actually picked (rows of
            `train.parquet`), one row per interaction.
        population_df: Feature values across the whole retrieval corpus
            (`benchmark_items.parquet`).
        binary_columns: 0/1 or boolean columns to compare by mean.
        numeric_columns: Continuous columns to compare by median.

    Returns:
        A DataFrame with one row per column: `statistic` (`"rate"` or
        `"median"`), `chosen`, `population`, `delta`.
    """
    rows = []
    for column in binary_columns:
        chosen_rate = chosen_df[column].astype(float).mean()
        population_rate = population_df[column].astype(float).mean()
        rows.append(
            {
                "feature": column,
                "statistic": "rate",
                "chosen": chosen_rate,
                "population": population_rate,
                "delta": chosen_rate - population_rate,
            }
        )
    for column in numeric_columns:
        chosen_median = chosen_df[column].median()
        population_median = population_df[column].median()
        rows.append(
            {
                "feature": column,
                "statistic": "median",
                "chosen": chosen_median,
                "population": population_median,
                "delta": chosen_median - population_median,
            }
        )
    return pd.DataFrame(rows)


def encode_for_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Builds a numeric-only view of a `train.parquet` row for `corr()`.

    Nominal ids (`search_location_id`, `item_category_id`, ...) are not
    themselves meaningful in a Pearson/Spearman correlation, so they are
    replaced by derived numeric signals: a location/category match
    indicator, and log1p of the heavy-tailed price/review-count columns
    (`item_price` also has a handful of non-positive placeholder values,
    clipped to `NaN` before the log).

    Args:
        df: A DataFrame with the columns loaded by
            `feature_loading.load_train_features` (query and item feature
            columns from the same `train.parquet` rows).

    Returns:
        A DataFrame of purely numeric columns, suitable for
        `DataFrame.corr(method="spearman")`.
    """
    price = df["item_price"].where(df["item_price"] > 0)
    return pd.DataFrame(
        {
            "location_match": (df["search_location_id"] == df["item_location_id"]).astype(int),
            "category_match": (df["search_category"] == df["item_category_id"]).astype(int),
            "search_is_delivery_search": df["search_is_delivery_search"].astype(int),
            "item_is_phone_hidden": df["item_is_phone_hidden"].astype(int),
            "item_is_message_forbidden": df["item_is_message_forbidden"].astype(int),
            "item_rating": df["item_rating"],
            "item_rating_reviews_count_log1p": np.log1p(df["item_rating_reviews_count"]),
            "item_price_log1p": np.log1p(price),
        }
    )
