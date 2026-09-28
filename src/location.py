"""A soft location-affinity signal for candidate generation.

`feature_analysis.ipynb` found that 83% of items users actually chose share
their `item_location_id` with the query's `search_location_id`, and that
`location_id` corresponds to a genuinely small area (median coordinate
spread within one id is ~0.01°, roughly 1 km) — so it is a real, if
imperfect, relevance signal. The first version of this module scored a
plain 0/1 exact-id match; this version keeps the exact match at full credit
(the previous ablation confirmed it is the single most valuable signal) but
grades everything else by geographic distance instead of a flat 0, using
each item's own coordinates and each `location_id`'s centroid coordinate —
so a query and an item in adjacent, but not identical, locations now get
partial credit instead of none.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

_EARTH_RADIUS_KM = 6371.0


def _haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Great-circle distance in km between elementwise (lat, lon) pairs."""
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class LocationMatchIndex:
    """Distance-decayed query/item location affinity, scored against a corpus.

    Attributes:
        n_documents: Number of documents in the corpus.
        decay_km: Distance at which a non-exact-match score decays to `1/e`.
    """

    def __init__(
        self,
        items: pd.DataFrame,
        location_column: str = "item_location_id",
        lat_column: str = "item_latitude",
        lon_column: str = "item_longitude",
        decay_km: float = 5.0,
    ) -> None:
        """Stores each item's location and builds per-location centroids.

        Args:
            items: Corpus DataFrame, in the same row order used by the
                other scoring indices.
            location_column: Column holding each item's location id.
            lat_column: Column holding each item's latitude. May be an
                `object` column of `Decimal` (as read from parquet); it is
                cast to `float64` here.
            lon_column: Column holding each item's longitude; same cast.
            decay_km: Distance (km) at which a non-exact-match item's score
                decays to `1/e` (~0.37); smaller values make the signal
                fall off faster with distance.
        """
        self.item_location = items[location_column].to_numpy()
        self.item_lat = items[lat_column].astype(float).to_numpy()
        self.item_lon = items[lon_column].astype(float).to_numpy()
        self.decay_km = decay_km
        self.n_documents = len(items)

        centroids = (
            items.assign(**{lat_column: self.item_lat, lon_column: self.item_lon})
            .dropna(subset=[lat_column, lon_column])
            .groupby(location_column)[[lat_column, lon_column]]
            .mean()
        )
        self._centroid_lookup: Dict[int, Tuple[float, float]] = {
            location_id: (row[lat_column], row[lon_column])
            for location_id, row in centroids.iterrows()
        }

    def score(
        self,
        search_location_id: int,
        row_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Scores every document by location match/distance to the query.

        Args:
            search_location_id: The query's `search_location_id`.
            row_mask: Optional boolean array of length `n_documents`
                selecting which documents to score.

        Returns:
            A float32 array in `(0, 1]`: `1.0` for an exact
            `item_location_id` match, `exp(-distance_km / decay_km)` for
            everything else (`0.0` where coordinates are missing, or where
            `search_location_id` has no known centroid in this corpus —
            i.e. nothing to compare distances against).
        """
        location = self.item_location if row_mask is None else self.item_location[row_mask]
        exact_match = location == search_location_id

        centroid = self._centroid_lookup.get(search_location_id)
        if centroid is None:
            return exact_match.astype(np.float32)

        lat = self.item_lat if row_mask is None else self.item_lat[row_mask]
        lon = self.item_lon if row_mask is None else self.item_lon[row_mask]
        distance_km = _haversine_km(
            np.full_like(lat, centroid[0]), np.full_like(lon, centroid[1]), lat, lon
        )
        decayed = np.exp(-distance_km / self.decay_km)
        decayed = np.nan_to_num(decayed, nan=0.0)  # items with missing coordinates
        return np.where(exact_match, 1.0, decayed).astype(np.float32)

    def distance_km(
        self,
        search_location_id: int,
        row_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Raw great-circle distance (km) to `search_location_id`'s centroid.

        `score` collapses distance through a decay function to keep it
        comparable to the other bounded `(0, 1]` signals; this method
        exposes the untransformed value as a reranker feature, where a
        model like CatBoost can learn its own (possibly non-monotonic or
        threshold-based) relationship with relevance instead of inheriting
        `score`'s hand-picked `decay_km`.

        Args:
            search_location_id: The query's `search_location_id`.
            row_mask: Optional boolean array of length `n_documents`
                selecting which documents to score.

        Returns:
            A float32 array of distances in km. `0.0` for an exact
            `item_location_id` match (by definition, not measured against
            the centroid). `NaN` where an item's coordinates are missing,
            or where `search_location_id` has no known centroid in this
            corpus.
        """
        location = self.item_location if row_mask is None else self.item_location[row_mask]
        exact_match = location == search_location_id

        n_out = self.n_documents if row_mask is None else int(row_mask.sum())
        centroid = self._centroid_lookup.get(search_location_id)
        if centroid is None:
            return np.where(exact_match, 0.0, np.nan).astype(np.float32)

        lat = self.item_lat if row_mask is None else self.item_lat[row_mask]
        lon = self.item_lon if row_mask is None else self.item_lon[row_mask]
        distance_km = _haversine_km(
            np.full_like(lat, centroid[0]), np.full_like(lon, centroid[1]), lat, lon
        )
        return np.where(exact_match, 0.0, distance_km).astype(np.float32)
