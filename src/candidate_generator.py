"""Top-level candidate generation: hard category filter + signal fusion.

For each query the generator:

1. Applies a hard filter: only items whose `item_category_id` matches the
   query's `search_category` survive (queries with no category selected are
   not filtered, since no item ever has `item_category_id == 0`).
2. Scores the surviving items with five signals: semantic coverage of the
   query's params tokens (`SemanticCoverageIndex`), lexical coverage of the
   query text against the item's title/description (`text_ranking_index`,
   a `CoverageIndex`), dense E5 cosine similarity between the query and the
   item's title+description (`DenseSimilarityIndex`), a distance-graded
   query/item location match (`LocationMatchIndex`), and query/
   microcategory affinity via centroid similarity (`MicrocatMatchIndex`).
3. Fuses the five rankings with Reciprocal Rank Fusion and returns the top
   `config.top_k` item ids.

`generate`/`generate_one` return only ranked item ids, for the candidate-
generation notebook. `generate_with_features`/`generate_one_with_features`
additionally return each signal's *raw* score per candidate — the reranking
notebook (`reranking.ipynb`) uses these as CatBoost features, since RRF's
fused rank throws away exactly the magnitude information a learned model
can use.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.config import CandidateGeneratorConfig
from src.coverage import SemanticCoverageIndex
from src.fusion import reciprocal_rank_fusion
from src.location import LocationMatchIndex
from src.microcat import MicrocatMatchIndex
from src.similarity import DenseSimilarityIndex
from src.text_processing import tokenize


class TextRankingIndex(Protocol):
    """Anything scorable by query tokens against a document corpus — the
    text-similarity slot `CoverageIndex` fills in the current pipeline."""

    def score(
        self, query_tokens, row_mask: Optional[np.ndarray] = None
    ) -> np.ndarray: ...


class CandidateGenerator:
    """Generates up to `config.top_k` candidate item ids per query.

    Attributes:
        item_ids: `item_id` of each corpus item, in corpus order.
        config: Hyperparameters controlling filtering and fusion.
    """

    def __init__(
        self,
        items: pd.DataFrame,
        params_coverage_index: SemanticCoverageIndex,
        text_ranking_index: TextRankingIndex,
        dense_similarity_index: DenseSimilarityIndex,
        location_index: LocationMatchIndex,
        microcat_index: MicrocatMatchIndex,
        config: Optional[CandidateGeneratorConfig] = None,
    ) -> None:
        """Wires up a corpus with its precomputed scoring indices.

        Args:
            items: Corpus DataFrame with at least `item_id` and
                `item_category_id`, in the same row order used to build
                `params_coverage_index`, `text_ranking_index`,
                `dense_similarity_index`, `location_index` and
                `microcat_index`.
            params_coverage_index: Semantic coverage index built over the
                corpus's `item_infm_params_text`.
            text_ranking_index: Query-text-vs-corpus ranking index (any
                object with a `.score(query_tokens, row_mask=None)` method
                — see `TextRankingIndex`), built over the corpus's title +
                description text. `CoverageIndex` in the current pipeline.
            dense_similarity_index: Dense embedding index for the corpus
                (title + description).
            location_index: Query/item location match index for the corpus.
            microcat_index: Query/item microcategory affinity index for the
                corpus.
            config: Hyperparameters; defaults to `CandidateGeneratorConfig()`.
        """
        self.items = items
        self.item_ids = items["item_id"].to_numpy()
        self.item_category = items["item_category_id"].to_numpy()
        self.params_coverage_index = params_coverage_index
        self.text_ranking_index = text_ranking_index
        self.dense_similarity_index = dense_similarity_index
        self.location_index = location_index
        self.microcat_index = microcat_index
        self.config = config or CandidateGeneratorConfig()

    def _category_mask(self, search_category: int) -> np.ndarray:
        """Boolean mask of items passing the hard category filter."""
        if search_category == self.config.unspecified_category_value:
            return np.ones(len(self.items), dtype=bool)
        mask = self.item_category == search_category
        if not mask.any():
            # The requested category has no items in this corpus (can
            # happen for rare/noisy search_category values): fall back to
            # the unfiltered corpus rather than returning nothing.
            return np.ones(len(self.items), dtype=bool)
        return mask

    def _score_all_signals(
        self,
        search_category: int,
        search_infm_params_text: Optional[str],
        search_query: Optional[str],
        search_location_id: int,
        query_embedding: np.ndarray,
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Scores every masked-in item on all five signals, plus the fusion.

        Shared by `generate_one` and `generate_one_with_features` so the
        filtering/scoring logic is defined once.

        Returns:
            A tuple `(mask, params_scores, text_scores, dense_scores,
            location_scores, microcat_scores, fused)`; every score array
            has one entry per `True` position in `mask`.
        """
        mask = self._category_mask(search_category)

        params_scores = self.params_coverage_index.score(
            tokenize(search_infm_params_text), row_mask=mask
        )
        text_scores = self.text_ranking_index.score(
            tokenize(search_query), row_mask=mask
        )
        dense_scores = self.dense_similarity_index.score(
            query_embedding, row_mask=mask
        )
        location_scores = self.location_index.score(
            search_location_id, row_mask=mask
        )
        microcat_scores = self.microcat_index.score(
            query_embedding, row_mask=mask
        )

        weights = self.config.signal_weights
        fused = reciprocal_rank_fusion(
            [params_scores, text_scores, dense_scores, location_scores, microcat_scores],
            weights=[
                weights["params_coverage"],
                weights["text_coverage"],
                weights["text_similarity"],
                weights["location_match"],
                weights["microcat_match"],
            ],
            k=self.config.rrf_k,
        )
        return (
            mask,
            params_scores,
            text_scores,
            dense_scores,
            location_scores,
            microcat_scores,
            fused,
        )

    def _top_k_positions(self, mask: np.ndarray, fused: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Corpus row indices of the top `config.top_k` items by `fused`."""
        top_k = min(self.config.top_k, fused.shape[0])
        top_local = np.argpartition(-fused, top_k - 1)[:top_k]
        top_local = top_local[np.argsort(-fused[top_local])]
        return np.flatnonzero(mask)[top_local], top_local

    def generate_one(
        self,
        search_category: int,
        search_infm_params_text: Optional[str],
        search_query: Optional[str],
        search_location_id: int,
        query_embedding: np.ndarray,
    ) -> List[str]:
        """Generates candidates for a single query.

        Args:
            search_category: The query's `search_category` value.
            search_infm_params_text: The query's params filter text.
            search_query: The query's free-text search string.
            search_location_id: The query's `search_location_id`.
            query_embedding: L2-normalized E5 "query:" embedding of
                `search_query`.

        Returns:
            Up to `config.top_k` `item_id`s, best first.
        """
        mask, _, _, _, _, _, fused = self._score_all_signals(
            search_category,
            search_infm_params_text,
            search_query,
            search_location_id,
            query_embedding,
        )
        candidate_positions, _ = self._top_k_positions(mask, fused)
        return self.item_ids[candidate_positions].tolist()

    def generate_one_with_features(
        self,
        search_category: int,
        search_infm_params_text: Optional[str],
        search_query: Optional[str],
        search_location_id: int,
        query_embedding: np.ndarray,
    ) -> pd.DataFrame:
        """Generates candidates for a single query, with raw signal scores.

        Args are the same as `generate_one`.

        Returns:
            A DataFrame with one row per candidate (up to `config.top_k`,
            in fused-rank order) and columns `item_id`,
            `params_coverage_score`, `text_coverage_score`,
            `text_similarity_score`, `location_match_score`,
            `location_distance_km`, `microcat_match_score`, `rrf_score`,
            `rrf_rank` (1-indexed).
        """
        mask, params_scores, text_scores, dense_scores, location_scores, microcat_scores, fused = (
            self._score_all_signals(
                search_category,
                search_infm_params_text,
                search_query,
                search_location_id,
                query_embedding,
            )
        )
        # Raw distance isn't part of the fusion (score() already encodes it
        # through decay_km) — computed separately, only for the surviving
        # top-k, purely as an extra reranker feature (see LocationMatchIndex
        # .distance_km's docstring for why it's useful alongside the decayed
        # score).
        distance_km = self.location_index.distance_km(search_location_id, row_mask=mask)
        candidate_positions, top_local = self._top_k_positions(mask, fused)
        return pd.DataFrame(
            {
                "item_id": self.item_ids[candidate_positions],
                "params_coverage_score": params_scores[top_local],
                "text_coverage_score": text_scores[top_local],
                "text_similarity_score": dense_scores[top_local],
                "location_match_score": location_scores[top_local],
                "location_distance_km": distance_km[top_local],
                "microcat_match_score": microcat_scores[top_local],
                "rrf_score": fused[top_local],
                "rrf_rank": np.arange(1, len(top_local) + 1),
            }
        )

    def generate(
        self,
        queries: pd.DataFrame,
        query_embeddings: np.ndarray,
        show_progress: bool = True,
    ) -> Dict[str, List[str]]:
        """Generates candidates for every row of `queries`.

        Args:
            queries: DataFrame with `query_id`, `search_category`,
                `search_infm_params_text`, `search_query` and
                `search_location_id` columns.
            query_embeddings: `(len(queries), hidden_size)` array of
                L2-normalized E5 "query:" embeddings, row-aligned with
                `queries`.
            show_progress: Whether to display a `tqdm` progress bar.

        Returns:
            A dict mapping `query_id` to its list of candidate `item_id`s.
        """
        results: Dict[str, List[str]] = {}
        for row, embedding in self._iter_queries(queries, query_embeddings, show_progress):
            results[row.query_id] = self.generate_one(
                search_category=row.search_category,
                search_infm_params_text=row.search_infm_params_text,
                search_query=row.search_query,
                search_location_id=row.search_location_id,
                query_embedding=embedding,
            )
        return results

    def generate_with_features(
        self,
        queries: pd.DataFrame,
        query_embeddings: np.ndarray,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Generates candidates with raw signal scores for every query.

        Args are the same as `generate`.

        Returns:
            The concatenation of `generate_one_with_features` over every
            query, with a `query_id` column prepended.
        """
        frames = []
        for row, embedding in self._iter_queries(queries, query_embeddings, show_progress):
            frame = self.generate_one_with_features(
                search_category=row.search_category,
                search_infm_params_text=row.search_infm_params_text,
                search_query=row.search_query,
                search_location_id=row.search_location_id,
                query_embedding=embedding,
            )
            frame.insert(0, "query_id", row.query_id)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def _iter_queries(
        self, queries: pd.DataFrame, query_embeddings: np.ndarray, show_progress: bool
    ):
        """Yields `(row, embedding)` pairs, optionally under a progress bar."""
        rows = queries.itertuples(index=False)
        iterator = zip(rows, query_embeddings)
        if show_progress:
            iterator = tqdm(iterator, total=len(queries), desc="Generating candidates")
        return iterator
