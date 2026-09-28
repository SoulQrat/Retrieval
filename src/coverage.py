"""Vectorized coverage(q, d) scoring over a fixed document corpus.

`coverage(q, d)` is defined, per the task brief, as the share of distinct
query tokens that are "found" in a document:

    coverage(q, d) = |{t in tokens(q) : found(t, d)}| / |tokens(q)|

`CoverageIndex` implements the exact-match version of `found` (token `t` is
found in `d` iff `t` is one of `d`'s tokens) with a binary term-document
matrix, so a single query is scored against the whole corpus (or a boolean
subset of it) with a handful of sparse-matrix column slices instead of a
per-document Python loop.

`SemanticCoverageIndex` extends this so `found` also fires when `t` is not
present verbatim but its E5 embedding is close to the document's E5
embedding: this is the embedding-based comparison the task asks for when
matching structured `*_infm_params_text` fields, where the query and the
item may use different wording for the same attribute (e.g. "уборка" vs
"клининг"). The comparison is token-vs-whole-document-embedding (not
token-vs-token): item params text is truncated, so the document embedding
stays short enough for this to reuse E5 close to how it was trained; the
threshold is calibrated empirically (see `candidate_generation.ipynb`).

In the current pipeline (`src/candidate_generator.py`), `CoverageIndex` is
used both for the `*_infm_params_text` fields (via `SemanticCoverageIndex`,
task requirement 2) and, on its own, for the query/title-description signal
(requirement 3).
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from src.embeddings import E5Embedder
from src.text_processing import tokenize


def _dedup(tokens: Iterable[str]) -> List[str]:
    """Deduplicates tokens while preserving first-seen order."""
    return list(dict.fromkeys(tokens))


class CoverageIndex:
    """Exact-match coverage(q, d) over a fixed corpus of documents.

    Attributes:
        n_documents: Number of documents in the corpus.
        vocabulary_: Mapping from token to its column index in `matrix`.
        matrix: A `(n_documents, vocabulary size)` binary CSC matrix; entry
            `(i, j)` is 1 iff document `i` contains vocabulary token `j`.
    """

    def __init__(
        self,
        documents: Sequence[Optional[str]],
        min_len: int = 2,
        tokenizer: Callable[[str], List[str]] = tokenize,
    ) -> None:
        """Builds the binary term-document matrix for `documents`.

        Args:
            documents: Raw text of each document, in corpus order.
            min_len: Minimum token length forwarded to `tokenizer`.
            tokenizer: Function turning raw text into a token list. Defaults
                to `text_processing.tokenize`.
        """
        self._tokenize = lambda text: tokenizer(text, min_len=min_len)
        vectorizer = CountVectorizer(
            tokenizer=self._tokenize,
            preprocessor=lambda text: text,
            lowercase=False,
            binary=True,
            token_pattern=None,
        )
        self.matrix: sp.csc_matrix = vectorizer.fit_transform(
            [doc or "" for doc in documents]
        ).tocsc()
        self.vocabulary_: Dict[str, int] = vectorizer.vocabulary_
        self.n_documents = self.matrix.shape[0]

    def score(
        self,
        query_tokens: Sequence[str],
        row_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Computes coverage(q, d) for every document (or a subset).

        Args:
            query_tokens: Tokens of the query (need not be pre-deduplicated
                or pre-filtered to the vocabulary).
            row_mask: Optional boolean array of length `n_documents`
                selecting which documents to score. Scoring only a subset
                (e.g. after a hard category filter) is both cheaper and
                what the candidate generator needs.

        Returns:
            A float32 array with one coverage value per selected document
            (length `n_documents` if `row_mask` is `None`, else
            `row_mask.sum()`). All zeros if the query has no usable tokens.
        """
        unique_tokens = _dedup(query_tokens)
        n_out = self.n_documents if row_mask is None else int(row_mask.sum())
        if not unique_tokens:
            return np.zeros(n_out, dtype=np.float32)

        columns = [
            self.vocabulary_[token]
            for token in unique_tokens
            if token in self.vocabulary_
        ]
        if not columns:
            return np.zeros(n_out, dtype=np.float32)

        block = self.matrix[:, columns]
        if row_mask is not None:
            block = block.tocsr()[row_mask]
        found_counts = np.asarray((block > 0).sum(axis=1)).ravel()
        return (found_counts / len(unique_tokens)).astype(np.float32)


class SemanticCoverageIndex(CoverageIndex):
    """Coverage(q, d) with E5-embedding-based semantic token matching.

    A query token counts as "found" in a document if the document contains
    that exact token (inherited exact-match check), OR the query token's E5
    embedding has at least `similarity_threshold` cosine similarity with
    the document's own E5 embedding. The latter is precomputed once as a
    `(n_query_vocab, n_documents)` boolean matrix, so scoring a query at
    lookup time is a handful of row look-ups, not new model inference.
    """

    def __init__(
        self,
        documents: Sequence[Optional[str]],
        query_vocabulary: Sequence[str],
        embedder: E5Embedder,
        document_embeddings: np.ndarray,
        similarity_threshold: float = 0.80,
        min_len: int = 2,
        tokenizer: Callable[[str], List[str]] = tokenize,
        query_cache_key: str = "coverage_query_vocab",
    ) -> None:
        """Builds the term-document matrix and the semantic match matrix.

        Args:
            documents: Raw text of each document, in corpus order (used
                only for the exact-match part; must match the order of
                `document_embeddings`).
            query_vocabulary: The closed set of tokens that queries may
                contain (e.g. all tokens seen in `search_infm_params_text`
                across train and benchmark queries). Precomputing matches
                for this fixed, small set is what keeps `score` cheap.
            embedder: E5 model used to embed `query_vocabulary`.
            document_embeddings: `(n_documents, hidden_size)` L2-normalized
                E5 "passage:" embeddings of `documents`, row-aligned with
                `documents` (typically from `E5Embedder.encode`, on a
                truncated version of the text — see the notebook).
            similarity_threshold: Minimum cosine similarity between a query
                token's embedding and a document's embedding for the
                document to count as a semantic match for that token.
            min_len: Minimum token length forwarded to `tokenizer`.
            tokenizer: Function turning raw text into a token list.
            query_cache_key: Cache key used when embedding `query_vocabulary`
                (see `E5Embedder.encode`).
        """
        super().__init__(documents, min_len=min_len, tokenizer=tokenizer)
        if document_embeddings.shape[0] != self.n_documents:
            raise ValueError(
                "document_embeddings must have one row per document in `documents`"
            )

        query_vocabulary = _dedup(query_vocabulary)
        query_embeddings = embedder.encode(
            query_vocabulary, prefix="query", cache_key=query_cache_key
        )

        # (n_query_vocab, n_documents) cosine similarity: both sides are
        # L2-normalized by E5Embedder, so a dot product suffices.
        similarity = query_embeddings @ document_embeddings.T
        self._semantic_match: np.ndarray = similarity >= similarity_threshold
        self._token_row: Dict[str, int] = {
            token: row for row, token in enumerate(query_vocabulary)
        }

    def score(
        self,
        query_tokens: Sequence[str],
        row_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Computes semantic coverage(q, d); see `CoverageIndex.score`."""
        unique_tokens = _dedup(query_tokens)
        n_out = self.n_documents if row_mask is None else int(row_mask.sum())
        if not unique_tokens:
            return np.zeros(n_out, dtype=np.float32)

        found_total = np.zeros(n_out, dtype=np.float32)
        for token in unique_tokens:
            found = self._exact_found(token, row_mask)
            semantic_row_idx = self._token_row.get(token)
            if semantic_row_idx is not None:
                semantic = self._semantic_match[semantic_row_idx]
                if row_mask is not None:
                    semantic = semantic[row_mask]
                found = semantic if found is None else (found | semantic)
            if found is not None:
                found_total += found.astype(np.float32)

        return (found_total / len(unique_tokens)).astype(np.float32)

    def _exact_found(
        self, token: str, row_mask: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Boolean array: does each (selected) document contain `token`?"""
        column = self.vocabulary_.get(token)
        if column is None:
            return None
        block = self.matrix[:, [column]]
        if row_mask is not None:
            block = block.tocsr()[row_mask]
        return np.asarray(block.todense()).ravel() > 0
