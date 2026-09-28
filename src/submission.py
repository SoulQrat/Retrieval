"""Building and validating `answer.csv` per the competition's format rules.

From `README.md`: exactly two columns (`query_id`, `answer`), one row per
`query_id` in `benchmark_queries.parquet`, `answer` a space-separated list
of up to 50 `item_id`s (each a 16-character lowercase hex string) with no
duplicates, no extra rows, no index column.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

import pandas as pd

_ITEM_ID_RE = re.compile(r"^[0-9a-f]{16}$")


def build_answer(
    predictions: Dict[str, List[str]],
    query_ids: Sequence[str],
    max_candidates: int = 50,
) -> pd.DataFrame:
    """Builds the `answer.csv` DataFrame from reranked predictions.

    Args:
        predictions: Mapping from `query_id` to a ranked list of `item_id`s
            (best first), e.g. from `reranker.rerank`.
        query_ids: Every `query_id` that must appear exactly once in the
            output, e.g. `benchmark_queries["query_id"]`.
        max_candidates: Truncate each row to at most this many `item_id`s.

    Returns:
        A DataFrame with exactly the columns `query_id`, `answer`, one row
        per entry of `query_ids`, in that order.
    """
    rows = []
    for query_id in query_ids:
        seen = set()
        deduped = []
        for item_id in predictions.get(query_id, [])[:max_candidates]:
            if item_id not in seen:
                seen.add(item_id)
                deduped.append(item_id)
        rows.append({"query_id": query_id, "answer": " ".join(deduped)})
    return pd.DataFrame(rows, columns=["query_id", "answer"])


def validate_answer(
    answer_df: pd.DataFrame,
    query_ids: Sequence[str],
    valid_item_ids: Sequence[str],
    max_candidates: int = 50,
) -> List[str]:
    """Checks `answer_df` against every rule in the task's `README.md`.

    Args:
        answer_df: A DataFrame to validate, as produced by `build_answer`.
        query_ids: Every `query_id` the file must cover exactly once.
        valid_item_ids: `item_id`s that actually exist in the corpus used
            (e.g. `benchmark_items["item_id"]`).
        max_candidates: Maximum `item_id`s allowed per row.

    Returns:
        A list of human-readable problem descriptions; empty means the
        file is valid.
    """
    problems: List[str] = []

    if list(answer_df.columns) != ["query_id", "answer"]:
        problems.append(f"unexpected columns: {list(answer_df.columns)}")
        return problems  # further checks assume the right shape

    expected_ids = set(query_ids)
    actual_ids = set(answer_df["query_id"])
    if len(answer_df) != len(expected_ids):
        problems.append(f"expected {len(expected_ids)} rows, got {len(answer_df)}")
    if answer_df["query_id"].duplicated().any():
        problems.append("duplicate query_id rows")
    missing = expected_ids - actual_ids
    if missing:
        problems.append(f"{len(missing)} query_id(s) missing, e.g. {sorted(missing)[:3]}")
    extra = actual_ids - expected_ids
    if extra:
        problems.append(f"{len(extra)} unexpected query_id(s), e.g. {sorted(extra)[:3]}")

    valid_item_id_set = set(valid_item_ids)
    malformed_examples: List[str] = []
    unknown_examples: List[str] = []
    too_long = 0
    has_duplicates = 0
    for query_id, answer in zip(answer_df["query_id"], answer_df["answer"]):
        item_ids = answer.split(" ") if answer else []
        if len(item_ids) > max_candidates:
            too_long += 1
        if len(item_ids) != len(set(item_ids)):
            has_duplicates += 1
        for item_id in item_ids:
            if not _ITEM_ID_RE.match(item_id):
                if len(malformed_examples) < 3:
                    malformed_examples.append(f"{query_id}: {item_id!r}")
            elif item_id not in valid_item_id_set and len(unknown_examples) < 3:
                unknown_examples.append(f"{query_id}: {item_id!r}")

    if too_long:
        problems.append(f"{too_long} row(s) with more than {max_candidates} item_ids")
    if has_duplicates:
        problems.append(f"{has_duplicates} row(s) with duplicate item_ids")
    if malformed_examples:
        problems.append(f"malformed item_id(s), e.g. {malformed_examples}")
    if unknown_examples:
        problems.append(f"item_id(s) not in the corpus, e.g. {unknown_examples}")

    return problems
