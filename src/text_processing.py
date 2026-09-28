"""Tokenization helpers shared by the coverage and embedding modules.

All text fields in the dataset (search query, search params, item title,
item description, item params) are free-form Russian text. We use a single,
simple tokenizer everywhere so that query-side and item-side tokens are
directly comparable.

Russian is heavily inflected, so a query token like "уборка" will not
appear verbatim in a title that instead has "уборки"/"уборку"/"уборкой" —
exact-match coverage would silently miss that. `tokenize`
therefore lemmatizes every token with `pymorphy3`, a dictionary+rules
morphological analyzer (no model inference, no network): ~65k words/sec,
so it is applied everywhere by default. Repeated tokens are cheap thanks to
`_lemmatize`'s cache — natural text is Zipfian, so a corpus of hundreds of
thousands of documents still only lemmatizes its few hundred-thousand
distinct tokens once.
"""

from __future__ import annotations

import re
from typing import Iterable, List

import pymorphy3

_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+")
_CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")

_morph = pymorphy3.MorphAnalyzer()
_lemma_cache: dict[str, str] = {}


def _lemmatize(token: str) -> str:
    """Returns the normal (dictionary) form of a single lowercase token.

    Non-Cyrillic tokens (numbers, Latin brand names) are returned unchanged
    — `pymorphy3` only models Russian morphology and would otherwise just
    add noise. Results are memoized process-wide.
    """
    if not _CYRILLIC_RE.search(token):
        return token
    cached = _lemma_cache.get(token)
    if cached is not None:
        return cached
    lemma = _morph.parse(token)[0].normal_form
    _lemma_cache[token] = lemma
    return lemma


# A short list of Russian function words that carry no retrieval signal but
# appear in almost every query/title/description. Removing them keeps
# coverage scores focused on content words. Lemmatized once at import
# time so filtering after `_lemmatize` stays correct regardless of which
# inflected form a stopword was written in here.
_RUSSIAN_STOPWORDS_RAW = frozenset(
    {
        "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как",
        "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к",
        "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было",
        "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь",
        "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или",
        "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять",
        "уж", "вам", "для", "их", "чтобы", "без", "при", "об",
    }
)
RUSSIAN_STOPWORDS = frozenset(_lemmatize(word) for word in _RUSSIAN_STOPWORDS_RAW)


def tokenize(
    text: str | None,
    min_len: int = 2,
    drop_stopwords: bool = True,
    lemmatize: bool = True,
) -> List[str]:
    """Splits free-form text into lowercase, lemmatized tokens.

    Args:
        text: Input string. `None` or non-string values yield an empty list.
        min_len: Minimum token length (in characters) to keep, checked
            before lemmatization.
        drop_stopwords: Whether to remove tokens in `RUSSIAN_STOPWORDS`
            (checked after lemmatization).
        lemmatize: Whether to normalize each token to its dictionary form
            with `pymorphy3` (e.g. "уборки" -> "уборка"). Disabling this
            falls back to plain surface-form tokens.

    Returns:
        A list of tokens, in the order they appear in `text`.
    """
    if not isinstance(text, str) or not text:
        return []
    tokens = [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]
    tokens = [token for token in tokens if len(token) >= min_len]
    if lemmatize:
        tokens = [_lemmatize(token) for token in tokens]
    if drop_stopwords:
        tokens = [token for token in tokens if token not in RUSSIAN_STOPWORDS]
    return tokens


def unique_tokens(texts: Iterable[str | None], **tokenize_kwargs) -> set:
    """Tokenizes many texts and returns the set of distinct tokens seen.

    Args:
        texts: An iterable of raw text values.
        **tokenize_kwargs: Forwarded to `tokenize`.

    Returns:
        The set of unique tokens across all `texts`.
    """
    vocabulary: set = set()
    for text in texts:
        vocabulary.update(tokenize(text, **tokenize_kwargs))
    return vocabulary
