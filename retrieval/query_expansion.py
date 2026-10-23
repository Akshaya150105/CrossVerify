"""Query expansion: WordNet synonyms + KeyBERT keywords.

Each expansion is returned as a separate query string so the pipeline can run
them independently and merge the ranked lists (guide section 7).
"""

from __future__ import annotations

import re

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "with",
    "by", "is", "are", "was", "were", "be", "from", "as", "it", "its", "this",
    "that", "near", "after", "before",
}


def _tokens(query: str) -> list[str]:
    return [t for t in re.findall(r"[a-zA-Z]+", query.lower()) if t not in STOPWORDS]


def wordnet_synonyms(word: str, limit: int = 3) -> list[str]:
    """Single-word WordNet synonyms for the most common senses of `word`."""

    try:
        from nltk.corpus import wordnet as wn

        synsets = wn.synsets(word)
    except (ImportError, LookupError):
        return []

    seen: set[str] = {word}
    out: list[str] = []
    for synset in synsets[:2]:  # most frequent senses only, to limit drift
        for lemma in synset.lemma_names():
            lemma = lemma.lower()
            if "_" in lemma or lemma in seen:
                continue
            seen.add(lemma)
            out.append(lemma)
            if len(out) >= limit:
                return out
    return out


_KEYBERT = None


def keybert_keywords(text: str, limit: int = 3) -> list[str]:
    """Top KeyBERT keywords. Returns [] if KeyBERT/model is unavailable."""

    global _KEYBERT
    try:
        if _KEYBERT is None:
            from keybert import KeyBERT

            _KEYBERT = KeyBERT()
        pairs = _KEYBERT.extract_keywords(
            text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=limit
        )
    except Exception:
        return []
    return [kw for kw, _ in pairs]


def expand_query(
    query: str,
    max_variants: int = 4,
    use_keybert: bool = False,
) -> list[str]:
    """Return [original, expansion1, ...]; the original is always first."""

    variants = [query]
    # Skip proper nouns (capitalised words after the first): WordNet turns
    # "Leo" into "lion", which would drift the query.
    raw = re.findall(r"[a-zA-Z]+", query)
    proper = {w.lower() for i, w in enumerate(raw) if i > 0 and w[0].isupper()}
    tokens = [t for t in _tokens(query) if t not in proper]

    # One variant per content word: swap in its best synonym.
    for token in tokens:
        for synonym in wordnet_synonyms(token, limit=1):
            variant = re.sub(rf"\b{re.escape(token)}\b", synonym, query, count=1, flags=re.I)
            if variant not in variants:
                variants.append(variant)

    if use_keybert:
        keywords = keybert_keywords(query)
        if keywords:
            variants.append(" ".join(keywords))

    return variants[:max_variants]