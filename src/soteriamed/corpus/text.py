"""Text cleaning for the sparse retrieval path.

``clean_text`` lowercases, strips noise, removes stopwords and lemmatises. That
is correct for BM25 and **wrong** for sentence embeddings, which want the
original surface form -- the dense path does not call this.

The stopword list deliberately subtracts negation words. NLTK's English list
contains "no", "not", "without" and friends; dropping them inverts clinical
meaning ("no chest pain" becomes "chest pain"). Do not "fix" that.

spaCy and NLTK load lazily, on first call. Loading them at import time put ~50s
on every pytest collection in the proof-of-concept; scispaCy joins them in
phase 3, so the pattern must not come back.
"""

import re

_NEGATION_WORDS = frozenset({
    "no", "not", "nor", "without", "none", "never", "neither",
    "cannot", "don", "doesn", "didn", "isn", "wasn", "weren",
    "hasn", "haven", "wouldn", "shouldn", "couldn",
})

_nlp = None
_stopwords: frozenset[str] | None = None


def _get_nlp():
    """Load the spaCy lemmatiser once, on first use."""
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _nlp


def _get_stopwords() -> frozenset[str]:
    """NLTK's English stopwords minus the negation words, loaded once."""
    global _stopwords
    if _stopwords is None:
        import nltk
        _stopwords = frozenset(nltk.corpus.stopwords.words("english")) - _NEGATION_WORDS
    return _stopwords


def clean_text(text: str) -> str:
    """Clean a clinical transcription: lowercase, remove noise, remove
    stopwords (keeping negation), and lemmatize with SpaCy."""
    import nltk

    text = text.lower()
    # Remove special characters but keep hyphens, periods, and apostrophes
    text = re.sub(r"[^a-z0-9\s\-\.']", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize with NLTK
    tokens = nltk.word_tokenize(text)

    # Remove stopwords (keep negation)
    stopwords = _get_stopwords()
    tokens = [t for t in tokens if t not in stopwords]

    # Lemmatize with SpaCy
    doc = _get_nlp()(" ".join(tokens))
    lemmas = [
        token.lemma_
        for token in doc
        if not token.is_punct and not token.is_space
    ]

    return " ".join(lemmas)
