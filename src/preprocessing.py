import re

import pandas as pd
import nltk
import spacy
from langchain_text_splitters import RecursiveCharacterTextSplitter

_nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

_nltk_stopwords = set(nltk.corpus.stopwords.words("english"))
_NEGATION_WORDS = {
    "no", "not", "nor", "without", "none", "never", "neither",
    "cannot", "don", "doesn", "didn", "isn", "wasn", "weren",
    "hasn", "haven", "wouldn", "shouldn", "couldn",
}
_STOPWORDS = _nltk_stopwords - _NEGATION_WORDS

_EXCLUDED_SPECIALTIES = {"SOAP / Chart / Progress Notes"}


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the mtsamples CSV and drop rows with empty transcriptions."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["transcription"])
    df = df[df["transcription"].str.strip() != ""]
    df["medical_specialty"] = df["medical_specialty"].str.strip()
    df = df[~df["medical_specialty"].isin(_EXCLUDED_SPECIALTIES)]
    df = df.reset_index(drop=True)
    return df


def clean_text(text: str) -> str:
    """Clean a clinical transcription: lowercase, remove noise, remove
    stopwords (keeping negation), and lemmatize with SpaCy."""
    text = text.lower()
    # Remove special characters but keep hyphens, periods, and apostrophes
    text = re.sub(r"[^a-z0-9\s\-\.']", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize with NLTK
    tokens = nltk.word_tokenize(text)

    # Remove stopwords (keep negation)
    tokens = [t for t in tokens if t not in _STOPWORDS]

    # Lemmatize with SpaCy
    doc = _nlp(" ".join(tokens))
    lemmas = [
        token.lemma_
        for token in doc
        if not token.is_punct and not token.is_space
    ]

    return " ".join(lemmas)


def chunk_documents(
    df: pd.DataFrame,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    """Split transcriptions into overlapping chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # try and split the text at paragraph breaks first
        # if paragraph is too long, it tries sentences
        # if a sentence is too long, it tries words
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )

    all_chunks = []
    for idx, row in df.iterrows():
        text = row["transcription"]
        specialty = row.get("medical_specialty", "")
        sample_name = row.get("sample_name", "")

        splits = splitter.split_text(text)
        for chunk_idx, chunk_text in enumerate(splits):
            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source_index": int(idx),
                    "medical_specialty": specialty,
                    "sample_name": sample_name,
                    "chunk_index": chunk_idx,
                },
            })

    return all_chunks


def get_token_lengths(texts: pd.Series) -> pd.Series:
    """Count tokens per text using NLTK word_tokenize."""
    return texts.apply(lambda t: len(nltk.word_tokenize(str(t))))
