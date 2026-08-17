"""Tests for the preprocessing module (src/preprocessing.py)."""

import os
import pickle

import pandas as pd
import pytest

from src.preprocessing import load_data, clean_text, chunk_documents

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data")
CSV_PATH = os.path.join(DATA_DIR, "mtsamples.csv")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.pkl")


# ---- load_data ------------------------------------------------------------

class TestLoadData:
    def test_returns_dataframe_with_expected_columns(self):
        df = load_data(CSV_PATH)
        assert isinstance(df, pd.DataFrame)
        assert "transcription" in df.columns
        assert "medical_specialty" in df.columns
        assert len(df) > 0

    def test_no_empty_transcriptions(self):
        df = load_data(CSV_PATH)
        assert df["transcription"].str.strip().ne("").all()


# ---- clean_text -----------------------------------------------------------

class TestCleanText:
    def test_lowercases_text(self):
        result = clean_text("The Patient Has CHEST PAIN.")
        assert result == result.lower()

    def test_preserves_negation(self):
        result = clean_text("Patient does not have fever.")
        assert "not" in result


# ---- chunk_documents ------------------------------------------------------

class TestChunkDocuments:
    @pytest.fixture()
    def _small_df(self):
        return pd.DataFrame({
            "transcription": ["This is a short transcription for testing."],
            "medical_specialty": ["General Medicine"],
            "sample_name": ["Test Sample"],
        })

    def test_chunk_metadata_keys(self, _small_df):
        chunks = chunk_documents(_small_df)
        assert len(chunks) > 0
        meta = chunks[0]["metadata"]
        assert "source_index" in meta
        assert "medical_specialty" in meta
        assert "sample_name" in meta
        assert "chunk_index" in meta

    def test_chunks_pkl_exists_and_is_list(self):
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        assert isinstance(chunks, list)
        assert len(chunks) > 1000  # should be ~38k
        assert "text" in chunks[0]
        assert "metadata" in chunks[0]
