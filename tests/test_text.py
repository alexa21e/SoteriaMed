"""Tests for sparse-path text cleaning (soteriamed/corpus/text.py).

The proof-of-concept's `load_data` and `chunk_documents` tests went with those
functions — they read an untracked CSV and pickle, and both were specific
to the old corpus.
"""

from soteriamed.corpus.text import clean_text


class TestCleanText:
    def test_lowercases_text(self):
        result = clean_text("The Patient Has CHEST PAIN.")
        assert result == result.lower()

    def test_preserves_negation(self):
        result = clean_text("Patient does not have fever.")
        assert "not" in result

    def test_preserves_without(self):
        """Dropping 'without' inverts clinical meaning. See CLAUDE.md."""
        assert "without" in clean_text("Chest tightness without radiation.")

    def test_removes_ordinary_stopwords(self):
        assert "the" not in clean_text("The patient and the doctor").split()

    def test_lemmatises(self):
        assert "cough" in clean_text("Patient reports coughing and coughs")

    def test_empty_input(self):
        assert clean_text("") == ""
