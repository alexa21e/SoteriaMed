"""Download the non-pip assets the package needs.

spaCy models and NLTK corpora are not pip-installable, so `pip install -e .`
leaves a fresh clone unable to call `clean_text`. A documented command in a
README rots silently; an executed one fails loudly, so this is a script.

Phase 3 extends it with scispaCy and the HPO `.obo`.

    python scripts/bootstrap.py
"""

from __future__ import annotations

import subprocess
import sys

SPACY_MODELS = ["en_core_web_sm"]
NLTK_CORPORA = ["stopwords", "punkt", "punkt_tab"]


def fetch_spacy_models() -> None:
    for model in SPACY_MODELS:
        print(f"spacy: {model}")
        subprocess.run([sys.executable, "-m", "spacy", "download", model], check=True)


def fetch_nltk_corpora() -> None:
    import nltk

    for corpus in NLTK_CORPORA:
        print(f"nltk: {corpus}")
        nltk.download(corpus, quiet=False, raise_on_error=True)


def main() -> int:
    fetch_spacy_models()
    fetch_nltk_corpora()

    # Prove it worked rather than trusting the downloads' own exit codes.
    from soteriamed.corpus.text import clean_text

    cleaned = clean_text("The patient has no fever.")
    assert "not" in cleaned or "no" in cleaned, cleaned
    print(f"\nok: clean_text('The patient has no fever.') -> {cleaned!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
