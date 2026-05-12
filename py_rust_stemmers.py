from __future__ import annotations


class SnowballStemmer:
    """Minimal compatibility shim for environments without compiled stemmers."""

    def __init__(self, language: str) -> None:
        self.language = language

    def stem_word(self, word: str) -> str:
        return word
