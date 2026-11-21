"""Nepali lexicon-first + rule-based G2P infrastructure adapter."""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path

from pawp_tts.domain.phonology.nepali_rules import (
    normalize_nepali_text,
    word_to_nepali_ipa,
)

logger = logging.getLogger(__name__)

DEFAULT_LEXICON_PATH = Path("data/lexicon/ne_lexicon_v1/ne_lexicon.tsv")
DEFAULT_HETERONYMS_PATH = Path("data/lexicon/ne_lexicon_v1/ne_heteronyms.tsv")

_CACHED_LEXICON: dict[str, tuple[str, ...]] | None = None
_CACHED_HETERONYM_WORDS: frozenset[str] | None = None


def _load_lexicon_tsv(path: Path) -> dict[str, tuple[str, ...]]:
    lexicon: dict[str, tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            word = row.get("word", "").strip()
            ipa_str = row.get("ipa", "").strip()
            if word and ipa_str:
                lexicon[word] = tuple(ipa_str.split())
    return lexicon


def _load_heteronym_words_tsv(path: Path) -> frozenset[str]:
    words: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            word = row.get("word", "").strip()
            if word:
                words.add(word)
    return frozenset(words)


class NepaliRuleG2PAdapter:
    """Nepali G2P adapter delegating to a verified pronunciation lexicon with rule-based fallback."""

    def __init__(
        self,
        override_lexicon_path: str | Path | None = None,
        lexicon_path: str | Path | None = DEFAULT_LEXICON_PATH,
        heteronyms_path: str | Path | None = DEFAULT_HETERONYMS_PATH,
    ) -> None:
        global _CACHED_LEXICON, _CACHED_HETERONYM_WORDS
        self._overrides: dict[str, tuple[str, ...]] = {}
        if override_lexicon_path is not None:
            path = Path(override_lexicon_path)
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                self._overrides = {str(k): tuple(str(v) for v in vals) for k, vals in raw.items()}

        self._lexicon: dict[str, tuple[str, ...]] = {}
        if lexicon_path is not None:
            lpath = Path(lexicon_path)
            if lpath.exists():
                if lpath == DEFAULT_LEXICON_PATH:
                    if _CACHED_LEXICON is None:
                        _CACHED_LEXICON = _load_lexicon_tsv(lpath)
                    self._lexicon = _CACHED_LEXICON
                else:
                    self._lexicon = _load_lexicon_tsv(lpath)

        self._heteronym_words: frozenset[str] = frozenset()
        if heteronyms_path is not None:
            hpath = Path(heteronyms_path)
            if hpath.exists():
                if hpath == DEFAULT_HETERONYMS_PATH:
                    if _CACHED_HETERONYM_WORDS is None:
                        _CACHED_HETERONYM_WORDS = _load_heteronym_words_tsv(hpath)
                    self._heteronym_words = _CACHED_HETERONYM_WORDS
                else:
                    self._heteronym_words = _load_heteronym_words_tsv(hpath)

    def normalize(self, text: str) -> str:
        """Normalize Nepali text into canonical Devanagari words."""
        return normalize_nepali_text(text)

    def phonemize(self, text: str) -> tuple[str, ...]:
        """Phonemize a Nepali span using context-aware domain phonological rules."""
        words = text.split()
        out: list[str] = []
        for idx, word in enumerate(words):
            prev_w = words[idx - 1] if idx > 0 else None
            next_w = words[idx + 1] if idx + 1 < len(words) else None
            next2_w = words[idx + 2] if idx + 2 < len(words) else None
            out.extend(self.phonemize_in_context(word, prev_w, next_w, next2_w))
        return tuple(out)

    def phonemize_in_context(
        self,
        word: str,
        prev_word: str | None = None,
        next_word: str | None = None,
        next_next_word: str | None = None,
    ) -> tuple[str, ...]:
        """Convert a single Devanagari word in context into unified IPA symbols."""
        clean = re.sub(r"[^\u0900-\u097F]", "", word)
        if not clean:
            return ()

        # 1. Explicit user/experiment overrides (highest precedence)
        if clean in self._overrides:
            return self._overrides[clean]

        # 2. Context-dependent heteronym words: delegate to domain phonological rules
        # to resolve morpho-syntactic and coarticulatory context (e.g. infinitive -न vs nominal).
        if clean in self._heteronym_words:
            return word_to_nepali_ipa(
                clean,
                prev_word=prev_word,
                next_word=next_word,
                override_lexicon=self._overrides,
                next_next_word=next_next_word,
            )

        # 3. Verified full-lexicon citation match (105,868 entries)
        if clean in self._lexicon:
            return self._lexicon[clean]

        # 4. Fallback to domain phonological rule engine
        return word_to_nepali_ipa(
            clean,
            prev_word=prev_word,
            next_word=next_word,
            override_lexicon=self._overrides,
            next_next_word=next_next_word,
        )
