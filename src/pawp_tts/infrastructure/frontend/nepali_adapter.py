"""Nepali rule-based + override-lexicon G2P infrastructure adapter."""

from __future__ import annotations

import json
from pathlib import Path

from pawp_tts.domain.phonology.nepali_rules import normalize_nepali_text, word_to_nepali_ipa


class NepaliRuleG2PAdapter:
    """Adapter loading an optional pronunciation override lexicon and delegating to domain rules."""

    def __init__(self, override_lexicon_path: str | Path | None = None) -> None:
        self._overrides: dict[str, tuple[str, ...]] = {}
        if override_lexicon_path is not None:
            path = Path(override_lexicon_path)
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                self._overrides = {str(k): tuple(str(v) for v in vals) for k, vals in raw.items()}

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
            out.extend(self.phonemize_in_context(word, prev_w, next_w))
        return tuple(out)

    def phonemize_in_context(
        self,
        word: str,
        prev_word: str | None = None,
        next_word: str | None = None,
    ) -> tuple[str, ...]:
        """Convert a single Devanagari word in context into unified IPA symbols."""
        return word_to_nepali_ipa(
            word,
            prev_word=prev_word,
            next_word=next_word,
            override_lexicon=self._overrides,
        )
