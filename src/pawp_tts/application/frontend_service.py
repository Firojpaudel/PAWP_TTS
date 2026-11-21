"""Application use case for deterministic bilingual and code-mixed phonemization."""

from __future__ import annotations

import re

from pawp_tts.domain.entities.models import (
    LanguageCode,
    PhonemeToken,
    PhonologicalSequence,
)
from pawp_tts.domain.phonology.codemix import sanitize_prompt, segment_language_spans
from pawp_tts.domain.phonology.inventory import symbol_to_id
from pawp_tts.domain.ports.interfaces import EnglishG2PPort, NepaliG2PPort

_PUNCT_BREAKS = frozenset({",", ".", "?", "!", ":"})


class PhonemizeUtteranceUseCase:
    """Orchestrates sanitization, span segmentation, and language-routed G2P."""

    def __init__(self, english_g2p: EnglishG2PPort, nepali_g2p: NepaliG2PPort) -> None:
        self._en = english_g2p
        self._ne = nepali_g2p

    def execute(self, raw_text: str) -> PhonologicalSequence:
        """Convert raw EN, NE, or code-mixed text into a unified phonological sequence."""
        sanitized = sanitize_prompt(raw_text)
        raw_spans = segment_language_spans(sanitized)

        tokens: list[PhonemeToken] = [
            PhonemeToken(
                symbol="<bos>",
                token_id=symbol_to_id("<bos>"),
                language=LanguageCode.SHARED,
                is_word_start=True,
                is_word_end=True,
            )
        ]

        normalized_parts: list[str] = []
        for idx, span in enumerate(raw_spans):
            if span.language == LanguageCode.SHARED:
                normalized_parts.append(span.text)
                for ch in span.text:
                    if ch in _PUNCT_BREAKS or ch == "-":
                        tokens.append(
                            PhonemeToken(
                                symbol=ch,
                                token_id=symbol_to_id(ch),
                                language=LanguageCode.SHARED,
                                is_phrase_break=ch in _PUNCT_BREAKS,
                            )
                        )
                continue

            # Detect tight clitics (`officeमा`) or tight inter-script hyphens (`check-गरिदिनुहोस्`)
            prev_span = raw_spans[idx - 1] if idx > 0 else None
            is_attached_to_prev = prev_span is not None and (
                span.start_idx == prev_span.end_idx
                or (
                    prev_span.language in {LanguageCode.EN, LanguageCode.NE}
                    and prev_span.language != span.language
                    and sanitized[prev_span.end_idx : span.start_idx] in {"-", "–", "—"}
                )
            )

            if span.language == LanguageCode.EN:
                norm_span = self._en.normalize(span.text)
                normalized_parts.append(norm_span)
                for w_idx, word in enumerate(norm_span.split()):
                    need_wb = (
                        w_idx > 0 or (len(tokens) > 1 and not tokens[-1].is_phrase_break)
                    ) and not (w_idx == 0 and is_attached_to_prev)
                    if need_wb:
                        tokens.append(
                            PhonemeToken(
                                symbol="<wb>",
                                token_id=symbol_to_id("<wb>"),
                                language=LanguageCode.EN,
                            )
                        )
                    self._append_word_tokens(
                        tokens,
                        self._en.phonemize(word),
                        LanguageCode.EN,
                    )
            else:
                norm_span = self._ne.normalize(span.text)
                normalized_parts.append(norm_span)
                words = [w for w in re.split(r"\s+", norm_span) if w]
                for w_idx, word in enumerate(words):
                    need_wb = (
                        w_idx > 0 or (len(tokens) > 1 and not tokens[-1].is_phrase_break)
                    ) and not (w_idx == 0 and is_attached_to_prev)
                    if need_wb:
                        tokens.append(
                            PhonemeToken(
                                symbol="<wb>",
                                token_id=symbol_to_id("<wb>"),
                                language=LanguageCode.NE,
                            )
                        )
                    prev_w = words[w_idx - 1] if w_idx > 0 else None
                    next_w = words[w_idx + 1] if w_idx + 1 < len(words) else None
                    next2_w = words[w_idx + 2] if w_idx + 2 < len(words) else None
                    if hasattr(self._ne, "phonemize_in_context"):
                        syms = self._ne.phonemize_in_context(word, prev_w, next_w, next2_w)
                    else:
                        syms = self._ne.phonemize(word)
                    self._append_word_tokens(tokens, syms, LanguageCode.NE)

        tokens.append(
            PhonemeToken(
                symbol="<eos>",
                token_id=symbol_to_id("<eos>"),
                language=LanguageCode.SHARED,
                is_phrase_break=True,
            )
        )

        return PhonologicalSequence(
            raw_text=raw_text,
            normalized_text=" ".join(normalized_parts).strip(),
            tokens=tuple(tokens),
            spans=raw_spans,
        )

    @staticmethod
    def _append_word_tokens(
        out: list[PhonemeToken],
        symbols: tuple[str, ...],
        language: LanguageCode,
    ) -> None:
        current_stress = 0
        filtered = [s for s in symbols if s]
        non_stress = [s for s in filtered if s not in {"ˈ", "ˌ"}]
        emitted_count = 0
        for sym in filtered:
            if sym == "ˈ":
                current_stress = 1
                continue
            if sym == "ˌ":
                current_stress = 2
                continue
            is_first = emitted_count == 0
            is_last = emitted_count == len(non_stress) - 1
            out.append(
                PhonemeToken(
                    symbol=sym,
                    token_id=symbol_to_id(sym),
                    language=language,
                    stress=current_stress,
                    is_word_start=is_first,
                    is_word_end=is_last,
                    is_syllable_start=is_first,
                    is_phrase_break=sym in _PUNCT_BREAKS,
                )
            )
            emitted_count += 1
            if sym not in {"ː", "̃", "̤"}:
                current_stress = 0
