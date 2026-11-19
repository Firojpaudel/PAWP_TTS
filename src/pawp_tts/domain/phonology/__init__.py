"""Phonology domain package."""

from pawp_tts.domain.phonology.codemix import (
    build_language_masks,
    sanitize_prompt,
    segment_language_spans,
)
from pawp_tts.domain.phonology.inventory import (
    ARTICULATORY_DIM,
    MAX_VOCAB_SIZE,
    UNIFIED_PHONEMES,
    parse_ipa_string,
    phoneme_to_articulatory_vector,
    symbol_to_id,
)
from pawp_tts.domain.phonology.nepali_rules import (
    normalize_nepali_text,
    segment_aksharas,
    word_to_nepali_ipa,
)

__all__ = [
    "ARTICULATORY_DIM",
    "MAX_VOCAB_SIZE",
    "UNIFIED_PHONEMES",
    "build_language_masks",
    "normalize_nepali_text",
    "parse_ipa_string",
    "phoneme_to_articulatory_vector",
    "sanitize_prompt",
    "segment_aksharas",
    "segment_language_spans",
    "symbol_to_id",
    "word_to_nepali_ipa",
]
