"""Unit tests for Nepali phonological rules, English contrasts, and code-mix routing."""

from __future__ import annotations

import time

from pawp_tts.application.frontend_service import PhonemizeUtteranceUseCase
from pawp_tts.domain.entities.models import LanguageCode
from pawp_tts.domain.phonology.codemix import (
    build_language_masks,
    sanitize_prompt,
    segment_language_spans,
)
from pawp_tts.domain.phonology.inventory import (
    ARTICULATORY_DIM,
    MAX_VOCAB_SIZE,
    SYMBOL_TO_ID,
    UNIFIED_PHONEMES,
    phoneme_to_articulatory_vector,
)
from pawp_tts.domain.phonology.nepali_rules import normalize_nepali_text, word_to_nepali_ipa
from pawp_tts.infrastructure.frontend.english_adapter import EnglishEspeakG2PAdapter
from pawp_tts.infrastructure.frontend.nepali_adapter import NepaliRuleG2PAdapter


def test_inventory_budget_and_articulatory_dims() -> None:
    assert len(UNIFIED_PHONEMES) <= MAX_VOCAB_SIZE
    vec = phoneme_to_articulatory_vector("ɡʱ")
    assert len(vec) == ARTICULATORY_DIM
    assert sum(vec) > 0


def test_nepali_sibilant_merger_prevents_hindi_bleed() -> None:
    for word in ("शहर", "सस्तो", "षड्यन्त्र"):
        ipa = word_to_nepali_ipa(word)
        assert "s" in ipa
        assert "ʃ" not in ipa
        assert "ə" not in ipa


def test_nepali_dental_vs_retroflex_and_four_way_stops() -> None:
    dental = word_to_nepali_ipa("तीन")
    retro = word_to_nepali_ipa("टिन")
    assert dental[0] == "t̪"
    assert retro[0] == "ʈ"

    four_way = [
        word_to_nepali_ipa("काल")[0],
        word_to_nepali_ipa("खाल")[0],
        word_to_nepali_ipa("गाल")[0],
        word_to_nepali_ipa("घाम")[0],
    ]
    assert four_way == ["k", "kʰ", "ɡ", "ɡʱ"]


def test_nepali_alveolar_affricates() -> None:
    affr = [
        word_to_nepali_ipa("चार")[0],
        word_to_nepali_ipa("छाल")[0],
        word_to_nepali_ipa("जाल")[0],
        word_to_nepali_ipa("झाल")[0],
    ]
    assert affr == ["ts", "tsʰ", "dz", "dzʱ"]
    assert "tʃ" not in affr
    assert "dʒ" not in affr


def test_nepali_gemination_and_vowel_length_neutralization() -> None:
    singleton = word_to_nepali_ipa("पाता")
    geminate = word_to_nepali_ipa("पत्ता")
    assert "ː" not in singleton
    assert "ː" in geminate

    # Orthographic इ/ई merge phonemically to /i/
    assert word_to_nepali_ipa("दिन") == word_to_nepali_ipa("दीन")


def test_nepali_schwa_syncope_tatsama_and_verbal_infinitive() -> None:
    # Nominal final consonant deletes schwa
    assert word_to_nepali_ipa("घर")[-1] != "ʌ"
    # Tatsama final consonant cluster retains schwa
    assert word_to_nepali_ipa("मित्र")[-1] == "ʌ"
    # Verbal infinitive '-न' before auxiliary retains schwa vs nominal after numeral
    assert word_to_nepali_ipa("दिन", prev_word=None, next_word="नपाएर")[-1] == "ʌ"
    assert word_to_nepali_ipa("दिन", prev_word="तीन", next_word="घर")[-1] != "ʌ"


def test_nepali_intervocalic_flapping_and_h_deletion() -> None:
    flapped = word_to_nepali_ipa("सडक")
    assert "ɽ" in flapped

    breathy_len = word_to_nepali_ipa("पहाड")
    assert "̤" in breathy_len and "ː" in breathy_len and "ɽ" in breathy_len


def test_nepali_number_and_currency_expansion() -> None:
    norm = normalize_nepali_text("रु १२५ तिर्नुहोस्।")
    assert "एक सय पच्चिस रुपैयाँ" in norm


def test_prompt_sanitization_and_codemix_segmentation() -> None:
    cleaned = sanitize_prompt("[urgent] Check NP-23-001... please (now)")
    assert "[urgent]" not in cleaned
    assert "(now)" not in cleaned
    assert "N P 2 3 0 0 1" in cleaned

    spans = segment_language_spans("आज मेरो office मा meeting छ।")
    langs = [s.language for s in spans]
    assert langs == [
        LanguageCode.NE,
        LanguageCode.EN,
        LanguageCode.NE,
        LanguageCode.EN,
        LanguageCode.NE,
        LanguageCode.SHARED,
    ]


def test_end_to_end_phonemize_use_case() -> None:
    use_case = PhonemizeUtteranceUseCase(
        english_g2p=EnglishEspeakG2PAdapter(),
        nepali_g2p=NepaliRuleG2PAdapter(),
    )
    seq = use_case.execute("आज मेरो office मा meeting छ।")
    assert seq.tokens[0].symbol == "<bos>"
    assert seq.tokens[-1].symbol == "<eos>"
    assert any(t.language == LanguageCode.EN for t in seq.tokens)
    assert any(t.language == LanguageCode.NE for t in seq.tokens)


def test_codemix_clitics_hyphens_context_digits_and_masks() -> None:
    use_case = PhonemizeUtteranceUseCase(
        english_g2p=EnglishEspeakG2PAdapter(),
        nepali_g2p=NepaliRuleG2PAdapter(),
    )

    # 1. Attached clitic (`officeमा`) joins without inserting a hard `<wb>` pause
    clitic_seq = use_case.execute("officeमा")
    assert "<wb>" not in clitic_seq.symbols
    assert clitic_seq.tokens[1].language == LanguageCode.EN
    assert clitic_seq.tokens[-2].language == LanguageCode.NE

    # 2. Hyphenated EN-NE verb compound (`check-गरिदिनुहोस्`) joins smoothly
    hyphen_seq = use_case.execute("balance check-गरिदिनुहोस्")
    assert "-" not in hyphen_seq.symbols

    # 3. Context-sensitive ASCII digits (`500` in NE vs EN context)
    ne_num_seq = use_case.execute("मेरो 500 रुपैयाँ छ")
    assert "पाँच सय" in ne_num_seq.normalized_text
    en_num_seq = use_case.execute("pay 500 dollars")
    assert "five hundred" in en_num_seq.normalized_text

    # 4. Comma numbers, consonant initialisms, and digit+letter tech tokens (`25,000`, `SMS`, `4G`)
    tech_seq = use_case.execute("तिम्रो 25,000 SMS र 4G छ")
    assert "S M S" in tech_seq.normalized_text
    assert "four G" in tech_seq.normalized_text
    assert "dʒ" in tech_seq.symbols

    # 4b. Prevent medial schwa deletion right before a geminate/conjunct (`अहिलेसम्म`)
    gem_compound = word_to_nepali_ipa("अहिलेसम्म")
    assert gem_compound[-4:] == ("ʌ", "m", "ː", "ʌ")

    # 5. Coarticulatory language mask blending at EN <-> NE boundary
    masks = build_language_masks(clitic_seq.tokens, boundary_blend=0.15)
    assert masks[0] == (0.0, 0.0)  # <bos>
    assert masks[1] == (1.0, 0.0)  # interior EN phoneme
    assert masks[-2] == (0.0, 1.0)  # interior NE phoneme
    assert any(abs(m_en - 0.85) < 1e-6 and abs(m_ne - 0.15) < 1e-6 for m_en, m_ne in masks)
    assert any(abs(m_en - 0.15) < 1e-6 and abs(m_ne - 0.85) < 1e-6 for m_en, m_ne in masks)


def test_english_unbiased_inprocess_espeak_g2p() -> None:
    adapter = EnglishEspeakG2PAdapter()
    unk_id = SYMBOL_TO_ID["<unk>"]
    use_case = PhonemizeUtteranceUseCase(english_g2p=adapter, nepali_g2p=NepaliRuleG2PAdapter())

    for rare_word in ("chrysanthemum", "xylophone", "abstracted", "certain", "pronunciation"):
        syms = adapter.phonemize(rare_word)
        assert syms, f"Expected non-empty IPA for {rare_word}"
        seq = use_case.execute(rare_word)
        assert unk_id not in seq.token_ids, f"Unexpected <unk> token for {rare_word}: {seq.symbols}"

    assert "ʃ" in adapter.phonemize("she")
    assert "s" in adapter.phonemize("sells") and "ʃ" not in adapter.phonemize("sells")
    assert "θ" in adapter.phonemize("thin")
    assert "t" in adapter.phonemize("tin") and "θ" not in adapter.phonemize("tin")
    assert "v" in adapter.phonemize("vine")
    assert "w" in adapter.phonemize("wine")

    sentence = "Artificial intelligence speech synthesis requires crisp pronunciation."
    use_case.execute(sentence)  # warmup
    t0 = time.perf_counter()
    for _ in range(50):
        use_case.execute(sentence)
    mean_ms = (time.perf_counter() - t0) * 1000.0 / 50.0
    assert mean_ms < 5.0
