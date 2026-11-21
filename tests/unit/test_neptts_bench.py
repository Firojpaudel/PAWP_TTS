"""Unit tests for NepTTS-Bench 77 minimal-pair benchmark and phonological duration alignment."""

from __future__ import annotations

from collections import Counter

from pawp_tts.application.frontend_service import PhonemizeUtteranceUseCase
from pawp_tts.domain.evaluation.neptts_bench import NEPTTS_MINIMAL_PAIRS
from pawp_tts.domain.phonology.inventory import SYMBOL_TO_ID
from pawp_tts.infrastructure.data.dataset_loader import compute_phonological_durations
from pawp_tts.infrastructure.frontend.english_adapter import EnglishEspeakG2PAdapter
from pawp_tts.infrastructure.frontend.nepali_adapter import NepaliRuleG2PAdapter


def test_neptts_77_minimal_pairs_completeness_and_phonological_contrasts() -> None:
    assert len(NEPTTS_MINIMAL_PAIRS) == 77
    cat_counts = Counter(p.category for p in NEPTTS_MINIMAL_PAIRS)
    assert len(cat_counts) == 7
    for cat, count in cat_counts.items():
        assert count == 11, f"Category {cat} has {count} pairs instead of 11"

    phonemize = PhonemizeUtteranceUseCase(
        english_g2p=EnglishEspeakG2PAdapter(),
        nepali_g2p=NepaliRuleG2PAdapter(),
    )
    unk_id = SYMBOL_TO_ID["<unk>"]

    for pair in NEPTTS_MINIMAL_PAIRS:
        seq_a = phonemize.execute(pair.word_a)
        seq_b = phonemize.execute(pair.word_b)

        assert unk_id not in seq_a.token_ids, f"<unk> in {pair.pair_id} word_a={pair.word_a}"
        assert unk_id not in seq_b.token_ids, f"<unk> in {pair.pair_id} word_b={pair.word_b}"

        if pair.category != "english_and_codemix_contrasts":
            assert "ʃ" not in seq_a.symbols, f"Hindi /ʃ/ bleed on {pair.word_a}"
            assert "ʃ" not in seq_b.symbols, f"Hindi /ʃ/ bleed on {pair.word_b}"

        if pair.expect_distinct_phonemes:
            assert seq_a.symbols != seq_b.symbols, (
                f"Expected distinct IPA for {pair.pair_id} ({pair.word_a} vs {pair.word_b}), "
                f"got {seq_a.symbols}"
            )
        else:
            assert seq_a.symbols == seq_b.symbols, (
                f"Expected sibilant merger for {pair.pair_id} ({pair.word_a} vs {pair.word_b}), "
                f"got {seq_a.symbols} vs {seq_b.symbols}"
            )


def test_compute_phonological_durations_exact_frame_sum() -> None:
    phonemize = PhonemizeUtteranceUseCase(
        english_g2p=EnglishEspeakG2PAdapter(),
        nepali_g2p=NepaliRuleG2PAdapter(),
    )
    seq = phonemize.execute("आज मेरो officeमा meeting छ।")
    for target_frames in (len(seq.token_ids), 85, 143, 250):
        durs = compute_phonological_durations(seq.token_ids, target_frames)
        assert len(durs) == len(seq.token_ids)
        assert int(durs.sum()) == target_frames
        assert all(int(d) >= 1 for d in durs)
