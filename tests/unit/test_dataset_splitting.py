"""Unit tests for pure dataset deduplication, filtering, and split-leakage prevention."""

from __future__ import annotations

from pawp_tts.domain.dataset.splitting import deduplicate_and_validate_splits
from pawp_tts.domain.entities.models import AudioSampleRecord, LanguageCode


def _make_record(
    sample_id: str,
    split: str,
    speaker_id: str,
    text: str,
    duration_sec: float = 2.5,
    phoneme_ids: tuple[int, ...] = (10, 11, 12, 13, 14),
    lang: LanguageCode = LanguageCode.EN,
    quality_tier: str = "tier1_human",
) -> AudioSampleRecord:
    return AudioSampleRecord(
        sample_id=sample_id,
        split=split,
        language=lang,
        speaker_id=speaker_id,
        text=text,
        normalized_text=text.strip(),
        phoneme_ids=phoneme_ids,
        audio_source_type="wav",
        audio_locator=f"dummy/{sample_id}.wav",
        sample_rate=24000,
        duration_sec=duration_sec,
        quality_tier=quality_tier,
        loss_weight=1.0,
    )


def test_filtering_and_same_speaker_dedup_priority() -> None:
    records = [
        _make_record("s_short", "train", "spk_1", "too short", duration_sec=0.4),
        _make_record("s_long", "train", "spk_1", "too long", duration_sec=18.0),
        _make_record("s_empty", "train", "spk_1", "empty phones", phoneme_ids=()),
        _make_record("s_unk", "train", "spk_1", "unk heavy", phoneme_ids=(3, 10, 11)),
        _make_record("s_dup_train", "train", "spk_1", "Hello world"),
        _make_record("s_dup_test", "test", "spk_1", "hello world"),
        _make_record("s_unique_val", "val", "spk_2", "Distinct validation line"),
    ]
    kept, summary = deduplicate_and_validate_splits(records)

    assert summary.dropped_duration == 2
    assert summary.dropped_unk == 2
    assert summary.dropped_duplicates == 1
    assert summary.leakage_text_count == 0

    kept_ids = {r.sample_id for r in kept}
    assert kept_ids == {"s_dup_test", "s_unique_val"}


def test_cross_split_text_and_multispeaker_disjointness_with_studio_preservation() -> None:
    records = [
        # Test speaker & text
        _make_record("t0", "test", "spk_shared", "Shared sentence across speakers"),
        # Val tries to reuse test speaker -> removed by speaker-disjointness
        _make_record("v_leak_spk", "val", "spk_shared", "Completely new validation sentence"),
        # Train tries to reuse test text with another speaker -> removed by text-disjointness
        _make_record("tr_leak_txt", "train", "spk_3", "  shared sentence across speakers "),
        # Studio speakers should remain allowed across train/val/test when texts are disjoint
        _make_record(
            "st_test",
            "test",
            "studio_female_01",
            "नेपाली स्टुडियो टेस्ट वाक्य।",
            lang=LanguageCode.NE,
            quality_tier="tier1_studio",
        ),
        _make_record(
            "st_val",
            "val",
            "studio_female_01",
            "नेपाली स्टुडियो भ्याल वाक्य।",
            lang=LanguageCode.NE,
            quality_tier="tier1_studio",
        ),
        _make_record(
            "st_train",
            "train",
            "studio_female_01",
            "नेपाली स्टुडियो ट्रेन वाक्य।",
            lang=LanguageCode.NE,
            quality_tier="tier1_studio",
        ),
    ]
    # Add enough distinct speakers so multi-speaker enforcement triggers (>= 10)
    for i in range(4, 13):
        records.append(
            _make_record(f"tr_extra_{i}", "train", f"spk_{i}", f"Training utterance {i}")
        )

    kept, summary = deduplicate_and_validate_splits(records, multi_speaker_min_speakers=10)

    assert summary.leakage_text_count == 0
    assert summary.leakage_speaker_count == 0
    assert summary.removed_cross_split == 2

    kept_ids = {r.sample_id for r in kept}
    assert "v_leak_spk" not in kept_ids
    assert "tr_leak_txt" not in kept_ids
    assert {"st_test", "st_val", "st_train"}.issubset(kept_ids)
