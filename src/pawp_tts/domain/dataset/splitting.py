"""Pure dataset deduplication and split leakage verification utilities."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from pawp_tts.domain.entities.models import AudioSampleRecord, DatasetSplitSummary


def deduplicate_and_validate_splits(
    records: Sequence[AudioSampleRecord],
    min_duration_sec: float = 0.8,
    max_duration_sec: float = 15.0,
    max_unk_ratio: float = 0.02,
    unk_token_id: int = 3,
    multi_speaker_min_speakers: int = 10,
) -> tuple[list[AudioSampleRecord], DatasetSplitSummary]:
    """
    Filter invalid durations/UNK ratios, deduplicate `(normalized_text, speaker_id)`,
    remove cross-split text and multi-speaker leakages (priority: test > val > train),
    and emit an explicit split audit summary.
    """
    dropped_duration = 0
    dropped_unk = 0
    dropped_duplicates = 0
    removed_cross_split = 0

    filtered: list[AudioSampleRecord] = []
    seen_pairs: set[tuple[str, str]] = set()
    speakers_all: set[str] = set()

    # Deterministic order: keep test, then val, then train so eval splits take precedence
    split_priority = {"test": 0, "val": 1, "train": 2}
    ordered = sorted(records, key=lambda r: (split_priority.get(r.split, 3), r.sample_id))

    for rec in ordered:
        if rec.duration_sec < min_duration_sec or rec.duration_sec > max_duration_sec:
            dropped_duration += 1
            continue
        if not rec.phoneme_ids:
            dropped_unk += 1
            continue
        unk_count = sum(1 for tid in rec.phoneme_ids if tid == unk_token_id)
        if unk_count / len(rec.phoneme_ids) > max_unk_ratio:
            dropped_unk += 1
            continue

        norm_key = rec.normalized_text.strip().lower()
        pair_key = (norm_key, rec.speaker_id)
        if pair_key in seen_pairs:
            dropped_duplicates += 1
            continue
        seen_pairs.add(pair_key)
        speakers_all.add(rec.speaker_id)
        filtered.append(rec)

    enforce_speaker_disjoint = len(speakers_all) >= multi_speaker_min_speakers

    eval_texts: set[str] = set()
    test_texts: set[str] = set()
    val_speakers: set[str] = set()
    test_speakers: set[str] = set()

    kept: list[AudioSampleRecord] = []
    for rec in filtered:
        norm_key = rec.normalized_text.strip().lower()
        if rec.split == "test":
            test_texts.add(norm_key)
            eval_texts.add(norm_key)
            test_speakers.add(rec.speaker_id)
            kept.append(rec)
        elif rec.split == "val":
            if norm_key in test_texts or (
                enforce_speaker_disjoint
                and rec.speaker_id in test_speakers
                and rec.quality_tier != "tier1_studio"
            ):
                removed_cross_split += 1
                continue
            eval_texts.add(norm_key)
            val_speakers.add(rec.speaker_id)
            kept.append(rec)
        else:
            eval_speakers = test_speakers | val_speakers
            if norm_key in eval_texts or (
                enforce_speaker_disjoint
                and rec.speaker_id in eval_speakers
                and rec.quality_tier != "tier1_studio"
            ):
                removed_cross_split += 1
                continue
            kept.append(rec)

    # Final independent verification pass
    by_split_texts: dict[str, set[str]] = defaultdict(set)
    by_split_speakers: dict[str, set[str]] = defaultdict(set)
    by_split_non_studio_speakers: dict[str, set[str]] = defaultdict(set)
    counts_by_split: dict[str, int] = defaultdict(int)
    secs_by_split: dict[str, float] = defaultdict(float)
    secs_by_lang: dict[str, float] = defaultdict(float)

    for rec in kept:
        by_split_texts[rec.split].add(rec.normalized_text.strip().lower())
        by_split_speakers[rec.split].add(rec.speaker_id)
        if rec.quality_tier != "tier1_studio":
            by_split_non_studio_speakers[rec.split].add(rec.speaker_id)
        counts_by_split[rec.split] += 1
        secs_by_split[rec.split] += rec.duration_sec
        secs_by_lang[rec.language.value] += rec.duration_sec

    leak_texts = len(by_split_texts["train"] & (by_split_texts["val"] | by_split_texts["test"]))
    leak_spk = (
        len(
            (
                by_split_non_studio_speakers["train"]
                & (by_split_non_studio_speakers["val"] | by_split_non_studio_speakers["test"])
            )
            | (by_split_non_studio_speakers["val"] & by_split_non_studio_speakers["test"])
        )
        if enforce_speaker_disjoint
        else 0
    )

    summary = DatasetSplitSummary(
        counts_by_split=dict(counts_by_split),
        hours_by_split={k: round(v / 3600.0, 3) for k, v in secs_by_split.items()},
        hours_by_language={k: round(v / 3600.0, 3) for k, v in secs_by_lang.items()},
        speakers_by_split={k: len(v) for k, v in by_split_speakers.items()},
        dropped_duplicates=dropped_duplicates,
        dropped_duration=dropped_duration,
        dropped_unk=dropped_unk,
        removed_cross_split=removed_cross_split,
        leakage_text_count=leak_texts,
        leakage_speaker_count=leak_spk,
    )
    return kept, summary
