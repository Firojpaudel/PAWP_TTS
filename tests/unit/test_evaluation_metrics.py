"""Unit tests for WER/CER alignment metrics."""

from __future__ import annotations

from pawp_tts.domain.evaluation.metrics import compute_cer, compute_wer


def test_exact_match_metrics() -> None:
    wer = compute_wer("नमस्ते, म दीपक हुँ।", "नमस्ते म दीपक हुँ")
    cer = compute_cer("नमस्ते, म दीपक हुँ।", "नमस्ते म दीपक हुँ")
    assert wer.error_rate == 0.0
    assert cer.error_rate == 0.0


def test_substitution_deletion_insertion_counts() -> None:
    wer = compute_wer("one two three four", "one five three four extra")
    assert wer.substitutions == 1
    assert wer.insertions == 1
    assert wer.deletions == 0
    assert wer.error_rate == 0.5
