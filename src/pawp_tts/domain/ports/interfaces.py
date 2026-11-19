"""Domain port protocols for dependency inversion."""

from __future__ import annotations

from typing import Protocol

from pawp_tts.domain.entities.models import (
    AsrTranscript,
    CheckpointSummary,
    DatasetSplitSummary,
    ExperimentRecord,
    LanguageCode,
    ParamCountReport,
)


class EnglishG2PPort(Protocol):
    """Port for deterministic English normalization and IPA grapheme-to-phoneme conversion."""

    def normalize(self, text: str) -> str:
        """Normalize English text (numbers, acronyms, abbreviations)."""

    def phonemize(self, text: str) -> tuple[str, ...]:
        """Convert normalized English span into unified IPA symbols."""


class NepaliG2PPort(Protocol):
    """Port for Nepali normalization and contrast-preserving IPA conversion."""

    def normalize(self, text: str) -> str:
        """Normalize Nepali text (Devanagari NFC, numbers, currency)."""

    def phonemize(self, text: str) -> tuple[str, ...]:
        """Convert normalized Nepali span into unified IPA symbols."""


class AsrEvaluatorPort(Protocol):
    """Port for independent speech recognition evaluation."""

    def transcribe(
        self,
        audio_path: str,
        language: LanguageCode,
        context_words: str = "string",
    ) -> AsrTranscript:
        """Transcribe an audio file for the requested language."""


class ParamAuditorPort(Protocol):
    """Port for 4-metric parameter and footprint auditing of neural models."""

    def audit_model(self, config_path: str) -> ParamCountReport:
        """Instantiate and measure the complete deployed inference graph."""


class CheckpointInspectorPort(Protocol):
    """Port for reading checkpoint metadata without loading full tensors."""

    def inspect(self, checkpoint_path: str) -> CheckpointSummary:
        """Return parameter counts, dtypes, and shape pools for a checkpoint."""


class ExperimentLedgerPort(Protocol):
    """Port for appending reproducible experiment records."""

    def append(self, record: ExperimentRecord) -> None:
        """Persist an experiment record."""


class ManifestBuilderPort(Protocol):
    """Port for building deduplicated, leak-free bilingual dataset manifests."""

    def build_and_write(
        self,
        output_dir: str = "data/manifests",
        include_supplemental: bool = False,
        max_en_shards: int | None = None,
    ) -> DatasetSplitSummary:
        """Scan sources, build splits, write JSONL manifests, and return split summary."""
