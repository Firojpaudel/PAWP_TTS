"""Application use case for building leak-free bilingual dataset manifests."""

from __future__ import annotations

from datetime import UTC, datetime

from pawp_tts.domain.entities.models import DatasetSplitSummary, ExperimentRecord
from pawp_tts.domain.ports.interfaces import ExperimentLedgerPort, ManifestBuilderPort


class BuildDatasetManifestsUseCase:
    """Orchestrates bilingual manifest compilation, leak validation, and ledger logging."""

    def __init__(
        self,
        builder: ManifestBuilderPort,
        ledger: ExperimentLedgerPort | None = None,
    ) -> None:
        self._builder = builder
        self._ledger = ledger

    def execute(
        self,
        output_dir: str = "data/manifests",
        include_supplemental: bool = False,
        max_en_shards: int | None = None,
        exp_id: str = "exp001-data",
        git_sha: str = "unknown",
    ) -> DatasetSplitSummary:
        """Build deduplicated, split-validated train/val/test manifests."""
        summary = self._builder.build_and_write(
            output_dir=output_dir,
            include_supplemental=include_supplemental,
            max_en_shards=max_en_shards,
        )
        if summary.leakage_text_count != 0 or summary.leakage_speaker_count != 0:
            raise AssertionError(
                f"Split leakage detected: text={summary.leakage_text_count}, "
                f"speaker={summary.leakage_speaker_count}"
            )
        if self._ledger is not None:
            self._ledger.append(
                ExperimentRecord(
                    exp_id=exp_id,
                    timestamp_utc=datetime.now(tz=UTC).isoformat(),
                    git_sha=git_sha,
                    config_path="configs/student_1_5m.yaml",
                    seed=42,
                    hypothesis="Compile leak-free bilingual EN/NE/CodeMix manifests without duplicating audio.",
                    metrics={
                        "train_samples": summary.counts_by_split.get("train", 0),
                        "val_samples": summary.counts_by_split.get("val", 0),
                        "test_samples": summary.counts_by_split.get("test", 0),
                        "train_hours": summary.hours_by_split.get("train", 0.0),
                        "val_hours": summary.hours_by_split.get("val", 0.0),
                        "test_hours": summary.hours_by_split.get("test", 0.0),
                        "leakage_text_count": summary.leakage_text_count,
                        "leakage_speaker_count": summary.leakage_speaker_count,
                    },
                    command="make data-manifest",
                )
            )
        return summary
