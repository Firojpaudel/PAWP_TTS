"""Application use cases for parameter auditing, checkpoint inspection, and ASR evaluation."""

from __future__ import annotations

from datetime import UTC, datetime

from pawp_tts.domain.entities.models import (
    AsrTranscript,
    CheckpointSummary,
    ExperimentRecord,
    LanguageCode,
    ParamCountReport,
    TextErrorStats,
)
from pawp_tts.domain.evaluation.metrics import compute_cer, compute_wer
from pawp_tts.domain.ports.interfaces import (
    AsrEvaluatorPort,
    CheckpointInspectorPort,
    ExperimentLedgerPort,
    ParamAuditorPort,
)


class CountModelParamsUseCase:
    """Audit full 4-metric parameter and storage footprints and optionally log to ledger."""

    def __init__(
        self,
        auditor: ParamAuditorPort,
        ledger: ExperimentLedgerPort | None = None,
    ) -> None:
        self._auditor = auditor
        self._ledger = ledger

    def execute(
        self,
        config_path: str,
        exp_id: str = "exp000",
        seed: int = 42,
        git_sha: str = "unknown",
        log_ledger: bool = False,
    ) -> ParamCountReport:
        """Measure deployed model parameters and record to the reproducibility ledger."""
        report = self._auditor.audit_model(config_path)
        if log_ledger and self._ledger is not None:
            record = ExperimentRecord(
                exp_id=exp_id,
                timestamp_utc=datetime.now(tz=UTC).isoformat(),
                git_sha=git_sha,
                config_path=config_path,
                seed=seed,
                hypothesis="Audit complete inference graph parameter budget against target limit.",
                metrics={
                    "total_params": report.total_params,
                    "unique_stored_params": report.unique_stored_params,
                    "effective_compute_params": report.effective_compute_params,
                    "macs_per_sec_audio": report.macs_per_sec_audio,
                    "fp32_bytes": report.fp32_bytes,
                    "fp16_bytes": report.fp16_bytes,
                    "int8_bytes": report.int8_bytes,
                },
                command=f"make count-params CONFIG={config_path} EXP_ID={exp_id}",
            )
            self._ledger.append(record)
        return report


class InspectCheckpointUseCase:
    """Inspect external or baseline checkpoints without loading full weights into RAM."""

    def __init__(self, inspector: CheckpointInspectorPort) -> None:
        self._inspector = inspector

    def execute(self, checkpoint_path: str) -> CheckpointSummary:
        """Inspect tensor metadata and shape-compatible pools."""
        return self._inspector.inspect(checkpoint_path)


class EvaluateAsrUseCase:
    """Transcribe synthesized or reference audio and score WER/CER if reference is supplied."""

    def __init__(self, asr: AsrEvaluatorPort) -> None:
        self._asr = asr

    def execute(
        self,
        audio_path: str,
        language: LanguageCode,
        reference_text: str | None = None,
        context_words: str = "string",
    ) -> tuple[AsrTranscript, TextErrorStats | None, TextErrorStats | None]:
        """Run ASR and optionally return WER and CER alignment breakdowns."""
        transcript = self._asr.transcribe(
            audio_path=audio_path,
            language=language,
            context_words=context_words,
        )
        if reference_text is None:
            return transcript, None, None
        wer = compute_wer(reference_text, transcript.text)
        cer = compute_cer(reference_text, transcript.text)
        return transcript, wer, cer
