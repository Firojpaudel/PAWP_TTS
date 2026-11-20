"""JSONL reproducibility experiment ledger storage."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from pawp_tts.domain.entities.models import ExperimentRecord


def current_git_sha() -> str:
    """Return short HEAD SHA or `nogit` when outside a git repo."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return proc.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "nogit"


class JsonlExperimentLedger:
    """Appends structured ExperimentRecord rows to a JSONL file."""

    def __init__(self, path: str | Path = "artifacts/reports/experiment_ledger.jsonl") -> None:
        self._path = Path(path)

    def append(self, record: ExperimentRecord) -> None:
        """Write record atomically as a single JSON line."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(record), ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(payload + "\n")
