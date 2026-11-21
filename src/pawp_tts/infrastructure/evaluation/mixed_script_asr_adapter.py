"""Code-mix-capable local ASR adapter (mixed-script output mode).

The model runs in its own isolated environment as a persistent subprocess, so its
dependencies never enter this package. Configure via `.env`:
`PAWP_MIXED_ASR_PYTHON` (interpreter of the isolated env) and `PAWP_MIXED_ASR_MODEL_DIR`.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from pawp_tts.domain.entities.models import AsrTranscript, LanguageCode


class MixedScriptLocalAsrAdapter:
    """`AsrEvaluatorPort` implementation backed by a local ASR worker process."""

    def __init__(self, python: str | None = None, model_dir: str | None = None) -> None:
        py = python or os.environ.get("PAWP_MIXED_ASR_PYTHON", "")
        mdir = model_dir or os.environ.get("PAWP_MIXED_ASR_MODEL_DIR", "")
        if not py or not mdir:
            raise RuntimeError(
                "PAWP_MIXED_ASR_PYTHON / PAWP_MIXED_ASR_MODEL_DIR not configured in .env"
            )
        worker = Path(__file__).with_name("mixed_script_asr_worker.py")
        self._proc = subprocess.Popen(
            [py, "-I", str(worker), mdir],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            cwd=mdir,
        )
        ready = self._readline()
        if not ready.get("ready"):
            raise RuntimeError(f"ASR worker failed to start: {ready}")

    def _readline(self) -> dict[str, object]:
        assert self._proc.stdout is not None
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("ASR worker exited unexpectedly")
            line = line.strip()
            if line.startswith("{"):
                parsed: dict[str, object] = json.loads(line)
                return parsed

    def transcribe(
        self, audio_path: str, language: LanguageCode, context_words: str = "string"
    ) -> AsrTranscript:
        assert self._proc.stdin is not None
        t0 = time.perf_counter()
        lang = "en" if language == LanguageCode.EN else "ne"
        req = {"path": str(Path(audio_path).resolve()), "lang": lang, "mode": "mixed"}
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()
        resp = self._readline()
        if "error" in resp:
            raise RuntimeError(f"Mixed-script ASR error: {resp['error']}")
        return AsrTranscript(
            text=str(resp.get("text", "")),
            language=language,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            provider="local-mixed-script-asr",
        )

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
