"""YAML configuration loader mapping files to typed neural config dataclasses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pawp_tts.infrastructure.neural.bilingual_tts import StudentConfig


def load_student_config(config_path: str | Path) -> StudentConfig:
    """Load StudentConfig from a YAML specification file."""
    path = Path(config_path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return StudentConfig(
        model_name=str(raw.get("model_name", "pawp-1.5m-baseline")),
        target_budget=int(raw.get("target_budget", 1_500_000)),
        sample_rate=int(raw.get("sample_rate", 24000)),
        hop_length=int(raw.get("hop_length", 240)),
        n_fft=int(raw.get("n_fft", 1024)),
        n_mels=int(raw.get("n_mels", 80)),
        vocab_size=int(raw.get("vocab_size", 128)),
        num_languages=int(raw.get("num_languages", 4)),
        num_stress_levels=int(raw.get("num_stress_levels", 4)),
        num_boundary_flags=int(raw.get("num_boundary_flags", 4)),
        num_speakers=int(raw.get("num_speakers", 1)),
        d_model=int(raw.get("d_model", 128)),
        enc_layers=int(raw.get("enc_layers", 4)),
        enc_ff=int(raw.get("enc_ff", 256)),
        enc_kernel=int(raw.get("enc_kernel", 5)),
        enc_lookahead=int(raw.get("enc_lookahead", 1)),
        duration_hidden=int(raw.get("duration_hidden", 96)),
        duration_kernel=int(raw.get("duration_kernel", 3)),
        pitch_hidden=int(raw.get("pitch_hidden", 96)),
        pitch_kernel=int(raw.get("pitch_kernel", 5)),
        energy_hidden=int(raw.get("energy_hidden", 64)),
        energy_kernel=int(raw.get("energy_kernel", 5)),
        acoustic_layers=int(raw.get("acoustic_layers", 4)),
        acoustic_ff=int(raw.get("acoustic_ff", 256)),
        acoustic_kernel=int(raw.get("acoustic_kernel", 7)),
        acoustic_lookahead=int(raw.get("acoustic_lookahead", 1)),
        residual_rank=int(raw.get("residual_rank", 16)),
        decoder_dim=int(raw.get("decoder_dim", 192)),
        decoder_ff=int(raw.get("decoder_ff", 384)),
        decoder_layers=int(raw.get("decoder_layers", 3)),
        decoder_kernel=int(raw.get("decoder_kernel", 7)),
        decoder_lookahead=int(raw.get("decoder_lookahead", 1)),
        num_shared_bases=int(raw.get("num_shared_bases", 0)),
        layer_residual_rank=int(raw.get("layer_residual_rank", 0)),
        use_articulatory_conditioning=bool(raw.get("use_articulatory_conditioning", False)),
    )
