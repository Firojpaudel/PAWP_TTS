"""Integration tests for the complete PAWP-1.5M bilingual PyTorch model."""

from __future__ import annotations

from pathlib import Path

import torch

from pawp_tts.infrastructure.neural.bilingual_tts import PAWPBilingualTTS
from pawp_tts.infrastructure.neural.convnext import GatedLowRankLinear
from pawp_tts.infrastructure.persistence.config_loader import load_student_config
from pawp_tts.infrastructure.weight_programming.param_auditor import TorchParamAuditor

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "student_1_5m.yaml"


def test_pawp_1_5m_strict_parameter_budget() -> None:
    cfg = load_student_config(CONFIG_PATH)
    auditor = TorchParamAuditor()
    report = auditor.audit_model(str(CONFIG_PATH))
    assert report.total_params <= cfg.target_budget
    assert report.total_params == 1_490_390
    assert report.effective_compute_params < report.total_params


def test_pawp_1_5m_bilingual_forward_and_monolingual_folding_parity() -> None:
    cfg = load_student_config(CONFIG_PATH)
    model = PAWPBilingualTTS(cfg).eval()

    batch, seq_len = 2, 12
    phoneme_ids = torch.randint(4, 90, (batch, seq_len))
    lang_ids = torch.ones((batch, seq_len), dtype=torch.long)
    stress_ids = torch.zeros((batch, seq_len), dtype=torch.long)
    boundary_flags = torch.zeros((batch, seq_len, 4))
    lang_masks = torch.zeros((batch, seq_len, 2))
    lang_masks[0, :, 0] = 1.0  # EN
    lang_masks[1, :, 1] = 1.0  # NE
    # Code-mix transition inside sample 0
    lang_masks[0, 6:, 0] = 0.0
    lang_masks[0, 6:, 1] = 1.0
    durations = torch.full((batch, seq_len), 3.0)

    with torch.no_grad():
        out = model(
            phoneme_ids=phoneme_ids,
            lang_ids=lang_ids,
            stress_ids=stress_ids,
            boundary_flags=boundary_flags,
            lang_masks=lang_masks,
            target_durations=durations,
        )

    assert out.mel.shape == (batch, 36, cfg.n_mels)
    assert out.waveform.ndim == 2
    assert out.waveform.shape[0] == batch
    assert out.waveform.shape[1] > 0
    assert torch.isfinite(out.waveform).all()

    # Verify exact algebraic parity of .fold_language("ne")
    layer = GatedLowRankLinear(128, 256, residual_rank=16).eval()
    with torch.no_grad():
        layer.u_ne.normal_()
        x = torch.randn(1, 5, 128)
        m_ne = torch.tensor([[[0.0, 1.0]]] * 5).transpose(0, 1)
        gated_out = layer(x, m_ne)
        folded_linear = layer.fold_language("ne")
        folded_out = folded_linear(x)
        assert torch.allclose(gated_out, folded_out, atol=1e-5)
