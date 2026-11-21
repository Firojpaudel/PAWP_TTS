"""Integration tests for SharedBasisPool, BasisBankLinear, PAWP-1M config, trainer, and eval suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pawp_tts.application.frontend_service import PhonemizeUtteranceUseCase
from pawp_tts.infrastructure.evaluation.benchmark_suite import run_evaluation_suite
from pawp_tts.infrastructure.frontend.english_adapter import EnglishEspeakG2PAdapter
from pawp_tts.infrastructure.frontend.nepali_adapter import NepaliRuleG2PAdapter
from pawp_tts.infrastructure.neural.articulatory import ArticulatoryConditioner
from pawp_tts.infrastructure.neural.bilingual_tts import PAWPBilingualTTS
from pawp_tts.infrastructure.persistence.config_loader import load_student_config
from pawp_tts.infrastructure.training.trainer import BilingualStudentTrainer
from pawp_tts.infrastructure.weight_programming.basis_bank import (
    BasisBankLinear,
    SharedBasisPool,
    decompose_weights_to_shared_basis,
)
from pawp_tts.infrastructure.weight_programming.param_auditor import (
    TorchParamAuditor,
    count_module_params,
)


def test_basis_bank_linear_folding_and_svd_decomposition() -> None:
    torch.manual_seed(11)
    pool = SharedBasisPool(num_bases=2, out_features=64, in_features=32)
    layer = BasisBankLinear(
        in_features=32,
        out_features=64,
        pool=pool,
        layer_residual_rank=8,
        lang_residual_rank=4,
    ).eval()
    with torch.no_grad():
        layer.u_layer.normal_(std=0.05)
        layer.u_en.normal_(std=0.05)
        layer.u_ne.normal_(std=0.05)

    x = torch.randn(2, 6, 32)
    mask_en = torch.tensor([1.0, 0.0]).view(1, 1, 2).expand(2, 6, 2)
    mask_ne = torch.tensor([0.0, 1.0]).view(1, 1, 2).expand(2, 6, 2)

    with torch.no_grad():
        out_en = layer(x, mask_en)
        folded_en = layer.fold_language("en")(x)
        assert torch.allclose(out_en, folded_en, atol=1e-5)

        out_ne = layer(x, mask_ne)
        folded_ne = layer.fold_language("ne")(x)
        assert torch.allclose(out_ne, folded_ne, atol=1e-5)

    # Check SVD shared-basis decomposition monotonic error reduction
    mats = [torch.randn(64, 32) for _ in range(4)]
    b1, c1, r_low = decompose_weights_to_shared_basis(mats, num_bases=2, residual_rank=2)
    b2, c2, r_high = decompose_weights_to_shared_basis(mats, num_bases=2, residual_rank=12)

    err_low = sum(
        float(
            torch.norm(mats[i] - (torch.einsum("k,koi->oi", c1[i], b1) + r_low[i][0] @ r_low[i][1]))
        )
        for i in range(4)
    )
    err_high = sum(
        float(
            torch.norm(
                mats[i] - (torch.einsum("k,koi->oi", c2[i], b2) + r_high[i][0] @ r_high[i][1])
            )
        )
        for i in range(4)
    )
    assert err_high < err_low


def test_pawp_1m_shared_basis_and_articulatory_parameter_audit() -> None:
    cond = ArticulatoryConditioner(vocab_size=128, d_model=128)
    assert count_module_params(cond) == 3_712

    auditor = TorchParamAuditor()
    report = auditor.audit_model("configs/student_1m_pawp.yaml")
    assert report.unique_stored_params <= 1_000_000
    assert report.unique_stored_params < report.total_params
    assert "7_shared_basis_pools" in report.breakdown_by_module

    cfg = load_student_config("configs/student_1m_pawp.yaml")
    model = PAWPBilingualTTS(cfg).eval()
    folded = model.fold_monolingual("en")

    p_ids = torch.randint(4, 70, (1, 8))
    l_ids = torch.ones((1, 8), dtype=torch.long)
    s_ids = torch.zeros((1, 8), dtype=torch.long)
    b_flags = torch.zeros((1, 8, 4))
    l_masks = torch.tensor([1.0, 0.0]).view(1, 1, 2).expand(1, 8, 2)
    durs = torch.full((1, 8), 3.0)

    with torch.no_grad():
        out_bi = model(p_ids, l_ids, s_ids, b_flags, l_masks, target_durations=durs)
        out_fold = folded(p_ids, l_ids, s_ids, b_flags, l_masks, target_durations=durs)
    assert torch.allclose(out_bi.mel, out_fold.mel, atol=1e-5)
    assert torch.allclose(out_bi.waveform, out_fold.waveform, atol=1e-5)


def test_trainer_checkpoint_hygiene_and_eval_suites(tmp_path: Path) -> None:
    import json

    root = tmp_path / "cache_v2"
    ckpt_dir = tmp_path / "ckpts"
    rng = np.random.default_rng(7)
    (root).mkdir()
    (root / "speakers.json").write_text(json.dumps({"spk_a": 0, "spk_target": 1}))
    for split in ("train", "val"):
        (root / split).mkdir()
        index = []
        for i in range(4):
            n_tok, n_frames = 8, 24 + 4 * i
            np.savez(
                root / split / f"{i:06d}.npz",
                phoneme_ids=rng.integers(5, 60, size=(n_tok,)).astype(np.int16),
                lang_ids=np.full((n_tok,), 2, dtype=np.int8),
                stress_ids=np.zeros((n_tok,), dtype=np.int8),
                boundary_flags=np.zeros((n_tok, 4), dtype=np.int8),
                lang_masks=np.tile(np.array([[0.0, 1.0]], dtype=np.float16), (n_tok, 1)),
                log_mel=(rng.standard_normal((n_frames, 80)) * 0.2).astype(np.float16),
                log_energy=(rng.standard_normal((n_frames, 1)) * 0.2).astype(np.float16),
                pitch_voicing=np.zeros((n_frames, 2), dtype=np.float16),
                waveform=(rng.standard_normal(n_frames * 240) * 1500).astype(np.int16),
                speaker=np.array([i % 2], dtype=np.int32),
                loss_weight=np.array([1.0], dtype=np.float32),
            )
            index.append(
                {"file": f"{i:06d}.npz", "frames": n_frames, "tokens": n_tok, "lang": "ne"}
            )
        (root / f"{split}_index.json").write_text(json.dumps(index))

    trainer = BilingualStudentTrainer(
        config_path="configs/student_1m_pawp.yaml",
        checkpoint_dir=ckpt_dir,
        cache_root=root,
        target_speaker="spk_target",
        device="cpu",
    )
    stats = trainer.train(
        steps=4,
        max_batch_frames=200,
        gan_start=2,
        bin_start=1,
        eval_every=2,
        log_every=2,
        workers=0,
    )
    assert float(stats["best_val_mel"]) > 0.0
    saved_ckpts = sorted(p.name for p in ckpt_dir.glob("*.safetensors"))
    assert saved_ckpts == ["best.safetensors", "latest.safetensors"]
    # resume continues from saved step
    stats2 = trainer.train(
        steps=6, max_batch_frames=200, gan_start=2, eval_every=2, log_every=2, workers=0
    )
    assert stats2["steps"] == 6

    phonemize = PhonemizeUtteranceUseCase(
        english_g2p=EnglishEspeakG2PAdapter(),
        nepali_g2p=NepaliRuleG2PAdapter(),
    )
    neptts_res = run_evaluation_suite(
        suite="neptts",
        config_path="configs/student_1m_pawp.yaml",
        checkpoint_path=ckpt_dir / "latest.safetensors",
        phonemize=phonemize,
    )
    assert neptts_res["total_pairs"] == 77
    assert neptts_res["frontend_g2p_contrast_rate"] == 1.0
    assert neptts_res["frontend_no_sh_symbol_rate"] == 1.0
