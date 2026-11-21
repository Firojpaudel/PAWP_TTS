"""Four-metric parameter, compute, and storage footprint auditor for PyTorch graphs."""

from __future__ import annotations

import torch.nn as nn

from pawp_tts.domain.entities.models import ParamCountReport
from pawp_tts.domain.pawp.accounting import serialized_sizes
from pawp_tts.infrastructure.neural.bilingual_tts import PAWPBilingualTTS
from pawp_tts.infrastructure.neural.convnext import GatedLowRankLinear
from pawp_tts.infrastructure.persistence.config_loader import load_student_config
from pawp_tts.infrastructure.weight_programming.basis_bank import BasisBankLinear


def count_module_params(module: nn.Module) -> int:
    """Count all unique learned parameters inside a module."""
    unique = {id(p): p for p in module.parameters()}
    return sum(p.numel() for p in unique.values())


def _lang_residual_param_count(module: nn.Module) -> int:
    total = 0
    for mod in module.modules():
        if (isinstance(mod, GatedLowRankLinear) and mod.residual_rank > 0) or (
            isinstance(mod, BasisBankLinear) and mod.lang_residual_rank > 0
        ):
            total += (
                mod.u_en.numel()
                + mod.v_en.numel()
                + mod.gate_en.numel()
                + mod.u_ne.numel()
                + mod.v_ne.numel()
                + mod.gate_ne.numel()
            )
    return total


class TorchParamAuditor:
    """Audits every learned tensor inside PAWPBilingualTTS (distinguishing shared vs unshared)."""

    def audit_model(self, config_path: str) -> ParamCountReport:
        """Instantiate PAWPBilingualTTS from YAML config and return 4-metric report."""
        cfg = load_student_config(config_path)
        model = PAWPBilingualTTS(cfg)
        return self.audit_nn_module(model, frames_per_sec=cfg.sample_rate // cfg.hop_length)

    def audit_nn_module(
        self,
        model: PAWPBilingualTTS,
        frames_per_sec: int = 100,
    ) -> ParamCountReport:
        """Compute exact module breakdown and 4-metric parameter audit."""
        m1_emb = (
            count_module_params(model.phoneme_emb)
            + count_module_params(model.lang_emb)
            + count_module_params(model.stress_emb)
            + count_module_params(model.boundary_proj)
            + count_module_params(model.speaker_emb)
            + count_module_params(model.input_norm)
            + (
                count_module_params(model.articulatory_cond)
                if model.articulatory_cond is not None
                else 0
            )
        )

        pool_params = (
            count_module_params(model.shared_pw1_pool) if model.shared_pw1_pool is not None else 0
        ) + (count_module_params(model.shared_pw2_pool) if model.shared_pw2_pool is not None else 0)

        enc_total = count_module_params(model.encoder_blocks)
        ac_total = (
            count_module_params(model.acoustic_blocks)
            + count_module_params(model.acoustic_norm)
            + count_module_params(model.mel_head)
        )
        res_params = _lang_residual_param_count(model)
        enc_res = _lang_residual_param_count(model.encoder_blocks)
        ac_res = res_params - enc_res

        m2_enc_shared = enc_total - enc_res
        m4_ac_shared = ac_total - ac_res
        m3_var = (
            count_module_params(model.duration_predictor)
            + count_module_params(model.length_regulator)
            + count_module_params(model.pitch_predictor)
            + count_module_params(model.energy_predictor)
        )
        m6_dec = count_module_params(model.decoder)

        unique_stored = count_module_params(model)

        # Compute unshared dense computation-graph total_params
        total_params = unique_stored - pool_params
        for mod in model.modules():
            if isinstance(mod, BasisBankLinear):
                stored_shared = mod.basis_coeffs.numel()
                if mod.layer_residual_rank > 0:
                    stored_shared += mod.u_layer.numel() + mod.v_layer.numel()
                unshared_dense = mod.in_features * mod.out_features
                total_params += unshared_dense - stored_shared

        effective_compute = count_module_params(model.fold_monolingual("en"))

        enc_macs = m2_enc_shared * 20
        frame_macs = (m4_ac_shared + m6_dec + m3_var) * frames_per_sec
        macs_per_sec = enc_macs + frame_macs

        fp32_b, fp16_b, int8_b, int4_b = serialized_sizes(unique_stored)

        breakdown: dict[str, int] = {
            "1_embeddings_conditioning": m1_emb,
            "2_shared_phoneme_encoder": m2_enc_shared,
            "3_variance_adaptor": m3_var,
            "4_shared_frame_acoustic": m4_ac_shared,
            "5_bilingual_gated_residuals": res_params,
            "6_streaming_istft_decoder": m6_dec,
        }
        if pool_params > 0:
            breakdown["7_shared_basis_pools"] = pool_params

        return ParamCountReport(
            total_params=total_params,
            unique_stored_params=unique_stored,
            effective_compute_params=effective_compute,
            macs_per_sec_audio=macs_per_sec,
            fp32_bytes=fp32_b,
            fp16_bytes=fp16_b,
            int8_bytes=int8_b,
            int4_bytes=int4_b,
            breakdown_by_module=breakdown,
        )
