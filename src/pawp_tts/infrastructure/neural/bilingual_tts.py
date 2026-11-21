"""PAWP-1.5M bilingual non-autoregressive streaming TTS model."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch
import torch.nn as nn

from pawp_tts.infrastructure.neural.articulatory import ArticulatoryConditioner
from pawp_tts.infrastructure.neural.convnext import ConvNeXt1DBlock, GatedLowRankLinear
from pawp_tts.infrastructure.neural.istft_decoder import StreamingConvNeXtISTFTDecoder
from pawp_tts.infrastructure.neural.variance_adaptor import (
    DurationPredictor,
    EnergyPredictor,
    LengthRegulator,
    PitchVoicingPredictor,
)
from pawp_tts.infrastructure.weight_programming.basis_bank import BasisBankLinear, SharedBasisPool


@dataclass(frozen=True)
class StudentConfig:
    """Configuration defining the complete bilingual inference architecture."""

    model_name: str = "pawp-1.5m-baseline"
    target_budget: int = 1_500_000
    sample_rate: int = 24000
    hop_length: int = 240
    n_fft: int = 1024
    n_mels: int = 80
    vocab_size: int = 128
    num_languages: int = 4
    num_stress_levels: int = 4
    num_boundary_flags: int = 4
    num_speakers: int = 1
    d_model: int = 128
    enc_layers: int = 4
    enc_ff: int = 256
    enc_kernel: int = 5
    enc_lookahead: int = 1
    duration_hidden: int = 96
    duration_kernel: int = 3
    pitch_hidden: int = 96
    pitch_kernel: int = 5
    energy_hidden: int = 64
    energy_kernel: int = 5
    acoustic_layers: int = 4
    acoustic_ff: int = 256
    acoustic_kernel: int = 7
    acoustic_lookahead: int = 1
    residual_rank: int = 16
    decoder_dim: int = 192
    decoder_ff: int = 384
    decoder_layers: int = 3
    decoder_kernel: int = 7
    decoder_lookahead: int = 1
    num_shared_bases: int = 0
    layer_residual_rank: int = 0
    use_articulatory_conditioning: bool = False


@dataclass(frozen=True)
class SynthesisOutput:
    """Intermediate and waveform outputs returned by PAWPBilingualTTS."""

    waveform: torch.Tensor
    mel: torch.Tensor
    durations: torch.Tensor
    pitch_voicing: torch.Tensor
    energy: torch.Tensor


class PAWPBilingualTTS(nn.Module):
    """Complete compact bilingual English-Nepali neural TTS inference graph."""

    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Module 1: Embeddings & Conditioning
        self.phoneme_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.lang_emb = nn.Embedding(cfg.num_languages, cfg.d_model)
        self.stress_emb = nn.Embedding(cfg.num_stress_levels, cfg.d_model)
        self.boundary_proj = nn.Linear(cfg.num_boundary_flags, cfg.d_model)
        self.speaker_emb = nn.Embedding(cfg.num_speakers, cfg.d_model)
        self.articulatory_cond: ArticulatoryConditioner | None = (
            ArticulatoryConditioner(cfg.vocab_size, cfg.d_model)
            if cfg.use_articulatory_conditioning
            else None
        )
        self.input_norm = nn.LayerNorm(cfg.d_model)

        self.shared_pw1_pool: SharedBasisPool | None = None
        self.shared_pw2_pool: SharedBasisPool | None = None
        if cfg.num_shared_bases > 0:
            if cfg.enc_ff != cfg.acoustic_ff:
                raise ValueError(
                    "Shared basis pooling requires enc_ff == acoustic_ff across encoder/acoustic blocks."
                )
            self.shared_pw1_pool = SharedBasisPool(cfg.num_shared_bases, cfg.enc_ff, cfg.d_model)
            self.shared_pw2_pool = SharedBasisPool(cfg.num_shared_bases, cfg.d_model, cfg.enc_ff)

        # Module 2 (+ part of Module 5): Shared Phoneme Encoder with gated EN/NE residuals
        self.encoder_blocks = nn.ModuleList(
            [
                ConvNeXt1DBlock(
                    dim=cfg.d_model,
                    ff_dim=cfg.enc_ff,
                    kernel_size=cfg.enc_kernel,
                    lookahead=cfg.enc_lookahead,
                    residual_rank=cfg.residual_rank,
                    pw1_pool=self.shared_pw1_pool,
                    pw2_pool=self.shared_pw2_pool,
                    layer_residual_rank=cfg.layer_residual_rank,
                )
                for _ in range(cfg.enc_layers)
            ]
        )

        # Module 3: Variance Adaptor
        self.duration_predictor = DurationPredictor(
            d_model=cfg.d_model,
            hidden=cfg.duration_hidden,
            kernel_size=cfg.duration_kernel,
        )
        self.length_regulator = LengthRegulator(d_model=cfg.d_model)
        self.pitch_predictor = PitchVoicingPredictor(
            d_model=cfg.d_model,
            hidden=cfg.pitch_hidden,
            kernel_size=cfg.pitch_kernel,
        )
        self.energy_predictor = EnergyPredictor(
            d_model=cfg.d_model,
            hidden=cfg.energy_hidden,
            kernel_size=cfg.energy_kernel,
        )

        # Module 4 (+ part of Module 5): Shared Frame-Acoustic Net + Mel Head
        self.acoustic_blocks = nn.ModuleList(
            [
                ConvNeXt1DBlock(
                    dim=cfg.d_model,
                    ff_dim=cfg.acoustic_ff,
                    kernel_size=cfg.acoustic_kernel,
                    lookahead=cfg.acoustic_lookahead,
                    residual_rank=cfg.residual_rank,
                    pw1_pool=self.shared_pw1_pool,
                    pw2_pool=self.shared_pw2_pool,
                    layer_residual_rank=cfg.layer_residual_rank,
                )
                for _ in range(cfg.acoustic_layers)
            ]
        )
        self.acoustic_norm = nn.LayerNorm(cfg.d_model)
        self.mel_head = nn.Linear(cfg.d_model, cfg.n_mels)

        # Module 6: Streaming Waveform Decoder (24 kHz ConvNeXt1D + iSTFT)
        self.decoder = StreamingConvNeXtISTFTDecoder(
            n_mels=cfg.n_mels,
            dim=cfg.decoder_dim,
            ff_dim=cfg.decoder_ff,
            num_layers=cfg.decoder_layers,
            kernel_size=cfg.decoder_kernel,
            lookahead=cfg.decoder_lookahead,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        )

    def encode_tokens(
        self,
        phoneme_ids: torch.Tensor,
        lang_ids: torch.Tensor,
        stress_ids: torch.Tensor,
        boundary_flags: torch.Tensor,
        lang_masks: torch.Tensor | None = None,
        speaker_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run phoneme embeddings, shared+gated encoder, and duration predictor."""
        batch_size, seq_len = phoneme_ids.shape
        if speaker_ids is None:
            speaker_ids = torch.zeros(batch_size, dtype=torch.long, device=phoneme_ids.device)

        spk = self.speaker_emb(speaker_ids).unsqueeze(1).expand(batch_size, seq_len, -1)
        x = (
            self.phoneme_emb(phoneme_ids)
            + self.lang_emb(lang_ids)
            + self.stress_emb(stress_ids)
            + self.boundary_proj(boundary_flags)
            + spk
        )
        if self.articulatory_cond is not None:
            x = x + self.articulatory_cond(phoneme_ids)
        x = self.input_norm(x)
        for block in self.encoder_blocks:
            x = block(x, lang_masks)
        pred_durations = self.duration_predictor(x)
        return x, pred_durations

    def prepare_frame_hidden(
        self,
        enc_hidden: torch.Tensor,
        lang_masks: torch.Tensor,
        durations: torch.Tensor,
        max_frames: int | None = None,
        target_pitch_voicing: torch.Tensor | None = None,
        target_energy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand token states to frames and apply pitch/energy conditioning."""
        frame_h, frame_masks = self.length_regulator(
            enc_hidden,
            lang_masks,
            durations,
            max_frames=max_frames,
        )
        pitch_voicing, pitch_cond = self.pitch_predictor(frame_h)
        if target_pitch_voicing is not None:  # teacher forcing during training
            pitch_cond = self.pitch_predictor.condition(target_pitch_voicing[:, : frame_h.shape[1]])
        frame_h = frame_h + pitch_cond
        energy, energy_cond = self.energy_predictor(frame_h)
        if target_energy is not None:
            energy_cond = self.energy_predictor.condition(target_energy[:, : frame_h.shape[1]])
        frame_h = frame_h + energy_cond
        return frame_h, frame_masks, pitch_voicing, energy

    def decode_frame_spectrum(
        self,
        frame_h: torch.Tensor,
        frame_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run frame-acoustic network and iSTFT spectral head on a frame window."""
        h = frame_h
        for block in self.acoustic_blocks:
            h = block(h, frame_masks)
        mel = self.mel_head(self.acoustic_norm(h))
        real_spec, imag_spec = self.decoder.forward_complex_spectrum(mel)
        return mel, real_spec, imag_spec

    def fold_monolingual(self, language: Literal["en", "ne"]) -> PAWPBilingualTTS:
        """
        Return a dense inference clone with shared bases & language residuals folded into
        dense `nn.Linear` weights for zero-overhead monolingual CPU/ONNX deployment.
        """
        folded_cfg = replace(
            self.cfg,
            residual_rank=0,
            num_shared_bases=0,
            layer_residual_rank=0,
        )
        folded = PAWPBilingualTTS(folded_cfg)
        folded.to(next(self.parameters()).device)

        with torch.no_grad():
            src_state = self.state_dict()
            dst_state = folded.state_dict()
            for key, tensor in dst_state.items():
                if key in src_state and src_state[key].shape == tensor.shape:
                    tensor.copy_(src_state[key])

            for src_blk, dst_blk in zip(self.encoder_blocks, folded.encoder_blocks, strict=True):
                assert isinstance(src_blk, ConvNeXt1DBlock) and isinstance(dst_blk, ConvNeXt1DBlock)
                if isinstance(src_blk.pw1, (GatedLowRankLinear, BasisBankLinear)):
                    dst_blk.pw1.load_state_dict(src_blk.pw1.fold_language(language).state_dict())
                if isinstance(src_blk.pw2, (GatedLowRankLinear, BasisBankLinear)):
                    dst_blk.pw2.load_state_dict(src_blk.pw2.fold_language(language).state_dict())

            for src_blk, dst_blk in zip(self.acoustic_blocks, folded.acoustic_blocks, strict=True):
                assert isinstance(src_blk, ConvNeXt1DBlock) and isinstance(dst_blk, ConvNeXt1DBlock)
                if isinstance(src_blk.pw1, (GatedLowRankLinear, BasisBankLinear)):
                    dst_blk.pw1.load_state_dict(src_blk.pw1.fold_language(language).state_dict())
                if isinstance(src_blk.pw2, (GatedLowRankLinear, BasisBankLinear)):
                    dst_blk.pw2.load_state_dict(src_blk.pw2.fold_language(language).state_dict())

        folded.eval()
        return folded

    def forward(
        self,
        phoneme_ids: torch.Tensor,
        lang_ids: torch.Tensor,
        stress_ids: torch.Tensor,
        boundary_flags: torch.Tensor,
        lang_masks: torch.Tensor,
        speaker_ids: torch.Tensor | None = None,
        target_durations: torch.Tensor | None = None,
        max_frames: int | None = None,
        target_pitch_voicing: torch.Tensor | None = None,
        target_energy: torch.Tensor | None = None,
    ) -> SynthesisOutput:
        """Run full end-to-end text-token-to-waveform inference."""
        x, pred_durations = self.encode_tokens(
            phoneme_ids=phoneme_ids,
            lang_ids=lang_ids,
            stress_ids=stress_ids,
            boundary_flags=boundary_flags,
            lang_masks=lang_masks,
            speaker_ids=speaker_ids,
        )
        used_durations = target_durations if target_durations is not None else pred_durations
        frame_h, frame_masks, pitch_voicing, energy = self.prepare_frame_hidden(
            x,
            lang_masks,
            used_durations,
            max_frames=max_frames,
            target_pitch_voicing=target_pitch_voicing,
            target_energy=target_energy,
        )
        for block in self.acoustic_blocks:
            frame_h = block(frame_h, frame_masks)
        mel = self.mel_head(self.acoustic_norm(frame_h))
        waveform = self.decoder(mel)

        return SynthesisOutput(
            waveform=waveform,
            mel=mel,
            durations=pred_durations,
            pitch_voicing=pitch_voicing,
            energy=energy,
        )
