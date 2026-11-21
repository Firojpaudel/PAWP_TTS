"""Compact streaming 24 kHz ConvNeXt1D + iSTFT neural waveform decoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from pawp_tts.infrastructure.neural.convnext import ConvNeXt1DBlock


class StreamingConvNeXtISTFTDecoder(nn.Module):
    """Vocos-style ConvNeXt1D magnitude+phase predictor followed by iSTFT."""

    hann_window: torch.Tensor
    spec_mask: torch.Tensor

    def __init__(
        self,
        n_mels: int = 80,
        dim: int = 192,
        ff_dim: int = 384,
        num_layers: int = 3,
        kernel_size: int = 7,
        lookahead: int = 1,
        n_fft: int = 1024,
        hop_length: int = 240,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bins = n_fft // 2 + 1

        self.stem_conv = nn.Conv1d(n_mels, dim, kernel_size=1)
        self.stem_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(
            [
                ConvNeXt1DBlock(
                    dim=dim,
                    ff_dim=ff_dim,
                    kernel_size=kernel_size,
                    lookahead=lookahead,
                    residual_rank=0,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, self.n_bins * 2)
        self.register_buffer("hann_window", torch.hann_window(n_fft), persistent=False)
        spec_mask = torch.ones(1, 1, self.n_bins)
        spec_mask[..., 0] = 0.0
        spec_mask[..., -1] = 0.0
        self.register_buffer("spec_mask", spec_mask, persistent=False)

    def forward_complex_spectrum(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict real and imaginary STFT bins `(B, n_bins, T_frames)` from mel frames."""
        h = self.stem_conv(mel.transpose(1, 2)).transpose(1, 2)
        h = self.stem_norm(h)
        for block in self.blocks:
            h = block(h, None)
        h = self.final_norm(h)

        spec_params = self.head(h)
        log_mag, phase = torch.split(spec_params, self.n_bins, dim=-1)
        # Zero DC and Nyquist bins cleanly via non-mutating mask
        mag = torch.exp(torch.clamp(log_mag, min=-12.0, max=8.0)) * self.spec_mask

        mag_t = mag.transpose(1, 2)
        phase_t = phase.transpose(1, 2)
        return mag_t * torch.cos(phase_t), mag_t * torch.sin(phase_t)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Synthesize 24 kHz time-domain waveform `(B, T_samples)` from `(B, T_frames, n_mels)`.
        """
        real, imag = self.forward_complex_spectrum(mel)
        complex_spec = torch.complex(real, imag)
        return torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.hann_window,
            center=True,
        )
