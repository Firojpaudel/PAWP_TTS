"""Compact variance predictors (duration, pitch/voicing, energy) and length regulator."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pawp_tts.infrastructure.neural.convnext import BoundedLookaheadConv1d


class _DWSepPredictorBlock(nn.Module):
    """Lightweight depthwise-separable 1D block for variance prediction."""

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int) -> None:
        super().__init__()
        self.dwconv = BoundedLookaheadConv1d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=kernel_size,
            groups=in_dim,
            lookahead=1,
            bias=True,
        )
        self.pw = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        h = self.pw(h)
        return F.gelu(self.norm(h))


class DurationPredictor(nn.Module):
    """Predicts log-duration per phonological token."""

    def __init__(self, d_model: int = 128, hidden: int = 96, kernel_size: int = 3) -> None:
        super().__init__()
        self.block1 = _DWSepPredictorBlock(d_model, hidden, kernel_size)
        self.block2 = _DWSepPredictorBlock(hidden, hidden, kernel_size)
        self.proj = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return positive predicted frame durations `(B, L)` from `(B, L, D)`."""
        h = self.block2(self.block1(x))
        log_dur = self.proj(h).squeeze(-1)
        return F.softplus(log_dur)


class PitchVoicingPredictor(nn.Module):
    """Predicts frame-level F0 + voicing logit and feeds smoothed conditioning back."""

    def __init__(self, d_model: int = 128, hidden: int = 96, kernel_size: int = 5) -> None:
        super().__init__()
        self.block1 = _DWSepPredictorBlock(d_model, hidden, kernel_size)
        self.block2 = _DWSepPredictorBlock(hidden, hidden, kernel_size)
        self.head = nn.Linear(hidden, 2)
        self.cond_conv = BoundedLookaheadConv1d(2, d_model, kernel_size=3, lookahead=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(pitch_voicing [B, T, 2], conditioning [B, T, D])`."""
        pv = self.head(self.block2(self.block1(x)))
        cond_in = torch.cat([pv[..., :1], torch.sigmoid(pv[..., 1:])], dim=-1)
        return pv, self.condition(cond_in)

    def condition(self, pitch_voicing: torch.Tensor) -> torch.Tensor:
        """Map `[B, T, 2]` (norm log-F0, voicing prob) to `[B, T, D]` conditioning."""
        out: torch.Tensor = self.cond_conv(pitch_voicing.transpose(1, 2)).transpose(1, 2)
        return out


class EnergyPredictor(nn.Module):
    """Predicts frame-level log-energy and feeds smoothed conditioning back."""

    def __init__(self, d_model: int = 128, hidden: int = 64, kernel_size: int = 5) -> None:
        super().__init__()
        self.block1 = _DWSepPredictorBlock(d_model, hidden, kernel_size)
        self.block2 = _DWSepPredictorBlock(hidden, hidden, kernel_size)
        self.head = nn.Linear(hidden, 1)
        self.cond_conv = BoundedLookaheadConv1d(1, d_model, kernel_size=3, lookahead=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(energy [B, T, 1], conditioning [B, T, D])`."""
        energy = self.head(self.block2(self.block1(x)))
        return energy, self.condition(energy)

    def condition(self, energy: torch.Tensor) -> torch.Tensor:
        """Map `[B, T, 1]` log-energy to `[B, T, D]` conditioning."""
        out: torch.Tensor = self.cond_conv(energy.transpose(1, 2)).transpose(1, 2)
        return out


class LengthRegulator(nn.Module):
    """Expands token representations and language masks cleanly to acoustic frames."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.progress_proj = nn.Linear(2, d_model)

    def forward(
        self,
        hidden: torch.Tensor,
        lang_masks: torch.Tensor,
        durations: torch.Tensor,
        max_frames: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Expand `(B, L, D)` and `(B, L, 2)` by integer frame counts `durations (B, L)`.
        Returns `(frame_hidden [B, T, D], frame_lang_masks [B, T, 2])`.
        """
        int_durs = torch.clamp(torch.round(durations).long(), min=0)
        zero_rows = int_durs.sum(dim=1) <= 0
        if torch.any(zero_rows):
            int_durs[zero_rows, 0] = 1
        batch_size, _, dim = hidden.shape
        frame_counts = int_durs.sum(dim=1)
        target_t = int(frame_counts.max().item())
        if max_frames is not None:
            target_t = min(target_t, max_frames)

        # Vectorized over the batch: map every output frame to its source token.
        ends = torch.cumsum(int_durs, dim=1)  # [B, L]
        starts = ends - int_durs
        frames = torch.arange(target_t, device=hidden.device).unsqueeze(0).expand(batch_size, -1)
        tok = torch.searchsorted(ends, frames.contiguous(), right=True).clamp(
            max=hidden.shape[1] - 1
        )
        valid = (frames < frame_counts.unsqueeze(1)).to(hidden.dtype).unsqueeze(-1)
        pos = (frames - torch.gather(starts, 1, tok)).to(hidden.dtype)
        dur = torch.gather(int_durs, 1, tok).to(hidden.dtype).clamp(min=1.0)
        out_progress = torch.stack([pos / dur, (dur - 1.0 - pos) / dur], dim=-1) * valid
        out_hidden = torch.gather(hidden, 1, tok.unsqueeze(-1).expand(-1, -1, dim)) * valid
        out_masks = torch.gather(lang_masks, 1, tok.unsqueeze(-1).expand(-1, -1, 2)) * valid

        out_hidden = out_hidden + self.progress_proj(out_progress)
        return out_hidden, out_masks
