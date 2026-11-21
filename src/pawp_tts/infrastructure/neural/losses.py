"""Training-only spectral, multi-period/multi-resolution adversarial, and acoustic losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT spectral convergence + log-magnitude L1 loss."""

    def __init__(
        self,
        fft_sizes: tuple[int, ...] = (512, 1024, 2048),
        hop_sizes: tuple[int, ...] = (120, 240, 480),
        win_lengths: tuple[int, ...] = (512, 1024, 2048),
    ) -> None:
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        for i, win_len in enumerate(win_lengths):
            self.register_buffer(f"win_{i}", torch.hann_window(win_len), persistent=False)

    def forward(self, pred_wav: torch.Tensor, target_wav: torch.Tensor) -> torch.Tensor:
        """Compute average spectral convergence + log-magnitude L1 across resolutions."""
        min_len = min(pred_wav.shape[-1], target_wav.shape[-1])
        pred = pred_wav[..., :min_len].float()
        target = target_wav[..., :min_len].float()

        total_loss = pred.new_zeros(())
        for i, (n_fft, hop, win_len) in enumerate(
            zip(self.fft_sizes, self.hop_sizes, self.win_lengths, strict=True)
        ):
            win = getattr(self, f"win_{i}")
            p_stft = torch.stft(
                pred,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win_len,
                window=win,
                center=True,
                return_complex=True,
            )
            t_stft = torch.stft(
                target,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win_len,
                window=win,
                center=True,
                return_complex=True,
            )
            p_mag = torch.clamp(p_stft.abs(), min=1e-6)
            t_mag = torch.clamp(t_stft.abs(), min=1e-6)

            sc = torch.norm(t_mag - p_mag, p="fro") / torch.clamp(
                torch.norm(t_mag, p="fro"), min=1e-6
            )
            log_l1 = F.l1_loss(torch.log(p_mag), torch.log(t_mag))
            total_loss = total_loss + sc + log_l1
        return total_loss / float(len(self.fft_sizes))


class _PeriodSubDiscriminator(nn.Module):
    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                nn.utils.parametrizations.weight_norm(
                    nn.Conv2d(1, 16, (5, 1), (3, 1), padding=(2, 0))
                ),
                nn.utils.parametrizations.weight_norm(
                    nn.Conv2d(16, 64, (5, 1), (3, 1), padding=(2, 0))
                ),
                nn.utils.parametrizations.weight_norm(
                    nn.Conv2d(64, 128, (5, 1), (3, 1), padding=(2, 0))
                ),
                nn.utils.parametrizations.weight_norm(
                    nn.Conv2d(128, 256, (5, 1), (3, 1), padding=(2, 0))
                ),
            ]
        )
        self.post = nn.utils.parametrizations.weight_norm(
            nn.Conv2d(256, 1, (3, 1), 1, padding=(1, 0))
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        b, c, t = x.shape
        if t % self.period != 0:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), mode="reflect")
            t += pad
        h = x.view(b, c, t // self.period, self.period)
        fmaps: list[torch.Tensor] = []
        for conv in self.convs:
            h = F.leaky_relu(conv(h), 0.1)
            fmaps.append(h)
        out = self.post(h)
        fmaps.append(out)
        return out.flatten(1, -1), fmaps


class MultiPeriodDiscriminator(nn.Module):
    """Training-only Multi-Period Discriminator for phase & harmonic crispness (0 inference params)."""

    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11)) -> None:
        super().__init__()
        self.sub_discs = nn.ModuleList([_PeriodSubDiscriminator(p) for p in periods])

    def forward(
        self, y_real: torch.Tensor, y_fake: torch.Tensor
    ) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]
    ]:
        r_outs: list[torch.Tensor] = []
        f_outs: list[torch.Tensor] = []
        r_fmaps: list[list[torch.Tensor]] = []
        f_fmaps: list[list[torch.Tensor]] = []
        for d in self.sub_discs:
            ro, rf = d(y_real)
            fo, ff = d(y_fake)
            r_outs.append(ro)
            f_outs.append(fo)
            r_fmaps.append(rf)
            f_fmaps.append(ff)
        return r_outs, f_outs, r_fmaps, f_fmaps


class _ResolutionSubDiscriminator(nn.Module):
    window: torch.Tensor

    def __init__(self, n_fft: int, hop_length: int) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.convs = nn.ModuleList(
            [
                nn.utils.parametrizations.weight_norm(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
                nn.utils.parametrizations.weight_norm(
                    nn.Conv2d(32, 64, (3, 9), stride=(1, 2), padding=(1, 4))
                ),
                nn.utils.parametrizations.weight_norm(
                    nn.Conv2d(64, 128, (3, 9), stride=(1, 2), padding=(1, 4))
                ),
            ]
        )
        self.post = nn.utils.parametrizations.weight_norm(nn.Conv2d(128, 1, (3, 3), padding=(1, 1)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        win = self.window
        spec = torch.stft(
            x.squeeze(1).float(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=win,
            center=True,
            return_complex=True,
        ).abs()
        h = spec.unsqueeze(1)
        fmaps: list[torch.Tensor] = []
        for conv in self.convs:
            h = F.leaky_relu(conv(h), 0.1)
            fmaps.append(h)
        out = self.post(h)
        fmaps.append(out)
        return out.flatten(1, -1), fmaps


class MultiResolutionDiscriminator(nn.Module):
    """Training-only Multi-Resolution Spectral Discriminator for sibilant clarity."""

    def __init__(
        self,
        resolutions: tuple[tuple[int, int], ...] = ((512, 120), (1024, 240), (2048, 480)),
    ) -> None:
        super().__init__()
        self.sub_discs = nn.ModuleList(
            [_ResolutionSubDiscriminator(n_fft, hop) for n_fft, hop in resolutions]
        )

    def forward(
        self, y_real: torch.Tensor, y_fake: torch.Tensor
    ) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]
    ]:
        r_outs: list[torch.Tensor] = []
        f_outs: list[torch.Tensor] = []
        r_fmaps: list[list[torch.Tensor]] = []
        f_fmaps: list[list[torch.Tensor]] = []
        for d in self.sub_discs:
            ro, rf = d(y_real)
            fo, ff = d(y_fake)
            r_outs.append(ro)
            f_outs.append(fo)
            r_fmaps.append(rf)
            f_fmaps.append(ff)
        return r_outs, f_outs, r_fmaps, f_fmaps


def discriminator_lsgan_loss(
    real_outputs: list[torch.Tensor],
    fake_outputs: list[torch.Tensor],
) -> torch.Tensor:
    """LSGAN discriminator objective."""
    loss = real_outputs[0].new_zeros(())
    for r_out, f_out in zip(real_outputs, fake_outputs, strict=True):
        loss = loss + torch.mean((1.0 - r_out) ** 2) + torch.mean(f_out**2)
    return loss


def generator_adversarial_and_fm_loss(
    fake_outputs: list[torch.Tensor],
    real_fmaps: list[list[torch.Tensor]],
    fake_fmaps: list[list[torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """LSGAN generator adversarial loss + L1 feature-matching loss."""
    adv_loss = fake_outputs[0].new_zeros(())
    fm_loss = fake_outputs[0].new_zeros(())
    for f_out in fake_outputs:
        adv_loss = adv_loss + torch.mean((1.0 - f_out) ** 2)
    for r_list, f_list in zip(real_fmaps, fake_fmaps, strict=True):
        for rf, ff in zip(r_list, f_list, strict=True):
            fm_loss = fm_loss + F.l1_loss(ff, rf.detach())
    return adv_loss, fm_loss


class PAWPTrainingLoss(nn.Module):
    """Combined acoustic + variance + multi-resolution STFT generator loss."""

    def __init__(
        self,
        w_mel: float = 15.0,
        w_stft: float = 2.5,
        w_dur: float = 1.0,
        w_pitch: float = 1.0,
        w_energy: float = 1.0,
    ) -> None:
        super().__init__()
        self.w_mel = w_mel
        self.w_stft = w_stft
        self.w_dur = w_dur
        self.w_pitch = w_pitch
        self.w_energy = w_energy
        self.mr_stft = MultiResolutionSTFTLoss()

    def forward(
        self,
        pred_wav: torch.Tensor,
        target_wav: torch.Tensor,
        pred_mel: torch.Tensor,
        target_mel: torch.Tensor,
        pred_dur: torch.Tensor,
        target_dur: torch.Tensor,
        pred_pv: torch.Tensor,
        target_pv: torch.Tensor,
        pred_energy: torch.Tensor,
        target_energy: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute weighted multi-term student loss and scalar metric breakdown."""
        mel_err = torch.abs(pred_mel - target_mel).mean(dim=(1, 2))
        dur_err = torch.square(torch.log1p(pred_dur) - torch.log1p(target_dur)).mean(dim=1)
        pv_err = torch.abs(pred_pv - target_pv).mean(dim=(1, 2))
        e_err = torch.abs(pred_energy - target_energy).mean(dim=(1, 2))

        if sample_weights is not None:
            w = sample_weights / torch.clamp(sample_weights.mean(), min=1e-6)
            l_mel = (mel_err * w).mean()
            l_dur = (dur_err * w).mean()
            l_pv = (pv_err * w).mean()
            l_e = (e_err * w).mean()
        else:
            l_mel = mel_err.mean()
            l_dur = dur_err.mean()
            l_pv = pv_err.mean()
            l_e = e_err.mean()

        l_stft = self.mr_stft(pred_wav, target_wav)
        total = (
            self.w_mel * l_mel
            + self.w_stft * l_stft
            + self.w_dur * l_dur
            + self.w_pitch * l_pv
            + self.w_energy * l_e
        )
        metrics = {
            "loss_total": float(total.detach()),
            "loss_mel": float(l_mel.detach()),
            "loss_stft": float(l_stft.detach()),
            "loss_dur": float(l_dur.detach()),
            "loss_pitch": float(l_pv.detach()),
            "loss_energy": float(l_e.detach()),
        }
        return total, metrics
