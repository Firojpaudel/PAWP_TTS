"""Vectorized acoustic feature extraction and length-bucketed batch sampling."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly
from torch.utils.data import Sampler


def resample_to_24k(audio: np.ndarray, orig_sr: int, target_sr: int = 24000) -> np.ndarray:
    """Fast polyphase anti-aliased resampling to target 24 kHz sample rate."""
    if orig_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return np.asarray(resample_poly(audio, up, down), dtype=np.float32)


def _hz_to_mel_slaney(freqs: torch.Tensor) -> torch.Tensor:
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    mels = freqs / f_sp
    return torch.where(
        freqs >= min_log_hz,
        min_log_mel + torch.log(torch.clamp(freqs, min=min_log_hz) / min_log_hz) / logstep,
        mels,
    )


def _mel_to_hz_slaney(mels: torch.Tensor) -> torch.Tensor:
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    freqs = f_sp * mels
    return torch.where(
        mels >= min_log_mel,
        min_log_hz * torch.exp(logstep * (mels - min_log_mel)),
        freqs,
    )


class MelSpectrogramExtractor(nn.Module):
    """GPU/CPU vectorized 24 kHz 80-bin log-mel, log-energy, and F0 feature extractor."""

    mel_fb: torch.Tensor
    hann_window: torch.Tensor

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 240,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 12000.0,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        n_freqs = n_fft // 2 + 1
        all_freqs = torch.linspace(0.0, sample_rate / 2.0, n_freqs)
        m_min = _hz_to_mel_slaney(torch.tensor(f_min))
        m_max = _hz_to_mel_slaney(torch.tensor(f_max))
        m_pts = torch.linspace(float(m_min), float(m_max), n_mels + 2)
        f_pts = _mel_to_hz_slaney(m_pts)

        f_diff = f_pts[1:] - f_pts[:-1]
        slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
        down_slopes = -slopes[:, :-2] / torch.clamp(f_diff[:-1], min=1e-8)
        up_slopes = slopes[:, 2:] / torch.clamp(f_diff[1:], min=1e-8)
        fb = torch.clamp(torch.minimum(down_slopes, up_slopes), min=0.0)
        # Slaney area normalization
        enorm = 2.0 / torch.clamp(f_pts[2 : n_mels + 2] - f_pts[:n_mels], min=1e-8)
        fb *= enorm.unsqueeze(0)

        self.register_buffer("mel_fb", fb, persistent=False)
        self.register_buffer("hann_window", torch.hann_window(n_fft), persistent=False)

    @torch.inference_mode()
    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(log_mel [B, T, n_mels], log_energy [B, T, 1])` from `(B, S)` PCM."""
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.hann_window,
            center=True,
            return_complex=True,
        )
        mag = spec.abs()  # (B, n_freqs, T)
        mel = torch.matmul(mag.transpose(1, 2), self.mel_fb)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        log_energy = torch.log(
            torch.clamp(mag.pow(2).sum(dim=1, keepdim=True), min=1e-6)
        ).transpose(1, 2)
        return log_mel, log_energy


def estimate_f0_autocorr(
    waveform: np.ndarray,
    sample_rate: int = 24000,
    hop_length: int = 240,
    fmin: float = 60.0,
    fmax: float = 550.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast framewise autocorrelation pitch (Hz) + binary voicing mask."""
    win_len = 4 * hop_length
    padded = np.pad(waveform.astype(np.float32), (win_len // 2, win_len // 2))
    n_frames = 1 + len(waveform) // hop_length
    min_lag = max(1, int(sample_rate / fmax))
    max_lag = min(win_len // 2, int(sample_rate / fmin))

    f0 = np.zeros(n_frames, dtype=np.float32)
    voiced = np.zeros(n_frames, dtype=np.float32)

    for idx in range(n_frames):
        start = idx * hop_length
        frame = padded[start : start + win_len]
        if len(frame) < win_len:
            break
        energy = float(np.dot(frame, frame))
        if energy < 1e-6:
            continue
        corr = np.correlate(frame, frame, mode="full")[win_len - 1 :]
        sub = corr[min_lag:max_lag]
        if len(sub) == 0:
            continue
        best_rel = int(np.argmax(sub))
        peak_ratio = float(sub[best_rel] / max(energy, 1e-8))
        if peak_ratio > 0.35:
            f0[idx] = sample_rate / float(min_lag + best_rel)
            voiced[idx] = 1.0
    return f0, voiced


class DynamicFrameBucketSampler(Sampler[list[int]]):
    """Groups length-similar utterances into frame-budgeted batches for fast RTX 4090 training."""

    def __init__(
        self,
        frame_lengths: Sequence[int],
        max_frames_per_batch: int = 8000,
        max_batch_size: int = 64,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self._lengths = list(frame_lengths)
        self._max_frames = max_frames_per_batch
        self._max_bs = max_batch_size
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0
        self._batches = self._build_batches()

    def set_epoch(self, epoch: int) -> None:
        """Update RNG epoch for deterministic reshuffling across epochs."""
        self._epoch = epoch
        self._batches = self._build_batches()

    def _build_batches(self) -> list[list[int]]:
        indices = sorted(range(len(self._lengths)), key=lambda i: self._lengths[i])
        batches: list[list[int]] = []
        cur_batch: list[int] = []
        cur_max_len = 0

        for idx in indices:
            length = max(1, self._lengths[idx])
            new_max = max(cur_max_len, length)
            if cur_batch and (
                (new_max * (len(cur_batch) + 1) > self._max_frames)
                or len(cur_batch) >= self._max_bs
            ):
                batches.append(cur_batch)
                cur_batch = [idx]
                cur_max_len = length
            else:
                cur_batch.append(idx)
                cur_max_len = new_max

        if cur_batch:
            batches.append(cur_batch)

        if self._shuffle and len(batches) > 1:
            rng = np.random.default_rng(self._seed + self._epoch)
            order = rng.permutation(len(batches))
            batches = [batches[int(i)] for i in order]
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


def pad_to_frame_batch(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Right-pad variable-length `(T, ...)` tensors cleanly into a batch."""
    max_t = max(t.shape[0] for t in tensors)
    out = tensors[0].new_zeros((len(tensors), max_t, *tensors[0].shape[1:]))
    for i, t in enumerate(tensors):
        if t.shape[0] > 0:
            out[i, : t.shape[0]] = F.pad(t, (0,) * (2 * (t.ndim - 1)))
    return out
