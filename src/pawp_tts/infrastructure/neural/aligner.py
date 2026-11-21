"""Training-only unsupervised text-to-mel aligner (RAD-TTS / "One TTS Alignment" style).

Not part of the inference graph and excluded from all parameter budgets. Produces hard
monotonic durations via Monotonic Alignment Search (MAS) over a soft attention trained with
a CTC forward-sum objective and a beta-binomial alignment prior (Badlani et al., ICASSP 2022).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class AlignmentEncoder(nn.Module):
    """Computes `log_attn [B, T_mel, N_text]` from phoneme ids and log-mel frames."""

    def __init__(self, vocab_size: int = 128, n_mels: int = 80, dim: int = 128, attn: int = 80):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.lang_emb = nn.Embedding(4, dim)
        self.key = nn.Sequential(
            nn.Conv1d(dim, dim * 2, 3, padding=1), nn.ReLU(), nn.Conv1d(dim * 2, attn, 1)
        )
        self.query = nn.Sequential(
            nn.Conv1d(n_mels, n_mels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(n_mels * 2, n_mels, 1),
            nn.ReLU(),
            nn.Conv1d(n_mels, attn, 1),
        )
        self.temperature = 0.0005

    def forward(
        self,
        phoneme_ids: torch.Tensor,
        lang_ids: torch.Tensor,
        mel: torch.Tensor,
        text_mask: torch.Tensor,
        log_prior: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(log_attn, attn_logits)`; both `[B, T, N]`, padded text keys set to -inf."""
        keys = self.key((self.emb(phoneme_ids) + self.lang_emb(lang_ids)).transpose(1, 2))
        queries = self.query(mel.transpose(1, 2))
        q, k = queries.float().transpose(1, 2), keys.float().transpose(1, 2)  # [B,T,A], [B,N,A]
        dist = (
            q.pow(2).sum(-1, keepdim=True)
            + k.pow(2).sum(-1).unsqueeze(1)
            - 2 * q @ k.transpose(1, 2)
        )
        logits = -self.temperature * dist
        neg = torch.finfo(logits.dtype).min / 4
        logits = logits.masked_fill(~text_mask.unsqueeze(1), neg)
        log_attn = F.log_softmax(logits, dim=-1)
        if log_prior is not None:
            log_attn = F.log_softmax(log_attn + log_prior, dim=-1).masked_fill(
                ~text_mask.unsqueeze(1), neg
            )
        return log_attn, logits


def beta_binomial_log_prior(
    text_lens: torch.Tensor, mel_lens: torch.Tensor, n_max: int, t_max: int, scale: float = 1.0
) -> torch.Tensor:
    """Log beta-binomial prior `[B, T, N]` encouraging a near-diagonal alignment."""
    dev = text_lens.device
    n = torch.arange(n_max, device=dev, dtype=torch.float32).view(1, 1, -1)
    t = torch.arange(t_max, device=dev, dtype=torch.float32).view(1, -1, 1)
    big_n = text_lens.float().view(-1, 1, 1)
    big_t = mel_lens.float().view(-1, 1, 1)
    a = scale * (t + 1.0)
    b = scale * (big_t - t)
    k = n
    nn_ = big_n - 1.0
    valid = (k <= nn_) & (t < big_t)
    k_c = torch.minimum(k, nn_.clamp(min=0))
    b_c = b.clamp(min=1e-3)
    log_p = (
        torch.lgamma(nn_ + 1)
        - torch.lgamma(k_c + 1)
        - torch.lgamma(nn_ - k_c + 1)
        + torch.lgamma(k_c + a)
        + torch.lgamma(nn_ - k_c + b_c)
        - torch.lgamma(nn_ + a + b_c)
        - (torch.lgamma(a) + torch.lgamma(b_c) - torch.lgamma(a + b_c))
    )
    return log_p.masked_fill(~valid, -1e4)


def forward_sum_loss(
    log_attn: torch.Tensor,
    text_lens: torch.Tensor,
    mel_lens: torch.Tensor,
    blank_logprob: float = -1.0,
) -> torch.Tensor:
    """CTC forward-sum loss: every text token must be visited monotonically across frames."""
    log_p = F.pad(log_attn.float(), (1, 0), value=blank_logprob)  # blank at index 0
    log_p = log_p.log_softmax(-1).transpose(0, 1)  # [T, B, N+1]; padded keys already ~ -inf
    n_max = log_attn.shape[-1]
    targets = (
        torch.arange(1, n_max + 1, device=log_attn.device)
        .unsqueeze(0)
        .expand(log_attn.shape[0], -1)
    )
    return F.ctc_loss(log_p, targets, mel_lens.long(), text_lens.long(), zero_infinity=True)


@torch.no_grad()
def monotonic_alignment_search(
    log_attn: torch.Tensor, text_lens: torch.Tensor, mel_lens: torch.Tensor
) -> torch.Tensor:
    """Batched MAS (vectorized NumPy on CPU; ~10x faster than per-frame CUDA launches).

    Returns hard alignment `[B, T, N]` (float 0/1): each frame maps to exactly one token,
    tokens are visited monotonically, and every token receives >= 1 frame.
    """
    import numpy as np

    bsz, t_max, n_max = log_attn.shape
    val = log_attn.float().cpu().numpy()
    tl = text_lens.cpu().numpy().astype(np.int64)
    ml = mel_lens.cpu().numpy().astype(np.int64)
    neg = np.float32(-1e9)
    val = np.where(np.arange(n_max)[None, None, :] >= tl[:, None, None], neg, val)
    q = np.full((bsz, n_max), neg, dtype=np.float32)
    q[:, 0] = val[:, 0, 0]
    from_prev = np.zeros((t_max, bsz, n_max), dtype=bool)
    shifted = np.empty_like(q)
    for t in range(1, t_max):
        shifted[:, 0] = neg
        shifted[:, 1:] = q[:, :-1]
        take = shifted > q
        from_prev[t] = take
        np.maximum(shifted, q, out=q)
        q += val[:, t]
    path = np.zeros((bsz, t_max, n_max), dtype=np.float32)
    cur = np.maximum(tl - 1, 0)
    ar = np.arange(bsz)
    for t in range(t_max - 1, -1, -1):
        active = t < ml
        path[ar[active], t, cur[active]] = 1.0
        cur = cur - (active & from_prev[t, ar, cur] & (cur > 0)).astype(np.int64)
    return torch.from_numpy(path).to(log_attn.device)


def binarization_loss(hard: torch.Tensor, log_attn: torch.Tensor) -> torch.Tensor:
    """Pull the soft attention toward the MAS path: `-mean log p(hard path)`."""
    return -(hard * log_attn.float()).sum() / hard.sum().clamp(min=1.0)
