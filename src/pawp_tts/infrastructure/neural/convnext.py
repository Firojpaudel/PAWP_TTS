"""Core streaming ConvNeXt1D and token/frame-gated bilingual residual blocks."""

from __future__ import annotations

import math
from typing import Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from pawp_tts.infrastructure.weight_programming.basis_bank import BasisBankLinear, SharedBasisPool


class BoundedLookaheadConv1d(nn.Module):
    """1D convolution with strictly bounded streaming future lookahead."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int = 1,
        lookahead: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.right_pad = min(max(lookahead, 0), kernel_size - 1)
        self.left_pad = (kernel_size - 1) - self.right_pad
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bounded-lookahead 1D convolution to `(B, C, T)` input."""
        if self.left_pad > 0 or self.right_pad > 0:
            x = F.pad(x, (self.left_pad, self.right_pad))
        return cast(torch.Tensor, self.conv(x))


class GatedLowRankLinear(nn.Module):
    """
    Linear layer implementing cross-lingual weight superposition:
        y_t = x_t @ W_shared^T + b
              + m_en(t) * g_en * (x_t @ V_en^T @ U_en^T)
              + m_ne(t) * g_ne * (x_t @ V_ne^T @ U_ne^T)
    Supports zero-overhead monolingual weight folding via `.fold_language(...)`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        residual_rank: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.residual_rank = residual_rank

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if residual_rank > 0:
            self.u_en = nn.Parameter(torch.zeros(out_features, residual_rank))
            self.v_en = nn.Parameter(torch.empty(residual_rank, in_features))
            self.gate_en = nn.Parameter(torch.ones(()))

            self.u_ne = nn.Parameter(torch.zeros(out_features, residual_rank))
            self.v_ne = nn.Parameter(torch.empty(residual_rank, in_features))
            self.gate_ne = nn.Parameter(torch.ones(()))

            nn.init.kaiming_uniform_(self.v_en, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.v_ne, a=math.sqrt(5))

    def forward(
        self,
        x: torch.Tensor,
        lang_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass on `(B, T, in_features)` with optional `(B, T, 2)` `[m_en, m_ne]` gates.
        """
        out = F.linear(x, self.weight, self.bias)
        if self.residual_rank <= 0 or lang_masks is None:
            return out

        m_en = lang_masks[..., 0:1]
        m_ne = lang_masks[..., 1:2]

        if torch.any(m_en != 0):
            res_en = F.linear(F.linear(x, self.v_en), self.u_en)
            out = out + (m_en * self.gate_en) * res_en
        if torch.any(m_ne != 0):
            res_ne = F.linear(F.linear(x, self.v_ne), self.u_ne)
            out = out + (m_ne * self.gate_ne) * res_ne
        return out

    def fold_language(self, language: Literal["en", "ne"]) -> nn.Linear:
        """Materialize dense monolingual weights `W_shared + g_lang * U_lang @ V_lang`."""
        folded = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        with torch.no_grad():
            merged = self.weight.detach().clone()
            if self.residual_rank > 0:
                if language == "en":
                    merged = merged + self.gate_en.detach() * (
                        self.u_en.detach() @ self.v_en.detach()
                    )
                else:
                    merged = merged + self.gate_ne.detach() * (
                        self.u_ne.detach() @ self.v_ne.detach()
                    )
            folded.weight.copy_(merged)
            if self.bias is not None and folded.bias is not None:
                folded.bias.copy_(self.bias.detach())
        return folded


class ConvNeXt1DBlock(nn.Module):
    """Streaming ConvNeXt1D block with bounded lookahead and optional shared-basis/bilingual residuals."""

    def __init__(
        self,
        dim: int,
        ff_dim: int,
        kernel_size: int = 7,
        lookahead: int = 1,
        residual_rank: int = 0,
        pw1_pool: SharedBasisPool | None = None,
        pw2_pool: SharedBasisPool | None = None,
        layer_residual_rank: int = 0,
    ) -> None:
        super().__init__()
        self.residual_rank = residual_rank
        self.dwconv = BoundedLookaheadConv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            groups=dim,
            lookahead=lookahead,
            bias=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.pw1: BasisBankLinear | GatedLowRankLinear | nn.Linear
        self.pw2: BasisBankLinear | GatedLowRankLinear | nn.Linear
        if pw1_pool is not None and pw2_pool is not None:
            self.pw1 = BasisBankLinear(
                dim,
                ff_dim,
                pool=pw1_pool,
                layer_residual_rank=layer_residual_rank,
                lang_residual_rank=residual_rank,
            )
            self.pw2 = BasisBankLinear(
                ff_dim,
                dim,
                pool=pw2_pool,
                layer_residual_rank=layer_residual_rank,
                lang_residual_rank=residual_rank,
            )
        elif residual_rank > 0:
            self.pw1 = GatedLowRankLinear(dim, ff_dim, residual_rank=residual_rank)
            self.pw2 = GatedLowRankLinear(ff_dim, dim, residual_rank=residual_rank)
        else:
            self.pw1 = nn.Linear(dim, ff_dim)
            self.pw2 = nn.Linear(ff_dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), 1e-2))

    def forward(
        self,
        x: torch.Tensor,
        lang_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply ConvNeXt1D residual transform on `(B, T, C)` tensors."""
        residual = x
        h = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        h = self.norm(h)
        h = (
            self.pw1(h, lang_masks)
            if isinstance(self.pw1, (GatedLowRankLinear, BasisBankLinear))
            else self.pw1(h)
        )
        h = F.gelu(h)
        h = (
            self.pw2(h, lang_masks)
            if isinstance(self.pw2, (GatedLowRankLinear, BasisBankLinear))
            else self.pw2(h)
        )
        return cast(torch.Tensor, residual + self.gamma * h)
