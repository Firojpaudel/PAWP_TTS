"""Shape-grouped shared weight basis pools and factorized linear operators for PAWP."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedBasisPool(nn.Module):
    """Stores `K` shape-compatible shared basis matrices `(K, out_features, in_features)`."""

    def __init__(self, num_bases: int, out_features: int, in_features: int) -> None:
        super().__init__()
        if num_bases <= 0:
            raise ValueError("num_bases must be positive.")
        self.num_bases = num_bases
        self.out_features = out_features
        self.in_features = in_features

        bases = torch.empty(num_bases, out_features, in_features)
        for k in range(num_bases):
            nn.init.orthogonal_(bases[k])
        self.bases = nn.Parameter(bases)


class BasisBankLinear(nn.Module):
    """
    PAWP programmable linear operator decomposing layer weights into:
        W_l = sum_k a_{l,k} * B_k + U_l @ V_l
    plus optional token/frame-gated bilingual EN/NE low-rank residuals:
        y_t = x_t @ W_l^T + b_l
              + m_en(t) * g_en * (x_t @ V_en^T @ U_en^T)
              + m_ne(t) * g_ne * (x_t @ V_ne^T @ U_ne^T)
    Supports zero-overhead dense monolingual folding via `.fold_language(...)`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        pool: SharedBasisPool,
        layer_residual_rank: int = 16,
        lang_residual_rank: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if pool.in_features != in_features or pool.out_features != out_features:
            raise ValueError(
                f"Pool shape ({pool.out_features}, {pool.in_features}) does not match "
                f"layer ({out_features}, {in_features})."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.layer_residual_rank = layer_residual_rank
        self.lang_residual_rank = lang_residual_rank
        # Store pool in tuple to avoid duplicate nn.Module parameter/state_dict aliases across layers
        self._pool_ref: tuple[SharedBasisPool] = (pool,)

        self.basis_coeffs = nn.Parameter(
            torch.full((pool.num_bases,), 1.0 / math.sqrt(pool.num_bases))
        )

        if layer_residual_rank > 0:
            self.u_layer = nn.Parameter(torch.zeros(out_features, layer_residual_rank))
            self.v_layer = nn.Parameter(torch.empty(layer_residual_rank, in_features))
            nn.init.kaiming_uniform_(self.v_layer, a=math.sqrt(5))

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        if lang_residual_rank > 0:
            self.u_en = nn.Parameter(torch.zeros(out_features, lang_residual_rank))
            self.v_en = nn.Parameter(torch.empty(lang_residual_rank, in_features))
            self.gate_en = nn.Parameter(torch.ones(()))

            self.u_ne = nn.Parameter(torch.zeros(out_features, lang_residual_rank))
            self.v_ne = nn.Parameter(torch.empty(lang_residual_rank, in_features))
            self.gate_ne = nn.Parameter(torch.ones(()))

            nn.init.kaiming_uniform_(self.v_en, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.v_ne, a=math.sqrt(5))

    @property
    def pool(self) -> SharedBasisPool:
        """Return associated shared basis pool."""
        return self._pool_ref[0]

    def reconstruct_shared_weight(self) -> torch.Tensor:
        """Reconstruct `sum_k a_k B_k + U_l V_l` as `(out_features, in_features)`."""
        w = torch.einsum("k,koi->oi", self.basis_coeffs, self.pool.bases)
        if self.layer_residual_rank > 0:
            w = w + (self.u_layer @ self.v_layer)
        return w

    def forward(
        self,
        x: torch.Tensor,
        lang_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass on `(B, T, in_features)` with optional `(B, T, 2)` `[m_en, m_ne]`."""
        out = F.linear(x, self.reconstruct_shared_weight(), self.bias)
        if self.lang_residual_rank <= 0 or lang_masks is None:
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
        """Materialize dense monolingual weights `sum_k a_k B_k + U_l V_l + g_lang U_lang V_lang`."""
        folded = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        with torch.no_grad():
            merged = self.reconstruct_shared_weight().detach().clone()
            if self.lang_residual_rank > 0:
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


def decompose_weights_to_shared_basis(
    weights: Sequence[torch.Tensor],
    num_bases: int,
    residual_rank: int,
    importance_weights: Sequence[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    """
    Decompose `L` shape-compatible weight matrices `W_l` into:
      - `bases`: `(K, d_out, d_in)` shared subspace basis
      - `coeffs`: `(L, K)` projection coefficients
      - `residuals`: list of `(U_l, V_l)` truncated-SVD remainders of rank `residual_rank`
    """
    if not weights:
        raise ValueError("weights sequence cannot be empty.")
    if num_bases <= 0:
        raise ValueError("num_bases must be positive.")

    d_out, d_in = weights[0].shape
    for w in weights:
        if w.shape != (d_out, d_in):
            raise ValueError(f"Mismatched weight shape {w.shape}; expected {(d_out, d_in)}.")

    flat_weights = torch.stack([w.detach().float().reshape(-1) for w in weights], dim=0)
    if importance_weights is not None:
        if len(importance_weights) != len(weights):
            raise ValueError("importance_weights length must match weights length.")
        flat_imp = torch.stack(
            [
                torch.clamp(imp.detach().float().reshape(-1), min=1e-6).sqrt()
                for imp in importance_weights
            ],
            dim=0,
        )
        weighted_mat = flat_weights * flat_imp
    else:
        weighted_mat = flat_weights

    _, _, vh = torch.linalg.svd(weighted_mat, full_matrices=False)
    k_eff = min(num_bases, vh.shape[0])
    bases_flat = vh[:k_eff]  # (k_eff, d_out * d_in)
    if k_eff < num_bases:
        pad = torch.zeros(
            num_bases - k_eff, d_out * d_in, dtype=bases_flat.dtype, device=bases_flat.device
        )
        bases_flat = torch.cat([bases_flat, pad], dim=0)

    # Orthogonal projection onto extracted basis directions
    coeffs = flat_weights @ bases_flat.T  # (L, num_bases)
    bases = bases_flat.view(num_bases, d_out, d_in)

    residuals: list[tuple[torch.Tensor, torch.Tensor]] = []
    for l_idx, w in enumerate(weights):
        recon = torch.einsum("k,koi->oi", coeffs[l_idx], bases)
        rem = w.detach().float() - recon
        if residual_rank <= 0:
            residuals.append(
                (
                    torch.zeros(d_out, 0, dtype=torch.float32, device=w.device),
                    torch.zeros(0, d_in, dtype=torch.float32, device=w.device),
                )
            )
            continue
        u, s, vh_rem = torch.linalg.svd(rem, full_matrices=False)
        r = min(residual_rank, s.shape[0])
        s_root = s[:r].sqrt()
        u_l = u[:, :r] * s_root.unsqueeze(0)
        v_l = s_root.unsqueeze(1) * vh_rem[:r, :]
        residuals.append((u_l, v_l))

    return bases, coeffs, residuals
