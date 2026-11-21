"""28-dimensional articulatory phonological conditioning module."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from pawp_tts.domain.phonology.inventory import (
    ARTICULATORY_DIM,
    UNIFIED_PHONEMES,
    phoneme_to_articulatory_vector,
)


class ArticulatoryConditioner(nn.Module):
    """
    Projects frozen 28-dim binary articulatory phonological vectors into `d_model`
    to share structural capacity across English & Nepali phonetic contrasts.
    """

    feature_table: torch.Tensor

    def __init__(self, vocab_size: int = 128, d_model: int = 128) -> None:
        super().__init__()
        table = torch.zeros(vocab_size, ARTICULATORY_DIM, dtype=torch.float32)
        for idx, sym in enumerate(UNIFIED_PHONEMES):
            if idx >= vocab_size:
                break
            table[idx] = torch.tensor(phoneme_to_articulatory_vector(sym), dtype=torch.float32)

        self.register_buffer("feature_table", table, persistent=False)
        self.proj = nn.Linear(ARTICULATORY_DIM, d_model)

    def forward(self, phoneme_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs `(B, T)` to `(B, T, d_model)` articulatory modulation."""
        clamped = torch.clamp(phoneme_ids, 0, self.feature_table.shape[0] - 1)
        features = F.embedding(clamped, self.feature_table)
        return cast(torch.Tensor, self.proj(features))
