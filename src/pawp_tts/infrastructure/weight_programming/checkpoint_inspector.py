"""Zero-RAM safetensors header inspector for baseline and teacher checkpoints."""

from __future__ import annotations

import json
import math
import struct
from collections import Counter
from pathlib import Path

from pawp_tts.domain.entities.models import CheckpointSummary


class SafetensorsCheckpointInspector:
    """Reads only the JSON header of `.safetensors` checkpoints to summarize tensors."""

    def inspect(self, checkpoint_path: str) -> CheckpointSummary:
        """Return parameter counts, dtypes, and top shape pools from checkpoint metadata."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with path.open("rb") as f:
            raw_len = f.read(8)
            if len(raw_len) != 8:
                raise ValueError(f"Invalid safetensors header length in {checkpoint_path}")
            header_size = struct.unpack("<Q", raw_len)[0]
            header_bytes = f.read(header_size)

        header = json.loads(header_bytes.decode("utf-8"))
        total_params = 0
        tensor_count = 0
        dtypes: Counter[str] = Counter()
        shape_pools: Counter[str] = Counter()

        for name, meta in header.items():
            if name == "__metadata__":
                continue
            shape = meta.get("shape", [])
            dtype = str(meta.get("dtype", "unknown"))
            numel = math.prod(shape) if shape else 1
            total_params += numel
            tensor_count += 1
            dtypes[dtype] += 1
            if len(shape) >= 2:
                shape_pools[str(tuple(shape))] += 1

        top_pools = dict(shape_pools.most_common(12))
        return CheckpointSummary(
            path=str(path),
            total_params=total_params,
            tensor_count=tensor_count,
            dtypes=dict(dtypes),
            shape_pools=top_pools,
        )
