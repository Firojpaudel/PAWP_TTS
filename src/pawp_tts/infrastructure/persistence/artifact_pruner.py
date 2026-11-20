"""Artifact cleanup utility enforcing the project's Disk Hygiene policy."""

from __future__ import annotations

import shutil
from pathlib import Path


def prune_artifacts(root_dir: str | Path = ".", finalize_release: bool = False) -> list[str]:
    """
    Clean temporary caches, generated audio renders, and optionally all non-final
    checkpoints so only the final release weights and reports remain.
    """
    root = Path(root_dir)
    removed: list[str] = []

    ephemeral_dirs = [
        root / "artifacts" / "audio",
        root / "artifacts" / "cache",
        root / "data" / "cache",
    ]
    for target in ephemeral_dirs:
        if not target.exists():
            continue
        for child in target.iterdir():
            if child.name == ".gitkeep":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            removed.append(str(child))

    if finalize_release:
        ckpt_dir = root / "artifacts" / "checkpoints"
        if ckpt_dir.exists():
            for child in ckpt_dir.iterdir():
                if child.name in {".gitkeep", "final"}:
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
                removed.append(str(child))

    return removed
