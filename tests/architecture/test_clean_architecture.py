"""AST-level tests enforcing strict Clean Architecture dependency boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "pawp_tts"

FORBIDDEN_DOMAIN_IMPORTS = {
    "torch",
    "torchaudio",
    "onnx",
    "onnxruntime",
    "httpx",
    "yaml",
    "soundfile",
    "safetensors",
    "dotenv",
    "numpy",
    "pyarrow",
    "scipy",
    "pawp_tts.application",
    "pawp_tts.infrastructure",
    "pawp_tts.interfaces",
}

FORBIDDEN_APPLICATION_IMPORTS = {
    "torch",
    "torchaudio",
    "onnx",
    "onnxruntime",
    "httpx",
    "yaml",
    "soundfile",
    "safetensors",
    "dotenv",
    "pyarrow",
    "scipy",
    "pawp_tts.infrastructure",
    "pawp_tts.interfaces",
}


def _imported_roots(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name)
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module)
            parts = node.module.split(".")
            roots.add(parts[0])
            if len(parts) >= 2:
                roots.add(".".join(parts[:2]))
    return roots


def test_domain_has_zero_framework_or_outer_layer_imports() -> None:
    for py_file in (SRC_ROOT / "domain").rglob("*.py"):
        imports = _imported_roots(py_file)
        violations = sorted(imports & FORBIDDEN_DOMAIN_IMPORTS)
        assert not violations, f"{py_file} imports forbidden modules: {violations}"


def test_application_depends_strictly_on_domain() -> None:
    for py_file in (SRC_ROOT / "application").rglob("*.py"):
        imports = _imported_roots(py_file)
        violations = sorted(imports & FORBIDDEN_APPLICATION_IMPORTS)
        assert not violations, f"{py_file} imports forbidden modules: {violations}"
