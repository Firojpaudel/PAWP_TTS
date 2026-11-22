"""Single standard Gradio UI for PAWP-TTS interactive A/B evaluation."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from pawp_tts.infrastructure.evaluation.benchmark_suite import _synthesize_utterance
from pawp_tts.infrastructure.neural.bilingual_tts import PAWPBilingualTTS
from pawp_tts.infrastructure.persistence.config_loader import load_student_config
from pawp_tts.infrastructure.training.trainer import CheckpointManager
from pawp_tts.interfaces.container import build_container

# Default to CPU so Gradio uses 0 MB GPU VRAM while training runs on the RTX 4090
DEVICE = torch.device(os.environ.get("PAWP_GRADIO_DEVICE", "cpu"))

MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "PAWP-1.5M Polished Default (EXP-012, HF Default)": (
        "configs/student_1_5m.yaml",
        "artifacts/checkpoints/exp012_polish/best.safetensors",
    ),
    "PAWP-300K Target Voice (EXP-011-FT, 281K Unique / 1.1 MB)": (
        "configs/student_300k_pawp.yaml",
        "artifacts/checkpoints/exp011_300k_ft/best.safetensors",
    ),
    "PAWP-300K Base (EXP-011, 80k Steps, 281K Unique)": (
        "configs/student_300k_pawp.yaml",
        "artifacts/checkpoints/exp011_300k/best.safetensors",
    ),
    "PAWP-1.5M + WSP E3 (EXP-009, HF WSP Variant)": (
        "configs/student_1_5m.yaml",
        "artifacts/checkpoints/exp009_wsp/pawp_1_5m_wsp_e3.safetensors",
    ),
    "PAWP-1.5M Target Voice (EXP-008-FT, Step 20k)": (
        "configs/student_1_5m.yaml",
        "artifacts/checkpoints/exp008/best.safetensors",
    ),
    "PAWP-1.5M Pre-FT Base (EXP-007, Step 92k)": (
        "configs/student_1_5m.yaml",
        "artifacts/checkpoints/exp007/latest.safetensors",
    ),
}

AVAILABLE_MODELS: dict[str, tuple[str, str]] = {
    label: pair for label, pair in MODEL_REGISTRY.items() if Path(pair[1]).exists()
}
DEFAULT_MODEL_LABEL = next(iter(AVAILABLE_MODELS.keys()))

_LOADED_MODELS: dict[str, PAWPBilingualTTS] = {}


def _get_model(label: str) -> PAWPBilingualTTS:
    if label not in _LOADED_MODELS:
        cfg_path, ckpt_path = AVAILABLE_MODELS[label]
        cfg = load_student_config(cfg_path)
        m = PAWPBilingualTTS(cfg).to(DEVICE).eval()
        CheckpointManager.load(m, ckpt_path)
        _LOADED_MODELS[label] = m
    return _LOADED_MODELS[label]


_default_model = _get_model(DEFAULT_MODEL_LABEL)
container = build_container()
phonemize_usecase = container.phonemize
SPEAKER_ID = 0


def synthesize(text: str, model_label: str) -> tuple[tuple[int, np.ndarray], str]:
    """Synthesize text in-memory using the selected checkpoint."""
    cleaned = text.strip()
    if not cleaned:
        return (24000, np.zeros(2400, dtype=np.float32)), ""

    model = _get_model(model_label)
    seq = phonemize_usecase.execute(cleaned)
    phonemes_str = " ".join(
        f"/{t.symbol}/" for t in seq.tokens if t.symbol not in {"<bos>", "<eos>"}
    )

    _, _, wav, _ = _synthesize_utterance(
        model, phonemize_usecase, cleaned, DEVICE, speaker_id=SPEAKER_ID
    )
    return (24000, wav), phonemes_str


EXAMPLES = [
    ["नेपालको प्राकृतिक सौन्दर्य र अग्ला हिमाल हेर्न संसारभरिबाट मानिसहरू आउँछन्।", DEFAULT_MODEL_LABEL],
    ["तातो चिया र च्याउको सुपले जाडो मौसममा शरीरलाई निकै आराम दिन्छ।", DEFAULT_MODEL_LABEL],
    ["पाता र पत्ता, पना र पन्नाको उच्चारणमा फरक स्पष्ट सुनिन्छ।", DEFAULT_MODEL_LABEL],
    ["Neural speech synthesis can now run completely on edge devices with very low latency.", DEFAULT_MODEL_LABEL],
    ["The sudden change in the weather made the evening walk surprisingly refreshing.", DEFAULT_MODEL_LABEL],
    ["Flight ticket confirm भइसक्यो, बाँकी details म तिमीलाई WhatsApp मा पठाइदिन्छु।", DEFAULT_MODEL_LABEL],
    ["आज हाम्रो team ले नयाँ AI model deploy गर्‍यो, performance एकदमै smooth देखिन्छ।", DEFAULT_MODEL_LABEL],
]

demo = gr.Interface(
    fn=synthesize,
    inputs=[
        gr.Textbox(
            label="Input Text (Nepali, English, or Code-mixed)",
            placeholder="यहाँ नेपाली, English वा दुवै mixed text लेख्नुहोस्...",
            value="नेपालको प्राकृतिक सौन्दर्य र अग्ला हिमाल हेर्न संसारभरिबाट मानिसहरू आउँछन्।",
            lines=3,
        ),
        gr.Dropdown(
            choices=list(AVAILABLE_MODELS.keys()),
            value=DEFAULT_MODEL_LABEL,
            label="Model Checkpoint",
        ),
    ],
    outputs=[
        gr.Audio(label="Synthesized 24 kHz Speech", type="numpy", autoplay=True),
        gr.Textbox(label="Phoneme Sequence & Pronunciation", interactive=False),
    ],
    title="PAWP-TTS: Interactive Model Comparison (1.5M vs 300K)",
    description=(
        "Compare PAWP-1.5M (EXP-012 Polish, EXP-009 WSP-E3, EXP-008-FT) and PAWP-300K (EXP-011-FT, 281K params) "
        "side by side. Runs on CPU (0 MB GPU VRAM)."
    ),
    examples=EXAMPLES,
)

if __name__ == "__main__":
    port = int(os.environ.get("PAWP_GRADIO_PORT", "7861"))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
