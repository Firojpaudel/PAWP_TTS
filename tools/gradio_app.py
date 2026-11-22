"""Single standard Gradio UI for PAWP-TTS interactive A/B evaluation."""

from __future__ import annotations

import hashlib
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
    "PAWP-1.5M v3 (EXP-015a — Best Overall, Clean v3 Alignment)": (
        "configs/student_1_5m.yaml",
        "artifacts/checkpoints/exp015_1_5m_v3/best.safetensors",
    ),
    "PAWP-1.5M v3 + Pause WSP-96 (EXP-016 — 27:1 <sp> Pause Edit)": (
        "configs/student_1_5m.yaml",
        "artifacts/checkpoints/exp016_pause_wsp/wsp_pause96.safetensors",
    ),
    "PAWP-1.5M v3 + Retroflex FT-0.4% (EXP-017 — ɳ vs n̪ NE Residual)": (
        "configs/student_1_5m.yaml",
        "artifacts/checkpoints/exp017_retroflex_wsp/control_sameparam_ne_ft.safetensors",
    ),
    "PAWP-1.5M v3 + Retroflex ROME Rank-1 (EXP-017 — Zero-Training)": (
        "configs/student_1_5m.yaml",
        "artifacts/checkpoints/exp017_retroflex_wsp/wsp_rome_ne_rank1.safetensors",
    ),
    "PAWP-300K v3 Shared-Basis (EXP-015b — Live, 281K Unique)": (
        "configs/student_300k_pawp.yaml",
        "artifacts/checkpoints/exp015_300k_shared_v3/latest.safetensors",
    ),
    "Dense-300K v3 Baseline (EXP-015c — Live, 281.4K Dense)": (
        "configs/student_300k_dense.yaml",
        "artifacts/checkpoints/exp015_300k_dense_v3/latest.safetensors",
    ),
    "PAWP-1.5M v2 Polished (EXP-012 — Current HF Default)": (
        "configs/student_1_5m.yaml",
        "artifacts/checkpoints/exp012_polish/best.safetensors",
    ),
    "PAWP-300K v2 Target Voice (EXP-011-FT — 281K Unique / 1.1 MB)": (
        "configs/student_300k_pawp.yaml",
        "artifacts/checkpoints/exp011_300k_ft/best.safetensors",
    ),
    "PAWP-1.5M + WSP E3 (EXP-009 — HF WSP Variant)": (
        "configs/student_1_5m.yaml",
        "artifacts/checkpoints/exp009_wsp/pawp_1_5m_wsp_e3.safetensors",
    ),
}

AVAILABLE_MODELS: dict[str, tuple[str, str]] = {
    label: pair for label, pair in MODEL_REGISTRY.items() if Path(pair[1]).exists()
}
DEFAULT_MODEL_LABEL = next(iter(AVAILABLE_MODELS.keys()))

_LOADED_MODELS: dict[str, tuple[float, PAWPBilingualTTS, int]] = {}


def _get_model(label: str) -> tuple[PAWPBilingualTTS, int]:
    cfg_path, ckpt_path = AVAILABLE_MODELS[label]
    mtime = Path(ckpt_path).stat().st_mtime
    cached = _LOADED_MODELS.get(label)
    if cached is None or cached[0] != mtime:
        cfg = load_student_config(cfg_path)
        m = PAWPBilingualTTS(cfg).to(DEVICE).eval()
        CheckpointManager.load(m, ckpt_path)
        n_params = sum(p.numel() for p in m.parameters())
        _LOADED_MODELS[label] = (mtime, m, n_params)
    return _LOADED_MODELS[label][1], _LOADED_MODELS[label][2]


_default_model = _get_model(DEFAULT_MODEL_LABEL)
container = build_container()
phonemize_usecase = container.phonemize
SPEAKER_ID = 0


def synthesize(text: str, model_label: str) -> tuple[tuple[int, np.ndarray], str, str]:
    """Synthesize text in-memory using the selected checkpoint (no audio caching)."""
    cleaned = text.strip()
    if not cleaned:
        return (24000, np.zeros(2400, dtype=np.float32)), "", "No input text."

    model, n_params = _get_model(model_label)
    seq = phonemize_usecase.execute(cleaned)
    phonemes_str = " ".join(
        f"/{t.symbol}/" for t in seq.tokens if t.symbol not in {"<bos>", "<eos>"}
    )

    _, _, wav, _ = _synthesize_utterance(
        model, phonemize_usecase, cleaned, DEVICE, speaker_id=SPEAKER_ID
    )
    rms = float(np.sqrt(np.mean(wav**2))) if len(wav) else 0.0
    digest = hashlib.sha1(wav.tobytes()).hexdigest()[:8]
    info = (
        f"Model: {model_label} | unique_params={n_params:,} | "
        f"duration={len(wav) / 24000:.2f}s | rms={rms:.4f} | audio_sha1={digest}"
    )
    return (24000, wav), info, phonemes_str


TEXT_EXAMPLES = [
    ["नेपालको प्राकृतिक सौन्दर्य र अग्ला हिमाल हेर्न संसारभरिबाट मानिसहरू आउँछन्।"],
    ["तातो चिया र च्याउको सुपले जाडो मौसममा शरीरलाई निकै आराम दिन्छ।"],
    ["पाता र पत्ता, पना र पन्नाको उच्चारणमा फरक स्पष्ट सुनिन्छ।"],
    ["नारायणगढको पूर्ण विवरण र कारण जान्न म अहिले कार्यालय जाँदै छु।"],
    ["Neural speech synthesis can now run completely on edge devices with very low latency."],
    ["The sudden change in the weather made the evening walk surprisingly refreshing."],
    ["Flight ticket confirm भइसक्यो, बाँकी details म तिमीलाई WhatsApp मा पठाइदिन्छु।"],
    ["आज हाम्रो team ले नयाँ AI model deploy गर्‍यो, performance एकदमै smooth देखिन्छ।"],
]

with gr.Blocks(title="PAWP-TTS: Interactive Model Comparison (1.5M vs 300K)") as demo:
    gr.Markdown(
        "# PAWP-TTS: Interactive Model Comparison (1.5M vs 300K)\n"
        "Clicking an example updates **only** the text box (never resets your Model Checkpoint). "
        "Each run prints the exact `unique_params` and `audio_sha1` so you can verify fresh synthesis."
    )
    with gr.Row():
        with gr.Column():
            text_in = gr.Textbox(
                label="Input Text (Nepali, English, or Code-mixed)",
                placeholder="यहाँ नेपाली, English वा दुवै mixed text लेख्नुहोस्...",
                value="नेपालको प्राकृतिक सौन्दर्य र अग्ला हिमाल हेर्न संसारभरिबाट मानिसहरू आउँछन्।",
                lines=3,
            )
            model_dd = gr.Dropdown(
                choices=list(AVAILABLE_MODELS.keys()),
                value=DEFAULT_MODEL_LABEL,
                label="Model Checkpoint",
            )
            btn = gr.Button("Synthesize", variant="primary")
            gr.Examples(examples=TEXT_EXAMPLES, inputs=[text_in], cache_examples=False)
        with gr.Column():
            audio_out = gr.Audio(label="Synthesized 24 kHz Speech", type="numpy", autoplay=True)
            info_out = gr.Textbox(label="Active Model & Audio Fingerprint", interactive=False)
            phon_out = gr.Textbox(label="Phoneme Sequence & Pronunciation", interactive=False)

    btn.click(fn=synthesize, inputs=[text_in, model_dd], outputs=[audio_out, info_out, phon_out])
    model_dd.change(fn=synthesize, inputs=[text_in, model_dd], outputs=[audio_out, info_out, phon_out])

if __name__ == "__main__":
    port = int(os.environ.get("PAWP_GRADIO_PORT", "7861"))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
