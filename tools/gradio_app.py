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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_PATH = "configs/student_1_5m.yaml"
cfg = load_student_config(CONFIG_PATH)

MODEL_REGISTRY: dict[str, str] = {
    "PAWP-1.5M Target Voice (EXP-008-FT, HF Release)": "artifacts/checkpoints/exp008/best.safetensors",
    "PAWP-1.5M + WSP E3 (Zero-Training Code-Mix Winner, 9.93% Mix WER)": "artifacts/checkpoints/exp009_wsp/pawp_1_5m_wsp_e3.safetensors",
    "PAWP-1.5M + WSP Null-64 (Nepali-Orthogonal Null-Space, 4.73% EN WER)": "artifacts/checkpoints/exp009_wsp/pawp_1_5m_wsp_null64.safetensors",
    "PAWP-1.0M Decoder-Centric (Zero-Shot 998.8K Params)": "artifacts/checkpoints/exp010_compress/pawp_1m_dc_zeroshot.safetensors",
    "PAWP-1.5M Pre-FT Base (EXP-007 Step 92k)": "artifacts/checkpoints/exp007/latest.safetensors",
}

AVAILABLE_MODELS: dict[str, str] = {
    label: path for label, path in MODEL_REGISTRY.items() if Path(path).exists()
}
DEFAULT_MODEL_LABEL = next(iter(AVAILABLE_MODELS.keys()))

_LOADED_MODELS: dict[str, PAWPBilingualTTS] = {}


def _get_model(label: str) -> PAWPBilingualTTS:
    if label not in _LOADED_MODELS:
        ckpt_path = AVAILABLE_MODELS[label]
        m = PAWPBilingualTTS(cfg).to(DEVICE).eval()
        CheckpointManager.load(m, ckpt_path)
        _LOADED_MODELS[label] = m
    return _LOADED_MODELS[label]


# Preload default model at startup
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
    ["तिमीले सोचेको काम समयमै पूरा हुन्छ, अब धेरै चिन्ता लिनु पर्दैन।", DEFAULT_MODEL_LABEL],
    ["शहर र सहर दुवै काठमाडौंको मानक बोलीमा सहर जस्तै उच्चारण हुन्छ।", DEFAULT_MODEL_LABEL],
    ["पाता र पत्ता, पना र पन्नाको उच्चारणमा फरक स्पष्ट सुनिन्छ।", DEFAULT_MODEL_LABEL],
    ["Neural speech synthesis can now run completely on edge devices with very low latency.", DEFAULT_MODEL_LABEL],
    ["The sudden change in the weather made the evening walk surprisingly refreshing.", DEFAULT_MODEL_LABEL],
    ["Our team has verified that the compact bilingual model preserves phonetic contrasts.", DEFAULT_MODEL_LABEL],
    ["Flight ticket confirm भइसक्यो, बाँकी details म तिमीलाई WhatsApp मा पठाइदिन्छु।", DEFAULT_MODEL_LABEL],
    ["आज हाम्रो team ले नयाँ AI model deploy गर्‍यो, performance एकदमै smooth देखिन्छ।", DEFAULT_MODEL_LABEL],
    ["यो weekend मा साथीहरूसँग movie हेर्न जाने plan बनिरहेको छ, तिमी पनि आउँछौ?", DEFAULT_MODEL_LABEL],
    ["PAWP-TTS ले English र Nepali दुवै भाषामा natural speech generate गर्न सक्छ।", DEFAULT_MODEL_LABEL],
]

demo = gr.Interface(
    fn=synthesize,
    inputs=[
        gr.Textbox(
            label="Input Text (Nepali, English, or Code-mixed)",
            placeholder="यहाँ नेपाली, English वा दुवै mixed text लेख्नुहोस्...",
            value="Flight ticket confirm भइसक्यो, बाँकी details म तिमीलाई WhatsApp मा पठाइदिन्छु।",
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
    title="PAWP-TTS: Interactive Model Comparison Playground",
    description=(
        "Compare EXP-008-FT, EXP-009 Weight-Space Programmed edits (WSP-E3, WSP-Null64), "
        "EXP-010 PAWP-1.0M compressed checkpoint, and EXP-007 base side by side."
    ),
    examples=EXAMPLES,
)

if __name__ == "__main__":
    port = int(os.environ.get("PAWP_GRADIO_PORT", "7861"))
    demo.launch(server_name=os.environ.get("PAWP_DEMO_HOST", "127.0.0.1"), server_port=port, share=False)
