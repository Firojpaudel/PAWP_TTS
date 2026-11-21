"""Lightweight interactive Gradio UI for temporary PAWP-TTS audio testing."""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import torch
import gradio as gr

from pawp_tts.infrastructure.persistence.config_loader import load_student_config
from pawp_tts.infrastructure.evaluation.benchmark_suite import _synthesize_utterance
from pawp_tts.infrastructure.neural.bilingual_tts import PAWPBilingualTTS
from pawp_tts.infrastructure.training.trainer import CheckpointManager
from pawp_tts.interfaces.container import build_container

# Load model once at startup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_PATH = "configs/student_1_5m.yaml"
CKPT_PATH = "artifacts/checkpoints/exp007/latest.safetensors"

cfg = load_student_config(CONFIG_PATH)
model = PAWPBilingualTTS(cfg).to(DEVICE).eval()
CheckpointManager.load(model, CKPT_PATH)

container = build_container()
phonemize_usecase = container.phonemize

target_speaker_name = os.environ.get("PAWP_TARGET_SPEAKER", "target_voice")
SPEAKER_ID = cfg.speaker_to_id.get(target_speaker_name, 0) if hasattr(cfg, "speaker_to_id") else 0
if model.speaker_emb is None or model.speaker_emb.num_embeddings <= 1:
    SPEAKER_ID = 0


def synthesize(text: str) -> tuple[int, np.ndarray]:
    """Synthesize text in-memory and return (sample_rate, wav_numpy) to Gradio."""
    if not text.strip():
        return (24000, np.zeros(2400, dtype=np.float32))

    _, _, wav, _ = _synthesize_utterance(
        model, phonemize_usecase, text, DEVICE, speaker_id=SPEAKER_ID
    )
    return (24000, wav)


demo = gr.Interface(
    fn=synthesize,
    inputs=gr.Textbox(
        label="Input Text (Nepali, English, or Code-mixed)",
        placeholder="यहाँ नेपाली, English वा mixed text लेख्नुहोस्...",
        value="नमस्ते, PAWP-TTS मा तपाईंलाई स्वागत छ!",
        lines=3,
    ),
    outputs=gr.Audio(label="Synthesized 24 kHz Speech", type="numpy", autoplay=True),
    title="PAWP-TTS: Phoneme-Aware Bilingual Neural Speech Synthesis (1.49M params)",
    description="Interactive browser playground. Audio is generated on-the-fly in-memory (no files permanently saved).",
    examples=[
        ["नमस्ते, PAWP-TTS मा तपाईंलाई स्वागत छ!"],
        ["आज काठमाडौंको मौसम निकै रमाइलो छ।"],
        ["Good morning. Your appointment is confirmed for ten thirty tomorrow."],
        ["आज मेरो office मा एउटा important meeting छ।"],
        ["हाम्रो नयाँ model ले धेरै राम्रो काम गरिरहेको छ।"],
    ],
    theme="soft",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
