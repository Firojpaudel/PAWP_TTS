"""Single standard Gradio UI for PAWP-TTS interactive evaluation."""

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
SPEAKER_ID = 0
if model.speaker_emb is not None and model.speaker_emb.num_embeddings > 1:
    SPEAKER_ID = (
        cfg.speaker_to_id.get(target_speaker_name, 0)
        if hasattr(cfg, "speaker_to_id")
        else 0
    )


def synthesize(text: str) -> tuple[tuple[int, np.ndarray], str]:
    """Synthesize text in-memory and return audio + phonemes without saving files."""
    cleaned = text.strip()
    if not cleaned:
        return (24000, np.zeros(2400, dtype=np.float32)), ""

    seq = phonemize_usecase.execute(cleaned)
    phonemes_str = " ".join(
        f"/{t.symbol}/" for t in seq.tokens if t.symbol not in {"<bos>", "<eos>"}
    )

    _, _, wav, _ = _synthesize_utterance(
        model, phonemize_usecase, cleaned, DEVICE, speaker_id=SPEAKER_ID
    )
    return (24000, wav), phonemes_str


EXAMPLES = [
    # Nepali
    ["नेपालको प्राकृतिक सौन्दर्य र अग्ला हिमाल हेर्न संसारभरिबाट मानिसहरू आउँछन्।"],
    ["तातो चिया र च्याउको सुपले जाडो मौसममा शरीरलाई निकै आराम दिन्छ।"],
    ["तिमीले सोचेको काम समयमै पूरा हुन्छ, अब धेरै चिन्ता लिनु पर्दैन।"],
    ["शहर र सहर दुवै काठमाडौंको मानक बोलीमा सहर जस्तै उच्चारण हुन्छ।"],
    ["पाता र पत्ता, पना र पन्नाको उच्चारणमा फरक स्पष्ट सुनिन्छ।"],
    # English
    ["Neural speech synthesis can now run completely on edge devices with very low latency."],
    ["The sudden change in the weather made the evening walk surprisingly refreshing."],
    ["Our team has verified that the compact bilingual model preserves phonetic contrasts."],
    # Code-mixed
    ["Flight ticket confirm भइसक्यो, बाँकी details म तिमीलाई WhatsApp मा पठाइदिन्छु।"],
    ["आज हाम्रो team ले नयाँ AI model deploy गर्‍यो, performance एकदमै smooth देखिन्छ।"],
    ["यो weekend मा साथीहरूसँग movie हेर्न जाने plan बनिरहेको छ, तिमी पनि आउँछौ?"],
    ["PAWP-TTS ले English र Nepali दुवै भाषामा natural speech generate गर्न सक्छ।"],
]

demo = gr.Interface(
    fn=synthesize,
    inputs=gr.Textbox(
        label="Input Text (Nepali, English, or Code-mixed)",
        placeholder="यहाँ नेपाली, English वा दुवै mixed text लेख्नुहोस्...",
        value="नेपालको प्राकृतिक सौन्दर्य र अग्ला हिमाल हेर्न संसारभरिबाट मानिसहरू आउँछन्।",
        lines=3,
    ),
    outputs=[
        gr.Audio(label="Synthesized 24 kHz Speech", type="numpy", autoplay=True),
        gr.Textbox(label="Phoneme Sequence & Pronunciation", interactive=False),
    ],
    title="PAWP-TTS: Bilingual Neural Speech Synthesis (1.49M params)",
    description=(
        "Standard interactive testing playground for PAWP-TTS. "
        "Audio is synthesized on-the-fly in memory with zero permanent file writes."
    ),
    examples=EXAMPLES,
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
