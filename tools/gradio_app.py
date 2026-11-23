"""Guided Side-by-Side A/B Listening Workbench & Model Explorer for PAWP-TTS."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from pawp_tts.application.frontend_service import PhonemizeUtteranceUseCase
from pawp_tts.domain.entities.models import LanguageCode
from pawp_tts.domain.phonology.codemix import build_language_masks
from pawp_tts.infrastructure.neural.bilingual_tts import PAWPBilingualTTS
from pawp_tts.infrastructure.persistence.config_loader import load_student_config
from pawp_tts.infrastructure.training.trainer import CheckpointManager
from pawp_tts.interfaces.container import build_container

# Use CUDA if available for instant real-time synthesis, with CPU fallback
DEVICE = torch.device(os.environ.get("PAWP_GRADIO_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
NOTES_PATH = Path("artifacts/reports/human_listening_notes.jsonl")

# Model Registry: label -> (config_path, ckpt_path, detailed_model_mention)
MODEL_REGISTRY: dict[str, tuple[str, str, str]] = {
    "PAWP-1.5M v3 (1.49M params — Warm Flagship)": (
        "configs/student_1_5m.yaml",
        "artifacts/checkpoints/exp015_1_5m_v3/best.safetensors",
        "1,490,390 params (5.98 MB FP32). Warm, expressive, natural flagship model with rich vocal resonance, 0.37% WER, and sub-4ms TTFA.",
    ),
    "Dense-300K v3 (281K params — Global #1 Speed King)": (
        "configs/student_300k_dense.yaml",
        "artifacts/checkpoints/exp015_300k_dense_v3/best.safetensors",
        "281,426 params (1.13 MB FP32, 5.3x smaller). Global #1 single-thread RTFx (245x-315x on CPU), sub-4.0 ms cloud streaming TTFA. Won decisively on phoneme clarity (5.24% EN / 5.54% NE PER).",
    ),
    "PAWP-300K v3 Shared (281K unique — Shared-Basis Twin)": (
        "configs/student_300k_pawp.yaml",
        "artifacts/checkpoints/exp015_300k_shared_v3/best.safetensors",
        "281,078 unique / 328,158 total params (1.12 MB FP32, K=2 shared weight bases). Demonstrates weight-space basis sharing with low-rank language residuals. Slightly smoother code-mix transitions.",
    ),
    "PAWP-1.5M v3 + Pause WSP-96 (96 Weights Edited)": (
        "configs/student_1_5m.yaml",
        "artifacts/checkpoints/exp016_pause_wsp/wsp_pause96.safetensors",
        "1,490,390 params. Surgical weight-space edit on 1.5M: only 96 weights (0.0064% of model) edited to add ~10 ms wider inter-word pauses without re-training.",
    ),
    "PAWP-1.5M v3 + Retroflex FT-0.4% (6K Weights Fine-Tuned)": (
        "configs/student_1_5m.yaml",
        "artifacts/checkpoints/exp017_retroflex_wsp/control_sameparam_ne_ft.safetensors",
        "1,490,390 params. 6,144 Nepali-only residual weights fine-tuned 60 steps for retroflex ण (ɳ) vs न (n̪) contrast.",
    ),
    "PAWP-1.5M v3 + Retroflex ROME Rank-1 (Zero-Training Edit)": (
        "configs/student_1_5m.yaml",
        "artifacts/checkpoints/exp017_retroflex_wsp/wsp_rome_ne_rank1.safetensors",
        "1,490,390 params. Zero-training closed-form Rank-1 ROME edit in 6,144 Nepali residual params (0.000% English drift).",
    ),
}

AVAILABLE_MODELS: dict[str, tuple[str, str, str]] = {
    label: info for label, info in MODEL_REGISTRY.items() if Path(info[1]).exists()
}
DEFAULT_MODEL_LABEL = next(iter(AVAILABLE_MODELS.keys()))

GUIDED_PRESETS: dict[str, dict[str, str]] = {
    "Comparison 1: Flagship 1.5M vs Dense-300K (5.3x smaller)": {
        "model_a": "PAWP-1.5M v3 (1.49M params — Warm Flagship)",
        "model_b": "Dense-300K v3 (281K params — Global #1 Speed King)",
        "guide": (
            "**What to listen for:** Compare the flagship **1.49M params (5.98 MB)** warm model against the ultra-compact **281K params (1.13 MB)** model. "
            "Notice the warm vocal resonance on 1.5M and how remarkably clear and intelligible the tiny 300K model remains."
        ),
        "default_text": "The sun dipped below the horizon, painting the sky with brilliant shades of orange and purple.",
    },
    "Comparison 2: Dense-300K vs Shared-300K (Both ~281K params)": {
        "model_a": "Dense-300K v3 (281K params — Global #1 Speed King)",
        "model_b": "PAWP-300K v3 Shared (281K unique — Shared-Basis Twin)",
        "guide": (
            "**What to listen for:** Both models operate at ~281K params. Dense-300K won on individual phoneme clarity (5.24% PER), "
            "while Shared-300K uses K=2 shared weight bases and edges ahead on code-mix transitions. Test both on English, Nepali, and code-mixed text."
        ),
        "default_text": "Flight ticket confirm भइसक्यो, बाँकी details म तिमीलाई WhatsApp मा पठाइदिन्छु।",
    },
    "Comparison 3: Pause Check — 1.5M Flagship vs Pause WSP-96": {
        "model_a": "PAWP-1.5M v3 (1.49M params — Warm Flagship)",
        "model_b": "PAWP-1.5M v3 + Pause WSP-96 (96 Weights Edited)",
        "guide": (
            "**What to listen for:** Model B has only 96 weights edited to create ~10 ms wider inter-word pauses. "
            "Listen to whether B feels more relaxed and well-paced on a longer sentence."
        ),
        "default_text": "पाता र पत्ता, पना र पन्नाको उच्चारणमा फरक स्पष्ट सुनिन्छ।",
    },
    "Comparison 4: Phonetic Contrast — 1.5M Flagship vs Retroflex ROME Rank-1": {
        "model_a": "PAWP-1.5M v3 (1.49M params — Warm Flagship)",
        "model_b": "PAWP-1.5M v3 + Retroflex ROME Rank-1 (Zero-Training Edit)",
        "guide": (
            "**What to listen for:** Model B incorporates a closed-form Rank-1 weight update to emphasize the retroflex ण (ɳ) vs dental न (n̪) contrast without altering English weights."
        ),
        "default_text": "नारायणगढको पूर्ण विवरण र कारण जान्न म अहिले कार्यालय जाँदै छु।",
    },
}

AVAILABLE_PRESETS = {
    k: v
    for k, v in GUIDED_PRESETS.items()
    if v["model_a"] in AVAILABLE_MODELS and v["model_b"] in AVAILABLE_MODELS
}

_LOADED_MODELS: dict[str, tuple[float, PAWPBilingualTTS, int]] = {}


def _get_model(label: str) -> tuple[PAWPBilingualTTS, int]:
    cfg_path, ckpt_path, _ = AVAILABLE_MODELS[label]
    mtime = Path(ckpt_path).stat().st_mtime
    cached = _LOADED_MODELS.get(label)
    if cached is None or cached[0] != mtime:
        cfg = load_student_config(cfg_path)
        m = PAWPBilingualTTS(cfg).to(DEVICE).eval()
        CheckpointManager.load(m, ckpt_path)
        n_params = sum(p.numel() for p in m.parameters())
        _LOADED_MODELS[label] = (mtime, m, n_params)
    return _LOADED_MODELS[label][1], _LOADED_MODELS[label][2]


container = build_container()
phonemize_usecase: PhonemizeUtteranceUseCase = container.phonemize
_get_model(DEFAULT_MODEL_LABEL)


def _synth_one(
    text: str,
    model_label: str,
    pacing_factor: float = 1.04,
    comfort_padding: bool = True,
) -> tuple[tuple[int, np.ndarray], str, str]:
    cleaned = text.strip()
    if not cleaned:
        return (24000, np.zeros(2400, dtype=np.float32)), "No input text.", ""

    t0 = time.perf_counter()
    model, n_params = _get_model(model_label)
    seq = phonemize_usecase.execute(cleaned)
    phonemes_str = " ".join(
        f"/{t.symbol}/" for t in seq.tokens if t.symbol not in {"<bos>", "<eos>"}
    )
    tokens = seq.tokens
    lang_map = {
        LanguageCode.SHARED: 0,
        LanguageCode.EN: 1,
        LanguageCode.NE: 2,
        LanguageCode.MIXED: 3,
    }
    p_ids = torch.tensor([[t.token_id for t in tokens]], dtype=torch.long, device=DEVICE)
    l_ids = torch.tensor([[lang_map.get(t.language, 0) for t in tokens]], dtype=torch.long, device=DEVICE)
    s_ids = torch.tensor([[min(t.stress, 3) for t in tokens]], dtype=torch.long, device=DEVICE)
    b_flags = torch.tensor(
        [
            [
                [
                    1.0 if t.is_word_start else 0.0,
                    1.0 if t.is_word_end else 0.0,
                    1.0 if t.is_syllable_start else 0.0,
                    1.0 if t.is_phrase_break else 0.0,
                ]
                for t in tokens
            ]
        ],
        dtype=torch.float32,
        device=DEVICE,
    )
    masks = torch.tensor([build_language_masks(tokens)], dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        _, pred_dur = model.encode_tokens(
            p_ids,
            l_ids,
            s_ids,
            b_flags,
            lang_masks=masks if model.cfg.residual_rank > 0 else None,
        )
        scaled_dur = pred_dur * float(pacing_factor) if pacing_factor != 1.0 else pred_dur
        out = model(
            phoneme_ids=p_ids,
            lang_ids=l_ids,
            stress_ids=s_ids,
            boundary_flags=b_flags,
            lang_masks=masks,
            target_durations=scaled_dur,
        )
        wav = out.waveform[0].detach().cpu().numpy()

    if comfort_padding:
        pad = np.zeros(int(model.cfg.sample_rate * 0.08), dtype=np.float32)
        wav = np.concatenate([pad, wav, pad])

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    dur_s = len(wav) / 24000.0
    rtfx = dur_s / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0
    rms = float(np.sqrt(np.mean(wav**2))) if len(wav) else 0.0
    digest = hashlib.sha1(wav.tobytes()).hexdigest()[:8]
    info = (
        f"{model_label}\n"
        f"• Unique Parameters: {n_params:,}\n"
        f"• Audio Duration: {dur_s:.2f}s | Device: {DEVICE.type.upper()}\n"
        f"• Latency: {elapsed_ms:.1f} ms | RTFx: {rtfx:.1f}x\n"
        f"• RMS: {rms:.4f} | Hash: {digest}"
    )
    return (24000, wav), info, phonemes_str


def _load_notes_markdown() -> str:
    if not NOTES_PATH.exists():
        return "_No listening notes saved yet._"
    lines = [ln.strip() for ln in NOTES_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return "_No listening notes saved yet._"
    rows = [
        "| # | Time | Mode / Preset | Verdict | Text (snippet) | Comment |",
        "|---:|---|---|---|---|---|",
    ]
    for idx, ln in enumerate(reversed(lines[-25:]), start=1):
        rec = json.loads(ln)
        t_short = rec.get("timestamp_utc", "")[11:19]
        preset = rec.get("preset", rec.get("model_a", ""))
        verdict = rec.get("verdict", "-")
        txt = rec.get("text", "").replace("|", "/")[:40]
        cmt = rec.get("comment", "").replace("|", "/").replace("\n", " ")
        rows.append(f"| {idx} | {t_short} | {preset} | **{verdict}** | {txt} | {cmt} |")
    return "\n".join(rows)


def on_preset_change(preset_name: str) -> tuple[str, str, str, str]:
    p = AVAILABLE_PRESETS[preset_name]
    return p["model_a"], p["model_b"], p["guide"], p["default_text"]


def synth_ab(
    text: str,
    model_a: str,
    model_b: str,
    pacing: float,
    padding: bool,
) -> tuple[tuple[int, np.ndarray], str, tuple[int, np.ndarray], str, str]:
    aud_a, info_a, phon = _synth_one(text, model_a, pacing_factor=pacing, comfort_padding=padding)
    aud_b, info_b, _ = _synth_one(text, model_b, pacing_factor=pacing, comfort_padding=padding)
    return aud_a, info_a, aud_b, info_b, phon


def save_ab_note(
    preset: str, model_a: str, model_b: str, text: str, verdict: str, comment: str
) -> tuple[str, str, str]:
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "mode": "ab_guided",
        "preset": preset,
        "model_a": model_a,
        "model_b": model_b,
        "text": text.strip(),
        "verdict": verdict,
        "comment": comment.strip(),
    }
    with NOTES_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return (
        f"Saved note ({verdict}) to {NOTES_PATH}",
        "",
        _load_notes_markdown(),
    )


def on_single_model_change(model_label: str) -> str:
    return f"**About this checkpoint:** {AVAILABLE_MODELS[model_label][2]}"


def save_single_note(model_label: str, text: str, comment: str) -> tuple[str, str, str]:
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "mode": "single",
        "preset": model_label,
        "model_a": model_label,
        "model_b": "",
        "text": text.strip(),
        "verdict": "Note",
        "comment": comment.strip(),
    }
    with NOTES_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return (
        f"Saved note to {NOTES_PATH}",
        "",
        _load_notes_markdown(),
    )


TEXT_EXAMPLES = [
    ["Welcome to PAWP-TTS version two. Ultra-low latency bilingual speech synthesis."],
    ["The sun dipped below the horizon, painting the sky with brilliant shades of orange and purple."],
    ["Neural text-to-speech models require careful weight-space programming and low-latency streaming inference."],
    ["Artificial intelligence transforms communication across different languages."],
    ["नेपालको प्राकृतिक सौन्दर्य र अग्ला हिमाल हेर्न संसारभरिबाट मानिसहरू आउँछन्।"],
    ["तातो चिया र च्याउको सुपले जाडो मौसममा शरीरलाई निकै आराम दिन्छ।"],
    ["पाता र पत्ता, पना र पन्नाको उच्चारणमा फरक स्पष्ट सुनिन्छ।"],
    ["नारायणगढको पूर्ण विवरण र कारण जान्न म अहिले कार्यालय जाँदै छु।"],
    ["Flight ticket confirm भइसक्यो, बाँकी details म तिमीलाई WhatsApp मा पठाइदिन्छु।"],
    ["आज हाम्रो team ले नयाँ AI model deploy गर्‍यो, performance एकदमै smooth देखिन्छ।"],
]

first_preset = next(iter(AVAILABLE_PRESETS.keys()))
first_p = AVAILABLE_PRESETS[first_preset]

custom_css = """
.model-box { border-radius: 8px; padding: 12px; margin-bottom: 8px; }
.model-card { font-size: 0.95em; line-height: 1.5; }
"""

with gr.Blocks(title="PAWP-TTS: Guided A/B Listening & Model Workbench", css=custom_css) as demo:
    gr.Markdown(
        "# PAWP-TTS: Guided Model Workbench & A/B Listening Test\n\n"
        "Explore and listen to the **PAWP-TTS** bilingual (English/Nepali) models side-by-side.\n\n"
        "### Available Models:\n"
        "- **PAWP-1.5M v3 (1.49M params)**: High-fidelity flagship (5.98 MB FP32), 0.37% WER on MiniMax, 115x single-thread RTFx.\n"
        "- **Dense-300K v3 (281K params)**: Ultra-compact speed king (1.13 MB FP32), Global #1 RTFx (245x-315x on CPU), sub-4ms cloud streaming TTFA.\n"
        "- **PAWP-300K v3 Shared (281K params)**: K=2 shared weight bases + low-rank language residuals for compact parameter efficiency.\n"
        "- **PAWP-1.5M v3 + WSP Edits**: Research models with surgical weight-space modifications (96-parameter pause edit and ROME Rank-1 retroflex edit)."
    )

    with gr.Tabs():
        with gr.Tab("1. Guided A/B Listening (Side-by-Side)"):
            preset_dd = gr.Dropdown(
                choices=list(AVAILABLE_PRESETS.keys()),
                value=first_preset,
                label="Step 1: Select Comparison Preset",
            )
            guide_md = gr.Markdown(first_p["guide"])

            with gr.Row():
                ab_model_a = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS.keys()),
                    value=first_p["model_a"],
                    label="Model A",
                )
                ab_model_b = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS.keys()),
                    value=first_p["model_b"],
                    label="Model B",
                )

            ab_text = gr.Textbox(
                label="Step 2: Input Text (English, Nepali, or Code-Mixed)",
                value=first_p["default_text"],
                lines=2,
            )
            gr.Examples(examples=TEXT_EXAMPLES, inputs=[ab_text], cache_examples=False)

            with gr.Row():
                ab_pacing = gr.Slider(
                    minimum=0.85,
                    maximum=1.25,
                    value=1.04,
                    step=0.01,
                    label="Pacing Calibration Factor (1.04 = calibrated relaxed pacing)",
                )
                ab_padding = gr.Checkbox(
                    value=True,
                    label="Comfort Silence Padding (80 ms boundary margin for clean ASR/naturalness)",
                )

            ab_btn = gr.Button("Generate Side-by-Side Audio", variant="primary")

            with gr.Row():
                with gr.Column():
                    ab_audio_a = gr.Audio(label="Model A Audio (24 kHz)", type="numpy")
                    ab_info_a = gr.Textbox(label="Model A Performance & Metadata", interactive=False, lines=4)
                with gr.Column():
                    ab_audio_b = gr.Audio(label="Model B Audio (24 kHz)", type="numpy")
                    ab_info_b = gr.Textbox(label="Model B Performance & Metadata", interactive=False, lines=4)

            ab_phon = gr.Textbox(label="Phoneme Sequence from In-Memory Fast Lexicon", interactive=False)

            gr.Markdown("### Step 3: Record Listening Note (Optional)")
            with gr.Row():
                ab_verdict = gr.Radio(
                    choices=["Prefer Model A", "Prefer Model B", "Tie (Both Great)", "Both Have Artifacts"],
                    value="Prefer Model A",
                    label="Perceptual Preference",
                )
                ab_comment = gr.Textbox(
                    label="Observations & Feedback",
                    placeholder="e.g., Model A is noticeably crisper on stops; Model B has slightly warmer tone...",
                    lines=2,
                )
            ab_save_btn = gr.Button("Save Listening Observation", variant="secondary")
            ab_save_status = gr.Markdown("")

        with gr.Tab("2. Single Model Explorer"):
            s_model = gr.Dropdown(
                choices=list(AVAILABLE_MODELS.keys()),
                value=DEFAULT_MODEL_LABEL,
                label="Select Model",
            )
            s_desc = gr.Markdown(on_single_model_change(DEFAULT_MODEL_LABEL))
            s_text = gr.Textbox(
                label="Input Text",
                value=TEXT_EXAMPLES[0][0],
                lines=2,
            )
            gr.Examples(examples=TEXT_EXAMPLES, inputs=[s_text], cache_examples=False)

            with gr.Row():
                s_pacing = gr.Slider(
                    minimum=0.85,
                    maximum=1.25,
                    value=1.04,
                    step=0.01,
                    label="Pacing Calibration Factor",
                )
                s_padding = gr.Checkbox(
                    value=True,
                    label="Comfort Silence Padding (80 ms)",
                )

            s_btn = gr.Button("Synthesize Speech", variant="primary")
            s_audio = gr.Audio(label="Synthesized 24 kHz Speech", type="numpy", autoplay=True)
            s_info = gr.Textbox(label="Model Performance & Footprint", interactive=False, lines=4)
            s_phon = gr.Textbox(label="Phoneme Sequence", interactive=False)

            with gr.Row():
                s_comment = gr.Textbox(
                    label="Observation Note",
                    placeholder="Record notes on pronunciation, cadence, or clarity...",
                    lines=2,
                )
                s_save_btn = gr.Button("Save Note")
            s_save_status = gr.Markdown("")

    gr.Markdown("### Saved Listening Notes History")
    notes_md = gr.Markdown(_load_notes_markdown())

    # Event Handlers
    preset_dd.change(
        fn=on_preset_change,
        inputs=[preset_dd],
        outputs=[ab_model_a, ab_model_b, guide_md, ab_text],
    )
    ab_btn.click(
        fn=synth_ab,
        inputs=[ab_text, ab_model_a, ab_model_b, ab_pacing, ab_padding],
        outputs=[ab_audio_a, ab_info_a, ab_audio_b, ab_info_b, ab_phon],
    )
    ab_save_btn.click(
        fn=save_ab_note,
        inputs=[preset_dd, ab_model_a, ab_model_b, ab_text, ab_verdict, ab_comment],
        outputs=[ab_save_status, ab_comment, notes_md],
    )
    s_model.change(fn=on_single_model_change, inputs=[s_model], outputs=[s_desc])
    s_btn.click(
        fn=_synth_one,
        inputs=[s_text, s_model, s_pacing, s_padding],
        outputs=[s_audio, s_info, s_phon],
    )
    s_save_btn.click(
        fn=save_single_note,
        inputs=[s_model, s_text, s_comment],
        outputs=[s_save_status, s_comment, notes_md],
    )

if __name__ == "__main__":
    port = int(os.environ.get("PAWP_GRADIO_PORT", "7861"))
    print(f"Launching PAWP-TTS Gradio Workbench on port {port}...")
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
