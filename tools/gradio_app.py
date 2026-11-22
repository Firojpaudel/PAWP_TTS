"""Guided Side-by-Side A/B Listening Workbench & Note Logger for PAWP-TTS."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from pawp_tts.infrastructure.evaluation.benchmark_suite import _synthesize_utterance
from pawp_tts.infrastructure.neural.bilingual_tts import PAWPBilingualTTS
from pawp_tts.infrastructure.persistence.config_loader import load_student_config
from pawp_tts.infrastructure.training.trainer import CheckpointManager
from pawp_tts.interfaces.container import build_container

# Default to CPU so Gradio uses 0 MB GPU VRAM
DEVICE = torch.device(os.environ.get("PAWP_GRADIO_DEVICE", "cpu"))
NOTES_PATH = Path("artifacts/reports/human_listening_notes.jsonl")

# label -> (config_path, ckpt_path, short_description)
MODEL_REGISTRY: dict[str, tuple[str, str, str]] = {
    "PAWP-1.5M v3 (EXP-015a — New Flagship)": (
        "configs/student_1_5m.yaml",
        "artifacts/checkpoints/exp015_1_5m_v3/best.safetensors",
        "1.49M params (5.98 MB). Trained on clean v3 alignment. Best overall numbers (EN CER 1.50%, NE CER 1.75%, NE PER 5.04%).",
    ),
    "Dense-300K v3 (EXP-015c — Best Tiny PER)": (
        "configs/student_300k_dense.yaml",
        "artifacts/checkpoints/exp015_300k_dense_v3/best.safetensors",
        "281.4K params (1.13 MB, 5.3x smaller, 0 shared bases). Statistically beat Shared-300K on 600 test sentences (NE PER 5.54%, EN PER 5.24%).",
    ),
    "PAWP-300K v3 Shared (EXP-015b — Shared-Basis Twin)": (
        "configs/student_300k_pawp.yaml",
        "artifacts/checkpoints/exp015_300k_shared_v3/best.safetensors",
        "281.1K unique / 328.2K total params (1.12 MB, K=2 shared bases). Slightly better Code-Mix ASR (12.0%) & val_mel; higher monolingual PER (6.28% NE / 7.09% EN).",
    ),
    "PAWP-1.5M v2 Polished (EXP-012 — Current HF Release)": (
        "configs/student_1_5m.yaml",
        "artifacts/checkpoints/exp012_polish/best.safetensors",
        "1.49M params (previous v2 cache). Currently live on Firoj112/pawp-tts-1.5m. Use to verify v3 sounds as good or better before updating HF.",
    ),
    "PAWP-1.5M v3 + Pause WSP-96 (EXP-016 — 96-Param Edit)": (
        "configs/student_1_5m.yaml",
        "artifacts/checkpoints/exp016_pause_wsp/wsp_pause96.safetensors",
        "Research edit on 1.5M v3: only 96 weights (0.0064%) edited to add ~10 ms longer pauses between words.",
    ),
    "PAWP-1.5M v3 + Retroflex FT-0.4% (EXP-017)": (
        "configs/student_1_5m.yaml",
        "artifacts/checkpoints/exp017_retroflex_wsp/control_sameparam_ne_ft.safetensors",
        "Research edit on 1.5M v3: 6,144 Nepali-only residual params fine-tuned 60 steps for retroflex ण (ɳ) vs न (n̪).",
    ),
    "PAWP-1.5M v3 + Retroflex ROME Rank-1 (EXP-017)": (
        "configs/student_1_5m.yaml",
        "artifacts/checkpoints/exp017_retroflex_wsp/wsp_rome_ne_rank1.safetensors",
        "Research edit on 1.5M v3: zero-training ROME Rank-1 edit in 6,144 Nepali-only residual params (0.000% EN drift).",
    ),
    "PAWP-300K v2 Target Voice (EXP-011-FT — Old v2)": (
        "configs/student_300k_pawp.yaml",
        "artifacts/checkpoints/exp011_300k_ft/best.safetensors",
        "Older v2 281K checkpoint (100k steps). Superseded by v3 300K models.",
    ),
    "PAWP-1.5M + WSP E3 (EXP-009 — Old HF WSP)": (
        "configs/student_1_5m.yaml",
        "artifacts/checkpoints/exp009_wsp/pawp_1_5m_wsp_e3.safetensors",
        "Older v2 WSP variant currently on Firoj112/pawp-tts-1.5m-wsp.",
    ),
}

AVAILABLE_MODELS: dict[str, tuple[str, str, str]] = {
    label: info for label, info in MODEL_REGISTRY.items() if Path(info[1]).exists()
}
DEFAULT_MODEL_LABEL = next(iter(AVAILABLE_MODELS.keys()))

GUIDED_PRESETS: dict[str, dict[str, str]] = {
    "Decision A: Should we update HF 1.5M? (1.5M v3 vs 1.5M v2)": {
        "model_a": "PAWP-1.5M v3 (EXP-015a — New Flagship)",
        "model_b": "PAWP-1.5M v2 Polished (EXP-012 — Current HF Release)",
        "guide": (
            "**What to listen for:** Compare **A (new v3)** vs **B (current HF v2)** on 1 Nepali + 1 English sentence. "
            "Check if Nepali consonants (especially व/ब and word endings) are cleaner in A and whether voice warmth/naturalness is equal or better."
        ),
        "default_text": "नेपालको प्राकृतिक सौन्दर्य र अग्ला हिमाल हेर्न संसारभरिबाट मानिसहरू आउँछन्।",
    },
    "Decision B: Which 300K for HF? (Dense-300K v3 vs Shared-300K v3)": {
        "model_a": "Dense-300K v3 (EXP-015c — Best Tiny PER)",
        "model_b": "PAWP-300K v3 Shared (EXP-015b — Shared-Basis Twin)",
        "guide": (
            "**What to listen for:** Both are ~281K unique params (~1.1 MB). "
            "**A (Dense)** won decisively on 600-sentence phoneme clarity (PER); **B (Shared)** edged ahead on small Code-Mix ASR. "
            "Test 1 Nepali + 1 Code-Mixed sentence and note which sounds crisper and smoother."
        ),
        "default_text": "Flight ticket confirm भइसक्यो, बाँकी details म तिमीलाई WhatsApp मा पठाइदिन्छु।",
    },
    "Scale Trade-Off: 1.5M v3 vs Dense-300K v3 (5.3x smaller)": {
        "model_a": "PAWP-1.5M v3 (EXP-015a — New Flagship)",
        "model_b": "Dense-300K v3 (EXP-015c — Best Tiny PER)",
        "guide": (
            "**What to listen for:** Hear how close the tiny **281K (1.13 MB)** model gets to the **1.49M (5.98 MB)** flagship. "
            "Note any muffling, metallic artifacts, or pronunciation slips in B."
        ),
        "default_text": "तातो चिया र च्याउको सुपले जाडो मौसममा शरीरलाई निकै आराम दिन्छ।",
    },
    "Pause Check: 1.5M v3 vs Pause WSP-96 (96 weights edited)": {
        "model_a": "PAWP-1.5M v3 (EXP-015a — New Flagship)",
        "model_b": "PAWP-1.5M v3 + Pause WSP-96 (EXP-016 — 96-Param Edit)",
        "guide": (
            "**What to listen for:** **B** edits only 96 weights to add ~10 ms wider gaps between words. "
            "Check if B feels better paced or slightly choppy on a long Nepali sentence."
        ),
        "default_text": "पाता र पत्ता, पना र पन्नाको उच्चारणमा फरक स्पष्ट सुनिन्छ।",
    },
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
phonemize_usecase = container.phonemize
SPEAKER_ID = 0
_get_model(DEFAULT_MODEL_LABEL)


def _synth_one(text: str, model_label: str) -> tuple[tuple[int, np.ndarray], str, str]:
    cleaned = text.strip()
    if not cleaned:
        return (24000, np.zeros(2400, dtype=np.float32)), "No input text.", ""
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
    info = f"{model_label} | unique_params={n_params:,} | dur={len(wav) / 24000:.2f}s | rms={rms:.4f} | sha1={digest}"
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
    p = GUIDED_PRESETS[preset_name]
    return p["model_a"], p["model_b"], p["guide"], p["default_text"]


def synth_ab(
    text: str, model_a: str, model_b: str
) -> tuple[tuple[int, np.ndarray], str, tuple[int, np.ndarray], str, str]:
    aud_a, info_a, phon = _synth_one(text, model_a)
    aud_b, info_b, _ = _synth_one(text, model_b)
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
        f"Saved note to {NOTES_PATH} ({verdict}).",
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
        f"Saved note to {NOTES_PATH}.",
        "",
        _load_notes_markdown(),
    )


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

first_preset = next(iter(GUIDED_PRESETS.keys()))
first_p = GUIDED_PRESETS[first_preset]

with gr.Blocks(title="PAWP-TTS: Guided A/B Listening & Notes") as demo:
    gr.Markdown(
        "# PAWP-TTS: Guided A/B Listening Workbench\n"
        "Use **Tab 1 (Guided A/B)** to listen side-by-side without getting overwhelmed. "
        "Any verdict or comment you submit is saved to `artifacts/reports/human_listening_notes.jsonl` for analysis."
    )

    with gr.Tabs():
        with gr.Tab("1. Guided A/B Listening (Recommended)"):
            preset_dd = gr.Dropdown(
                choices=list(GUIDED_PRESETS.keys()),
                value=first_preset,
                label="Step 1: Pick a Guided Decision",
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
                label="Step 2: Input Text (Nepali, English, or Code-Mixed)",
                value=first_p["default_text"],
                lines=2,
            )
            gr.Examples(examples=TEXT_EXAMPLES, inputs=[ab_text], cache_examples=False)
            ab_btn = gr.Button("Generate Side-by-Side (Model A & Model B)", variant="primary")

            with gr.Row():
                with gr.Column():
                    ab_audio_a = gr.Audio(label="Model A Audio (24 kHz)", type="numpy")
                    ab_info_a = gr.Textbox(label="Model A Fingerprint", interactive=False)
                with gr.Column():
                    ab_audio_b = gr.Audio(label="Model B Audio (24 kHz)", type="numpy")
                    ab_info_b = gr.Textbox(label="Model B Fingerprint", interactive=False)

            ab_phon = gr.Textbox(label="Frontend Phoneme Sequence", interactive=False)

            gr.Markdown("### Step 3: Save Your Verdict & Comments")
            with gr.Row():
                ab_verdict = gr.Radio(
                    choices=["Prefer A", "Prefer B", "Tie (Same)", "Both Have Issues"],
                    value="Prefer A",
                    label="Which sounded better?",
                )
                ab_comment = gr.Textbox(
                    label="Your comments (pronunciation, naturalness, pauses, glitches...)",
                    placeholder="e.g., A is clearer on Nepali consonants, B sounds slightly muffled on English...",
                    lines=2,
                )
            ab_save_btn = gr.Button("Submit & Save Note", variant="secondary")
            ab_save_status = gr.Markdown("")

        with gr.Tab("2. Single Model Explorer (All Checkpoints)"):
            s_model = gr.Dropdown(
                choices=list(AVAILABLE_MODELS.keys()),
                value=DEFAULT_MODEL_LABEL,
                label="Model Checkpoint",
            )
            s_desc = gr.Markdown(on_single_model_change(DEFAULT_MODEL_LABEL))
            s_text = gr.Textbox(
                label="Input Text",
                value=TEXT_EXAMPLES[0][0],
                lines=2,
            )
            gr.Examples(examples=TEXT_EXAMPLES, inputs=[s_text], cache_examples=False)
            s_btn = gr.Button("Synthesize", variant="primary")
            s_audio = gr.Audio(label="Synthesized 24 kHz Speech", type="numpy", autoplay=True)
            s_info = gr.Textbox(label="Model & Audio Fingerprint", interactive=False)
            s_phon = gr.Textbox(label="Phoneme Sequence", interactive=False)

            with gr.Row():
                s_comment = gr.Textbox(
                    label="Quick note on this model/sentence",
                    placeholder="Write any observation here...",
                    lines=2,
                )
                s_save_btn = gr.Button("Save Note")
            s_save_status = gr.Markdown("")

    gr.Markdown("### Your Saved Listening Notes (latest 25)")
    notes_md = gr.Markdown(_load_notes_markdown())

    # Wiring
    preset_dd.change(
        fn=on_preset_change,
        inputs=[preset_dd],
        outputs=[ab_model_a, ab_model_b, guide_md, ab_text],
    )
    ab_btn.click(
        fn=synth_ab,
        inputs=[ab_text, ab_model_a, ab_model_b],
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
        inputs=[s_text, s_model],
        outputs=[s_audio, s_info, s_phon],
    )
    s_save_btn.click(
        fn=save_single_note,
        inputs=[s_model, s_text, s_comment],
        outputs=[s_save_status, s_comment, notes_md],
    )

if __name__ == "__main__":
    port = int(os.environ.get("PAWP_GRADIO_PORT", "7861"))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
