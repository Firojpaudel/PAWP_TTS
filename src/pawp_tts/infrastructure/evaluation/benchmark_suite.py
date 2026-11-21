"""Bilingual (EN, NE, Code-Mix) and NepTTS-Bench 77-minimal-pair evaluation harness."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import torch

from pawp_tts.application.frontend_service import PhonemizeUtteranceUseCase
from pawp_tts.domain.entities.models import LanguageCode
from pawp_tts.domain.evaluation.metrics import compute_cer, compute_wer
from pawp_tts.domain.evaluation.neptts_bench import NEPTTS_MINIMAL_PAIRS
from pawp_tts.domain.phonology.codemix import build_language_masks
from pawp_tts.domain.phonology.inventory import SYMBOL_TO_ID
from pawp_tts.domain.ports.interfaces import AsrEvaluatorPort
from pawp_tts.infrastructure.neural.bilingual_tts import PAWPBilingualTTS
from pawp_tts.infrastructure.persistence.config_loader import load_student_config
from pawp_tts.infrastructure.training.trainer import CheckpointManager

_SUITE_PROMPTS: dict[str, tuple[str, ...]] = {
    "en": (
        "She sells thin shells by the vine and wine.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence speech synthesis requires crisp pronunciation.",
        "Please confirm your account balance and transfer fifty dollars today.",
        "Modern neural acoustic decoders convert spectral frames into crisp waveforms.",
        "Customer service operates twenty four seven with low latency.",
        "Verify the security code before approving the online transaction.",
        "High fidelity bilingual voice synthesis preserves speaker identity.",
        "Compact weight space models run fast on edge processors.",
        "Natural sentence rhythm and clear consonant articulation matter.",
    ),
    "ne": (
        "आज मौसम धेरै राम्रो र सफा छ।",
        "शहरमा सस्तो सामान किन्न मानिसहरू बजार गए।",
        "पाता र पत्ताको उच्चारण स्पष्ट हुनुपर्छ।",
        "तीन दिन घर जान दिन नपाएर मित्र रोयो।",
        "नेपालका अग्ला पहाड र सडकहरू सुन्दर छन्।",
        "चार जना साथीहरूले चिया र खाजा खाए।",
        "मेरो खातामा एक सय पच्चिस रुपैयाँ बाँकी छ।",
        "शिक्षा र स्वास्थ्य सबै नागरिकको आधारभूत अधिकार हो।",
        "काठमाडौँ उपत्यकामा आज बिहान चिसो हावा चल्यो।",
        "हामीले समयमै काम सम्पन्न गर्नुपर्छ।",
    ),
    "codemix": (
        "आज मेरो officeमा meeting छ।",
        "मेरो बैंक balance check-गरिदिनुहोस्।",
        "मेरो 500 रुपैयाँको bill तिर्न बाँकी छ।",
        "Please मेरो mobile app मा 50% discount कोड पठाइदिनुहोस्।",
        "तिम्रो SMS र 4G data सक्रिय भयो?",
        "दुवै जनाको narrative अहिलेसम्म पनि different छ।",
        "यो project को deadline भोलि बेलुका सम्म मात्र छ।",
        "हाम्रो system ले real time streaming audio सपोर्ट गर्छ।",
        "Online payment गर्दा QR कोड scan गर्नुहोस्।",
        "आजको podcast episode धेरै informative थियो।",
    ),
}


def _synthesize_utterance(
    model: PAWPBilingualTTS,
    phonemize: PhonemizeUtteranceUseCase,
    text: str,
    device: torch.device,
    speaker_id: int | None = None,
) -> tuple[tuple[str, ...], tuple[int, ...], np.ndarray, np.ndarray]:
    seq = phonemize.execute(text)
    tokens = seq.tokens
    lang_map = {
        LanguageCode.SHARED: 0,
        LanguageCode.EN: 1,
        LanguageCode.NE: 2,
        LanguageCode.MIXED: 3,
    }
    p_ids = torch.tensor([[t.token_id for t in tokens]], dtype=torch.long, device=device)
    l_ids = torch.tensor(
        [[lang_map.get(t.language, 0) for t in tokens]], dtype=torch.long, device=device
    )
    s_ids = torch.tensor([[min(t.stress, 3) for t in tokens]], dtype=torch.long, device=device)
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
        device=device,
    )
    masks = torch.tensor([build_language_masks(tokens)], dtype=torch.float32, device=device)

    with torch.no_grad():
        out = model(
            phoneme_ids=p_ids,
            lang_ids=l_ids,
            stress_ids=s_ids,
            boundary_flags=b_flags,
            lang_masks=masks,
            speaker_ids=None
            if speaker_id is None
            else torch.tensor([speaker_id], dtype=torch.long, device=device),
        )
    wav = out.waveform[0].detach().cpu().numpy()
    mel = out.mel[0].detach().cpu().numpy()
    return seq.symbols, seq.token_ids, wav, mel


def run_evaluation_suite(
    suite: Literal["en", "ne", "neptts", "codemix"],
    config_path: str | Path,
    checkpoint_path: str | Path | None,
    phonemize: PhonemizeUtteranceUseCase,
    asr: AsrEvaluatorPort | None = None,
    live_asr: bool = False,
    out_dir: str | Path | None = None,
) -> dict[str, float | int | str]:
    """
    Evaluate phonological contrast accuracy, synthesis stability, and optional ASR WER/CER
    across `en`, `ne`, `neptts` (77 minimal pairs), or `codemix`.
    """
    cfg = load_student_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PAWPBilingualTTS(cfg).to(device).eval()

    ckpt = (
        Path(checkpoint_path) if checkpoint_path else Path("artifacts/checkpoints/best.safetensors")
    )
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt} (refusing to evaluate random weights)"
        )
    CheckpointManager.load(model, ckpt)  # strict; raises on any shape/key mismatch
    ckpt_loaded = True

    unk_id = SYMBOL_TO_ID["<unk>"]

    if suite == "neptts":
        total_pairs = len(NEPTTS_MINIMAL_PAIRS)
        phon_pass = 0
        acoustic_pass = 0
        zero_unk_pairs = 0
        sibilant_clean = 0

        for pair in NEPTTS_MINIMAL_PAIRS:
            syms_a, ids_a, wav_a, mel_a = _synthesize_utterance(
                model, phonemize, pair.word_a, device
            )
            syms_b, ids_b, wav_b, mel_b = _synthesize_utterance(
                model, phonemize, pair.word_b, device
            )

            if unk_id not in ids_a and unk_id not in ids_b:
                zero_unk_pairs += 1

            if pair.category != "english_and_codemix_contrasts":
                if "ʃ" not in syms_a and "ʃ" not in syms_b:
                    sibilant_clean += 1
            else:
                sibilant_clean += 1

            is_distinct = syms_a != syms_b
            if is_distinct == pair.expect_distinct_phonemes:
                phon_pass += 1

            wav_ok = (
                wav_a.size > 0
                and wav_b.size > 0
                and bool(np.all(np.isfinite(wav_a)))
                and bool(np.all(np.isfinite(wav_b)))
                and float(np.std(wav_a)) > 1e-5
                and float(np.std(wav_b)) > 1e-5
            )
            if pair.expect_distinct_phonemes:
                min_t = min(mel_a.shape[0], mel_b.shape[0])
                mel_diff = (
                    float(np.mean(np.abs(mel_a[:min_t] - mel_b[:min_t]))) if min_t > 0 else 0.0
                )
                mel_distinct = (mel_a.shape[0] != mel_b.shape[0]) or (mel_diff > 1e-5)
            else:
                mel_distinct = True

            if wav_ok and mel_distinct:
                acoustic_pass += 1

        return {
            "suite": "neptts",
            "note": "internal G2P/audio smoke test; NOT the published NepTTS-Bench; no listening/ASR",
            "checkpoint_loaded": int(ckpt_loaded),
            "total_pairs": total_pairs,
            "frontend_g2p_contrast_rate": round(phon_pass / max(total_pairs, 1), 4),
            "audio_nonsilent_and_mel_differs_rate": round(acoustic_pass / max(total_pairs, 1), 4),
            "zero_unk_rate": round(zero_unk_pairs / max(total_pairs, 1), 4),
            "frontend_no_sh_symbol_rate": round(sibilant_clean / max(total_pairs, 1), 4),
        }

    prompts = _SUITE_PROMPTS[suite]
    valid_wavs = 0
    zero_unk = 0
    total_dur_sec = 0.0
    rms_vals: list[float] = []
    wer_vals: list[float] = []
    cer_vals: list[float] = []

    asr_lang = LanguageCode.EN if suite == "en" else LanguageCode.NE

    for idx, prompt in enumerate(prompts):
        _, ids, wav, _ = _synthesize_utterance(model, phonemize, prompt, device)
        if unk_id not in ids:
            zero_unk += 1
        finite_and_active = (
            wav.size > 0 and bool(np.all(np.isfinite(wav))) and float(np.std(wav)) > 1e-5
        )
        if finite_and_active:
            valid_wavs += 1
            total_dur_sec += wav.size / float(cfg.sample_rate)
            rms_vals.append(float(np.sqrt(np.mean(np.square(wav)))))

        if out_dir is not None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            sf.write(Path(out_dir) / f"{suite}_{idx:02d}.wav", wav, cfg.sample_rate)
        if live_asr and asr is not None and finite_and_active:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, wav, cfg.sample_rate)
                tx = asr.transcribe(tmp.name, language=asr_lang)
                wer_vals.append(compute_wer(prompt, tx.text).error_rate)
                cer_vals.append(compute_cer(prompt, tx.text).error_rate)

    n_p = max(len(prompts), 1)
    result: dict[str, float | int | str] = {
        "suite": suite,
        "checkpoint_loaded": int(ckpt_loaded),
        "num_prompts": len(prompts),
        "audio_nonsilent_rate": round(valid_wavs / n_p, 4),
        "zero_unk_rate": round(zero_unk / n_p, 4),
        "total_audio_sec": round(total_dur_sec, 2),
        "mean_waveform_rms": round(float(np.mean(rms_vals)) if rms_vals else 0.0, 5),
    }
    if wer_vals:
        result["asr_scored_prompts"] = len(wer_vals)
        result["live_asr_wer"] = round(float(np.mean(wer_vals)), 4)
        result["live_asr_cer"] = round(float(np.mean(cer_vals)), 4)
    return result


def run_heldout_asr_eval(
    config_path: str | Path,
    checkpoint_path: str | Path,
    phonemize: PhonemizeUtteranceUseCase,
    asr: AsrEvaluatorPort,
    manifest_path: str | Path = "data/manifests/test.jsonl",
    sources_dir: str | Path = "data/sources",
    per_language: int = 30,
    seed: int = 0,
    out_dir: str | Path | None = None,
    asr_mixed: AsrEvaluatorPort | None = None,
) -> dict[str, float | int | str]:
    """Synthesize held-out test sentences and score them with ASR (WER/CER).

    The same ASR also transcribes the original human recordings of those sentences, giving an
    ASR-error floor (`gt_*`) so TTS errors can be separated from recognizer errors.
    """
    import json
    import random

    from pawp_tts.infrastructure.data.bulk_cache import _read_audio
    from pawp_tts.infrastructure.data.feature_extractor import resample_to_24k

    cfg = load_student_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PAWPBilingualTTS(cfg).to(device).eval()
    CheckpointManager.load(model, checkpoint_path)

    rows = [
        json.loads(x)
        for x in Path(manifest_path).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    rows = [r for r in rows if 1.5 <= float(r["duration_sec"]) <= 10.0]
    random.Random(seed).shuffle(rows)
    picked: dict[str, list[dict[str, object]]] = {"en": [], "ne": [], "mixed": []}
    for r in rows:
        lang = str(r["language"])
        if lang in picked and len(picked[lang]) < per_language:
            picked[lang].append(r)

    result: dict[str, float | int | str] = {"checkpoint": str(checkpoint_path)}
    for lang, items in picked.items():
        asr_lang = LanguageCode.EN if lang == "en" else LanguageCode.NE
        asr_used = asr_mixed if (lang == "mixed" and asr_mixed is not None) else asr
        result[f"{lang}_asr"] = "mixed_script_asr" if asr_used is asr_mixed else "primary_asr"
        acc: dict[str, list[float]] = {"wer": [], "cer": [], "gt_wer": [], "gt_cer": []}
        for i, r in enumerate(items):
            text = str(r["text"])
            _, _, wav, _ = _synthesize_utterance(model, phonemize, text, device)
            with tempfile.TemporaryDirectory() as td:
                syn_p = Path(td) / "syn.wav"
                sf.write(syn_p, wav, cfg.sample_rate)
                tx = asr_used.transcribe(str(syn_p), language=asr_lang).text
                acc["wer"].append(compute_wer(text, tx).error_rate)
                acc["cer"].append(compute_cer(text, tx).error_rate)
                gt_raw, gt_sr = _read_audio(dict(r), Path(sources_dir))
                gt_p = Path(td) / "gt.wav"
                sf.write(gt_p, resample_to_24k(gt_raw, gt_sr), 24000)
                gtx = asr_used.transcribe(str(gt_p), language=asr_lang).text
                acc["gt_wer"].append(compute_wer(text, gtx).error_rate)
                acc["gt_cer"].append(compute_cer(text, gtx).error_rate)
            if out_dir is not None and i < 5:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                sf.write(Path(out_dir) / f"heldout_{lang}_{i:02d}.wav", wav, cfg.sample_rate)
        result[f"{lang}_n"] = len(items)
        for k, v in acc.items():
            result[f"{lang}_{k}"] = round(float(np.mean(v)), 4) if v else -1.0
    return result
