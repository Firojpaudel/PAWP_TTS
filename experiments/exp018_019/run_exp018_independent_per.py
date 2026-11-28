"""EXP-018 & EXP-019: Independent phoneme recognizer (facebook/wav2vec2-xlsr-53-espeak-cv-ft) PER + retroflex recheck."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from scipy import stats as scipy_stats
from scipy.signal import resample_poly
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

from pawp_tts.domain.entities.models import LanguageCode
from pawp_tts.domain.phonology.codemix import build_language_masks
from pawp_tts.infrastructure.data.bulk_cache import _read_audio
from pawp_tts.infrastructure.neural.bilingual_tts import PAWPBilingualTTS
from pawp_tts.infrastructure.persistence.config_loader import load_student_config
from pawp_tts.infrastructure.training.trainer import CheckpointManager
from pawp_tts.interfaces.container import build_container

MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"

CHECKPOINTS: list[tuple[str, str, str, int]] = [
    ("pawp_1_5m_v3", "configs/student_1_5m.yaml", "artifacts/checkpoints/exp015_1_5m_v3/best.safetensors", 42),
    ("shared_300k_s42", "configs/student_300k_pawp.yaml", "artifacts/checkpoints/exp015_300k_shared_v3/best.safetensors", 42),
    ("shared_300k_s101", "configs/student_300k_pawp.yaml", "artifacts/checkpoints/exp015_300k_shared_v3_s101/best.safetensors", 101),
    ("shared_300k_s202", "configs/student_300k_pawp.yaml", "artifacts/checkpoints/exp015_300k_shared_v3_s202/best.safetensors", 202),
    ("dense_300k_s42", "configs/student_300k_dense.yaml", "artifacts/checkpoints/exp015_300k_dense_v3/best.safetensors", 42),
    ("dense_300k_s101", "configs/student_300k_dense.yaml", "artifacts/checkpoints/exp015_300k_dense_v3_s101/best.safetensors", 101),
    ("dense_300k_s202", "configs/student_300k_dense.yaml", "artifacts/checkpoints/exp015_300k_dense_v3_s202/best.safetensors", 202),
]

IGNORE_TOKENS = {
    "<pad>", "<bos>", "<eos>", "<unk>", "<sp>", "<wb>", "<sb>", "<pb>",
    "<s>", "</s>", "|", ",", ".", "?", "!", ":", "-", "ˈ", "ˌ", "ː", "̃", "̤",
}

# Documented reduced shared phone set mapping (applied symmetrically to frontend & XLSR-eSpeak IPA)
REDUCED_PHONE_MAP: dict[str, list[str]] = {
    # Stops (dental/alveolar/retroflex and plain/aspirated/breathy collapsed to shared base stops)
    "p": ["p"], "pʰ": ["p"], "ph": ["p"],
    "b": ["b"], "bʱ": ["b"], "bh": ["b"],
    "t": ["t"], "t̪": ["t"], "t̪ʰ": ["t"], "ʈ": ["t"], "ʈʰ": ["t"], "th": ["t"], "tʲ": ["t"],
    "d": ["d"], "d̪": ["d"], "d̪ʱ": ["d"], "ɖ": ["d"], "ɖʱ": ["d"], "dh": ["d"], "dʲ": ["d"],
    "k": ["k"], "kʰ": ["k"], "kh": ["k"], "q": ["k"], "c": ["k"],
    "ɡ": ["g"], "g": ["g"], "ɡʱ": ["g"], "gh": ["g"], "ɟ": ["g"],
    # Affricates
    "tʃ": ["tʃ"], "tS": ["tʃ"], "tɕ": ["tʃ"], "tɕh": ["tʃ"], "tʃʲ": ["tʃ"],
    "dʒ": ["dʒ"], "dZ": ["dʒ"], "dʑ": ["dʒ"],
    "ts": ["ts"], "tsʰ": ["ts"], "tsh": ["ts"],
    "dz": ["dz"], "dzʱ": ["dz"], "dzh": ["dz"],
    # Fricatives
    "f": ["f"], "v": ["v"], "ʋ": ["v"],
    "θ": ["θ"], "ð": ["ð"],
    "s": ["s"], "z": ["z"],
    "ʃ": ["ʃ"], "ʂ": ["ʃ"], "ɕ": ["ʃ"], "S": ["ʃ"],
    "ʒ": ["ʒ"], "ʐ": ["ʒ"], "ʑ": ["ʒ"], "Z": ["ʒ"],
    "h": ["h"], "ɦ": ["h"], "x": ["h"], "χ": ["h"], "ħ": ["h"],
    # Nasals
    "m": ["m"], "mʲ": ["m"],
    "n": ["n"], "n̪": ["n"], "ɳ": ["n"], "ɲ": ["n"], "nʲ": ["n"],
    "ŋ": ["ŋ"],
    # Liquids & Glides
    "l": ["l"], "ɭ": ["l"], "ʎ": ["l"], "ɫ": ["l"],
    "r": ["r"], "ɾ": ["r"], "ɹ": ["r"], "ɽ": ["r"], "ɽʱ": ["r"], "ʁ": ["r"], "ʀ": ["r"], "ɻ": ["r"],
    "w": ["w"], "j": ["j"],
    # Monophthong vowels
    "i": ["i"], "y": ["i"],
    "ɪ": ["ɪ"], "ɨ": ["ɪ"], "ᵻ": ["ɪ"],
    "e": ["e"], "ø": ["e"],
    "ɛ": ["ɛ"], "œ": ["ɛ"],
    "æ": ["æ"],
    "a": ["a"], "ɑ": ["a"], "ɐ": ["a"], "ɒ": ["a"],
    "ʌ": ["ʌ"], "ɜ": ["ɜ"],
    "ə": ["ə"],
    "ɚ": ["ə", "r"], "ɝ": ["ɜ", "r"],
    "ɔ": ["ɔ"],
    "o": ["o"], "ɤ": ["o"],
    "ʊ": ["ʊ"],
    "u": ["u"], "ɯ": ["u"],
    # Diphthongs split into component vowels
    "aɪ": ["a", "i"], "aʊ": ["a", "u"], "eɪ": ["e", "i"], "oʊ": ["o", "u"], "ɔɪ": ["ɔ", "i"],
    "ɪə": ["ɪ", "ə"], "eə": ["ɛ", "ə"], "ʊə": ["ʊ", "ə"], "ʌɪ": ["ʌ", "i"], "ʌʊ": ["ʌ", "u"],
    "əʊ": ["ə", "u"], "aɪə": ["a", "i", "ə"], "aʊə": ["a", "u", "ə"],
}

STRIP_RE = re.compile(r"[ːˈˌ̤̃ʲˤ̩\^\[\]\.0-9]")


def map_to_reduced(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for raw in tokens:
        if raw in IGNORE_TOKENS:
            continue
        if raw in REDUCED_PHONE_MAP:
            out.extend(REDUCED_PHONE_MAP[raw])
            continue
        cleaned = STRIP_RE.sub("", raw)
        if not cleaned or cleaned in IGNORE_TOKENS:
            continue
        if cleaned in REDUCED_PHONE_MAP:
            out.extend(REDUCED_PHONE_MAP[cleaned])
        else:
            # Fallback: strip dental diacritic and map character by character
            cleaned2 = cleaned.replace("\u032a", "").replace("ʰ", "").replace("ʱ", "")
            if cleaned2 in REDUCED_PHONE_MAP:
                out.extend(REDUCED_PHONE_MAP[cleaned2])
            elif cleaned2:
                for ch in cleaned2:
                    if ch in REDUCED_PHONE_MAP:
                        out.extend(REDUCED_PHONE_MAP[ch])
                    else:
                        out.append(ch)
    return out


def edit_distance_sdi(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    s = d = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            s += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, d, ins, n


def _bootstrap_per_ci(per_items: list[tuple[int, int]], n_boot: int = 1000, seed: int = 42) -> dict[str, float]:
    """Micro-PER bootstrap over (errors, ref_len) pairs."""
    rng = np.random.default_rng(seed)
    errs = np.array([e for e, _ in per_items], dtype=np.float64)
    lens = np.array([l for _, l in per_items], dtype=np.float64)
    n = len(errs)
    idx = rng.integers(0, n, size=(n_boot, n))
    rates = errs[idx].sum(axis=1) / np.maximum(lens[idx].sum(axis=1), 1.0)
    return {
        "per": round(float(errs.sum() / max(lens.sum(), 1.0)), 5),
        "ci95_low": round(float(np.percentile(rates, 2.5)), 5),
        "ci95_high": round(float(np.percentile(rates, 97.5)), 5),
    }


def _paired_bootstrap_per(
    a_items: list[tuple[int, int]], b_items: list[tuple[int, int]], n_boot: int = 1000, seed: int = 42
) -> dict[str, float]:
    """Paired bootstrap of (PER_a - PER_b). Positive means b < a (b is better)."""
    rng = np.random.default_rng(seed)
    ea = np.array([e for e, _ in a_items], dtype=np.float64)
    la = np.array([l for _, l in a_items], dtype=np.float64)
    eb = np.array([e for e, _ in b_items], dtype=np.float64)
    lb = np.array([l for _, l in b_items], dtype=np.float64)
    n = len(ea)
    idx = rng.integers(0, n, size=(n_boot, n))
    ra = ea[idx].sum(axis=1) / np.maximum(la[idx].sum(axis=1), 1.0)
    rb = eb[idx].sum(axis=1) / np.maximum(lb[idx].sum(axis=1), 1.0)
    diff = ra - rb
    return {
        "per_a": round(float(ea.sum() / max(la.sum(), 1.0)), 5),
        "per_b": round(float(eb.sum() / max(lb.sum(), 1.0)), 5),
        "diff_a_minus_b": round(float(ea.sum() / max(la.sum(), 1.0) - eb.sum() / max(lb.sum(), 1.0)), 5),
        "ci95_low": round(float(np.percentile(diff, 2.5)), 5),
        "ci95_high": round(float(np.percentile(diff, 97.5)), 5),
        "p_value_b_lt_a": round(float(np.mean(diff <= 0.0)), 4),
    }


class XlsrEspeakRecognizer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device).eval()
        vocab = json.loads(Path(hf_hub_download(MODEL_ID, "vocab.json")).read_text(encoding="utf-8"))
        self.id2tok = {v: k for k, v in vocab.items()}
        self.blank_id = vocab.get("<pad>", 0)

    @torch.inference_mode()
    def decode_batch(self, wavs_16k: list[np.ndarray], batch_size: int = 16) -> list[list[str]]:
        results: list[list[str]] = []
        for i in range(0, len(wavs_16k), batch_size):
            batch = wavs_16k[i : i + batch_size]
            inp = self.fe(batch, sampling_rate=16000, return_tensors="pt", padding=True)
            logits = self.model(
                inp.input_values.to(self.device),
                attention_mask=inp.attention_mask.to(self.device) if "attention_mask" in inp else None,
            ).logits
            pred_ids = logits.argmax(-1).cpu().tolist()
            for seq_ids in pred_ids:
                toks: list[str] = []
                prev = None
                for idx in seq_ids:
                    if idx != prev and idx != self.blank_id:
                        tok = self.id2tok.get(idx, "")
                        if tok not in ("<pad>", "<s>", "</s>", "|", "<unk>"):
                            toks.append(tok)
                    prev = idx
                results.append(toks)
        return results


def _synth_16k(model: PAWPBilingualTTS, phonemize, texts: list[str], device: torch.device) -> list[np.ndarray]:
    lmap = {LanguageCode.SHARED: 0, LanguageCode.EN: 1, LanguageCode.NE: 2, LanguageCode.MIXED: 3}
    wavs_16k: list[np.ndarray] = []
    with torch.no_grad():
        for txt in texts:
            seq = phonemize.execute(txt)
            toks = seq.tokens
            p_ids = torch.tensor([[t.token_id for t in toks]], dtype=torch.long, device=device)
            l_ids = torch.tensor([[lmap.get(t.language, 0) for t in toks]], dtype=torch.long, device=device)
            s_ids = torch.tensor([[min(t.stress, 3) for t in toks]], dtype=torch.long, device=device)
            b_flags = torch.tensor(
                [
                    [
                        [
                            1.0 if t.is_word_start else 0.0,
                            1.0 if t.is_word_end else 0.0,
                            1.0 if t.is_syllable_start else 0.0,
                            1.0 if t.is_phrase_break else 0.0,
                        ]
                        for t in toks
                    ]
                ],
                dtype=torch.float32,
                device=device,
            )
            masks = torch.tensor([build_language_masks(toks)], dtype=torch.float32, device=device)
            out = model(p_ids, l_ids, s_ids, b_flags, lang_masks=masks)
            wav24 = out.waveform[0].detach().cpu().numpy().astype(np.float32)
            # 24 kHz -> 16 kHz = resample_poly(wav, 2, 3)
            wavs_16k.append(resample_poly(wav24, 2, 3).astype(np.float32))
    return wavs_16k


def _score_refs_hyps(refs_red: list[list[str]], hyps_raw: list[list[str]]) -> tuple[dict[str, object], list[tuple[int, int]]]:
    items: list[tuple[int, int]] = []
    tot_s = tot_d = tot_i = tot_n = 0
    for ref, h_raw in zip(refs_red, hyps_raw, strict=True):
        hyp = map_to_reduced(h_raw)
        s, d, ins, n = edit_distance_sdi(ref, hyp)
        tot_s += s
        tot_d += d
        tot_i += ins
        tot_n += n
        items.append((s + d + ins, n))
    boot = _bootstrap_per_ci(items)
    boot.update({"S": tot_s, "D": tot_d, "I": tot_i, "N": tot_n})
    return boot, items


def main() -> None:
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phonemize = build_container().phonemize
    rec = XlsrEspeakRecognizer(device)

    # 1. Load exact 300 NE + 300 EN from test.jsonl (matching run_exp015_multiseed.sh)
    ne_rows, en_rows = [], []
    with Path("data/manifests/test.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["language"] == "ne" and len(ne_rows) < 300:
                ne_rows.append(r)
            elif r["language"] == "en" and len(en_rows) < 300:
                en_rows.append(r)

    en_texts = [str(r["text"]) for r in en_rows]
    ne_texts = [str(r["text"]) for r in ne_rows]
    en_refs = [map_to_reduced([t.symbol for t in phonemize.execute(t).tokens]) for t in en_texts]
    ne_refs = [map_to_reduced([t.symbol for t in phonemize.execute(t).tokens]) for t in ne_texts]

    # Expanded codemix_v1.jsonl (n=160)
    cm_texts = [
        str(json.loads(x)["text"])
        for x in Path("data/evaluation/codemix_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    cm_refs = [map_to_reduced([t.symbol for t in phonemize.execute(t).tokens]) for t in cm_texts]

    # 2. Score Human Floor on exact 300 EN + 300 NE
    sources = Path("data/sources")
    en_human_wavs, ne_human_wavs = [], []
    en_human_valid_idx, ne_human_valid_idx = [], []
    for i, r in enumerate(en_rows):
        try:
            w, sr = _read_audio(r, sources)
            w16 = resample_poly(w, 16000, sr).astype(np.float32) if sr != 16000 else w.astype(np.float32)
            en_human_wavs.append(w16)
            en_human_valid_idx.append(i)
        except Exception:
            pass
    for i, r in enumerate(ne_rows):
        try:
            w, sr = _read_audio(r, sources)
            w16 = resample_poly(w, 16000, sr).astype(np.float32) if sr != 16000 else w.astype(np.float32)
            ne_human_wavs.append(w16)
            ne_human_valid_idx.append(i)
        except Exception:
            pass

    en_human_hyps = rec.decode_batch(en_human_wavs)
    ne_human_hyps = rec.decode_batch(ne_human_wavs)
    human_en_stats, human_en_items = _score_refs_hyps([en_refs[i] for i in en_human_valid_idx], en_human_hyps)
    human_ne_stats, human_ne_items = _score_refs_hyps([ne_refs[i] for i in ne_human_valid_idx], ne_human_hyps)
    print(f"[HUMAN EN n={len(en_human_valid_idx)}] XLSR-eSpeak PER={human_en_stats['per']:.4f}", flush=True)
    print(f"[HUMAN NE n={len(ne_human_valid_idx)}] XLSR-eSpeak PER={human_ne_stats['per']:.4f}", flush=True)

    # 3. Retroflex-nasal recheck on human Nepali test utterances containing 'ण'
    retro_utts = []
    retro_wavs = []
    retro_total_occ = 0
    for idx_in_loaded, orig_i in enumerate(ne_human_valid_idx):
        r = ne_rows[orig_i]
        txt = str(r["text"])
        if "ण" in txt:
            occ = txt.count("ण")
            retro_total_occ += occ
            retro_utts.append((orig_i, txt, occ, ne_human_hyps[idx_in_loaded]))

    # Check if we need all test.jsonl NE utterances with ण (not just first 300) to match 52 utts / 62 occs
    # Note: audit_003 loaded 178 NE audio files from first 300; let's count raw ɳ vs n across all loaded NE utts with ण
    raw_retro_count_in_retro_utts = sum(sum(1 for tok in hyps if "ɳ" in tok) for _, _, _, hyps in retro_utts)
    raw_dental_n_count_in_retro_utts = sum(sum(1 for tok in hyps if tok == "n" or tok == "n̪") for _, _, _, hyps in retro_utts)
    raw_retro_count_all_ne = sum(sum(1 for tok in hyps if "ɳ" in tok) for hyps in ne_human_hyps)
    raw_dental_n_count_all_ne = sum(sum(1 for tok in hyps if tok == "n" or tok == "n̪") for hyps in ne_human_hyps)

    # Align each retroflex utterance's raw frontend tokens (preserving ɳ vs n̪) against raw XLSR tokens
    aligned_retro = 0
    aligned_dental = 0
    aligned_other_or_del = 0
    for _orig_i, txt, _occ, h_raw in retro_utts:
        f_toks = [t.symbol for t in phonemize.execute(txt).tokens if t.symbol not in IGNORE_TOKENS]
        # Map everything EXCEPT ɳ and n̪/n so alignment finds the nasal slots cleanly
        f_semi = ["ɳ" if x == "ɳ" else ("n" if x in ("n", "n̪") else (map_to_reduced([x])[0] if map_to_reduced([x]) else x)) for x in f_toks]
        h_semi = ["ɳ" if "ɳ" in x else ("n" if x in ("n", "n̪") else (map_to_reduced([x])[0] if map_to_reduced([x]) else x)) for x in h_raw]
        # DP alignment
        n_l, m_l = len(f_semi), len(h_semi)
        dp = [[0] * (m_l + 1) for _ in range(n_l + 1)]
        for i in range(n_l + 1):
            dp[i][0] = i
        for j in range(m_l + 1):
            dp[0][j] = j
        for i in range(1, n_l + 1):
            for j in range(1, m_l + 1):
                c = 0 if f_semi[i - 1] == h_semi[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c)
        i, j = n_l, m_l
        while i > 0 or j > 0:
            if i > 0 and j > 0 and f_semi[i - 1] == h_semi[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
                if f_semi[i - 1] == "ɳ":
                    aligned_retro += 1
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                if f_semi[i - 1] == "ɳ":
                    if h_semi[j - 1] == "n":
                        aligned_dental += 1
                    else:
                        aligned_other_or_del += 1
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                if f_semi[i - 1] == "ɳ":
                    aligned_other_or_del += 1
                i -= 1
            else:
                j -= 1

    retro_recheck = {
        "ne_human_audio_loaded": len(ne_human_valid_idx),
        "utts_with_retroflex_nasal": len(retro_utts),
        "human_occ_orthographic_na": retro_total_occ,
        "aligned_frontend_retroflex_total": aligned_retro + aligned_dental + aligned_other_or_del,
        "decoded_as_retroflex_ɳ": aligned_retro,
        "decoded_as_dental_n": aligned_dental,
        "decoded_as_other_or_deleted": aligned_other_or_del,
        "raw_ɳ_emitted_in_retro_utts": raw_retro_count_in_retro_utts,
        "raw_n_emitted_in_retro_utts": raw_dental_n_count_in_retro_utts,
        "raw_ɳ_emitted_across_all_loaded_ne_human": raw_retro_count_all_ne,
        "raw_n_emitted_across_all_loaded_ne_human": raw_dental_n_count_all_ne,
        "can_separate_ɳ_and_n": bool(raw_retro_count_all_ne > 0),
    }
    print(f"[RETROFLEX RECHECK] {retro_recheck}", flush=True)

    # 4. Score all 7 checkpoints on EN (300), NE (300), and codemix_v1 (160)
    ckpt_results: dict[str, dict[str, object]] = {}
    ckpt_items: dict[str, dict[str, list[tuple[int, int]]]] = {}

    for name, cfg_p, ckpt_p, seed in CHECKPOINTS:
        cfg = load_student_config(cfg_p)
        model = PAWPBilingualTTS(cfg).to(device).eval()
        CheckpointManager.load(model, ckpt_p)

        en_hyps = rec.decode_batch(_synth_16k(model, phonemize, en_texts, device))
        ne_hyps = rec.decode_batch(_synth_16k(model, phonemize, ne_texts, device))
        cm_hyps = rec.decode_batch(_synth_16k(model, phonemize, cm_texts, device))

        en_st, en_it = _score_refs_hyps(en_refs, en_hyps)
        ne_st, ne_it = _score_refs_hyps(ne_refs, ne_hyps)
        cm_st, cm_it = _score_refs_hyps(cm_refs, cm_hyps)

        ckpt_items[name] = {"en": en_it, "ne": ne_it, "codemix_v1": cm_it}
        ckpt_results[name] = {
            "checkpoint": ckpt_p,
            "seed": seed,
            "en_300": en_st,
            "ne_300": ne_st,
            "codemix_v1_160": cm_st,
            "en_vs_human": _paired_bootstrap_per([en_it[i] for i in en_human_valid_idx], human_en_items),
            "ne_vs_human": _paired_bootstrap_per([ne_it[i] for i in ne_human_valid_idx], human_ne_items),
        }
        print(
            f"[{name}] EN PER={en_st['per']:.4f} | NE PER={ne_st['per']:.4f} | CM160 PER={cm_st['per']:.4f}",
            flush=True,
        )

    # 5. Shared vs Dense statistical tests across 3 seeds
    seeds = [42, 101, 202]
    s_en = [ckpt_results[f"shared_300k_s{s}"]["en_300"]["per"] for s in seeds]
    d_en = [ckpt_results[f"dense_300k_s{s}"]["en_300"]["per"] for s in seeds]
    s_ne = [ckpt_results[f"shared_300k_s{s}"]["ne_300"]["per"] for s in seeds]
    d_ne = [ckpt_results[f"dense_300k_s{s}"]["ne_300"]["per"] for s in seeds]
    s_cm = [ckpt_results[f"shared_300k_s{s}"]["codemix_v1_160"]["per"] for s in seeds]
    d_cm = [ckpt_results[f"dense_300k_s{s}"]["codemix_v1_160"]["per"] for s in seeds]

    diffs_en = [a - b for a, b in zip(s_en, d_en, strict=True)]
    diffs_ne = [a - b for a, b in zip(s_ne, d_ne, strict=True)]
    diffs_pooled = diffs_en + diffs_ne

    t_pool, p_pool = scipy_stats.ttest_1samp(diffs_pooled, 0.0)
    t_en, p_en = scipy_stats.ttest_1samp(diffs_en, 0.0)
    t_ne, p_ne = scipy_stats.ttest_1samp(diffs_ne, 0.0)
    try:
        w_stat, w_p = scipy_stats.wilcoxon(diffs_pooled)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    shared_vs_dense = {
        "en_300": {
            "shared_per_seeds": s_en,
            "dense_per_seeds": d_en,
            "shared_mean_std": [round(float(np.mean(s_en)), 5), round(float(np.std(s_en, ddof=1)), 5)],
            "dense_mean_std": [round(float(np.mean(d_en)), 5), round(float(np.std(d_en, ddof=1)), 5)],
            "per_seed_diff_shared_minus_dense": [round(x, 5) for x in diffs_en],
            "paired_ttest_3seeds": {"t": round(float(t_en), 4), "p": round(float(p_en), 5)},
            "seed42_within_seed_paired_bootstrap": _paired_bootstrap_per(
                ckpt_items["shared_300k_s42"]["en"], ckpt_items["dense_300k_s42"]["en"]
            ),
        },
        "ne_300": {
            "shared_per_seeds": s_ne,
            "dense_per_seeds": d_ne,
            "shared_mean_std": [round(float(np.mean(s_ne)), 5), round(float(np.std(s_ne, ddof=1)), 5)],
            "dense_mean_std": [round(float(np.mean(d_ne)), 5), round(float(np.std(d_ne, ddof=1)), 5)],
            "per_seed_diff_shared_minus_dense": [round(x, 5) for x in diffs_ne],
            "paired_ttest_3seeds": {"t": round(float(t_ne), 4), "p": round(float(p_ne), 5)},
            "seed42_within_seed_paired_bootstrap": _paired_bootstrap_per(
                ckpt_items["shared_300k_s42"]["ne"], ckpt_items["dense_300k_s42"]["ne"]
            ),
        },
        "pooled_3seeds_x_2langs": {
            "diffs_shared_minus_dense": [round(x, 5) for x in diffs_pooled],
            "dense_better_count": sum(1 for x in diffs_pooled if x > 0),
            "paired_ttest_n6": {"t": round(float(t_pool), 4), "p": round(float(p_pool), 5)},
            "wilcoxon_n6": {"stat": float(w_stat), "p": round(float(w_p), 5)},
        },
        "codemix_v1_160": {
            "shared_mean_std": [round(float(np.mean(s_cm)), 5), round(float(np.std(s_cm, ddof=1)), 5)],
            "dense_mean_std": [round(float(np.mean(d_cm)), 5), round(float(np.std(d_cm, ddof=1)), 5)],
            "seed42_within_seed_paired_bootstrap": _paired_bootstrap_per(
                ckpt_items["shared_300k_s42"]["codemix_v1"], ckpt_items["dense_300k_s42"]["codemix_v1"]
            ),
        },
    }

    out = {
        "recognizer": MODEL_ID,
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "reduced_phone_mapping": REDUCED_PHONE_MAP,
        "human_floor": {
            "en": {"n_loaded": len(en_human_valid_idx), **human_en_stats},
            "ne": {"n_loaded": len(ne_human_valid_idx), **human_ne_stats},
        },
        "retroflex_nasal_recheck": retro_recheck,
        "checkpoints": ckpt_results,
        "shared_vs_dense": shared_vs_dense,
    }
    out_p = Path("artifacts/reports/exp018_independent_per.json")
    out_p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONE EXP-018 independent PER in {out['elapsed_sec']}s -> {out_p}", flush=True)


if __name__ == "__main__":
    main()
