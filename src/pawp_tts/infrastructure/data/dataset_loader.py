"""Zero-copy lazy waveform decoding, phonological alignment, and 24 kHz feature caching."""

from __future__ import annotations

import io
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

from pawp_tts.application.frontend_service import PhonemizeUtteranceUseCase
from pawp_tts.domain.entities.models import AudioSampleRecord, LanguageCode
from pawp_tts.domain.phonology.codemix import build_language_masks
from pawp_tts.domain.phonology.inventory import (
    CONSONANT_SYMBOLS,
    ID_TO_SYMBOL,
    VOWEL_SYMBOLS,
)
from pawp_tts.infrastructure.data.feature_extractor import (
    MelSpectrogramExtractor,
    estimate_f0_autocorr,
    resample_to_24k,
)
from pawp_tts.infrastructure.frontend.english_adapter import EnglishEspeakG2PAdapter
from pawp_tts.infrastructure.frontend.nepali_adapter import NepaliRuleG2PAdapter

_CONTROL_SET = frozenset({"<pad>", "<bos>", "<eos>", "<unk>", "<sp>", "<wb>", "<sb>", "<pb>"})
_STOP_FLAP_SET = frozenset(
    {
        "p",
        "pʰ",
        "b",
        "bʱ",
        "t",
        "d",
        "t̪",
        "t̪ʰ",
        "d̪",
        "d̪ʱ",
        "ʈ",
        "ʈʰ",
        "ɖ",
        "ɖʱ",
        "k",
        "kʰ",
        "ɡ",
        "ɡʱ",
        "ɽ",
        "ɽʱ",
    }
)
_VOWEL_SET = frozenset(VOWEL_SYMBOLS)
_CONSONANT_SET = frozenset(CONSONANT_SYMBOLS)
_PUNCT_PAUSE_SET = frozenset({",", ".", "?", "!", ":"})


def _phoneme_weight(token_id: int) -> float:
    sym = ID_TO_SYMBOL.get(token_id, "<unk>")
    if sym in _CONTROL_SET:
        return 0.4
    if sym in _PUNCT_PAUSE_SET:
        return 2.0
    if sym == "-":
        return 0.5
    if sym == "ː":
        return 1.6
    if sym in {"ˈ", "ˌ", "̃", "̤"}:
        return 0.6
    if sym in _VOWEL_SET:
        return 1.6 if len(sym) >= 2 else 1.3
    if sym in _STOP_FLAP_SET:
        return 0.8
    if sym in _CONSONANT_SET:
        return 1.0
    return 1.0


def compute_phonological_durations(phoneme_ids: Sequence[int], total_frames: int) -> np.ndarray:
    """
    Distribute `total_frames` deterministically across `phoneme_ids` according to
    phonological class priors using Hamilton largest-remainder rounding so the
    output integer frame counts sum strictly to `total_frames`.
    """
    n_tok = len(phoneme_ids)
    if n_tok == 0:
        return np.zeros(0, dtype=np.int64)
    if total_frames <= 0:
        return np.zeros(n_tok, dtype=np.int64)

    if total_frames < n_tok:
        counts = np.zeros(n_tok, dtype=np.int64)
        counts[:total_frames] = 1
        return counts

    weights = np.array([_phoneme_weight(int(tid)) for tid in phoneme_ids], dtype=np.float64)
    w_sum = float(weights.sum())
    if w_sum <= 0.0:
        weights[:] = 1.0
        w_sum = float(n_tok)

    counts = np.ones(n_tok, dtype=np.int64)
    rem = total_frames - n_tok
    if rem == 0:
        return counts

    exact_extra = weights * (rem / w_sum)
    floored = np.floor(exact_extra).astype(np.int64)
    counts += floored
    leftover = int(rem - floored.sum())
    if leftover > 0:
        remainders = exact_extra - floored
        order = np.argsort(-remainders, kind="stable")
        counts[order[:leftover]] += 1
    return counts


def load_raw_waveform_from_record(
    record: AudioSampleRecord,
    sources_dir: str | Path = "data/sources",
    target_sr: int = 24000,
) -> tuple[np.ndarray, int]:
    """Lazily decode a single audio sample from symlinked parquet/FLAC/WAV without copying."""
    root = Path(sources_dir)
    if record.audio_source_type in {"parquet", "parquet_flac"}:
        rel_path, row_str = record.audio_locator.split("#", maxsplit=1)
        target_row = int(row_str)
        pq_path = root / rel_path
        pf = pq.ParquetFile(pq_path)

        acc = 0
        raw_bytes: bytes | None = None
        for rg_idx in range(pf.num_row_groups):
            n_rows = pf.metadata.row_group(rg_idx).num_rows
            if acc + n_rows > target_row:
                tbl = pf.read_row_group(rg_idx, columns=["audio"])
                cell = tbl.column("audio")[target_row - acc].as_py()
                if isinstance(cell, dict):
                    raw_bytes = cell.get("bytes")
                elif isinstance(cell, (bytes, bytearray)):
                    raw_bytes = bytes(cell)
                break
            acc += n_rows

        if raw_bytes is None:
            raise ValueError(
                f"Failed to locate audio bytes for {record.sample_id} ({record.audio_locator})"
            )
        audio_raw, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    else:
        wav_path = root / record.audio_locator
        audio_raw, sr = sf.read(wav_path, dtype="float32")

    arr = np.asarray(audio_raw, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    arr_24k = resample_to_24k(arr, int(sr), target_sr=target_sr)
    arr_24k = np.clip(arr_24k, -1.0, 1.0).astype(np.float32)
    return arr_24k, target_sr


def _record_from_json(raw: dict[str, object]) -> AudioSampleRecord:
    lang_str = str(raw["language"])
    lang = LanguageCode(lang_str)
    raw_pids = raw.get("phoneme_ids", [])
    pids = tuple(int(x) for x in raw_pids) if isinstance(raw_pids, list) else ()
    return AudioSampleRecord(
        sample_id=str(raw["sample_id"]),
        split=str(raw["split"]),
        language=lang,
        speaker_id=str(raw["speaker_id"]),
        text=str(raw["text"]),
        normalized_text=str(raw["normalized_text"]),
        phoneme_ids=pids,
        audio_source_type=str(raw["audio_source_type"]),
        audio_locator=str(raw["audio_locator"]),
        sample_rate=int(raw["sample_rate"]),  # type: ignore[call-overload]
        duration_sec=float(raw["duration_sec"]),  # type: ignore[arg-type]
        quality_tier=str(raw["quality_tier"]),
        loss_weight=float(raw["loss_weight"]),  # type: ignore[arg-type]
    )


def _select_balanced_records(
    manifest_path: Path,
    max_samples: int,
) -> list[AudioSampleRecord]:
    buckets: dict[str, list[AudioSampleRecord]] = defaultdict(list)
    per_lang_cap = max(max_samples, 32)
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            lang = str(raw.get("language", "en"))
            if len(buckets[lang]) < per_lang_cap:
                buckets[lang].append(_record_from_json(raw))
            if all(len(buckets[k]) >= per_lang_cap for k in ("ne", "en", "mixed")):
                break

    selected: list[AudioSampleRecord] = []
    idx_map: dict[str, int] = defaultdict(int)
    keys = [k for k in ("ne", "en", "mixed") if buckets.get(k)]
    for extra in sorted(buckets.keys()):
        if extra not in keys:
            keys.append(extra)

    while len(selected) < max_samples and keys:
        next_keys: list[str] = []
        for k in keys:
            pos = idx_map[k]
            if pos < len(buckets[k]) and len(selected) < max_samples:
                selected.append(buckets[k][pos])
                idx_map[k] = pos + 1
                if idx_map[k] < len(buckets[k]):
                    next_keys.append(k)
        keys = next_keys
    return selected


def extract_and_cache_features(
    manifest_path: str | Path = "data/manifests/train.jsonl",
    cache_dir: str | Path = "data/cache/features",
    sources_dir: str | Path = "data/sources",
    max_samples: int = 64,
) -> dict[str, float | int | str]:
    """
    Lazily load 24 kHz audio for a language-balanced subset of `manifest_path`, extract
    mel/energy/F0 + phonological durations, and write compressed `.npz` bundles into `cache_dir`.
    """
    m_path = Path(manifest_path)
    if not m_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {m_path}")

    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear old feature files when regenerating cache cleanly
    for old_npz in out_dir.glob("*.npz"):
        old_npz.unlink()

    records = _select_balanced_records(m_path, max_samples=max_samples)
    extractor = MelSpectrogramExtractor(sample_rate=24000, n_fft=1024, hop_length=240, n_mels=80)
    phonemize = PhonemizeUtteranceUseCase(
        english_g2p=EnglishEspeakG2PAdapter(),
        nepali_g2p=NepaliRuleG2PAdapter(),
    )
    lang_map = {
        LanguageCode.SHARED: 0,
        LanguageCode.EN: 1,
        LanguageCode.NE: 2,
        LanguageCode.MIXED: 3,
    }
    pitch_scale = math.log1p(550.0)

    cached_samples = 0
    total_frames = 0
    cache_bytes = 0
    lang_counts: dict[str, int] = defaultdict(int)

    for idx, rec in enumerate(records):
        wav_24k, _ = load_raw_waveform_from_record(rec, sources_dir=sources_dir, target_sr=24000)
        if wav_24k.size < 1024:
            continue

        wav_t = torch.from_numpy(wav_24k).unsqueeze(0)
        log_mel, log_energy = extractor(wav_t)
        mel_np = log_mel[0].numpy().astype(np.float32)
        energy_np = log_energy[0].numpy().astype(np.float32)
        n_frames = int(mel_np.shape[0])

        f0, voiced = estimate_f0_autocorr(wav_24k, sample_rate=24000, hop_length=240)
        if len(f0) < n_frames:
            pad_len = n_frames - len(f0)
            f0 = np.pad(f0, (0, pad_len))
            voiced = np.pad(voiced, (0, pad_len))
        pitch_norm = (np.log1p(np.maximum(f0[:n_frames], 0.0)) / pitch_scale).astype(np.float32)
        pv_np = np.stack([pitch_norm, voiced[:n_frames].astype(np.float32)], axis=-1)

        seq = phonemize.execute(rec.text)
        tokens = seq.tokens
        p_ids = np.array([tok.token_id for tok in tokens], dtype=np.int64)
        l_ids = np.array([lang_map.get(tok.language, 0) for tok in tokens], dtype=np.int64)
        s_ids = np.array([min(tok.stress, 3) for tok in tokens], dtype=np.int64)
        b_flags = np.array(
            [
                [
                    1.0 if tok.is_word_start else 0.0,
                    1.0 if tok.is_word_end else 0.0,
                    1.0 if tok.is_syllable_start else 0.0,
                    1.0 if tok.is_phrase_break else 0.0,
                ]
                for tok in tokens
            ],
            dtype=np.float32,
        )
        l_masks = np.array(build_language_masks(tokens), dtype=np.float32)
        durations = compute_phonological_durations(p_ids.tolist(), n_frames)

        safe_id = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in rec.sample_id)[:48]
        out_file = out_dir / f"{idx:05d}_{safe_id}.npz"
        np.savez_compressed(
            out_file,
            phoneme_ids=p_ids,
            lang_ids=l_ids,
            stress_ids=s_ids,
            boundary_flags=b_flags,
            lang_masks=l_masks,
            durations=durations,
            log_mel=mel_np,
            log_energy=energy_np,
            pitch_voicing=pv_np,
            waveform=wav_24k,
            loss_weight=np.array([rec.loss_weight], dtype=np.float32),
        )
        cached_samples += 1
        total_frames += n_frames
        cache_bytes += int(out_file.stat().st_size)
        lang_counts[rec.language.value] += 1

    return {
        "cached_samples": cached_samples,
        "total_frames": total_frames,
        "cache_bytes": cache_bytes,
        "mean_mel_frames": round(total_frames / max(cached_samples, 1), 1),
        "ne_samples": lang_counts.get("ne", 0),
        "en_samples": lang_counts.get("en", 0),
        "mixed_samples": lang_counts.get("mixed", 0),
    }
