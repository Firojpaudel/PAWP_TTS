"""Zero-copy bilingual dataset manifest builder over symlinked corpora."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

from pawp_tts.application.frontend_service import PhonemizeUtteranceUseCase
from pawp_tts.domain.dataset.splitting import deduplicate_and_validate_splits
from pawp_tts.domain.entities.models import AudioSampleRecord, DatasetSplitSummary, LanguageCode
from pawp_tts.infrastructure.frontend.english_adapter import EnglishEspeakG2PAdapter
from pawp_tts.infrastructure.frontend.nepali_adapter import NepaliRuleG2PAdapter


def _flac_header_info(raw_bytes: bytes) -> tuple[int, float]:
    """Parse sample rate and duration from 38-byte FLAC STREAMINFO header in O(1)."""
    if len(raw_bytes) >= 26 and raw_bytes[:4] == b"fLaC":
        val = int.from_bytes(raw_bytes[18:26], "big")
        sr = (val >> 44) & 0xFFFFF
        total_samples = val & 0xFFFFFFFFF
        if sr > 0 and total_samples > 0:
            return sr, total_samples / float(sr)
    info = sf.info(io.BytesIO(raw_bytes))
    return int(info.samplerate), float(info.duration)


def _normalize_split_name(raw_split: str) -> str:
    s = raw_split.strip().lower()
    if s in {"validation", "val", "dev"}:
        return "val"
    if s == "test":
        return "test"
    return "train"


def _assign_deterministic_split(speaker_id: str) -> str:
    digest = hashlib.sha1(speaker_id.encode("utf-8")).digest()
    bucket = digest[0] % 20
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "val"
    return "train"


def _normalize_english_case(text: str) -> str:
    stripped = text.strip()
    if stripped.isupper():
        return stripped.lower()
    return stripped


class BilingualManifestBuilder:
    """Scans symlinked EN, NE, and Code-Mix sources without duplicating audio files."""

    def __init__(self, sources_dir: str | Path = "data/sources") -> None:
        self._sources = Path(sources_dir)

    def build_and_write(
        self,
        output_dir: str | Path = "data/manifests",
        include_supplemental: bool = False,
        max_en_shards: int | None = None,
    ) -> DatasetSplitSummary:
        """
        Scan sources, phonemize all utterances using fast in-process G2P,
        validate leak-free splits, and persist JSONL manifests.
        """
        raw_items = list(
            self._iter_raw_metadata(
                include_supplemental=include_supplemental,
                max_en_shards=max_en_shards,
            )
        )

        en_adapter = EnglishEspeakG2PAdapter()
        ne_adapter = NepaliRuleG2PAdapter()
        phonemize = PhonemizeUtteranceUseCase(english_g2p=en_adapter, nepali_g2p=ne_adapter)

        records: list[AudioSampleRecord] = []
        for item in raw_items:
            text = str(item["text"])
            seq = phonemize.execute(text)
            records.append(
                AudioSampleRecord(
                    sample_id=str(item["sample_id"]),
                    split=str(item["split"]),
                    language=item["language"],  # type: ignore[arg-type]
                    speaker_id=str(item["speaker_id"]),
                    text=text,
                    normalized_text=seq.normalized_text,
                    phoneme_ids=seq.token_ids,
                    audio_source_type=str(item["audio_source_type"]),
                    audio_locator=str(item["audio_locator"]),
                    sample_rate=int(item["sample_rate"]),  # type: ignore[call-overload]
                    duration_sec=float(item["duration_sec"]),  # type: ignore[arg-type]
                    quality_tier=str(item["quality_tier"]),
                    loss_weight=float(item["loss_weight"]),  # type: ignore[arg-type]
                )
            )

        clean_records, summary = deduplicate_and_validate_splits(records)

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        writers = {
            split: (out_path / f"{split}.jsonl").open("w", encoding="utf-8")
            for split in ("train", "val", "test")
        }
        try:
            for rec in clean_records:
                fh = writers.get(rec.split)
                if fh is None:
                    continue
                payload = asdict(rec)
                payload["language"] = rec.language.value
                payload["phoneme_ids"] = list(rec.phoneme_ids)
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        finally:
            for fh in writers.values():
                fh.close()

        (out_path / "summary.json").write_text(
            json.dumps(asdict(summary), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return summary

    def _iter_raw_metadata(
        self,
        include_supplemental: bool,
        max_en_shards: int | None,
    ) -> Iterable[dict[str, object]]:
        # 1. Nepali Studio Parquet (24 kHz human speech)
        ne_root = self._sources / "nepali_studio_parquet"
        if ne_root.exists():
            for tier_dir in ("finetuning", "pretraining"):
                tier_path = ne_root / tier_dir
                if not tier_path.exists():
                    continue
                for pq_file in sorted(tier_path.glob("*.parquet")):
                    table = pq.read_table(
                        pq_file,
                        columns=[
                            "id",
                            "split",
                            "speaker_id",
                            "text",
                            "duration",
                            "snr",
                            "sample_rate",
                        ],
                    )
                    data = table.to_pydict()
                    rel = pq_file.relative_to(self._sources)
                    n_rows = len(data["id"])
                    for idx in range(n_rows):
                        snr_val = data["snr"][idx]
                        if snr_val is not None and float(snr_val) < 18.0:
                            continue
                        raw_txt = str(data["text"][idx] or "").strip()
                        if not raw_txt:
                            continue
                        yield {
                            "sample_id": f"ne_{tier_dir}_{data['id'][idx]}",
                            "split": _normalize_split_name(str(data["split"][idx])),
                            "language": LanguageCode.NE,
                            "speaker_id": f"ne_{data['speaker_id'][idx]}",
                            "text": raw_txt,
                            "audio_source_type": "parquet",
                            "audio_locator": f"{rel}#{idx}",
                            "sample_rate": int(data["sample_rate"][idx] or 24000),
                            "duration_sec": float(data["duration"][idx] or 0.0),
                            "quality_tier": "tier1_studio"
                            if tier_dir == "finetuning"
                            else "tier1_human",
                            "loss_weight": 1.0,
                        }

        # 2. English LibriSpeech Clean Parquet (16 kHz human speech)
        en_root = self._sources / "english_librispeech_clean"
        if en_root.exists():
            for split_dir, target_split in (
                ("test", "test"),
                ("train.100", "train"),
                ("train.360", "train"),
            ):
                sp_path = en_root / split_dir
                if not sp_path.exists():
                    continue
                shards = sorted(sp_path.glob("*.parquet"))
                if max_en_shards is not None and split_dir != "test":
                    shards = shards[:max_en_shards]
                for pq_file in shards:
                    pf = pq.ParquetFile(pq_file)
                    rel = pq_file.relative_to(self._sources)
                    row_offset = 0
                    for batch in pf.iter_batches(
                        batch_size=512,
                        columns=["id", "speaker_id", "text", "audio"],
                    ):
                        cols = batch.to_pydict()
                        n_rows = len(cols["id"])
                        for i in range(n_rows):
                            audio_obj = cols["audio"][i]
                            raw_bytes = (
                                audio_obj.get("bytes") if isinstance(audio_obj, dict) else None
                            )
                            if not raw_bytes:
                                continue
                            try:
                                sr, dur = _flac_header_info(raw_bytes)
                            except RuntimeError:
                                continue
                            spk = f"en_ls_{cols['speaker_id'][i]}"
                            is_val = hashlib.sha1(spk.encode("utf-8")).digest()[0] % 25 == 0
                            en_split = (
                                "test" if target_split == "test" else ("val" if is_val else "train")
                            )
                            yield {
                                "sample_id": f"en_ls_{cols['id'][i]}",
                                "split": en_split,
                                "language": LanguageCode.EN,
                                "speaker_id": spk,
                                "text": _normalize_english_case(str(cols["text"][i])),
                                "audio_source_type": "parquet_flac",
                                "audio_locator": f"{rel}#{row_offset + i}",
                                "sample_rate": sr,
                                "duration_sec": dur,
                                "quality_tier": "tier1_human",
                                "loss_weight": 1.0,
                            }
                        row_offset += n_rows

        # 3. Code-Mix Real Human Podcasts (16 kHz)
        cm_root = self._sources / "codemix_podcasts"
        cm_manifest = cm_root / "manifest.jsonl"
        if cm_manifest.exists():
            with cm_manifest.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    txt = str(row.get("text") or row.get("clean_verbatim_text") or "").strip()
                    if not txt:
                        continue
                    spk = f"cm_{row.get('speaker_id', 'unknown')}"
                    yield {
                        "sample_id": f"cm_{row['utterance_id']}",
                        "split": _assign_deterministic_split(spk),
                        "language": LanguageCode.MIXED,
                        "speaker_id": spk,
                        "text": txt,
                        "audio_source_type": "wav",
                        "audio_locator": f"codemix_podcasts/{row['audio_path']}",
                        "sample_rate": 16000,
                        "duration_sec": float(row.get("duration_seconds", 0.0)),
                        "quality_tier": "tier2_human",
                        "loss_weight": 0.8,
                    }

        # 4. Optional Supplemental Code-Mix (disabled by default for open-source purity)
        if include_supplemental:
            sup_root = self._sources / "codemix_synth_extra"
            sup_manifest = sup_root / "manifest.jsonl"
            if sup_manifest.exists():
                with sup_manifest.open("r", encoding="utf-8") as fh:
                    for idx, line in enumerate(fh):
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        txt = str(row.get("text") or "").strip()
                        rel_audio = str(row.get("audio") or "")
                        wav_file = sup_root / rel_audio
                        if not txt or not wav_file.exists():
                            continue
                        info = sf.info(wav_file)
                        yield {
                            "sample_id": f"sup_cm_{idx:06d}",
                            "split": "train",
                            "language": LanguageCode.MIXED,
                            "speaker_id": "sup_cm_pool",
                            "text": txt,
                            "audio_source_type": "wav",
                            "audio_locator": f"codemix_synth_extra/{rel_audio}",
                            "sample_rate": int(info.samplerate),
                            "duration_sec": float(info.duration),
                            "quality_tier": "supplemental",
                            "loss_weight": 0.3,
                        }
