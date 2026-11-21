"""Parallel, file-grouped feature cache builder for full-scale student training (v2 cache).

Stores per-utterance `.npz` bundles with fp16 log-mel, int16 waveform, F0/voicing, energy,
frontend token arrays and a speaker index. Durations are NOT cached: they are learned online
by the training-only alignment module (see `neural/aligner.py`).
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

_STATE: dict[str, Any] = {}


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)
    from pawp_tts.application.frontend_service import PhonemizeUtteranceUseCase
    from pawp_tts.infrastructure.data.feature_extractor import MelSpectrogramExtractor
    from pawp_tts.infrastructure.frontend.english_adapter import EnglishEspeakG2PAdapter
    from pawp_tts.infrastructure.frontend.nepali_adapter import NepaliRuleG2PAdapter

    _STATE["extractor"] = MelSpectrogramExtractor(
        sample_rate=24000, n_fft=1024, hop_length=240, n_mels=80
    )
    _STATE["phonemize"] = PhonemizeUtteranceUseCase(
        english_g2p=EnglishEspeakG2PAdapter(), nepali_g2p=NepaliRuleG2PAdapter()
    )
    _STATE["rg_cache"] = (None, None)


def _read_audio(rec: dict[str, Any], sources_dir: Path) -> tuple[np.ndarray, int]:
    import io

    import pyarrow.parquet as pq
    import soundfile as sf

    loc = str(rec["audio_locator"])
    if rec["audio_source_type"] in {"parquet", "parquet_flac"}:
        rel, row_s = loc.split("#", maxsplit=1)
        row = int(row_s)
        pf = pq.ParquetFile(sources_dir / rel)
        acc = 0
        for rg in range(pf.num_row_groups):
            n = pf.metadata.row_group(rg).num_rows
            if acc + n > row:
                key = (rel, rg)
                cached_key, tbl = _STATE["rg_cache"]
                if cached_key != key:
                    tbl = pf.read_row_group(rg, columns=["audio"])
                    _STATE["rg_cache"] = (key, tbl)
                cell = tbl.column("audio")[row - acc].as_py()
                raw = cell.get("bytes") if isinstance(cell, dict) else bytes(cell)
                audio, sr = sf.read(io.BytesIO(bytes(raw or b"")), dtype="float32")
                break
            acc += n
        else:
            raise ValueError(f"row not found: {loc}")
    else:
        audio, sr = sf.read(sources_dir / loc, dtype="float32")
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr, int(sr)


def _iter_audio(records: list[dict[str, Any]], sources_dir: Path) -> Any:
    """Yield `(record, audio|None, sr)`; parquet files are streamed in small row batches
    (single-row-group parquet files can be >1 GB and must never be loaded whole)."""
    import io

    import pyarrow.parquet as pq
    import soundfile as sf

    by_file: dict[str, dict[int, dict[str, Any] | None]] = defaultdict(dict)
    for rec in records:
        if rec["audio_source_type"] in {"parquet", "parquet_flac"}:
            rel, row_s = str(rec["audio_locator"]).split("#", maxsplit=1)
            by_file[rel][int(row_s)] = rec
        else:
            try:
                arr, sr = _read_audio(rec, sources_dir)
                yield rec, arr, sr
            except Exception:
                yield rec, None, 0
    for rel, wanted in by_file.items():
        pf = pq.ParquetFile(sources_dir / rel)
        last = max(wanted)
        row = 0
        for batch in pf.iter_batches(batch_size=32, columns=["audio"]):
            col = batch.column(0)
            for j in range(batch.num_rows):
                hit = wanted.pop(row + j, None)
                if hit is None:
                    continue
                rec = hit
                cell = col[j].as_py()
                raw = cell.get("bytes") if isinstance(cell, dict) else cell
                try:
                    audio, sr = sf.read(io.BytesIO(bytes(raw or b"")), dtype="float32")
                    arr = np.asarray(audio, dtype=np.float32)
                    yield rec, (arr.mean(axis=1) if arr.ndim > 1 else arr), int(sr)
                except Exception:
                    yield rec, None, 0
            row += batch.num_rows
            if row > last:
                break
        for left in wanted.values():
            if left is not None:
                yield left, None, 0


def _process_chunk(args: tuple[list[dict[str, Any]], str, str]) -> list[dict[str, Any]]:
    import torch

    from pawp_tts.domain.entities.models import LanguageCode
    from pawp_tts.domain.phonology.codemix import build_language_masks
    from pawp_tts.infrastructure.data.feature_extractor import (
        estimate_f0_autocorr,
        resample_to_24k,
    )

    records, out_dir_s, sources_s = args
    out_dir, sources_dir = Path(out_dir_s), Path(sources_s)
    lang_map = {
        LanguageCode.SHARED: 0,
        LanguageCode.EN: 1,
        LanguageCode.NE: 2,
        LanguageCode.MIXED: 3,
    }
    pitch_scale = math.log1p(550.0)
    results: list[dict[str, Any]] = []
    todo: list[dict[str, Any]] = []
    for rec in records:
        out_file = out_dir / f"{rec['_name']}.npz"
        if not out_file.exists():
            todo.append(rec)
            continue
        with np.load(out_file) as d:
            results.append(
                {
                    "file": out_file.name,
                    "frames": int(d["log_mel"].shape[0]),
                    "tokens": int(d["phoneme_ids"].shape[0]),
                    "lang": rec["language"],
                }
            )
    for rec, arr, sr in _iter_audio(todo, sources_dir):
        try:
            out_file = out_dir / f"{rec['_name']}.npz"
            if arr is None:
                raise ValueError("audio not found")
            wav = np.clip(resample_to_24k(arr, sr), -1.0, 1.0).astype(np.float32)
            peak = float(np.abs(wav).max())
            if wav.size < 24000 or peak < 1e-3:
                continue
            wav = wav * (0.95 / peak) if peak > 0.95 else wav
            log_mel, log_energy = _STATE["extractor"](torch.from_numpy(wav).unsqueeze(0))
            mel = log_mel[0].numpy()
            n_frames = int(mel.shape[0])
            f0, voiced = estimate_f0_autocorr(wav, sample_rate=24000, hop_length=240)
            f0 = np.pad(f0, (0, max(0, n_frames - len(f0))))[:n_frames]
            voiced = np.pad(voiced, (0, max(0, n_frames - len(voiced))))[:n_frames]
            pv = np.stack(
                [np.log1p(np.maximum(f0, 0.0)) / pitch_scale, voiced.astype(np.float32)], -1
            )
            tokens = _STATE["phonemize"].execute(str(rec["text"])).tokens
            if len(tokens) < 2 or n_frames < len(tokens):
                continue
            np.savez(
                out_file,
                phoneme_ids=np.array([t.token_id for t in tokens], dtype=np.int16),
                lang_ids=np.array([lang_map.get(t.language, 0) for t in tokens], dtype=np.int8),
                stress_ids=np.array([min(t.stress, 3) for t in tokens], dtype=np.int8),
                boundary_flags=np.array(
                    [
                        [t.is_word_start, t.is_word_end, t.is_syllable_start, t.is_phrase_break]
                        for t in tokens
                    ],
                    dtype=np.int8,
                ),
                lang_masks=np.array(build_language_masks(tokens), dtype=np.float16),
                log_mel=mel.astype(np.float16),
                log_energy=log_energy[0].numpy().astype(np.float16),
                pitch_voicing=pv.astype(np.float16),
                waveform=(wav * 32767.0).astype(np.int16),
                speaker=np.array([rec["_spk"]], dtype=np.int32),
                loss_weight=np.array([float(rec["loss_weight"])], dtype=np.float32),
            )
            results.append(
                {
                    "file": out_file.name,
                    "frames": n_frames,
                    "tokens": len(tokens),
                    "lang": rec["language"],
                }
            )
        except Exception as exc:  # skip corrupt samples; counted in summary
            results.append({"error": f"{rec.get('sample_id')}: {exc}"})
    return results


_Job = tuple[list[dict[str, Any]], str, str]


def _run_jobs_crash_tolerant(jobs: list[_Job], workers: int) -> list[list[dict[str, Any]]]:
    """Run jobs; if a native crash kills a worker, restart the pool and retry unfinished jobs.

    A job that crashes twice is split into single-record jobs; a single record that still
    crashes is reported as an error and skipped.
    """
    from concurrent.futures import as_completed
    from concurrent.futures.process import BrokenProcessPool

    results: list[list[dict[str, Any]]] = []
    pending = list(jobs)
    crashes: dict[int, int] = defaultdict(int)
    while pending:
        retry: list[_Job] = []
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            futs = {pool.submit(_process_chunk, job): job for job in pending}
            for fut in as_completed(futs):
                job = futs[fut]
                try:
                    results.append(fut.result())
                except BrokenProcessPool:
                    retry.append(job)
        for job in retry:
            key = id(job[0][0]) if job[0] else 0
            crashes[key] += 1
        new_pending: list[_Job] = []
        for job in retry:
            recs = job[0]
            n_crash = crashes[id(recs[0])] if recs else 0
            if len(recs) == 1 and n_crash >= 2:
                results.append([{"error": f"{recs[0].get('sample_id')}: worker crash"}])
            elif n_crash >= 2:
                new_pending.extend(([r], job[1], job[2]) for r in recs)
            else:
                new_pending.append(job)
        pending = new_pending
    return results


def build_extra_nepali_manifest(
    manifests_dir: str | Path = "data/manifests",
    sources_dir: str | Path = "data/sources",
    out_name: str = "train_extra.jsonl",
) -> dict[str, Any]:
    """Collect additional Nepali sources into a train-only manifest.

    Drops any sentence present in val/test (leakage) or already in train (duplicates).
    Supplementary (non-studio) sets get `loss_weight=0.5`, so human studio speech dominates.
    """
    src = Path(sources_dir)
    mdir = Path(manifests_dir)
    held: set[str] = set()
    seen: set[str] = set()
    for split in ("val", "test", "train"):
        for line in (mdir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                text = str(json.loads(line)["text"]).strip()
                (seen if split == "train" else held).add(text)

    specs: list[tuple[str, str, str, float, str]] = [
        # (symlink dir, manifest rel path, audio key, loss weight, quality tier)
        (
            "nepali_studio_tts_manifests",
            "pure_studio_nepali_train.jsonl",
            "audio",
            1.0,
            "tier1_studio",
        ),
        (
            "nepali_synth_multidomain",
            "manifests/all_samples.jsonl",
            "audio_path",
            0.5,
            "tier3_supplement",
        ),
        (
            "nepali_synth_expressive",
            "manifests/all_samples.jsonl",
            "audio_path",
            0.5,
            "tier3_supplement",
        ),
        (
            "nepali_synth_polysemous",
            "manifests/all_samples.jsonl",
            "audio_path",
            0.5,
            "tier3_supplement",
        ),
    ]
    out_rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    for link, rel, key, weight, tier in specs:
        root = (src / link).resolve()
        kept = dropped_leak = dropped_dup = 0
        for line in (src / link / rel).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            text = str(r.get("clean_text") or r["text"]).strip()
            if text in held:
                dropped_leak += 1
                continue
            if text in seen and tier == "tier1_studio":  # expressive re-readings are kept
                dropped_dup += 1
                continue
            seen.add(text)
            audio = Path(str(r[key]))
            try:
                loc = f"{link}/{audio.resolve().relative_to(root)}"
            except ValueError:
                continue
            fname = audio.name
            if link == "nepali_studio_tts_manifests":
                spk = (
                    "ne_openslr_studio_extra" if fname.startswith("openslr") else "ne_studio_extra"
                )
            else:
                spk = f"ne_{link}"
            dur = float(r.get("duration") or r.get("audio_duration") or 0.0)
            if dur <= 0:
                import soundfile as sf

                dur = float(sf.info(str(audio)).duration)
            out_rows.append(
                {
                    "sample_id": f"{link}_{r['id']}",
                    "split": "train",
                    "language": "ne",
                    "speaker_id": spk,
                    "text": text,
                    "normalized_text": text,
                    "phoneme_ids": [],
                    "audio_source_type": "wav",
                    "audio_locator": loc,
                    "sample_rate": 0,
                    "duration_sec": dur,
                    "quality_tier": tier,
                    "loss_weight": weight,
                }
            )
            kept += 1
        stats[link] = {"kept": kept, "dropped_leak": dropped_leak, "dropped_dup": dropped_dup}
    (mdir / out_name).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8"
    )
    stats["hours"] = round(sum(r["duration_sec"] for r in out_rows) / 3600, 2)
    return stats


def build_codemix_synth_manifest(
    manifests_dir: str | Path = "data/manifests",
    sources_dir: str | Path = "data/sources",
    link: str = "codemix_synth_extra",
) -> dict[str, Any]:
    """Index the generated code-mix set (single voice) as a train-only manifest."""
    import soundfile as sf

    root = Path(sources_dir) / link
    held: set[str] = set()
    for split in ("val", "test"):
        for line in (
            (Path(manifests_dir) / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ):
            if line.strip():
                held.add(str(json.loads(line)["text"]).strip())
    rows: list[dict[str, Any]] = []
    missing = leak = 0
    for i, line in enumerate((root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        text = str(r["text"]).strip()
        wav = root / str(r["audio"])
        if text in held:
            leak += 1
            continue
        if not wav.exists():
            missing += 1
            continue
        info = sf.info(str(wav))
        rows.append(
            {
                "sample_id": f"cmsynth_{i:06d}",
                "split": "train",
                "language": "mixed" if str(r.get("language")) != "ne" else "ne",
                "speaker_id": "ne_codemix_synth",
                "text": text,
                "normalized_text": text,
                "phoneme_ids": [],
                "audio_source_type": "wav",
                "audio_locator": f"{link}/{r['audio']}",
                "sample_rate": int(info.samplerate),
                "duration_sec": float(info.duration),
                "quality_tier": "tier3_supplement",
                "loss_weight": 0.7,
            }
        )
    (Path(manifests_dir) / "codemix_synth.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8"
    )
    return {
        "kept": len(rows),
        "missing_audio": missing,
        "dropped_leak": leak,
        "hours": round(sum(x["duration_sec"] for x in rows) / 3600, 2),
    }


def _select(
    manifest: Path, en_hours: float, min_s: float, max_s: float, seed: int, max_items: int | None
) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in manifest.open(encoding="utf-8") if line.strip()]
    rows = [r for r in rows if min_s <= float(r["duration_sec"]) <= max_s]
    random.Random(seed).shuffle(rows)
    en_budget = en_hours * 3600.0
    out: list[dict[str, Any]] = []
    for r in rows:
        if r["language"] == "en":
            if en_budget <= 0:
                continue
            en_budget -= float(r["duration_sec"])
        out.append(r)
    if max_items is not None:
        out = out[:max_items]
    return out


def build_feature_cache(
    out_root: str | Path = "data/cache/v2",
    manifests_dir: str | Path = "data/manifests",
    sources_dir: str | Path = "data/sources",
    en_hours: float = 80.0,
    min_sec: float = 1.0,
    max_sec: float = 12.0,
    val_items: int = 400,
    workers: int = 24,
    seed: int = 1234,
    only_splits: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build train/val(/train_extra) feature caches; speaker index shared across splits."""
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": _select(
            Path(manifests_dir) / "train.jsonl", en_hours, min_sec, max_sec, seed, None
        ),
        "val": _select(
            Path(manifests_dir) / "val.jsonl", en_hours / 40, min_sec, max_sec, seed, val_items
        ),
    }
    extra = Path(manifests_dir) / "train_extra.jsonl"
    if extra.exists():
        splits["train_extra"] = _select(extra, 0.0, min_sec, max_sec, seed, None)
    # Fine-tune split: single-voice teacher corpus + code-mix supplement (grows over time).
    ft_rows: list[dict[str, Any]] = []
    for name in sorted(
        [p.name for p in Path(manifests_dir).glob("teacher_*.jsonl")] + ["codemix_synth.jsonl"]
    ):
        mp = Path(manifests_dir) / name
        if mp.exists():
            ft_rows += _select(mp, 1e9, min_sec, max_sec, seed, None)
    if ft_rows:
        splits["train_ft"] = ft_rows
    if only_splits is not None:
        splits = {k: v for k, v in splits.items() if k in only_splits}
    spk_path = root / "speakers.json"
    speakers: dict[str, int] = json.loads(spk_path.read_text()) if spk_path.exists() else {}
    for split_rows in splits.values():
        for r in split_rows:
            speakers.setdefault(str(r["speaker_id"]), len(speakers))
    spk_path.write_text(json.dumps(speakers, indent=0))

    sum_path = root / "summary.json"
    summary: dict[str, Any] = json.loads(sum_path.read_text()) if sum_path.exists() else {}
    summary["num_speakers"] = len(speakers)
    for split, rows in splits.items():
        out_dir = root / split
        out_dir.mkdir(exist_ok=True)
        for i, r in enumerate(rows):
            r["_idx"] = i
            # train/val keep historical index names; other splits use stable sample ids
            r["_name"] = (
                f"{i:06d}"
                if split in ("train", "val", "train_extra")
                else "".join(c if c.isalnum() or c in "-_" else "_" for c in str(r["sample_id"]))
            )
            r["_spk"] = speakers[str(r["speaker_id"])]
        # One job per parquet file (streamed); WAV/other sources in chunks of 200.
        pq_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        other: list[dict[str, Any]] = []
        for r in rows:
            if r["audio_source_type"] in {"parquet", "parquet_flac"}:
                pq_groups[str(r["audio_locator"]).split("#")[0]].append(r)
            else:
                other.append(r)
        jobs = [(g, str(out_dir), str(sources_dir)) for g in pq_groups.values()]
        jobs += [
            (other[i : i + 200], str(out_dir), str(sources_dir)) for i in range(0, len(other), 200)
        ]
        index: list[dict[str, Any]] = []
        errors: list[str] = []
        for res in _run_jobs_crash_tolerant(jobs, workers):
            for item in res:
                (errors.append(item["error"]) if "error" in item else index.append(item))
        (root / f"{split}_index.json").write_text(json.dumps(index))
        hours: defaultdict[str, float] = defaultdict(float)
        for it in index:
            hours[it["lang"]] += it["frames"] / 100.0 / 3600.0
        summary[split] = {
            "items": len(index),
            "errors": len(errors),
            "hours": {k: round(v, 2) for k, v in hours.items()},
            "first_errors": errors[:3],
        }
    (root / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":  # pragma: no cover
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    print(json.dumps(build_feature_cache(), indent=1))
