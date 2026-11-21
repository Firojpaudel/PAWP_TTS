"""Single-voice teacher corpus generation via a configurable teacher TTS service, ASR-filtered.

Produces English / Nepali / code-mixed speech in ONE target voice so the student learns a
consistent cross-lingual identity. Every clip is transcribed by an independent ASR; clips
whose CER exceeds a per-language threshold are discarded (teacher errors never become
targets). All service details (URL, request template, voice) live only in the git-ignored
`.env`: `PAWP_TTS_API_URL`, `PAWP_TTS_FORM_TEMPLATE` (JSON with `{text}`, `{voice}`, `{lang}`
placeholders and `{lang_en}`/`{lang_ne}` codes) and `PAWP_TTS_VOICE`.
"""

from __future__ import annotations

import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import soundfile as sf

from pawp_tts.domain.entities.models import LanguageCode
from pawp_tts.domain.evaluation.metrics import compute_cer
from pawp_tts.infrastructure.evaluation.http_asr_adapter import HttpMultipartAsrAdapter


def _render_form(text: str, voice: str, lang: str) -> dict[str, str]:
    """Fill the request template from `.env` (keeps service schema out of the codebase)."""
    template = json.loads(os.environ.get("PAWP_TTS_FORM_TEMPLATE", "{}"))
    codes = json.loads(os.environ.get("PAWP_TTS_LANG_CODES", '{"en": "en", "ne": "ne"}'))
    values = {"text": text, "voice": voice, "lang": codes.get(lang, lang)}
    return {k: str(v).format(**values) for k, v in template.items()}


_CER_MAX = {"en": 0.10, "ne": 0.12, "mixed": 0.30}


def _collect_texts(
    manifests_dir: Path, ljspeech_dir: Path | None, seed: int, per_lang: dict[str, int]
) -> list[tuple[str, str]]:
    held: set[str] = set()
    pools: dict[str, list[str]] = {"en": [], "ne": [], "mixed": []}
    for split in ("val", "test"):
        for line in (manifests_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                held.add(str(json.loads(line)["text"]).strip())
    for line in (manifests_dir / "train.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        text = str(r["text"]).strip()
        if text not in held and 20 <= len(text) <= 220:
            pools[str(r["language"])].append(text)
    if ljspeech_dir is not None and (ljspeech_dir / "metadata.csv").exists():
        for line in (ljspeech_dir / "metadata.csv").read_text(encoding="utf-8").splitlines():
            parts = line.split("|")
            text = parts[-1].strip()
            if 20 <= len(text) <= 220 and text not in held:
                pools["en"].append(text)
    rng = random.Random(seed)
    out: list[tuple[str, str]] = []
    for lang, n in per_lang.items():
        uniq = sorted(set(pools[lang]))
        rng.shuffle(uniq)
        out.extend((lang, t) for t in uniq[:n])
    rng.shuffle(out)
    return out


def generate_teacher_corpus(
    voice: str | None = None,
    out_dir: str | Path | None = None,
    manifests_dir: str | Path = "data/manifests",
    ljspeech_dir: str | Path | None = "data/sources/english_ljspeech",
    per_lang: dict[str, int] | None = None,
    workers: int = 8,
    seed: int = 7,
) -> dict[str, Any]:
    """Generate, ASR-filter and index a single-voice corpus; resumable (skips existing clips)."""
    voice = voice or os.environ.get("PAWP_TTS_VOICE", "target")
    out_dir = out_dir or f"data/cache/teacher_{voice}"
    url = os.environ.get("PAWP_TTS_API_URL", "")
    if not url:
        raise RuntimeError("PAWP_TTS_API_URL is not configured in .env")
    per_lang = per_lang or {"en": 9000, "ne": 9000, "mixed": 7000}
    out = Path(out_dir)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    texts = _collect_texts(
        Path(manifests_dir), Path(ljspeech_dir) if ljspeech_dir else None, seed, per_lang
    )
    asr = HttpMultipartAsrAdapter()
    lock = threading.Lock()
    done_path = out / "generated.jsonl"
    done = (
        {json.loads(x)["id"] for x in done_path.read_text().splitlines()}
        if done_path.exists()
        else set()
    )
    stats: dict[str, Any] = {"requested": len(texts), "kept": 0, "rejected_cer": 0, "failed": 0}
    client = httpx.Client(timeout=120)

    def work(item: tuple[int, tuple[str, str]]) -> None:
        idx, (lang, text) = item
        cid = f"{voice}_{lang}_{idx:06d}"
        if cid in done:
            return
        wav = out / "wavs" / f"{cid}.wav"
        try:
            resp = client.post(url, data=_render_form(text, voice, lang))
            resp.raise_for_status()
            wav.write_bytes(resp.content)
            info = sf.info(str(wav))
            hyp = asr.transcribe(
                str(wav), LanguageCode.EN if lang == "en" else LanguageCode.NE
            ).text
            cer = compute_cer(text, hyp).error_rate
            keep = cer <= _CER_MAX[lang]
            if not keep:
                wav.unlink(missing_ok=True)
            rec = {
                "id": cid,
                "lang": lang,
                "text": text,
                "cer": round(cer, 4),
                "keep": keep,
                "duration": round(info.duration, 3),
                "sr": info.samplerate,
            }
        except Exception as exc:  # network / decode failure: record and continue
            wav.unlink(missing_ok=True)
            rec = {"id": cid, "lang": lang, "text": text, "keep": False, "error": str(exc)[:200]}
        with lock, done_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, enumerate(texts)))

    stats.update(write_teacher_manifest(voice, out, Path(manifests_dir)))
    return stats


def write_teacher_manifest(
    voice: str | None = None,
    out_dir: str | Path | None = None,
    manifests_dir: str | Path = "data/manifests",
) -> dict[str, Any]:
    """(Re)write the project-schema manifest from the generation log (safe while running)."""
    voice = voice or os.environ.get("PAWP_TTS_VOICE", "target")
    out_dir = out_dir or f"data/cache/teacher_{voice}"
    done_path = Path(out_dir) / "generated.jsonl"
    stats: dict[str, Any] = {"kept": 0, "rejected_cer": 0, "failed": 0}
    # Build a manifest in the project schema (train-only; voice = dedicated speaker id).
    rows = []
    for x in done_path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(x))
        except json.JSONDecodeError:
            continue  # line being written concurrently
    manifest: list[dict[str, Any]] = []
    hours: dict[str, float] = {}
    for r in rows:
        if r.get("error"):
            stats["failed"] += 1
            continue
        if not r["keep"]:
            stats["rejected_cer"] += 1
            continue
        stats["kept"] += 1
        hours[r["lang"]] = hours.get(r["lang"], 0.0) + r["duration"] / 3600
        manifest.append(
            {
                "sample_id": r["id"],
                "split": "train",
                "language": r["lang"],
                "speaker_id": f"target_{voice}",
                "text": r["text"],
                "normalized_text": r["text"],
                "phoneme_ids": [],
                "audio_source_type": "wav",
                "audio_locator": f"teacher_{voice}/wavs/{r['id']}.wav",
                "sample_rate": r["sr"],
                "duration_sec": r["duration"],
                "quality_tier": "tier2_teacher",
                "loss_weight": 1.0,
            }
        )
    Path(manifests_dir, f"teacher_{voice}.jsonl").write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest), encoding="utf-8"
    )
    stats["hours"] = {k: round(v, 2) for k, v in hours.items()}
    return stats
