"""Thin CLI entrypoint delegating all workflows to Clean Architecture use cases."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import soundfile as sf
import torch

from pawp_tts.domain.entities.models import ExperimentRecord, LanguageCode
from pawp_tts.domain.evaluation.frontend_validation import GOLD_300_CASES
from pawp_tts.domain.ports.interfaces import AsrEvaluatorPort
from pawp_tts.infrastructure.data.bulk_cache import (
    build_extra_nepali_manifest,
    build_feature_cache,
)
from pawp_tts.infrastructure.evaluation.benchmark_suite import (
    _synthesize_utterance,
    run_evaluation_suite,
    run_heldout_asr_eval,
)
from pawp_tts.infrastructure.evaluation.http_asr_adapter import HttpMultipartAsrAdapter
from pawp_tts.infrastructure.evaluation.latency_profiler import run_latency_benchmarks
from pawp_tts.infrastructure.inference.streaming import OnnxStreamingExporter
from pawp_tts.infrastructure.neural.bilingual_tts import PAWPBilingualTTS
from pawp_tts.infrastructure.persistence.artifact_pruner import prune_artifacts
from pawp_tts.infrastructure.persistence.config_loader import load_student_config
from pawp_tts.infrastructure.persistence.ledger import JsonlExperimentLedger, current_git_sha
from pawp_tts.infrastructure.training.trainer import BilingualStudentTrainer, CheckpointManager
from pawp_tts.interfaces.container import AppContainer, build_container


def _cmd_frontend(container: AppContainer, args: argparse.Namespace) -> int:
    if args.validate:
        cases = [
            ("EN contrast", "She sells thin shells by the vine and wine."),
            ("NE sibilant merger", "शहरमा सस्तो षड्यन्त्र चल्यो।"),
            ("NE gemination", "पाता र पत्ता अलग शब्द हुन्।"),
            ("NE schwa & infinitive", "तीन दिन घर जान दिन नपाएर मित्र रोयो।"),
            ("Code-mix clitic", "आज मेरो officeमा meeting छ।"),
            ("Code-mix hyphen", "मेरो बैंक balance check-गरिदिनुहोस्।"),
            ("Code-mix digits NE", "मेरो 500 रुपैयाँको bill छ।"),
            ("Code-mix digits EN", "Please pay 500 dollars via mobile app."),
            ("Code-mix percent", "आज 50% discount छ।"),
            ("Code-mix tech", "तिम्रो SMS र 4G data सक्रिय भयो?"),
            ("Code-mix podcast", "दुवै जनाको narrative अहिलेसम्म पनि different छ।"),
            ("Sanitization", "[urgent] Check IBAN NP-23-001... please!"),
        ]
        for label, prompt in cases:
            seq = container.phonemize.execute(prompt)
            print(f"[{label}] norm='{seq.normalized_text}' -> {' '.join(seq.symbols)}")
        # Assert core Nepali sibilant merger invariant (no postalveolar /ʃ/ on native Nepali)
        sib_seq = container.phonemize.execute("शहर सस्तो षड्यन्त्र")
        if "ʃ" in sib_seq.symbols:
            raise AssertionError("Hindi sibilant /ʃ/ bleed detected in Nepali frontend!")

        # Validate 300-word native gold Kathmandu pronunciation sample
        gold_passed = 0
        gold_failures: list[str] = []
        for g_case in GOLD_300_CASES:
            g_seq = container.phonemize.execute(g_case.word)
            # filter out <bos> and <eos>
            g_syms = tuple(s for s in g_seq.symbols if not s.startswith("<"))
            if g_syms == g_case.expected_ipa:
                gold_passed += 1
            else:
                gold_failures.append(
                    f"{g_case.word} ({g_case.pos}): exp={' '.join(g_case.expected_ipa)} got={' '.join(g_syms)}"
                )
        print(
            f"[Gold 300 Native Sample] {gold_passed}/{len(GOLD_300_CASES)} passed "
            f"({gold_passed / len(GOLD_300_CASES) * 100:.1f}%)"
        )
        if gold_failures:
            print("First 5 mismatches:\n  " + "\n  ".join(gold_failures[:5]))
            raise AssertionError(f"Gold sample failures: {len(gold_failures)}/300")

        print("Frontend validation passed successfully.")
        return 0

    if not args.text:
        raise SystemExit("Pass --text '...' or --validate")
    seq = container.phonemize.execute(args.text)
    print(f"Normalized: {seq.normalized_text}")
    print("Spans:", [(s.language.value, s.text) for s in seq.spans])
    print("IPA:", " ".join(seq.symbols))
    return 0


def _cmd_count_params(container: AppContainer, args: argparse.Namespace) -> int:
    cfg = load_student_config(args.config)
    report = container.count_params.execute(
        config_path=args.config,
        exp_id=args.exp_id,
        seed=args.seed,
        git_sha=current_git_sha(),
        log_ledger=args.log_ledger,
    )
    print(f"Model: {cfg.model_name} (target <= {cfg.target_budget:,})")
    print(f"  total_params:             {report.total_params:,}")
    print(f"  unique_stored_params:     {report.unique_stored_params:,}")
    print(f"  effective_compute_params: {report.effective_compute_params:,}")
    print(f"  macs_per_sec_audio:       {report.macs_per_sec_audio:,}")
    print(
        f"  serialized_bytes:         FP32={report.fp32_bytes / 1e6:.2f} MB | "
        f"FP16={report.fp16_bytes / 1e6:.2f} MB | INT8={report.int8_bytes / 1e6:.2f} MB | "
        f"INT4={report.int4_bytes / 1e6:.2f} MB"
    )
    print("  module_breakdown:")
    for name, val in report.breakdown_by_module.items():
        pct = 100.0 * val / max(report.total_params, 1)
        print(f"    {name:<30} {val:>10,} ({pct:5.1f}%)")
    budget_metric = report.unique_stored_params if cfg.num_shared_bases > 0 else report.total_params
    if budget_metric > cfg.target_budget:
        raise AssertionError(
            f"Parameter budget exceeded: {budget_metric:,} > {cfg.target_budget:,}"
        )
    return 0


def _cmd_inspect_ckpt(container: AppContainer, args: argparse.Namespace) -> int:
    summary = container.inspect_checkpoint.execute(args.path)
    print(f"Checkpoint: {summary.path}")
    print(f"  total_params: {summary.total_params:,} across {summary.tensor_count} tensors")
    print(f"  dtypes:       {summary.dtypes}")
    print("  top_shape_pools:")
    for shape_str, cnt in summary.shape_pools.items():
        print(f"    {shape_str:<20} x{cnt}")
    return 0


def _cmd_asr_transcribe(container: AppContainer, args: argparse.Namespace) -> int:
    lang = LanguageCode.EN if args.lang == "en" else LanguageCode.NE
    transcript, wer, cer = container.evaluate_asr.execute(
        audio_path=args.audio,
        language=lang,
        reference_text=args.ref,
    )
    print(f"[{lang.value}] ({transcript.latency_ms:.1f} ms) -> {transcript.text}")
    if wer is not None and cer is not None:
        print(
            f"  WER={wer.error_rate:.3f} (S={wer.substitutions}, D={wer.deletions}, "
            f"I={wer.insertions}) | CER={cer.error_rate:.3f}"
        )
    return 0


def _cmd_asr_smoke(container: AppContainer, _args: argparse.Namespace) -> int:
    ne_dir = Path("data/sources/nepali_studio_wavs")
    ne_wav = next(ne_dir.glob("*.wav"), None) if ne_dir.exists() else None
    if ne_wav is None:
        raise FileNotFoundError(f"No WAV files found in {ne_dir}")

    ne_tx, _, _ = container.evaluate_asr.execute(str(ne_wav), LanguageCode.NE)
    print(f"NE smoke [{ne_wav.name}] ({ne_tx.latency_ms:.1f} ms): {ne_tx.text}")

    espeak = shutil.which("espeak-ng")
    if espeak is None:
        raise RuntimeError("espeak-ng not found to synthesize temporary EN smoke WAV")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        subprocess.run(
            [espeak, "-v", "en-us", "-w", tmp.name, "Hello, this is a speech test."],
            check=True,
            capture_output=True,
        )
        en_tx, wer, cer = container.evaluate_asr.execute(
            tmp.name,
            LanguageCode.EN,
            reference_text="Hello, this is a speech test.",
        )
        print(f"EN smoke [temp_espeak.wav] ({en_tx.latency_ms:.1f} ms): {en_tx.text}")
        if wer is not None and cer is not None:
            print(f"  EN smoke WER={wer.error_rate:.3f} | CER={cer.error_rate:.3f}")
    return 0


def _cmd_prune_artifacts(_container: AppContainer, args: argparse.Namespace) -> int:
    removed = prune_artifacts(".", finalize_release=args.finalize_release)
    print(f"Pruned {len(removed)} artifact path(s) (finalize_release={args.finalize_release}).")
    return 0


def _print_split_summary(data: dict[str, object]) -> None:
    print("Dataset Split Summary:")
    print(f"  counts_by_split:      {data['counts_by_split']}")
    print(f"  hours_by_split:       {data['hours_by_split']}")
    print(f"  hours_by_language:    {data['hours_by_language']}")
    print(f"  speakers_by_split:    {data['speakers_by_split']}")
    print(
        f"  dropped:              duplicates={data['dropped_duplicates']}, "
        f"duration={data['dropped_duration']}, unk={data['dropped_unk']}, "
        f"cross_split={data['removed_cross_split']}"
    )
    print(
        f"  leakage_check:        text={data['leakage_text_count']}, "
        f"speaker={data['leakage_speaker_count']}"
    )


def _cmd_build_manifests(container: AppContainer, args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    if args.verify_only:
        summary_file = out_dir / "summary.json"
        if not summary_file.exists():
            raise FileNotFoundError(
                f"Missing {summary_file}; run `make data-manifest` before `make data-splits`."
            )
        for split_name in ("train", "val", "test"):
            split_file = out_dir / f"{split_name}.jsonl"
            if not split_file.exists() or split_file.stat().st_size == 0:
                raise AssertionError(f"Expected non-empty split manifest: {split_file}")
        data = json.loads(summary_file.read_text(encoding="utf-8"))
        if int(data["leakage_text_count"]) != 0 or int(data["leakage_speaker_count"]) != 0:
            raise AssertionError(f"Split leakage detected in {summary_file}: {data}")
        _print_split_summary(data)
        print("Split verification passed (zero text/speaker leakage).")
        return 0

    summary = container.build_manifests.execute(
        output_dir=args.output_dir,
        include_supplemental=args.include_supplemental,
        max_en_shards=args.max_en_shards,
        exp_id=args.exp_id,
        git_sha=current_git_sha(),
    )
    _print_split_summary(
        {
            "counts_by_split": summary.counts_by_split,
            "hours_by_split": summary.hours_by_split,
            "hours_by_language": summary.hours_by_language,
            "speakers_by_split": summary.speakers_by_split,
            "dropped_duplicates": summary.dropped_duplicates,
            "dropped_duration": summary.dropped_duration,
            "dropped_unk": summary.dropped_unk,
            "removed_cross_split": summary.removed_cross_split,
            "leakage_text_count": summary.leakage_text_count,
            "leakage_speaker_count": summary.leakage_speaker_count,
        }
    )
    return 0


def _cmd_export_onnx(_container: AppContainer, args: argparse.Namespace) -> int:
    cfg = load_student_config(args.config)
    model = PAWPBilingualTTS(cfg).eval()
    if args.checkpoint:
        CheckpointManager.load(model, args.checkpoint)
    paths = OnnxStreamingExporter.export(model, args.lang, args.output_dir)
    enc_kb = paths.encoder_dur_path.stat().st_size / 1024.0
    dec_kb = paths.frame_decoder_path.stat().st_size / 1024.0
    print(f"Exported folded '{args.lang}' ONNX streaming graphs to {args.output_dir}:")
    print(f"  encoder_dur:    {paths.encoder_dur_path} ({enc_kb:.1f} KB)")
    print(f"  frame_decoder:  {paths.frame_decoder_path} ({dec_kb:.1f} KB)")
    print(f"  progress_npz:   {paths.progress_weights_path}")
    return 0


def _cmd_bench_latency(container: AppContainer, args: argparse.Namespace) -> int:
    reports = run_latency_benchmarks(
        config_path=args.config,
        phonemize=container.phonemize,
        num_samples=args.num_samples,
        warmup_samples=args.warmup_samples,
        cpu_threads=args.cpu_threads,
        first_chunk_frames=args.first_chunk_frames,
        chunk_frames=args.chunk_frames,
    )
    metrics: dict[str, float | int | str] = {}
    for rep in reports:
        print(
            f"[{rep.backend}] device={rep.device} threads={rep.threads} "
            f"samples={rep.num_samples} (warmup={rep.warmup_samples}) chunk={rep.chunk_ms:.1f}ms | "
            f"FE p50={rep.frontend_p50_ms:.3f}ms p95={rep.frontend_p95_ms:.3f}ms | "
            f"TTFA p50={rep.ttfa_p50_ms:.2f}ms p95={rep.ttfa_p95_ms:.2f}ms "
            f"p99={rep.ttfa_p99_ms:.2f}ms | RTFx={rep.rtfx:.2f} RTF={rep.rtf:.4f} | "
            f"RSS={rep.peak_rss_mb:.1f}MB"
        )
        if rep.backend in {"onnx_cpu_folded", "pytorch_cuda_folded"}:
            metrics[f"{rep.backend}_frontend_p50_ms"] = round(rep.frontend_p50_ms, 4)
            metrics[f"{rep.backend}_ttfa_p50_ms"] = round(rep.ttfa_p50_ms, 3)
            metrics[f"{rep.backend}_ttfa_p95_ms"] = round(rep.ttfa_p95_ms, 3)
            metrics[f"{rep.backend}_ttfa_p99_ms"] = round(rep.ttfa_p99_ms, 3)
            metrics[f"{rep.backend}_rtfx"] = round(rep.rtfx, 2)
            metrics[f"{rep.backend}_rtf"] = round(rep.rtf, 5)
            metrics[f"{rep.backend}_peak_rss_mb"] = round(rep.peak_rss_mb, 1)

    if args.log_ledger:
        JsonlExperimentLedger().append(
            ExperimentRecord(
                exp_id=args.exp_id,
                timestamp_utc=datetime.now(tz=UTC).isoformat(),
                git_sha=current_git_sha(),
                config_path=args.config,
                seed=42,
                hypothesis=(
                    "Monolingual folding + bounded-halo ONNX streaming yields low CPU/GPU TTFA and high RTFx."
                ),
                metrics=metrics,
                command="make bench-latency",
            )
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the root CLI argument parser."""
    parser = argparse.ArgumentParser(prog="pawp", description="PAWP-TTS CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fe = sub.add_parser("frontend", help="Normalize and phonemize text")
    p_fe.add_argument("--text", type=str, default="")
    p_fe.add_argument("--validate", action="store_true")

    p_cp = sub.add_parser("count-params", help="Audit 4-metric parameter budget")
    p_cp.add_argument("--config", type=str, default="configs/student_1_5m.yaml")
    p_cp.add_argument("--exp-id", type=str, default="exp000")
    p_cp.add_argument("--seed", type=int, default=42)
    p_cp.add_argument("--log-ledger", action="store_true")

    p_ic = sub.add_parser("inspect-ckpt", help="Inspect safetensors checkpoint headers")
    p_ic.add_argument("--path", type=str, required=True)

    p_asr = sub.add_parser("asr-transcribe", help="Transcribe a WAV file via ASR API")
    p_asr.add_argument("--audio", type=str, required=True)
    p_asr.add_argument("--lang", choices=["en", "ne"], required=True)
    p_asr.add_argument("--ref", type=str, default=None)

    sub.add_parser("asr-smoke", help="Smoke test EN and NE ASR endpoints")

    p_pr = sub.add_parser("prune-artifacts", help="Remove ephemeral caches/checkpoints")
    p_pr.add_argument("--finalize-release", action="store_true")

    p_bm = sub.add_parser("build-manifests", help="Compile leak-free EN/NE/CodeMix manifests")
    p_bm.add_argument("--output-dir", type=str, default="data/manifests")
    p_bm.add_argument("--include-supplemental", action="store_true")
    p_bm.add_argument("--max-en-shards", type=int, default=None)
    p_bm.add_argument("--exp-id", type=str, default="exp001-data")
    p_bm.add_argument("--verify-only", action="store_true")

    p_ox = sub.add_parser("export-onnx", help="Export folded monolingual ONNX streaming graphs")
    p_ox.add_argument("--config", type=str, default="configs/student_1_5m.yaml")
    p_ox.add_argument("--checkpoint", type=str, default=None)
    p_ox.add_argument("--lang", choices=["en", "ne"], default="en")
    p_ox.add_argument("--output-dir", type=str, default="artifacts/cache/onnx_export")

    p_bl = sub.add_parser("bench-latency", help="Benchmark CPU/ONNX/GPU streaming TTFA & RTFx")
    p_bl.add_argument("--config", type=str, default="configs/student_1_5m.yaml")
    p_bl.add_argument("--num-samples", type=int, default=30)
    p_bl.add_argument("--warmup-samples", type=int, default=3)
    p_bl.add_argument("--cpu-threads", type=int, default=4)
    p_bl.add_argument("--first-chunk-frames", type=int, default=6)
    p_bl.add_argument("--chunk-frames", type=int, default=16)
    p_bl.add_argument("--exp-id", type=str, default="exp001-latency")
    p_bl.add_argument("--log-ledger", action="store_true")

    p_te = sub.add_parser(
        "teacher-extract", help="Extract & cache 24 kHz acoustic/duration features"
    )
    p_te.add_argument("--cache-root", type=str, default="data/cache/v2")
    p_te.add_argument("--en-hours", type=float, default=80.0)
    p_te.add_argument("--workers", type=int, default=12)
    p_te.add_argument(
        "--extras", action="store_true", help="Add extra Nepali sources as split train_extra"
    )
    p_te.add_argument("--finetune", action="store_true", help="(Re)build fine-tune split train_ft")
    p_te.add_argument("--exp-id", type=str, default="exp002-extract")
    p_te.add_argument("--log-ledger", action="store_true")

    p_tg = sub.add_parser("teacher-generate", help="Generate ASR-filtered single-voice corpus")
    p_tg.add_argument("--voice", type=str, default=None, help="defaults to PAWP_TTS_VOICE")
    p_tg.add_argument("--workers", type=int, default=8)
    p_tg.add_argument("--langs", type=str, default="en,ne", help="comma list of en,ne,mixed")
    p_tg.add_argument("--exp-id", type=str, default="exp004-teacher")
    p_tg.add_argument("--log-ledger", action="store_true")

    for name, help_txt in (
        ("train", "Train bilingual student (v2: aligner + full cache)"),
        ("finetune-bilingual", "Continue/resume bilingual training"),
    ):
        p_tr = sub.add_parser(name, help=help_txt)
        p_tr.add_argument("--config", type=str, default="configs/student_1_5m.yaml")
        p_tr.add_argument("--steps", type=int, default=200_000)
        p_tr.add_argument("--max-batch-frames", type=int, default=16_000)
        p_tr.add_argument("--max-batch", type=int, default=32)
        p_tr.add_argument("--lr", type=float, default=1e-3)
        p_tr.add_argument("--gan-start", type=int, default=30_000)
        p_tr.add_argument("--eval-every", type=int, default=2_000)
        p_tr.add_argument("--cache-root", type=str, default="data/cache/v2")
        p_tr.add_argument("--checkpoint-dir", type=str, default="artifacts/checkpoints")
        p_tr.add_argument(
            "--target-speaker", type=str, default=os.environ.get("PAWP_TARGET_SPEAKER", "")
        )
        p_tr.add_argument("--no-resume", action="store_true")
        p_tr.add_argument("--train-splits", type=str, default="train,train_extra")
        p_tr.add_argument("--replay-items", type=int, default=0)
        p_tr.add_argument("--init-from", type=str, default=None)
        p_tr.add_argument("--aligner-warmup", type=int, default=0)
        p_tr.add_argument("--fresh-aligner", action="store_true")
        p_tr.add_argument("--prosody-free-prob", type=float, default=0.0)
        p_tr.add_argument("--seed", type=int, default=42)
        p_tr.add_argument("--exp-id", type=str, default="exp003-train")
        p_tr.add_argument("--log-ledger", action="store_true")

    p_ev = sub.add_parser("eval-suite", help="Run EN, NE, NepTTS-Bench, or CodeMix evaluation")
    p_ev.add_argument(
        "--suite", choices=["en", "ne", "neptts", "codemix", "heldout"], required=True
    )
    p_ev.add_argument("--per-language", type=int, default=30)
    p_ev.add_argument("--config", type=str, default="configs/student_1_5m.yaml")
    p_ev.add_argument("--checkpoint", type=str, default=None)
    p_ev.add_argument("--live-asr", action="store_true")
    p_ev.add_argument("--exp-id", type=str, default="exp002-eval")
    p_ev.add_argument("--log-ledger", action="store_true")

    p_sy = sub.add_parser("synth", help="Synthesize text to a 24 kHz WAV audio file")
    p_sy.add_argument(
        "--text", "-t", required=True, help="Text to synthesize (EN, NE, or code-mixed)"
    )
    p_sy.add_argument(
        "--output", "-o", default="output.wav", help="Output WAV path (default: output.wav)"
    )
    p_sy.add_argument("--checkpoint", "-c", default=None, help="Path to .safetensors checkpoint")
    p_sy.add_argument(
        "--config", default="configs/student_1_5m.yaml", help="Path to student config YAML"
    )
    p_sy.add_argument("--speaker-id", type=int, default=None, help="Optional integer speaker ID")
    p_sy.add_argument("--device", default=None, help="Execution device (cuda or cpu)")

    return parser


def _cmd_teacher_extract(_container: AppContainer, args: argparse.Namespace) -> int:
    if args.extras:
        print(f"Extra Nepali manifest: {json.dumps(build_extra_nepali_manifest())}")
    if args.finetune:
        from pawp_tts.infrastructure.data.bulk_cache import build_codemix_synth_manifest
        from pawp_tts.infrastructure.data.teacher_generation import write_teacher_manifest

        print(f"Teacher manifest: {json.dumps(write_teacher_manifest())}")
        print(f"Code-mix synth manifest: {json.dumps(build_codemix_synth_manifest())}")
    stats = build_feature_cache(
        out_root=args.cache_root,
        en_hours=args.en_hours,
        workers=args.workers,
        only_splits=("train_ft",) if args.finetune else (("train_extra",) if args.extras else None),
    )
    print(f"Feature cache summary ({args.cache_root}): {json.dumps(stats)}")
    if args.log_ledger:
        JsonlExperimentLedger().append(
            ExperimentRecord(
                exp_id=args.exp_id,
                timestamp_utc=datetime.now(tz=UTC).isoformat(),
                git_sha=current_git_sha(),
                config_path="configs/student_1_5m.yaml",
                seed=42,
                hypothesis="Build full v2 24 kHz feature cache (mel/F0/energy/tokens/speaker); durations learned online.",
                metrics={"summary": json.dumps(stats)},
                command="make teacher-extract",
            )
        )
    return 0


def _cmd_train(_container: AppContainer, args: argparse.Namespace) -> int:
    trainer = BilingualStudentTrainer(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        cache_root=args.cache_root,
        target_speaker=args.target_speaker,
    )
    log_path = Path(args.checkpoint_dir) / f"{args.exp_id}.log.jsonl"
    stats = trainer.train(
        steps=args.steps,
        max_batch_frames=args.max_batch_frames,
        max_batch=args.max_batch,
        lr=args.lr,
        seed=args.seed,
        resume=not args.no_resume,
        gan_start=args.gan_start,
        eval_every=args.eval_every,
        log_path=log_path,
        run_name=args.exp_id,
        train_splits=tuple(args.train_splits.split(",")),
        replay_items=args.replay_items,
        init_from=args.init_from,
        aligner_warmup=args.aligner_warmup,
        fresh_aligner=args.fresh_aligner,
        prosody_free_prob=args.prosody_free_prob,
    )
    print(f"Training summary [{args.config}]: {stats}")
    if args.log_ledger:
        JsonlExperimentLedger().append(
            ExperimentRecord(
                exp_id=args.exp_id,
                timestamp_utc=datetime.now(tz=UTC).isoformat(),
                git_sha=current_git_sha(),
                config_path=args.config,
                seed=args.seed,
                hypothesis="v2 student training: learned MAS alignment, masked losses, "
                "teacher-forced prosody, val-selected checkpoint.",
                metrics=stats,
                command=f"make train CONFIG={args.config} SEED={args.seed}",
            )
        )
    return 0


_cmd_finetune_bilingual = _cmd_train


def _cmd_eval_suite(container: AppContainer, args: argparse.Namespace) -> int:
    asr: AsrEvaluatorPort | None = HttpMultipartAsrAdapter() if args.live_asr else None
    mixed_asr: AsrEvaluatorPort | None = None
    if (
        args.live_asr
        and args.suite in {"codemix", "heldout"}
        and os.environ.get("PAWP_MIXED_ASR_PYTHON")
    ):
        from pawp_tts.infrastructure.evaluation.mixed_script_asr_adapter import (
            MixedScriptLocalAsrAdapter,
        )

        mixed_asr = MixedScriptLocalAsrAdapter()
        if args.suite == "codemix":
            asr = mixed_asr  # code-mixed speech needs a mixed-script recognizer
    out_dir = Path("artifacts/audio") / args.exp_id
    if args.suite == "heldout":
        if asr is None:
            raise SystemExit("heldout suite requires --live-asr")
        stats = run_heldout_asr_eval(
            config_path=args.config,
            checkpoint_path=args.checkpoint or "artifacts/checkpoints/best.safetensors",
            phonemize=container.phonemize,
            asr=asr,
            per_language=args.per_language,
            out_dir=out_dir,
            asr_mixed=mixed_asr,
        )
    else:
        stats = run_evaluation_suite(
            suite=args.suite,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            phonemize=container.phonemize,
            asr=asr,
            live_asr=args.live_asr,
            out_dir=out_dir,
        )
    print(f"Evaluation suite [{args.suite}]: {stats}")
    if args.log_ledger:
        JsonlExperimentLedger().append(
            ExperimentRecord(
                exp_id=args.exp_id,
                timestamp_utc=datetime.now(tz=UTC).isoformat(),
                git_sha=current_git_sha(),
                config_path=args.config,
                seed=42,
                hypothesis=f"Evaluate bilingual phonological & synthesis fidelity on suite '{args.suite}'.",
                metrics=stats,
                command=f"make eval-{args.suite} CONFIG={args.config}",
            )
        )
    return 0


def _cmd_teacher_generate(_container: AppContainer, args: argparse.Namespace) -> int:
    from pawp_tts.infrastructure.data.teacher_generation import generate_teacher_corpus

    sizes = {"en": 9000, "ne": 9000, "mixed": 7000}
    per_lang = {k: sizes[k] for k in args.langs.split(",") if k in sizes}
    stats = generate_teacher_corpus(voice=args.voice, workers=args.workers, per_lang=per_lang)
    print(f"Teacher corpus [{args.voice}]: {json.dumps(stats)}")
    if args.log_ledger:
        JsonlExperimentLedger().append(
            ExperimentRecord(
                exp_id=args.exp_id,
                timestamp_utc=datetime.now(tz=UTC).isoformat(),
                git_sha=current_git_sha(),
                config_path="n/a",
                seed=7,
                hypothesis="Single-voice EN/NE/code-mix teacher corpus, ASR-CER filtered.",
                metrics={"summary": json.dumps(stats)},
                command="make teacher-generate",
            )
        )
    return 0


def _cmd_synth(container: AppContainer, args: argparse.Namespace) -> int:
    ckpt: str | None = args.checkpoint
    if not ckpt:
        candidates = [
            Path("artifacts/checkpoints/exp008/best.safetensors"),
            Path("artifacts/checkpoints/exp007/best.safetensors"),
            Path("artifacts/checkpoints/exp007/latest.safetensors"),
            Path("artifacts/checkpoints/best.safetensors"),
        ]
        for c in candidates:
            if c.exists():
                ckpt = str(c)
                break
    if not ckpt or not Path(ckpt).exists():
        raise SystemExit(f"No checkpoint found at {ckpt}")

    cfg = load_student_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = PAWPBilingualTTS(cfg).to(device).eval()
    CheckpointManager.load(model, ckpt)

    speaker_id: int | None = args.speaker_id
    if speaker_id is None and model.speaker_emb is not None:
        speaker_id = 0

    _, _, wav, _ = _synthesize_utterance(
        model, container.phonemize, args.text, device, speaker_id=speaker_id
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wav, 24000)
    print(f"Synthesized audio: {out_path} ({len(wav) / 24000:.2f}s, 24 kHz)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    container = build_container()

    handlers = {
        "frontend": _cmd_frontend,
        "count-params": _cmd_count_params,
        "inspect-ckpt": _cmd_inspect_ckpt,
        "asr-transcribe": _cmd_asr_transcribe,
        "asr-smoke": _cmd_asr_smoke,
        "prune-artifacts": _cmd_prune_artifacts,
        "build-manifests": _cmd_build_manifests,
        "export-onnx": _cmd_export_onnx,
        "bench-latency": _cmd_bench_latency,
        "teacher-extract": _cmd_teacher_extract,
        "teacher-generate": _cmd_teacher_generate,
        "train": _cmd_train,
        "finetune-bilingual": _cmd_finetune_bilingual,
        "eval-suite": _cmd_eval_suite,
        "synth": _cmd_synth,
    }
    return handlers[args.command](container, args)


if __name__ == "__main__":
    raise SystemExit(main())
