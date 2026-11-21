"""Open TTS Leaderboard–aligned CPU/ONNX/GPU streaming TTFA & RTFx latency profiler."""

from __future__ import annotations

import resource
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import torch

from pawp_tts.application.frontend_service import PhonemizeUtteranceUseCase
from pawp_tts.domain.entities.models import LanguageCode, LatencyBenchmarkReport
from pawp_tts.domain.phonology.codemix import build_language_masks
from pawp_tts.infrastructure.inference.streaming import (
    OnnxStreamingExporter,
    OnnxStreamingRunner,
    PyTorchStreamingRunner,
)
from pawp_tts.infrastructure.neural.bilingual_tts import PAWPBilingualTTS
from pawp_tts.infrastructure.persistence.config_loader import load_student_config

DEFAULT_BENCHMARK_PROMPTS: tuple[str, ...] = (
    "She sells seashells by the seashore while testing low latency streaming audio.",
    "The quick brown fox jumps over the lazy dog before sunrise.",
    "Artificial intelligence speech synthesis requires crisp pronunciation and natural rhythm.",
    "Please confirm your account balance and transfer fifty dollars today.",
    "Modern neural acoustic decoders convert spectral frames into crisp waveforms.",
)


def _tensor_inputs_from_text(
    phonemize: PhonemizeUtteranceUseCase,
    text: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seq = phonemize.execute(text)
    tokens = seq.tokens
    p_ids = np.array([[tok.token_id for tok in tokens]], dtype=np.int64)
    lang_map = {
        LanguageCode.SHARED: 0,
        LanguageCode.EN: 1,
        LanguageCode.NE: 2,
        LanguageCode.MIXED: 3,
    }
    l_ids = np.array([[lang_map.get(tok.language, 0) for tok in tokens]], dtype=np.int64)
    s_ids = np.array([[min(tok.stress, 3) for tok in tokens]], dtype=np.int64)
    b_flags = np.array(
        [
            [
                [
                    1.0 if tok.is_word_start else 0.0,
                    1.0 if tok.is_word_end else 0.0,
                    1.0 if tok.is_syllable_start else 0.0,
                    1.0 if tok.is_phrase_break else 0.0,
                ]
                for tok in tokens
            ]
        ],
        dtype=np.float32,
    )
    masks = np.array([build_language_masks(tokens)], dtype=np.float32)
    return p_ids, l_ids, s_ids, b_flags, masks


def run_latency_benchmarks(
    config_path: str,
    phonemize: PhonemizeUtteranceUseCase,
    num_samples: int = 30,
    warmup_samples: int = 3,
    cpu_threads: int = 4,
    first_chunk_frames: int = 6,
    chunk_frames: int = 16,
) -> list[LatencyBenchmarkReport]:
    """
    Benchmark batch-size-1 streaming TTFA (p50/p95/p99) and RTFx across:
      1. ONNX Runtime CPU (folded monolingual)
      2. PyTorch CPU (folded monolingual)
      3. PyTorch CUDA RTX 4090 (if CUDA available)
    """
    cfg = load_student_config(config_path)
    base_model = PAWPBilingualTTS(cfg).eval()
    folded_cpu = base_model.fold_monolingual("en").cpu().eval()
    chunk_ms = 1000.0 * first_chunk_frames * cfg.hop_length / cfg.sample_rate

    prompts = [
        DEFAULT_BENCHMARK_PROMPTS[i % len(DEFAULT_BENCHMARK_PROMPTS)]
        for i in range(warmup_samples + num_samples)
    ]
    reports: list[LatencyBenchmarkReport] = []

    # 1. ONNX Runtime CPU (folded monolingual)
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = OnnxStreamingExporter.export(base_model, "en", Path(tmpdir))
        onnx_runner = OnnxStreamingRunner(paths, intra_op_threads=cpu_threads)
        reports.append(
            _bench_stream_runner(
                backend="onnx_cpu_folded",
                device="cpu",
                threads=cpu_threads,
                prompts=prompts,
                warmup_samples=warmup_samples,
                chunk_ms=chunk_ms,
                sample_rate=cfg.sample_rate,
                phonemize=phonemize,
                stream_fn=lambda p, lang_arr, s, b, _m: onnx_runner.stream_pcm(
                    p,
                    lang_arr,
                    s,
                    b,
                    first_chunk_frames=first_chunk_frames,
                    chunk_frames=chunk_frames,
                ),
            )
        )

    # 2. PyTorch CPU (folded monolingual)
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(cpu_threads)
    try:
        pt_cpu = PyTorchStreamingRunner(folded_cpu, device="cpu")
        reports.append(
            _bench_stream_runner(
                backend="pytorch_cpu_folded",
                device="cpu",
                threads=cpu_threads,
                prompts=prompts,
                warmup_samples=warmup_samples,
                chunk_ms=chunk_ms,
                sample_rate=cfg.sample_rate,
                phonemize=phonemize,
                stream_fn=lambda p, lang_arr, s, b, m: pt_cpu.stream_pcm(
                    torch.from_numpy(p),
                    torch.from_numpy(lang_arr),
                    torch.from_numpy(s),
                    torch.from_numpy(b),
                    torch.from_numpy(m),
                    first_chunk_frames=first_chunk_frames,
                    chunk_frames=chunk_frames,
                ),
            )
        )
    finally:
        torch.set_num_threads(prev_threads)

    # 3. PyTorch GPU (CUDA) if available
    if torch.cuda.is_available():
        folded_gpu = base_model.fold_monolingual("en").cuda().eval()
        pt_gpu = PyTorchStreamingRunner(folded_gpu, device="cuda")

        def _gpu_stream(
            p: np.ndarray,
            lang_arr: np.ndarray,
            s: np.ndarray,
            b: np.ndarray,
            m: np.ndarray,
        ) -> Iterator[np.ndarray]:
            for ch in pt_gpu.stream_pcm(
                torch.from_numpy(p),
                torch.from_numpy(lang_arr),
                torch.from_numpy(s),
                torch.from_numpy(b),
                torch.from_numpy(m),
                first_chunk_frames=first_chunk_frames,
                chunk_frames=chunk_frames,
            ):
                torch.cuda.synchronize()
                yield ch

        reports.append(
            _bench_stream_runner(
                backend="pytorch_cuda_folded",
                device=torch.cuda.get_device_name(0),
                threads=1,
                prompts=prompts,
                warmup_samples=warmup_samples,
                chunk_ms=chunk_ms,
                sample_rate=cfg.sample_rate,
                phonemize=phonemize,
                stream_fn=_gpu_stream,
            )
        )

    return reports


def _bench_stream_runner(
    backend: str,
    device: str,
    threads: int,
    prompts: list[str],
    warmup_samples: int,
    chunk_ms: float,
    sample_rate: int,
    phonemize: PhonemizeUtteranceUseCase,
    stream_fn: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Iterator[np.ndarray],
    ],
) -> LatencyBenchmarkReport:
    fe_times: list[float] = []
    ttfa_times: list[float] = []
    total_compute_s = 0.0
    total_audio_s = 0.0

    for idx, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        p, lang_arr, s, b, m = _tensor_inputs_from_text(phonemize, prompt)
        t_fe = time.perf_counter()

        it = iter(stream_fn(p, lang_arr, s, b, m))
        first_chunk = next(it, None)
        t_first = time.perf_counter()

        rest_chunks = list(it)
        t_end = time.perf_counter()

        if idx >= warmup_samples:
            fe_times.append((t_fe - t0) * 1000.0)
            ttfa_times.append((t_first - t0) * 1000.0)
            total_compute_s += t_end - t0
            n_samples = (len(first_chunk) if first_chunk is not None else 0) + sum(
                len(c) for c in rest_chunks
            )
            total_audio_s += n_samples / float(sample_rate)

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    rtfx = total_audio_s / max(total_compute_s, 1e-9)
    rtf = total_compute_s / max(total_audio_s, 1e-9)

    return LatencyBenchmarkReport(
        backend=backend,
        device=device,
        threads=threads,
        num_samples=len(prompts) - warmup_samples,
        warmup_samples=warmup_samples,
        chunk_ms=chunk_ms,
        frontend_p50_ms=float(np.percentile(fe_times, 50)),
        frontend_p95_ms=float(np.percentile(fe_times, 95)),
        ttfa_p50_ms=float(np.percentile(ttfa_times, 50)),
        ttfa_p95_ms=float(np.percentile(ttfa_times, 95)),
        ttfa_p99_ms=float(np.percentile(ttfa_times, 99)),
        rtfx=rtfx,
        rtf=rtf,
        peak_rss_mb=peak_rss_mb,
    )
