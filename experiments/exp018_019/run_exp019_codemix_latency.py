"""EXP-019: Code-mix unfolded (both residuals active) vs folded monolingual latency benchmark."""

from __future__ import annotations

import json
import resource
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from pawp_tts.domain.entities.models import LanguageCode
from pawp_tts.domain.phonology.codemix import build_language_masks
from pawp_tts.infrastructure.inference.streaming import (
    OnnxStreamingExporter,
    OnnxStreamingRunner,
    _expand_frames_numpy,
    _stream_frame_windows,
)
from pawp_tts.infrastructure.neural.bilingual_tts import PAWPBilingualTTS
from pawp_tts.infrastructure.persistence.config_loader import load_student_config
from pawp_tts.infrastructure.training.trainer import CheckpointManager
from pawp_tts.interfaces.container import build_container


class _GatedEnc(torch.nn.Module):
    def __init__(self, m: PAWPBilingualTTS) -> None:
        super().__init__()
        self.m = m

    def forward(self, phoneme_ids, lang_ids, stress_ids, boundary_flags, lang_masks):
        return self.m.encode_tokens(
            phoneme_ids, lang_ids, stress_ids, boundary_flags, lang_masks=lang_masks
        )


class _GatedDec(torch.nn.Module):
    def __init__(self, m: PAWPBilingualTTS) -> None:
        super().__init__()
        self.m = m

    def forward(self, frame_h, frame_masks):
        _, pitch_cond = self.m.pitch_predictor(frame_h)
        h = frame_h + pitch_cond
        _, energy_cond = self.m.energy_predictor(h)
        h = h + energy_cond
        return self.m.decode_frame_spectrum(h, frame_masks=frame_masks)


def _export_gated_onnx(model: PAWPBilingualTTS, out_dir: Path) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    model.eval().cpu()
    enc_p = out_dir / "gated_enc.onnx"
    dec_p = out_dir / "gated_dec.onnx"
    T, F, dm = 12, 24, model.cfg.d_model
    torch.onnx.export(
        _GatedEnc(model).eval(),
        (
            torch.zeros(1, T, dtype=torch.long),
            torch.zeros(1, T, dtype=torch.long),
            torch.zeros(1, T, dtype=torch.long),
            torch.zeros(1, T, 4, dtype=torch.float32),
            torch.full((1, T, 2), 0.5, dtype=torch.float32),
        ),
        str(enc_p),
        input_names=["phoneme_ids", "lang_ids", "stress_ids", "boundary_flags", "lang_masks"],
        output_names=["enc_h", "durations"],
        dynamic_axes={
            "phoneme_ids": {1: "T"},
            "lang_ids": {1: "T"},
            "stress_ids": {1: "T"},
            "boundary_flags": {1: "T"},
            "lang_masks": {1: "T"},
            "enc_h": {1: "T"},
            "durations": {1: "T"},
        },
        opset_version=17,
        dynamo=False,
    )
    torch.onnx.export(
        _GatedDec(model).eval(),
        (
            torch.zeros(1, F, dm, dtype=torch.float32),
            torch.full((1, F, 2), 0.5, dtype=torch.float32),
        ),
        str(dec_p),
        input_names=["frame_h", "frame_masks"],
        output_names=["mel", "real_spec", "imag_spec"],
        dynamic_axes={
            "frame_h": {1: "F"},
            "frame_masks": {1: "F"},
            "mel": {1: "F"},
            "real_spec": {2: "F"},
            "imag_spec": {2: "F"},
        },
        opset_version=17,
        dynamo=False,
    )
    pw = model.length_regulator.progress_proj.weight.detach().cpu().numpy().astype(np.float32)
    pb = model.length_regulator.progress_proj.bias.detach().cpu().numpy().astype(np.float32)
    return enc_p, dec_p, pw, pb


def _frontend_inputs(phonemize, text: str):
    t0 = time.perf_counter()
    seq = phonemize.execute(text)
    fe_ms = (time.perf_counter() - t0) * 1000.0
    toks = seq.tokens
    lmap = {LanguageCode.SHARED: 0, LanguageCode.EN: 1, LanguageCode.NE: 2, LanguageCode.MIXED: 3}
    p_ids = np.array([[t.token_id for t in toks]], dtype=np.int64)
    l_ids = np.array([[lmap.get(t.language, 0) for t in toks]], dtype=np.int64)
    s_ids = np.array([[min(t.stress, 3) for t in toks]], dtype=np.int64)
    b_flags = np.array(
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
        dtype=np.float32,
    )
    l_masks = np.array([build_language_masks(toks)], dtype=np.float32)
    return p_ids, l_ids, s_ids, b_flags, l_masks, fe_ms


def _bench_gated(
    enc_p: Path,
    dec_p: Path,
    pw: np.ndarray,
    pb: np.ndarray,
    n_fft: int,
    hop: int,
    sr: int,
    phonemize,
    texts: list[str],
    first_chunk_frames: int,
    threads: int = 4,
    warmups: int = 3,
    runs: int = 30,
) -> dict[str, float]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    enc_s = ort.InferenceSession(str(enc_p), sess_options=opts, providers=["CPUExecutionProvider"])
    dec_s = ort.InferenceSession(str(dec_p), sess_options=opts, providers=["CPUExecutionProvider"])
    hann = np.hanning(n_fft + 1)[:-1].astype(np.float32)

    ttfas, totals, aud_durs, fes = [], [], [], []
    for i in range(warmups + runs):
        txt = texts[i % len(texts)]
        p_ids, l_ids, s_ids, b_flags, l_masks, fe_ms = _frontend_inputs(phonemize, txt)
        t0 = time.perf_counter()
        enc_h, pred_dur = enc_s.run(
            None,
            {
                "phoneme_ids": p_ids,
                "lang_ids": l_ids,
                "stress_ids": s_ids,
                "boundary_flags": b_flags,
                "lang_masks": l_masks,
            },
        )
        int_durs = np.maximum(np.round(pred_dur[0]).astype(np.int64), 1)
        frame_h = _expand_frames_numpy(enc_h, pred_dur, pw, pb)
        frame_m = np.repeat(l_masks[0], int_durs, axis=0)[None, :, :].astype(np.float32)
        total_frames = frame_h.shape[1]

        first_ms = None
        n_samp = 0
        for pcm in _stream_frame_windows(
            total_frames=total_frames,
            first_chunk_frames=first_chunk_frames,
            chunk_frames=16,
            left_halo=24,
            right_halo=11,
            n_fft=n_fft,
            hop_length=hop,
            window=hann,
            decode_window_fn=lambda w0, w1: (
                lambda res: (res[1][0], res[2][0])
            )(
                dec_s.run(
                    None,
                    {
                        "frame_h": frame_h[:, w0:w1, :].astype(np.float32),
                        "frame_masks": frame_m[:, w0:w1, :].astype(np.float32),
                    },
                )
            ),
        ):
            if first_ms is None:
                first_ms = (time.perf_counter() - t0) * 1000.0
            n_samp += len(pcm)
        tot_s = time.perf_counter() - t0
        if i >= warmups:
            fes.append(fe_ms)
            ttfas.append(first_ms or 0.0)
            totals.append(tot_s)
            aud_durs.append(n_samp / sr)

    rtf = sum(totals) / max(sum(aud_durs), 1e-6)
    return {
        "threads": threads,
        "warmups": warmups,
        "runs": runs,
        "first_chunk_ms": first_chunk_frames * 10.0,
        "fe_p50_ms": round(float(np.percentile(fes, 50)), 3),
        "ttfa_p50_ms": round(float(np.percentile(ttfas, 50)), 3),
        "ttfa_p95_ms": round(float(np.percentile(ttfas, 95)), 3),
        "ttfa_p99_ms": round(float(np.percentile(ttfas, 99)), 3),
        "rtf": round(rtf, 5),
        "rtfx": round(1.0 / max(rtf, 1e-9), 2),
        "rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
    }


def _bench_folded(
    model: PAWPBilingualTTS,
    phonemize,
    texts: list[str],
    out_dir: Path,
    first_chunk_frames: int,
    threads: int = 4,
    warmups: int = 3,
    runs: int = 30,
) -> dict[str, float]:
    paths = OnnxStreamingExporter.export(model, language="en", output_dir=out_dir)
    runner = OnnxStreamingRunner(paths, intra_op_threads=threads)
    ttfas, totals, aud_durs, fes = [], [], [], []
    for i in range(warmups + runs):
        txt = texts[i % len(texts)]
        p_ids, l_ids, s_ids, b_flags, _l_masks, fe_ms = _frontend_inputs(phonemize, txt)
        t0 = time.perf_counter()
        first_ms = None
        n_samp = 0
        for pcm in runner.stream_pcm(
            p_ids,
            l_ids,
            s_ids,
            b_flags,
            first_chunk_frames=first_chunk_frames,
            chunk_frames=16,
        ):
            if first_ms is None:
                first_ms = (time.perf_counter() - t0) * 1000.0
            n_samp += len(pcm)
        tot_s = time.perf_counter() - t0
        if i >= warmups:
            fes.append(fe_ms)
            ttfas.append(first_ms or 0.0)
            totals.append(tot_s)
            aud_durs.append(n_samp / model.cfg.sample_rate)

    rtf = sum(totals) / max(sum(aud_durs), 1e-6)
    return {
        "threads": threads,
        "warmups": warmups,
        "runs": runs,
        "first_chunk_ms": first_chunk_frames * 10.0,
        "fe_p50_ms": round(float(np.percentile(fes, 50)), 3),
        "ttfa_p50_ms": round(float(np.percentile(ttfas, 50)), 3),
        "ttfa_p95_ms": round(float(np.percentile(ttfas, 95)), 3),
        "ttfa_p99_ms": round(float(np.percentile(ttfas, 99)), 3),
        "rtf": round(rtf, 5),
        "rtfx": round(1.0 / max(rtf, 1e-9), 2),
        "rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
    }


def main() -> None:
    phonemize = build_container().phonemize
    cm_texts = [
        json.loads(x)["text"]
        for x in Path("data/evaluation/codemix_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ][:30]

    models = [
        ("pawp_1_5m_v3", "configs/student_1_5m.yaml", "artifacts/checkpoints/exp015_1_5m_v3/best.safetensors"),
        ("shared_300k_s42", "configs/student_300k_pawp.yaml", "artifacts/checkpoints/exp015_300k_shared_v3/best.safetensors"),
        ("dense_300k_s42", "configs/student_300k_dense.yaml", "artifacts/checkpoints/exp015_300k_dense_v3/best.safetensors"),
    ]

    report: dict[str, object] = {"protocol": "4 CPU threads, 3 warmups, 30 timed runs on codemix_v1 sentences", "results": {}}
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        for name, cfg_p, ckpt_p in models:
            cfg = load_student_config(cfg_p)
            model = PAWPBilingualTTS(cfg).eval()
            CheckpointManager.load(model, ckpt_p)
            mdir = td_p / name
            mdir.mkdir(parents=True, exist_ok=True)
            enc_p, dec_p, pw, pb = _export_gated_onnx(model, mdir)

            report["results"][name] = {}
            for fcf in (6, 3):
                folded_m = _bench_folded(model, phonemize, cm_texts, mdir / f"folded_{fcf}", first_chunk_frames=fcf)
                unfolded_m = _bench_gated(
                    enc_p, dec_p, pw, pb, cfg.n_fft, cfg.hop_length, cfg.sample_rate, phonemize, cm_texts, first_chunk_frames=fcf
                )
                overhead_ttfa = round((unfolded_m["ttfa_p50_ms"] / max(folded_m["ttfa_p50_ms"], 1e-6) - 1.0) * 100.0, 1)
                overhead_rtf = round((unfolded_m["rtf"] / max(folded_m["rtf"], 1e-6) - 1.0) * 100.0, 1)
                report["results"][name][f"chunk_{fcf * 10}ms"] = {
                    "folded_monolingual": folded_m,
                    "unfolded_codemix_gated": unfolded_m,
                    "unfolded_vs_folded_ttfa_p50_overhead_pct": overhead_ttfa,
                    "unfolded_vs_folded_rtf_overhead_pct": overhead_rtf,
                }
                print(
                    f"[{name} {fcf*10}ms] folded TTFA={folded_m['ttfa_p50_ms']}ms RTFx={folded_m['rtfx']}x | "
                    f"unfolded TTFA={unfolded_m['ttfa_p50_ms']}ms ({overhead_ttfa:+.1f}%) RTFx={unfolded_m['rtfx']}x ({overhead_rtf:+.1f}%)",
                    flush=True,
                )

    out_p = Path("artifacts/reports/exp019_codemix_latency.json")
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {out_p}")


if __name__ == "__main__":
    main()
