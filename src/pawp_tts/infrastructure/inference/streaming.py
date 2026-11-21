"""Stateful chunked streaming synthesis and folded monolingual ONNX export."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from pawp_tts.infrastructure.neural.bilingual_tts import PAWPBilingualTTS


def hann_istft_numpy(
    real_spec: np.ndarray,
    imag_spec: np.ndarray,
    n_fft: int = 1024,
    hop_length: int = 240,
    window: np.ndarray | None = None,
) -> np.ndarray:
    """
    Fast NumPy vectorized inverse STFT matching `torch.istft(..., center=True)`.
    Expects `(n_bins, T_frames)` real and imaginary arrays.
    """
    if window is None:
        window = np.hanning(n_fft + 1)[:-1].astype(np.float32)

    n_frames = real_spec.shape[1]
    if n_frames <= 0:
        return np.zeros(0, dtype=np.float32)

    complex_spec = real_spec.astype(np.float32) + 1j * imag_spec.astype(np.float32)
    frames = np.fft.irfft(complex_spec, n=n_fft, axis=0).astype(np.float32)
    frames *= window[:, None]

    raw_len = (n_frames - 1) * hop_length + n_fft
    audio = np.zeros(raw_len, dtype=np.float32)
    win_env = np.zeros(raw_len, dtype=np.float32)
    win_sq = window * window

    for t in range(n_frames):
        offset = t * hop_length
        audio[offset : offset + n_fft] += frames[:, t]
        win_env[offset : offset + n_fft] += win_sq

    nonzero = win_env > 1e-8
    audio[nonzero] /= win_env[nonzero]
    pad = n_fft // 2
    if raw_len <= 2 * pad:
        return np.zeros(0, dtype=np.float32)
    return audio[pad : raw_len - pad].astype(np.float32)


class _OnnxEncoderDurWrapper(nn.Module):
    """Exportable encoder + duration predictor graph (folded monolingual weights)."""

    def __init__(self, model: PAWPBilingualTTS) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        phoneme_ids: torch.Tensor,
        lang_ids: torch.Tensor,
        stress_ids: torch.Tensor,
        boundary_flags: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enc_h, durations = self.model.encode_tokens(
            phoneme_ids=phoneme_ids,
            lang_ids=lang_ids,
            stress_ids=stress_ids,
            boundary_flags=boundary_flags,
            lang_masks=None,
            speaker_ids=None,
        )
        return enc_h, durations


class _OnnxFrameDecoderWrapper(nn.Module):
    """Exportable frame-level pitch/energy + acoustic + complex STFT head."""

    def __init__(self, model: PAWPBilingualTTS) -> None:
        super().__init__()
        self.model = model

    def forward(self, frame_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pitch_voicing, pitch_cond = self.model.pitch_predictor(frame_h)
        _ = pitch_voicing
        h = frame_h + pitch_cond
        energy, energy_cond = self.model.energy_predictor(h)
        _ = energy
        h = h + energy_cond
        return self.model.decode_frame_spectrum(h, frame_masks=None)


@dataclass(frozen=True)
class ExportedOnnxPaths:
    """File paths for exported monolingual ONNX streaming artifacts."""

    encoder_dur_path: Path
    frame_decoder_path: Path
    progress_weights_path: Path
    sample_rate: int
    hop_length: int
    n_fft: int


class OnnxStreamingExporter:
    """Exports folded monolingual PAWPBilingualTTS into lightweight ONNX streaming graphs."""

    @staticmethod
    def export(
        model: PAWPBilingualTTS,
        language: Literal["en", "ne"],
        output_dir: str | Path,
    ) -> ExportedOnnxPaths:
        """Fold language residuals into dense weights and export encoder + frame decoder ONNX."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        folded = model.fold_monolingual(language).cpu().eval()
        enc_wrapper = _OnnxEncoderDurWrapper(folded).eval()
        dec_wrapper = _OnnxFrameDecoderWrapper(folded).eval()

        dummy_ids = torch.ones((1, 16), dtype=torch.long)
        dummy_flags = torch.zeros((1, 16, 4), dtype=torch.float32)
        enc_path = out_dir / f"pawp_{language}_encoder_dur.onnx"

        with torch.no_grad():
            torch.onnx.export(
                enc_wrapper,
                (dummy_ids, dummy_ids, dummy_ids, dummy_flags),
                str(enc_path),
                input_names=["phoneme_ids", "lang_ids", "stress_ids", "boundary_flags"],
                output_names=["enc_h", "durations"],
                dynamic_axes={
                    "phoneme_ids": {1: "seq_len"},
                    "lang_ids": {1: "seq_len"},
                    "stress_ids": {1: "seq_len"},
                    "boundary_flags": {1: "seq_len"},
                    "enc_h": {1: "seq_len"},
                    "durations": {1: "seq_len"},
                },
                opset_version=17,
                dynamo=False,
            )

            dummy_frames = torch.zeros((1, 24, folded.cfg.d_model), dtype=torch.float32)
            dec_path = out_dir / f"pawp_{language}_frame_decoder.onnx"
            torch.onnx.export(
                dec_wrapper,
                (dummy_frames,),
                str(dec_path),
                input_names=["frame_h"],
                output_names=["mel", "real_spec", "imag_spec"],
                dynamic_axes={
                    "frame_h": {1: "num_frames"},
                    "mel": {1: "num_frames"},
                    "real_spec": {2: "num_frames"},
                    "imag_spec": {2: "num_frames"},
                },
                opset_version=17,
                dynamo=False,
            )

        prog_path = out_dir / f"pawp_{language}_progress.npz"
        np.savez(
            prog_path,
            weight=folded.length_regulator.progress_proj.weight.detach().cpu().numpy(),
            bias=folded.length_regulator.progress_proj.bias.detach().cpu().numpy(),
        )

        return ExportedOnnxPaths(
            encoder_dur_path=enc_path,
            frame_decoder_path=dec_path,
            progress_weights_path=prog_path,
            sample_rate=folded.cfg.sample_rate,
            hop_length=folded.cfg.hop_length,
            n_fft=folded.cfg.n_fft,
        )


def _expand_frames_numpy(
    enc_h: np.ndarray,
    durations: np.ndarray,
    prog_weight: np.ndarray,
    prog_bias: np.ndarray,
) -> np.ndarray:
    """Vectorized NumPy length regulation for ultra-low CPU overhead."""
    int_durs = np.maximum(np.round(durations[0]).astype(np.int64), 1)
    expanded = np.repeat(enc_h[0], int_durs, axis=0)
    prog_chunks = []
    for d in int_durs:
        steps = np.arange(d, dtype=np.float32)
        denom = float(max(d, 1))
        prog_chunks.append(np.stack([steps / denom, (d - 1 - steps) / denom], axis=-1))
    prog = np.concatenate(prog_chunks, axis=0)
    return cast(np.ndarray, (expanded + (prog @ prog_weight.T + prog_bias))[None, :, :])


class OnnxStreamingRunner:
    """Batch-size-1 CPU ONNX Runtime chunked streaming synthesizer."""

    def __init__(self, paths: ExportedOnnxPaths, intra_op_threads: int = 4) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._enc_sess = ort.InferenceSession(
            str(paths.encoder_dur_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._dec_sess = ort.InferenceSession(
            str(paths.frame_decoder_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        prog_data = np.load(paths.progress_weights_path)
        self._prog_w = prog_data["weight"].astype(np.float32)
        self._prog_b = prog_data["bias"].astype(np.float32)
        self._n_fft = paths.n_fft
        self._hop_length = paths.hop_length
        self._hann = np.hanning(self._n_fft + 1)[:-1].astype(np.float32)

    def stream_pcm(
        self,
        phoneme_ids: np.ndarray,
        lang_ids: np.ndarray,
        stress_ids: np.ndarray,
        boundary_flags: np.ndarray,
        first_chunk_frames: int = 6,
        chunk_frames: int = 16,
        left_halo: int = 24,
        right_halo: int = 11,
        target_durations: np.ndarray | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield 24 kHz float32 PCM chunks incrementally with exact halo context."""
        enc_h, pred_dur = self._enc_sess.run(
            None,
            {
                "phoneme_ids": phoneme_ids.astype(np.int64),
                "lang_ids": lang_ids.astype(np.int64),
                "stress_ids": stress_ids.astype(np.int64),
                "boundary_flags": boundary_flags.astype(np.float32),
            },
        )
        used_dur = target_durations if target_durations is not None else pred_dur
        frame_h = _expand_frames_numpy(enc_h, used_dur, self._prog_w, self._prog_b)
        total_frames = frame_h.shape[1]

        yield from _stream_frame_windows(
            total_frames=total_frames,
            first_chunk_frames=first_chunk_frames,
            chunk_frames=chunk_frames,
            left_halo=left_halo,
            right_halo=right_halo,
            n_fft=self._n_fft,
            hop_length=self._hop_length,
            window=self._hann,
            decode_window_fn=lambda w_start, w_end: self._decode_window(
                frame_h[:, w_start:w_end, :]
            ),
        )

    def _decode_window(self, frame_win: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, real_spec, imag_spec = self._dec_sess.run(
            None,
            {"frame_h": frame_win.astype(np.float32)},
        )
        return cast(np.ndarray, real_spec[0]), cast(np.ndarray, imag_spec[0])


class PyTorchStreamingRunner:
    """PyTorch CPU/GPU chunked streaming synthesizer supporting both folded and bilingual models."""

    def __init__(self, model: PAWPBilingualTTS, device: str = "cpu") -> None:
        self._device = torch.device(device)
        self._model = model.to(self._device).eval()
        self._n_fft = model.cfg.n_fft
        self._hop_length = model.cfg.hop_length
        self._hann = np.hanning(self._n_fft + 1)[:-1].astype(np.float32)

    @torch.inference_mode()
    def stream_pcm(
        self,
        phoneme_ids: torch.Tensor,
        lang_ids: torch.Tensor,
        stress_ids: torch.Tensor,
        boundary_flags: torch.Tensor,
        lang_masks: torch.Tensor,
        first_chunk_frames: int = 6,
        chunk_frames: int = 16,
        left_halo: int = 24,
        right_halo: int = 11,
        target_durations: torch.Tensor | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield 24 kHz float32 PCM chunks incrementally on CPU or CUDA."""
        p_ids = phoneme_ids.to(self._device)
        l_ids = lang_ids.to(self._device)
        s_ids = stress_ids.to(self._device)
        b_flags = boundary_flags.to(self._device)
        l_masks = lang_masks.to(self._device)

        enc_h, pred_dur = self._model.encode_tokens(
            phoneme_ids=p_ids,
            lang_ids=l_ids,
            stress_ids=s_ids,
            boundary_flags=b_flags,
            lang_masks=l_masks if self._model.cfg.residual_rank > 0 else None,
        )
        used_dur = target_durations.to(self._device) if target_durations is not None else pred_dur
        frame_h, frame_masks, _, _ = self._model.prepare_frame_hidden(enc_h, l_masks, used_dur)
        total_frames = frame_h.shape[1]

        def _decode_fn(w_start: int, w_end: int) -> tuple[np.ndarray, np.ndarray]:
            fm = frame_masks[:, w_start:w_end, :] if self._model.cfg.residual_rank > 0 else None
            _, real_s, imag_s = self._model.decode_frame_spectrum(frame_h[:, w_start:w_end, :], fm)
            return real_s[0].detach().cpu().numpy(), imag_s[0].detach().cpu().numpy()

        yield from _stream_frame_windows(
            total_frames=total_frames,
            first_chunk_frames=first_chunk_frames,
            chunk_frames=chunk_frames,
            left_halo=left_halo,
            right_halo=right_halo,
            n_fft=self._n_fft,
            hop_length=self._hop_length,
            window=self._hann,
            decode_window_fn=_decode_fn,
        )


def _stream_frame_windows(
    total_frames: int,
    first_chunk_frames: int,
    chunk_frames: int,
    left_halo: int,
    right_halo: int,
    n_fft: int,
    hop_length: int,
    window: np.ndarray,
    decode_window_fn: object,
) -> Iterator[np.ndarray]:
    """Slice bounded receptive-field windows and yield clean overlap-free PCM chunks."""
    from collections.abc import Callable

    decode_cb = cast(Callable[[int, int], tuple[np.ndarray, np.ndarray]], decode_window_fn)
    cursor = 0
    step = 0
    while cursor < total_frames:
        cur_size = first_chunk_frames if step == 0 else chunk_frames
        end = min(cursor + cur_size, total_frames)
        w_start = max(0, cursor - left_halo)
        w_end = min(total_frames, end + right_halo)

        real_win, imag_win = decode_cb(w_start, w_end)
        # Keep tiny iSTFT context frame overlap (2 frames on each side) to eliminate edge ripple
        spec_pad_l = min(2, cursor - w_start)
        spec_pad_r = min(2, w_end - end)
        rel_l = cursor - w_start
        rel_r = end - w_start

        sub_real = real_win[:, rel_l - spec_pad_l : rel_r + spec_pad_r]
        sub_imag = imag_win[:, rel_l - spec_pad_l : rel_r + spec_pad_r]
        audio_win = hann_istft_numpy(
            sub_real, sub_imag, n_fft=n_fft, hop_length=hop_length, window=window
        )

        trim_l = spec_pad_l * hop_length
        target_samples = (
            (end - cursor) * hop_length
            if end < total_frames
            else max((end - cursor - 1) * hop_length, 0)
        )
        if len(audio_win) >= trim_l + target_samples and target_samples > 0:
            yield audio_win[trim_l : trim_l + target_samples]
        elif len(audio_win) > trim_l:
            yield audio_win[trim_l:]
        cursor = end
        step += 1
