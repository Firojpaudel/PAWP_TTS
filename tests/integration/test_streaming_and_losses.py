"""Integration tests for monolingual folding, ONNX/PyTorch streaming, features, and losses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pawp_tts.infrastructure.data.feature_extractor import (
    DynamicFrameBucketSampler,
    MelSpectrogramExtractor,
    pad_to_frame_batch,
)
from pawp_tts.infrastructure.inference.streaming import (
    OnnxStreamingExporter,
    OnnxStreamingRunner,
    PyTorchStreamingRunner,
    hann_istft_numpy,
)
from pawp_tts.infrastructure.neural.bilingual_tts import PAWPBilingualTTS
from pawp_tts.infrastructure.neural.losses import (
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    PAWPTrainingLoss,
    discriminator_lsgan_loss,
    generator_adversarial_and_fm_loss,
)
from pawp_tts.infrastructure.persistence.config_loader import load_student_config
from pawp_tts.infrastructure.weight_programming.param_auditor import count_module_params


def _sample_inputs(
    batch_size: int = 1, seq_len: int = 8, lang_id: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    p_ids = torch.randint(5, 80, (batch_size, seq_len), dtype=torch.long)
    l_ids = torch.full((batch_size, seq_len), lang_id, dtype=torch.long)
    s_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    b_flags = torch.zeros((batch_size, seq_len, 4), dtype=torch.float32)
    mask_vec = [1.0, 0.0] if lang_id == 1 else [0.0, 1.0]
    l_masks = (
        torch.tensor(mask_vec, dtype=torch.float32)
        .view(1, 1, 2)
        .expand(batch_size, seq_len, 2)
        .clone()
    )
    durations = torch.full((batch_size, seq_len), 4.0, dtype=torch.float32)
    return p_ids, l_ids, s_ids, b_flags, l_masks, durations


def test_fold_monolingual_param_count_and_parity() -> None:
    cfg = load_student_config("configs/student_1_5m.yaml")
    model = PAWPBilingualTTS(cfg).eval()
    folded_en = model.fold_monolingual("en")

    assert count_module_params(model) == 1_490_390
    assert count_module_params(folded_en) == 1_293_750

    p_ids, l_ids, s_ids, b_flags, l_masks, durations = _sample_inputs(
        batch_size=1, seq_len=10, lang_id=1
    )
    with torch.no_grad():
        out_bi = model(p_ids, l_ids, s_ids, b_flags, l_masks, target_durations=durations)
        out_fold = folded_en(p_ids, l_ids, s_ids, b_flags, l_masks, target_durations=durations)

    assert torch.allclose(out_bi.durations, out_fold.durations, atol=1e-5)
    assert torch.allclose(out_bi.mel, out_fold.mel, atol=1e-5)
    assert torch.allclose(out_bi.waveform, out_fold.waveform, atol=1e-5)


def test_hann_istft_numpy_matches_torch_istft() -> None:
    rng = np.random.default_rng(123)
    real = rng.standard_normal((513, 28), dtype=np.float32) * 0.1
    imag = rng.standard_normal((513, 28), dtype=np.float32) * 0.1
    # Zero DC and Nyquist bins as produced by decoder
    real[0, :] = 0.0
    imag[0, :] = 0.0
    real[-1, :] = 0.0
    imag[-1, :] = 0.0

    np_wav = hann_istft_numpy(real, imag, n_fft=1024, hop_length=240)
    pt_complex = torch.complex(torch.from_numpy(real), torch.from_numpy(imag)).unsqueeze(0)
    pt_wav = (
        torch.istft(
            pt_complex,
            n_fft=1024,
            hop_length=240,
            win_length=1024,
            window=torch.hann_window(1024),
            center=True,
        )
        .squeeze(0)
        .numpy()
    )
    assert np_wav.shape == pt_wav.shape
    assert np.allclose(np_wav, pt_wav, atol=1e-4)


def test_onnx_export_and_streaming_runner_parity(tmp_path: Path) -> None:
    cfg = load_student_config("configs/student_1_5m.yaml")
    model = PAWPBilingualTTS(cfg).eval()
    folded_en = model.fold_monolingual("en")
    paths = OnnxStreamingExporter.export(model, "en", tmp_path)

    assert paths.encoder_dur_path.exists()
    assert paths.frame_decoder_path.exists()
    assert paths.progress_weights_path.exists()

    p_ids, l_ids, s_ids, b_flags, l_masks, durations = _sample_inputs(
        batch_size=1, seq_len=10, lang_id=1
    )

    pt_runner = PyTorchStreamingRunner(folded_en, device="cpu")
    pt_chunks = list(
        pt_runner.stream_pcm(
            p_ids,
            l_ids,
            s_ids,
            b_flags,
            l_masks,
            first_chunk_frames=6,
            chunk_frames=16,
            target_durations=durations,
        )
    )
    ort_runner = OnnxStreamingRunner(paths, intra_op_threads=1)
    ort_chunks = list(
        ort_runner.stream_pcm(
            p_ids.numpy(),
            l_ids.numpy(),
            s_ids.numpy(),
            b_flags.numpy(),
            first_chunk_frames=6,
            chunk_frames=16,
            target_durations=durations.numpy(),
        )
    )

    assert len(pt_chunks) >= 2
    assert len(pt_chunks) == len(ort_chunks)
    for chunk in ort_chunks:
        assert chunk.dtype == np.float32
        assert chunk.ndim == 1
        assert chunk.size > 0
        assert np.all(np.isfinite(chunk))

    assert np.allclose(np.concatenate(pt_chunks), np.concatenate(ort_chunks), atol=1e-4)


def test_feature_extractor_sampler_and_losses_backward() -> None:
    extractor = MelSpectrogramExtractor(sample_rate=24000, n_fft=1024, hop_length=240, n_mels=80)
    wav = torch.randn(2, 24000)
    log_mel, log_energy = extractor(wav)
    assert log_mel.shape[0] == 2
    assert log_mel.shape[2] == 80
    assert log_energy.shape == (2, log_mel.shape[1], 1)
    assert torch.all(torch.isfinite(log_mel))

    lengths = [120, 250, 140, 410, 110, 260]
    sampler = DynamicFrameBucketSampler(
        frame_lengths=lengths,
        max_frames_per_batch=550,
        max_batch_size=2,
        shuffle=True,
        seed=42,
    )
    batches = list(sampler)
    seen = [idx for batch in batches for idx in batch]
    assert sorted(seen) == list(range(len(lengths)))
    for batch in batches:
        assert len(batch) <= 2
        assert max(lengths[i] for i in batch) * len(batch) <= 550

    padded = pad_to_frame_batch([torch.randn(3, 8), torch.randn(6, 8)])
    assert padded.shape == (2, 6, 8)

    # Full differentiable loss + discriminators smoke check
    cfg = load_student_config("configs/student_1_5m.yaml")
    student = PAWPBilingualTTS(cfg).train()
    mpd = MultiPeriodDiscriminator()
    mrd = MultiResolutionDiscriminator()
    criterion = PAWPTrainingLoss()

    p_ids, l_ids, s_ids, b_flags, l_masks, durations = _sample_inputs(
        batch_size=2, seq_len=8, lang_id=1
    )
    out = student(p_ids, l_ids, s_ids, b_flags, l_masks, target_durations=durations)

    target_wav = torch.randn_like(out.waveform)
    target_mel = torch.randn_like(out.mel)
    target_pv = torch.randn_like(out.pitch_voicing)
    target_energy = torch.randn_like(out.energy)
    weights = torch.tensor([1.0, 0.8], dtype=torch.float32)

    g_loss, metrics = criterion(
        pred_wav=out.waveform,
        target_wav=target_wav,
        pred_mel=out.mel,
        target_mel=target_mel,
        pred_dur=out.durations,
        target_dur=durations,
        pred_pv=out.pitch_voicing,
        target_pv=target_pv,
        pred_energy=out.energy,
        target_energy=target_energy,
        sample_weights=weights,
    )
    mpd_ro, mpd_fo, mpd_rf, mpd_ff = mpd(target_wav.unsqueeze(1), out.waveform.unsqueeze(1))
    mrd_ro, mrd_fo, mrd_rf, mrd_ff = mrd(target_wav.unsqueeze(1), out.waveform.unsqueeze(1))
    d_loss = discriminator_lsgan_loss(mpd_ro + mrd_ro, [t.detach() for t in mpd_fo + mrd_fo])
    adv_loss, fm_loss = generator_adversarial_and_fm_loss(
        mpd_fo + mrd_fo, mpd_rf + mrd_rf, mpd_ff + mrd_ff
    )

    total_g = g_loss + 0.1 * (adv_loss + fm_loss)
    total_g.backward()
    d_loss.backward()

    assert torch.isfinite(total_g)
    assert torch.isfinite(d_loss)
    assert metrics["loss_total"] > 0.0
    assert any(
        p.grad is not None and torch.all(torch.isfinite(p.grad)) for p in student.parameters()
    )
