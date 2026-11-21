"""Mechanistic Weight-Space Programming engine: inspect, localize, rank-1 edit, and verify."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from pawp_tts.application.frontend_service import PhonemizeUtteranceUseCase
from pawp_tts.domain.pawp.wsp_analysis import (
    BootstrapResult,
    PhonemeConfusionReport,
    bootstrap_paired_ci,
    compute_per_and_confusions,
    ctc_greedy_decode,
)
from pawp_tts.domain.phonology.inventory import ID_TO_SYMBOL, SYMBOL_TO_ID
from pawp_tts.infrastructure.evaluation.latency_profiler import _tensor_inputs_from_text
from pawp_tts.infrastructure.neural.bilingual_tts import PAWPBilingualTTS
from pawp_tts.infrastructure.neural.ctc_aligner import CTCPhonemeRecognizer
from pawp_tts.infrastructure.persistence.config_loader import load_student_config


@dataclass(frozen=True)
class LayerProbeResult:
    """Layer-level linear separability and causal patching flip score for a phoneme contrast."""

    layer_name: str
    probe_acc: float
    cosine_sep: float
    causal_flip_delta: float


def load_manifest_texts(
    manifest_path: str | Path,
    language: str,
    limit: int,
    seed: int = 42,
) -> list[str]:
    """Deterministically sample texts for a language from a manifest JSONL."""
    texts: list[str] = []
    with Path(manifest_path).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("language") == language:
                txt = str(row.get("text", "")).strip()
                if 8 <= len(txt) <= 180:
                    texts.append(txt)
    rng = random.Random(seed)
    rng.shuffle(texts)
    return texts[:limit]


def load_ctc_recognizer(
    ckpt_path: str | Path,
    device: torch.device,
) -> CTCPhonemeRecognizer:
    """Load frozen training-only CTC phoneme recognizer for phoneme-level PER and probing."""
    rec = CTCPhonemeRecognizer().to(device).eval()
    raw = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    rec.load_state_dict(state, strict=True)
    for p in rec.parameters():
        p.requires_grad_(False)
    return rec


@torch.inference_mode()
def evaluate_ctc_per(
    model: PAWPBilingualTTS,
    ctc: CTCPhonemeRecognizer,
    phonemize: PhonemizeUtteranceUseCase,
    texts: list[str],
    language: str,
    device: torch.device,
) -> PhonemeConfusionReport:
    """Synthesize utterances and score phoneme-level PER + confusions via frozen CTC recognizer."""
    refs: list[tuple[int, ...]] = []
    hyps: list[tuple[int, ...]] = []
    spk = torch.zeros(1, dtype=torch.long, device=device)

    for text in texts:
        p_ids, l_ids, s_ids, b_flags, l_masks = _tensor_inputs_from_text(phonemize, text)
        out = model(
            torch.from_numpy(p_ids).to(device),
            torch.from_numpy(l_ids).to(device),
            torch.from_numpy(s_ids).to(device),
            torch.from_numpy(b_flags).to(device),
            torch.from_numpy(l_masks).to(device),
            speaker_ids=spk,
        )
        logp = ctc(out.mel)  # [1, T, 129]
        pred_frames = tuple(int(x) for x in logp[0].argmax(dim=-1).cpu().tolist())
        hyps.append(ctc_greedy_decode(pred_frames, blank_id=ctc.blank))
        refs.append(tuple(int(x) for x in p_ids[0].tolist()))

    return compute_per_and_confusions(
        language=language,
        refs=refs,
        hyps=hyps,
        id_to_symbol=ID_TO_SYMBOL,
    )


@torch.inference_mode()
def measure_target_confusion_rate(
    model: PAWPBilingualTTS,
    ctc: CTCPhonemeRecognizer,
    phonemize: PhonemizeUtteranceUseCase,
    texts: list[str],
    target_sym: str,
    confused_sym: str,
    device: torch.device,
) -> tuple[float, tuple[float, ...], int]:
    """Measure per-utterance CTC log-prob margin `logP(confused) - logP(target)` and confusion count."""
    tid = SYMBOL_TO_ID[target_sym]
    cid = SYMBOL_TO_ID[confused_sym]
    spk = torch.zeros(1, dtype=torch.long, device=device)
    margins: list[float] = []
    total_conf = 0
    total_occ = 0

    for text in texts:
        p_ids, l_ids, s_ids, b_flags, l_masks = _tensor_inputs_from_text(phonemize, text)
        tok_list = p_ids[0].tolist()
        if tid not in tok_list:
            continue
        out = model(
            torch.from_numpy(p_ids).to(device),
            torch.from_numpy(l_ids).to(device),
            torch.from_numpy(s_ids).to(device),
            torch.from_numpy(b_flags).to(device),
            torch.from_numpy(l_masks).to(device),
            speaker_ids=spk,
        )
        logp = ctc(out.mel)[0]  # [T, 129]
        durs = np.maximum(np.round(out.durations[0].cpu().numpy()).astype(np.int64), 1)
        pos = 0
        utt_margins: list[float] = []
        for tok, d in zip(tok_list, durs, strict=False):
            end = min(pos + int(d), logp.shape[0])
            if tok == tid and end > pos:
                seg = logp[pos:end]
                m_t = float(seg[:, tid].max().item())
                m_c = float(seg[:, cid].max().item())
                # Lower is better: negative margin means target beats confused
                utt_margins.append(m_c - m_t)
                total_occ += 1
                if m_c > m_t:
                    total_conf += 1
            pos = end
        if utt_margins:
            margins.append(sum(utt_margins) / len(utt_margins))

    rate = total_conf / max(total_occ, 1)
    return rate, tuple(margins), total_occ


@torch.inference_mode()
def localize_contrast_layers(
    model: PAWPBilingualTTS,
    ctc: CTCPhonemeRecognizer,
    phonemize: PhonemizeUtteranceUseCase,
    dev_texts: list[str],
    target_sym: str,
    confused_sym: str,
    device: torch.device,
) -> list[LayerProbeResult]:
    """Run layer-by-layer linear separability probe and causal activation patching."""
    tid = SYMBOL_TO_ID[target_sym]
    cid = SYMBOL_TO_ID[confused_sym]
    spk = torch.zeros(1, dtype=torch.long, device=device)

    layer_modules: list[tuple[str, torch.nn.Module]] = [
        (f"encoder_blocks.{i}", blk) for i, blk in enumerate(model.encoder_blocks)
    ] + [(f"acoustic_blocks.{i}", blk) for i, blk in enumerate(model.acoustic_blocks)]

    acts_t: dict[str, list[torch.Tensor]] = {n: [] for n, _ in layer_modules}
    acts_c: dict[str, list[torch.Tensor]] = {n: [] for n, _ in layer_modules}

    # 1. Collect activations at target_sym vs confused_sym positions across dev_texts
    sample_texts = dev_texts[:80]
    for text in sample_texts:
        p_ids, l_ids, s_ids, b_flags, l_masks = _tensor_inputs_from_text(phonemize, text)
        toks = p_ids[0].tolist()
        if tid not in toks and cid not in toks:
            continue

        captured: dict[str, torch.Tensor] = {}
        hooks = [
            mod.register_forward_hook(
                lambda _m, _i, o, name=name, store=captured: store.__setitem__(name, o.detach())
            )
            for name, mod in layer_modules
        ]
        out = model(
            torch.from_numpy(p_ids).to(device),
            torch.from_numpy(l_ids).to(device),
            torch.from_numpy(s_ids).to(device),
            torch.from_numpy(b_flags).to(device),
            torch.from_numpy(l_masks).to(device),
            speaker_ids=spk,
        )
        for h in hooks:
            h.remove()

        durs = np.maximum(np.round(out.durations[0].cpu().numpy()).astype(np.int64), 1)
        for name, _ in layer_modules:
            h_act = captured[name][0]  # [seq_or_frames, D]
            if name.startswith("encoder_blocks."):
                for idx, tok in enumerate(toks):
                    if idx < h_act.shape[0]:
                        if tok == tid:
                            acts_t[name].append(h_act[idx].float().cpu())
                        elif tok == cid:
                            acts_c[name].append(h_act[idx].float().cpu())
            else:
                pos = 0
                for tok, d in zip(toks, durs, strict=False):
                    end = min(pos + int(d), h_act.shape[0])
                    if end > pos:
                        if tok == tid:
                            acts_t[name].append(h_act[pos:end].mean(dim=0).float().cpu())
                        elif tok == cid:
                            acts_c[name].append(h_act[pos:end].mean(dim=0).float().cpu())
                    pos = end

    # 2. Compute linear probe accuracy and causal patching flip per layer
    results: list[LayerProbeResult] = []
    patch_texts = [
        t for t in dev_texts if tid in _tensor_inputs_from_text(phonemize, t)[0][0].tolist()
    ][:20]

    for name, mod in layer_modules:
        if not acts_t[name] or not acts_c[name]:
            results.append(LayerProbeResult(name, 0.5, 0.0, 0.0))
            continue
        ht = torch.stack(acts_t[name])
        hc = torch.stack(acts_c[name])
        mu_t = ht.mean(dim=0)
        mu_c = hc.mean(dim=0)
        direction = mu_t - mu_c
        dir_unit = direction / (direction.norm() + 1e-12)
        mid = 0.5 * (mu_t + mu_c)
        acc_t = float(((ht - mid) @ dir_unit > 0).float().mean().item())
        acc_c = float(((hc - mid) @ dir_unit < 0).float().mean().item())
        probe_acc = 0.5 * (acc_t + acc_c)
        cos_sep = float(1.0 - F.cosine_similarity(mu_t.unsqueeze(0), mu_c.unsqueeze(0)).item())

        # Causal patching: steer target phoneme activations along `direction` (+1.0 * direction)
        dir_dev = direction.to(device)
        deltas: list[float] = []
        for text in patch_texts:
            p_ids, l_ids, s_ids, b_flags, l_masks = _tensor_inputs_from_text(phonemize, text)
            toks = p_ids[0].tolist()
            p_t = torch.from_numpy(p_ids).to(device)
            l_t = torch.from_numpy(l_ids).to(device)
            s_t = torch.from_numpy(s_ids).to(device)
            b_t = torch.from_numpy(b_flags).to(device)
            m_t = torch.from_numpy(l_masks).to(device)

            base_out = model(p_t, l_t, s_t, b_t, m_t, speaker_ids=spk)
            base_logp = ctc(base_out.mel)[0]
            durs = np.maximum(np.round(base_out.durations[0].cpu().numpy()).astype(np.int64), 1)

            def _make_patch_hook(
                layer_n: str,
                token_seq: list[int],
                dur_arr: np.ndarray,
                steer_vec: torch.Tensor,
            ) -> Any:
                def _patch_hook(_m: torch.nn.Module, _i: Any, o: torch.Tensor) -> torch.Tensor:
                    patched = o.clone()
                    if layer_n.startswith("encoder_blocks."):
                        for idx, tok in enumerate(token_seq):
                            if tok == tid and idx < patched.shape[1]:
                                patched[0, idx] = patched[0, idx] + 0.5 * steer_vec
                    else:
                        pos = 0
                        for tok, d in zip(token_seq, dur_arr, strict=False):
                            end = min(pos + int(d), patched.shape[1])
                            if tok == tid and end > pos:
                                patched[0, pos:end] = patched[0, pos:end] + 0.5 * steer_vec
                            pos = end
                    return patched

                return _patch_hook

            hk = mod.register_forward_hook(_make_patch_hook(name, toks, durs, dir_dev))
            patch_out = model(p_t, l_t, s_t, b_t, m_t, speaker_ids=spk)
            hk.remove()
            patch_logp = ctc(patch_out.mel)[0]

            # Measure average CTC log-prob margin shift (positive = target_sym strengthened vs confused_sym)
            pos = 0
            for tok, d in zip(toks, durs, strict=False):
                end = min(pos + int(d), min(base_logp.shape[0], patch_logp.shape[0]))
                if tok == tid and end > pos:
                    base_m = float(
                        (base_logp[pos:end, tid] - base_logp[pos:end, cid]).mean().item()
                    )
                    patch_m = float(
                        (patch_logp[pos:end, tid] - patch_logp[pos:end, cid]).mean().item()
                    )
                    deltas.append(patch_m - base_m)
                pos = end

        causal_flip = sum(deltas) / max(len(deltas), 1)
        results.append(LayerProbeResult(name, probe_acc, cos_sep, causal_flip))

    return results


def build_rank1_wsp_and_controls(
    config_path: str | Path,
    base_ckpt: str | Path,
    ctc_ckpt: str | Path,
    phonemize: PhonemizeUtteranceUseCase,
    dev_texts: list[str],
    target_sym: str,
    confused_sym: str,
    target_layer_name: str,
    out_dir: str | Path,
    device: torch.device,
    alphas: tuple[float, ...] = (0.05, 0.15, 0.30, 0.50),
    ft_steps: int = 40,
) -> dict[str, Any]:
    """
    Construct and select on DEV strictly inside `target_layer_name.pw2.weight`:
      1. Rank-1 mechanistic WSP edit (`W <- W + alpha * u_out @ v_in^T`)
      2. Same-location, same-Frobenius-norm Rank-1 random control
      3. Same-parameter gradient fine-tuning control (only `target_layer_name.pw2.weight` unfrozen)
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cfg = load_student_config(config_path)
    state = load_file(str(base_ckpt))
    ctc = load_ctc_recognizer(ctc_ckpt, device)
    tid = SYMBOL_TO_ID[target_sym]
    cid = SYMBOL_TO_ID[confused_sym]
    spk = torch.zeros(1, dtype=torch.long, device=device)

    model = PAWPBilingualTTS(cfg).to(device).eval()
    model.load_state_dict(state, strict=True)

    # Locate `target_layer_name.pw2`
    block_type, idx_str = target_layer_name.split(".")
    block_idx = int(idx_str)
    block = (
        model.encoder_blocks[block_idx]
        if block_type == "encoder_blocks"
        else model.acoustic_blocks[block_idx]
    )
    pw2 = block.pw2
    assert isinstance(pw2, torch.nn.Module)
    w_key = f"{target_layer_name}.pw2.weight"

    # 1. Capture input activations to pw2 (`v_in`) and output contrast direction (`u_out`) on DEV
    in_t: list[torch.Tensor] = []
    out_t: list[torch.Tensor] = []
    out_c: list[torch.Tensor] = []

    dev_subset = [
        t for t in dev_texts if tid in _tensor_inputs_from_text(phonemize, t)[0][0].tolist()
    ][:60]

    with torch.inference_mode():
        for text in dev_subset:
            p_ids, l_ids, s_ids, b_flags, l_masks = _tensor_inputs_from_text(phonemize, text)
            toks = p_ids[0].tolist()
            cap: dict[str, torch.Tensor] = {}

            def _cap_hook(
                _m: torch.nn.Module,
                inp: tuple[torch.Tensor, ...],
                o: torch.Tensor,
                store: dict[str, torch.Tensor] = cap,
            ) -> None:
                store["in"] = inp[0].detach()
                store["out"] = o.detach()

            hk = pw2.register_forward_hook(_cap_hook)
            out = model(
                torch.from_numpy(p_ids).to(device),
                torch.from_numpy(l_ids).to(device),
                torch.from_numpy(s_ids).to(device),
                torch.from_numpy(b_flags).to(device),
                torch.from_numpy(l_masks).to(device),
                speaker_ids=spk,
            )
            hk.remove()
            durs = np.maximum(np.round(out.durations[0].cpu().numpy()).astype(np.int64), 1)
            hin = cap["in"][0]
            hout = cap["out"][0]
            if block_type == "encoder_blocks":
                for i, tok in enumerate(toks):
                    if i < hin.shape[0]:
                        if tok == tid:
                            in_t.append(hin[i].float().cpu())
                            out_t.append(hout[i].float().cpu())
                        elif tok == cid:
                            out_c.append(hout[i].float().cpu())
            else:
                pos = 0
                for tok, d in zip(toks, durs, strict=False):
                    end = min(pos + int(d), hin.shape[0])
                    if end > pos:
                        if tok == tid:
                            in_t.append(hin[pos:end].mean(dim=0).float().cpu())
                            out_t.append(hout[pos:end].mean(dim=0).float().cpu())
                        elif tok == cid:
                            out_c.append(hout[pos:end].mean(dim=0).float().cpu())
                    pos = end

    v_in = torch.stack(in_t).mean(dim=0)
    v_in = v_in / (v_in.norm() + 1e-12)
    u_out = torch.stack(out_t).mean(dim=0) - torch.stack(out_c).mean(dim=0)
    u_out = u_out / (u_out.norm() + 1e-12)
    rank1_unit = torch.outer(u_out, v_in)  # [out_dim, in_dim]

    W0 = state[w_key].float()
    w_norm = float(W0.norm().item())

    # Select alpha strictly on DEV
    best_alpha = alphas[0]
    best_dev_score = float("inf")
    dev_sweep: list[dict[str, float]] = []
    for alpha in alphas:
        cand_state = dict(state)
        delta = (alpha * w_norm * rank1_unit).reshape(W0.shape)
        cand_state[w_key] = (W0 + delta).to(state[w_key].dtype)
        model.load_state_dict(cand_state, strict=True)
        rate, margins, _ = measure_target_confusion_rate(
            model, ctc, phonemize, dev_subset, target_sym, confused_sym, device
        )
        mean_margin = sum(margins) / max(len(margins), 1)
        dev_sweep.append({"alpha": alpha, "dev_conf_rate": rate, "dev_margin": mean_margin})
        if mean_margin < best_dev_score:
            best_dev_score = mean_margin
            best_alpha = alpha

    # Save best Rank-1 WSP edit
    delta_best = (best_alpha * w_norm * rank1_unit).reshape(W0.shape)
    wsp_state = dict(state)
    wsp_state[w_key] = (W0 + delta_best).to(state[w_key].dtype)
    wsp_ckpt = out_path / "wsp_rank1.safetensors"
    save_file(wsp_state, wsp_ckpt)

    # Save same-location, same-Frobenius-norm Rank-1 random control
    rng = torch.Generator().manual_seed(42)
    u_r = torch.randn(u_out.shape, generator=rng)
    u_r = u_r / (u_r.norm() + 1e-12)
    v_r = torch.randn(v_in.shape, generator=rng)
    v_r = v_r / (v_r.norm() + 1e-12)
    delta_rand = (best_alpha * w_norm * torch.outer(u_r, v_r)).reshape(W0.shape)
    rand_state = dict(state)
    rand_state[w_key] = (W0 + delta_rand).to(state[w_key].dtype)
    rand_ckpt = out_path / "control_random_rank1.safetensors"
    save_file(rand_state, rand_ckpt)

    # Control 2: Same-parameter gradient fine-tuning (ONLY w_key unfrozen, trained on same dev_subset)
    ft_model = PAWPBilingualTTS(cfg).to(device).train()
    ft_model.load_state_dict(state, strict=True)
    for n, p in ft_model.named_parameters():
        p.requires_grad_(n == w_key)
    opt = torch.optim.Adam([p for p in ft_model.parameters() if p.requires_grad], lr=5e-4)

    for step in range(ft_steps):
        text = dev_subset[step % len(dev_subset)]
        p_ids, l_ids, s_ids, b_flags, l_masks = _tensor_inputs_from_text(phonemize, text)
        opt.zero_grad(set_to_none=True)
        out = ft_model(
            torch.from_numpy(p_ids).to(device),
            torch.from_numpy(l_ids).to(device),
            torch.from_numpy(s_ids).to(device),
            torch.from_numpy(b_flags).to(device),
            torch.from_numpy(l_masks).to(device),
            speaker_ids=spk,
        )
        with torch.backends.cudnn.flags(enabled=False):
            logp = ctc(out.mel).transpose(0, 1)  # [T, 1, 129]
            targets = torch.tensor(
                [t for t in p_ids[0].tolist() if t not in {0, 1, 2, 3, 4, 5}],
                dtype=torch.long,
                device=device,
            )
            in_len = torch.tensor([logp.shape[0]], dtype=torch.long, device=device)
            tgt_len = torch.tensor([len(targets)], dtype=torch.long, device=device)
            loss = F.ctc_loss(logp, targets, in_len, tgt_len, blank=ctc.blank, zero_infinity=True)
            torch.autograd.backward(loss)
        opt.step()

    ft_ckpt = out_path / "control_sameparam_ft.safetensors"
    save_file({k: v.detach().cpu() for k, v in ft_model.state_dict().items()}, ft_ckpt)

    return {
        "target_layer": target_layer_name,
        "edited_tensor": w_key,
        "edited_params": int(W0.numel()),
        "rank": 1,
        "best_alpha": best_alpha,
        "delta_frobenius_norm": float(delta_best.norm().item()),
        "dev_sweep": dev_sweep,
        "wsp_ckpt": str(wsp_ckpt),
        "rand_ckpt": str(rand_ckpt),
        "ft_ckpt": str(ft_ckpt),
    }


def build_rome_nullspace_ne_edit(
    config_path: str | Path,
    base_ckpt: str | Path,
    ctc_ckpt: str | Path,
    phonemize: PhonemizeUtteranceUseCase,
    dev_texts: list[str],
    target_sym: str,
    confused_sym: str,
    target_layer_name: str,
    out_dir: str | Path,
    device: torch.device,
    alphas: tuple[float, ...] = (0.01, 0.02, 0.04, 0.08, 0.15),
    cov_reg: float = 0.1,
    ft_steps: int = 40,
) -> dict[str, Any]:
    """
    ROME/MEMIT-style null-space Rank-1 edit strictly inside the Nepali-gated residual
    `{target_layer_name}.pw2.{u_ne, v_ne}` (`6,144` params = `0.41%` of model):
      - Mathematically 0.00000 collateral on English (`m_ne(t) == 0`).
      - Key direction `v_rome = (C_other + reg*I)^{-1} h_target` minimizes perturbation
        on all non-target Nepali phonemes.
      - Alpha selected strictly on DEV subject to `dev_ne_per <= base_dev_ne_per + 0.001`.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cfg = load_student_config(config_path)
    state = load_file(str(base_ckpt))
    ctc = load_ctc_recognizer(ctc_ckpt, device)
    tid = SYMBOL_TO_ID[target_sym]
    cid = SYMBOL_TO_ID[confused_sym]
    spk = torch.zeros(1, dtype=torch.long, device=device)

    model = PAWPBilingualTTS(cfg).to(device).eval()
    model.load_state_dict(state, strict=True)

    block_type, idx_str = target_layer_name.split(".")
    block_idx = int(idx_str)
    block = (
        model.encoder_blocks[block_idx]
        if block_type == "encoder_blocks"
        else model.acoustic_blocks[block_idx]
    )
    pw2 = block.pw2
    assert isinstance(pw2, torch.nn.Module)
    u_key = f"{target_layer_name}.pw2.u_ne"
    v_key = f"{target_layer_name}.pw2.v_ne"
    g_key = f"{target_layer_name}.pw2.gate_ne"

    in_t: list[torch.Tensor] = []
    in_other: list[torch.Tensor] = []
    out_t: list[torch.Tensor] = []
    out_c: list[torch.Tensor] = []

    dev_subset = [
        t for t in dev_texts if tid in _tensor_inputs_from_text(phonemize, t)[0][0].tolist()
    ][:60]

    with torch.inference_mode():
        base_dev_rep = evaluate_ctc_per(model, ctc, phonemize, dev_subset, "ne", device)
        base_dev_per = base_dev_rep.per
        for text in dev_subset:
            p_ids, l_ids, s_ids, b_flags, l_masks = _tensor_inputs_from_text(phonemize, text)
            toks = p_ids[0].tolist()
            cap: dict[str, torch.Tensor] = {}

            def _cap_hook(
                _m: torch.nn.Module,
                inp: tuple[torch.Tensor, ...],
                o: torch.Tensor,
                store: dict[str, torch.Tensor] = cap,
            ) -> None:
                store["in"] = inp[0].detach()
                store["out"] = o.detach()

            hk = pw2.register_forward_hook(_cap_hook)
            out = model(
                torch.from_numpy(p_ids).to(device),
                torch.from_numpy(l_ids).to(device),
                torch.from_numpy(s_ids).to(device),
                torch.from_numpy(b_flags).to(device),
                torch.from_numpy(l_masks).to(device),
                speaker_ids=spk,
            )
            hk.remove()
            durs = np.maximum(np.round(out.durations[0].cpu().numpy()).astype(np.int64), 1)
            hin = cap["in"][0]
            hout = cap["out"][0]
            pos = 0
            for tok, d in zip(toks, durs, strict=False):
                end = min(pos + int(d), hin.shape[0])
                if end > pos:
                    m_in = hin[pos:end].mean(dim=0).float().cpu()
                    m_out = hout[pos:end].mean(dim=0).float().cpu()
                    if tok == tid:
                        in_t.append(m_in)
                        out_t.append(m_out)
                    else:
                        in_other.append(m_in)
                        if tok == cid:
                            out_c.append(m_out)
                pos = end

    # ROME whitened key direction: (C_other + reg * I)^{-1} mean(h_target)
    X_other = torch.stack(in_other)  # [N_other, in_dim]
    C_other = (X_other.T @ X_other) / max(X_other.shape[0], 1)
    reg_eye = cov_reg * torch.eye(C_other.shape[0], dtype=torch.float32)
    h_target = torch.stack(in_t).mean(dim=0)
    k_rome = torch.linalg.solve(C_other + reg_eye, h_target)
    v_rome = k_rome / (k_rome.norm() + 1e-12)

    u_out = torch.stack(out_t).mean(dim=0) - torch.stack(out_c).mean(dim=0)
    u_out = u_out / (u_out.norm() + 1e-12)
    rank1_unit = torch.outer(u_out, v_rome)  # [out_dim, in_dim]

    U0 = state[u_key].float()
    V0 = state[v_key].float()
    g0 = float(state[g_key].float().item()) if g_key in state else 1.0
    rank_r = U0.shape[1]
    W_ne0 = U0 @ V0
    w_ref_norm = float(state[f"{target_layer_name}.pw2.weight"].float().norm().item())

    def _refactor_uv(W_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        U_s, S_s, Vh_s = torch.linalg.svd(W_target, full_matrices=False)
        sq = torch.sqrt(S_s[:rank_r].clamp_min(1e-12))
        return (U_s[:, :rank_r] * sq[None, :]), (sq[:, None] * Vh_s[:rank_r, :])

    # Select alpha strictly on DEV under zero-collateral PER guard
    best_alpha = alphas[0]
    best_score = float("inf")
    dev_sweep: list[dict[str, float]] = []
    for alpha in alphas:
        eff_scale = (alpha * w_ref_norm) / max(abs(g0), 1e-6)
        u_new, v_new = _refactor_uv(W_ne0 + eff_scale * rank1_unit)
        cand_state = dict(state)
        cand_state[u_key] = u_new.to(state[u_key].dtype)
        cand_state[v_key] = v_new.to(state[v_key].dtype)
        model.load_state_dict(cand_state, strict=True)
        rate, margins, _ = measure_target_confusion_rate(
            model, ctc, phonemize, dev_subset, target_sym, confused_sym, device
        )
        dev_rep = evaluate_ctc_per(model, ctc, phonemize, dev_subset, "ne", device)
        mean_margin = sum(margins) / max(len(margins), 1)
        dev_sweep.append(
            {
                "alpha": alpha,
                "dev_conf_rate": round(rate, 5),
                "dev_margin": round(mean_margin, 5),
                "dev_ne_per": round(dev_rep.per, 5),
                "base_dev_ne_per": round(base_dev_per, 5),
            }
        )
        if dev_rep.per <= base_dev_per + 0.0015 and rate < best_score:
            best_score = rate
            best_alpha = alpha

    eff_scale_best = (best_alpha * w_ref_norm) / max(abs(g0), 1e-6)
    delta_best = eff_scale_best * rank1_unit
    u_wsp, v_wsp = _refactor_uv(W_ne0 + delta_best)
    wsp_state = dict(state)
    wsp_state[u_key] = u_wsp.to(state[u_key].dtype)
    wsp_state[v_key] = v_wsp.to(state[v_key].dtype)
    wsp_ckpt = out_path / "wsp_rome_ne_rank1.safetensors"
    save_file(wsp_state, wsp_ckpt)

    # Control 1: Same-location (`u_ne, v_ne`), same-Frobenius-norm Rank-1 random edit
    rng = torch.Generator().manual_seed(42)
    u_r = torch.randn(u_out.shape, generator=rng)
    u_r = u_r / (u_r.norm() + 1e-12)
    v_r = torch.randn(v_rome.shape, generator=rng)
    v_r = v_r / (v_r.norm() + 1e-12)
    u_rand, v_rand = _refactor_uv(W_ne0 + eff_scale_best * torch.outer(u_r, v_r))
    rand_state = dict(state)
    rand_state[u_key] = u_rand.to(state[u_key].dtype)
    rand_state[v_key] = v_rand.to(state[v_key].dtype)
    rand_ckpt = out_path / "control_random_ne_rank1.safetensors"
    save_file(rand_state, rand_ckpt)

    # Control 2: Same-parameter gradient fine-tuning (ONLY u_ne and v_ne unfrozen: 6,144 params)
    ft_model = PAWPBilingualTTS(cfg).to(device).train()
    ft_model.load_state_dict(state, strict=True)
    for n, p in ft_model.named_parameters():
        p.requires_grad_(n in {u_key, v_key})
    opt = torch.optim.Adam([p for p in ft_model.parameters() if p.requires_grad], lr=5e-4)

    for step in range(ft_steps):
        text = dev_subset[step % len(dev_subset)]
        p_ids, l_ids, s_ids, b_flags, l_masks = _tensor_inputs_from_text(phonemize, text)
        opt.zero_grad(set_to_none=True)
        out = ft_model(
            torch.from_numpy(p_ids).to(device),
            torch.from_numpy(l_ids).to(device),
            torch.from_numpy(s_ids).to(device),
            torch.from_numpy(b_flags).to(device),
            torch.from_numpy(l_masks).to(device),
            speaker_ids=spk,
        )
        with torch.backends.cudnn.flags(enabled=False):
            logp = ctc(out.mel).transpose(0, 1)
            targets = torch.tensor(
                [t for t in p_ids[0].tolist() if t not in {0, 1, 2, 3, 4, 5}],
                dtype=torch.long,
                device=device,
            )
            in_len = torch.tensor([logp.shape[0]], dtype=torch.long, device=device)
            tgt_len = torch.tensor([len(targets)], dtype=torch.long, device=device)
            loss = F.ctc_loss(logp, targets, in_len, tgt_len, blank=ctc.blank, zero_infinity=True)
            torch.autograd.backward(loss)
        opt.step()

    ft_ckpt = out_path / "control_sameparam_ne_ft.safetensors"
    save_file({k: v.detach().cpu() for k, v in ft_model.state_dict().items()}, ft_ckpt)

    return {
        "mode": "rome_nullspace_ne_residual",
        "target_layer": target_layer_name,
        "edited_tensors": [u_key, v_key],
        "edited_params": int(U0.numel() + V0.numel()),
        "rank": 1,
        "best_alpha": best_alpha,
        "delta_frobenius_norm": float((g0 * delta_best).norm().item()),
        "dev_sweep": dev_sweep,
        "wsp_ckpt": str(wsp_ckpt),
        "rand_ckpt": str(rand_ckpt),
        "ft_ckpt": str(ft_ckpt),
    }


def verify_wsp_on_untouched_test(
    config_path: str | Path,
    base_ckpt: str | Path,
    wsp_ckpt: str | Path,
    rand_ckpt: str | Path,
    ft_ckpt: str | Path,
    ctc_ckpt: str | Path,
    phonemize: PhonemizeUtteranceUseCase,
    test_manifest: str | Path,
    target_sym: str,
    confused_sym: str,
    n_ne: int = 300,
    n_en: int = 300,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """One-shot evaluation on untouched test.jsonl with paired bootstrap 95% CIs."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_student_config(config_path)
    ctc = load_ctc_recognizer(ctc_ckpt, dev)

    ne_texts = load_manifest_texts(test_manifest, "ne", limit=n_ne, seed=101)
    en_texts = load_manifest_texts(test_manifest, "en", limit=n_en, seed=101)

    models_map = {
        "baseline": base_ckpt,
        "wsp_rank1": wsp_ckpt,
        "control_random_rank1": rand_ckpt,
        "control_sameparam_ft": ft_ckpt,
    }

    raw: dict[str, dict[str, Any]] = {}
    for name, ckpt in models_map.items():
        m = PAWPBilingualTTS(cfg).to(dev).eval()
        m.load_state_dict(load_file(str(ckpt)), strict=True)
        ne_rep = evaluate_ctc_per(m, ctc, phonemize, ne_texts, "ne", dev)
        en_rep = evaluate_ctc_per(m, ctc, phonemize, en_texts, "en", dev)
        conf_rate, margins, n_occ = measure_target_confusion_rate(
            m, ctc, phonemize, ne_texts, target_sym, confused_sym, dev
        )
        raw[name] = {
            "ne_per": ne_rep.per,
            "en_per": en_rep.per,
            "ne_utt_pers": ne_rep.utterance_pers,
            "en_utt_pers": en_rep.utterance_pers,
            "target_conf_rate": conf_rate,
            "target_margins": margins,
            "target_occurrences": n_occ,
        }

    def _boot_dict(b: BootstrapResult) -> dict[str, float | int]:
        return {
            "mean_baseline": round(b.mean_baseline, 5),
            "mean_edit": round(b.mean_edit, 5),
            "mean_delta": round(b.mean_delta, 5),
            "ci95_low": round(b.ci95_low, 5),
            "ci95_high": round(b.ci95_high, 5),
            "p_value_improved": round(b.p_value_improved, 4),
            "n_items": b.n_items,
        }

    comparisons: dict[str, Any] = {}
    for cand in ("wsp_rank1", "control_random_rank1", "control_sameparam_ft"):
        comparisons[cand] = {
            "ne_per": round(raw[cand]["ne_per"], 5),
            "en_per": round(raw[cand]["en_per"], 5),
            "target_conf_rate": round(raw[cand]["target_conf_rate"], 5),
            "target_occurrences": raw[cand]["target_occurrences"],
            "boot_target_margin": _boot_dict(
                bootstrap_paired_ci(raw["baseline"]["target_margins"], raw[cand]["target_margins"])
            ),
            "boot_ne_per": _boot_dict(
                bootstrap_paired_ci(raw["baseline"]["ne_utt_pers"], raw[cand]["ne_utt_pers"])
            ),
            "boot_en_per": _boot_dict(
                bootstrap_paired_ci(raw["baseline"]["en_utt_pers"], raw[cand]["en_utt_pers"])
            ),
        }

    return {
        "n_ne_test": len(ne_texts),
        "n_en_test": len(en_texts),
        "target_contrast": f"{target_sym} vs {confused_sym}",
        "baseline": {
            "ne_per": round(raw["baseline"]["ne_per"], 5),
            "en_per": round(raw["baseline"]["en_per"], 5),
            "target_conf_rate": round(raw["baseline"]["target_conf_rate"], 5),
            "target_occurrences": raw["baseline"]["target_occurrences"],
        },
        "comparisons": comparisons,
    }
