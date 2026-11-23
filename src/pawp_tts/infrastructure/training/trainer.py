"""RTX 4090 BF16 bilingual student trainer (v2).

- Durations come from a jointly trained, training-only aligner (CTC forward-sum + MAS);
  no heuristic duration targets.
- Full utterances (1-12 s) in length-bucketed batches; text and audio are never cropped apart.
- All losses are masked by true token / frame / sample lengths.
- Pitch/energy conditioning is teacher-forced; voicing uses BCE.
- Multi-speaker training; exported checkpoints are collapsed to one target speaker so the
  deployed inference graph holds a single speaker vector.
- Full resume (model, aligner, discriminators, optimizers, schedulers, step).
- `best.safetensors` is selected on held-out validation mel L1, never on a training batch.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset, Sampler

from pawp_tts.infrastructure.neural.aligner import (
    AlignmentEncoder,
    beta_binomial_log_prior,
    binarization_loss,
    forward_sum_loss,
    monotonic_alignment_search,
)
from pawp_tts.infrastructure.neural.bilingual_tts import PAWPBilingualTTS
from pawp_tts.infrastructure.neural.losses import (
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    MultiResolutionSTFTLoss,
    discriminator_lsgan_loss,
    generator_adversarial_and_fm_loss,
)
from pawp_tts.infrastructure.persistence.config_loader import load_student_config

HOP = 240
MEL_MEAN, MEL_STD = -5.0, 2.5  # approx. log-mel statistics, used only to condition the aligner


class CheckpointManager:
    """Loads inference checkpoints (safetensors) into a model."""

    @staticmethod
    def load(model: PAWPBilingualTTS, path: str | Path) -> None:
        model.load_state_dict(load_file(str(path)), strict=True)


# --------------------------------------------------------------------------- data
class CachedFeatureDataset(Dataset[dict[str, torch.Tensor]]):
    """Reads v2 `.npz` bundles written by `data/bulk_cache.py`."""

    def __init__(self, root: str | Path, split: str | tuple[str, ...]) -> None:
        splits = (split,) if isinstance(split, str) else split
        self.index: list[dict[str, Any]] = []
        for sp in splits:
            idx_path = Path(root) / f"{sp}_index.json"
            if not idx_path.exists():
                continue
            for it in json.loads(idx_path.read_text()):
                it["path"] = str(Path(root) / sp / it["file"])
                dur_p = Path(root) / "durations" / sp / (Path(it["file"]).stem + ".npy")
                if dur_p.exists():
                    it["dur_path"] = str(dur_p)
                self.index.append(it)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        dur_path = self.index[i].get("dur_path")
        with np.load(self.index[i]["path"]) as d:
            mel = d["log_mel"].astype(np.float32)
            t = mel.shape[0]
            wav = d["waveform"].astype(np.float32)[: t * HOP] / 32767.0
            return {
                "phoneme_ids": torch.from_numpy(d["phoneme_ids"].astype(np.int64)),
                "lang_ids": torch.from_numpy(d["lang_ids"].astype(np.int64)),
                "stress_ids": torch.from_numpy(d["stress_ids"].astype(np.int64)),
                "boundary_flags": torch.from_numpy(d["boundary_flags"].astype(np.float32)),
                "lang_masks": torch.from_numpy(d["lang_masks"].astype(np.float32)),
                "log_mel": torch.from_numpy(mel),
                "log_energy": torch.from_numpy(d["log_energy"].astype(np.float32)[:t]),
                "pitch_voicing": torch.from_numpy(d["pitch_voicing"].astype(np.float32)[:t]),
                "waveform": torch.from_numpy(wav),
                "speaker": torch.tensor(int(d["speaker"][0])),
                "loss_weight": torch.tensor(float(d["loss_weight"][0])),
                **(
                    {"durations": torch.from_numpy(np.load(dur_path).astype(np.float32))}
                    if dur_path
                    else {}
                ),
            }


class FrameBucketSampler(Sampler[list[int]]):
    """Groups similar-length utterances into batches capped at `max_frames` total frames."""

    def __init__(
        self, frames: list[int], max_frames: int, max_batch: int, seed: int, shuffle: bool = True
    ) -> None:
        self.frames, self.max_frames, self.max_batch = frames, max_frames, max_batch
        self.seed, self.shuffle, self.epoch = seed, shuffle, 0
        self._batches = self._build()

    def _build(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        order = sorted(
            range(len(self.frames)),
            key=lambda i: self.frames[i] + (rng.random() * 30 if self.shuffle else 0),
        )
        batches: list[list[int]] = []
        cur: list[int] = []
        for i in order:
            longest = max([self.frames[j] for j in cur] + [self.frames[i]])
            if cur and (longest * (len(cur) + 1) > self.max_frames or len(cur) >= self.max_batch):
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._batches = self._build()

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


def collate(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key in items[0]:
        vals = [it[key] for it in items]
        out[key] = (
            torch.stack(vals)
            if vals[0].dim() == 0
            else torch.nn.utils.rnn.pad_sequence(vals, batch_first=True)
        )
    out["text_lens"] = torch.tensor([len(it["phoneme_ids"]) for it in items])
    out["mel_lens"] = torch.tensor([it["log_mel"].shape[0] for it in items])
    return out


def _seq_mask(lens: torch.Tensor, max_len: int) -> torch.Tensor:
    return torch.arange(max_len, device=lens.device).unsqueeze(0) < lens.unsqueeze(1)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    m = mask.expand_as(x).to(x.dtype)
    return (x * m).sum() / m.sum().clamp(min=1.0)


def collapse_to_speaker(model: PAWPBilingualTTS, speaker_idx: int) -> dict[str, torch.Tensor]:
    """State dict of the deployable single-speaker graph (`num_speakers=1`)."""
    state = {k: v.detach().float().contiguous().cpu() for k, v in model.state_dict().items()}
    state["speaker_emb.weight"] = state["speaker_emb.weight"][speaker_idx : speaker_idx + 1].clone()
    return state


# --------------------------------------------------------------------------- trainer
class BilingualStudentTrainer:
    """Full-scale student trainer over the v2 feature cache."""

    def __init__(
        self,
        config_path: str | Path = "configs/student_1_5m.yaml",
        checkpoint_dir: str | Path = "artifacts/checkpoints",
        cache_root: str | Path = "data/cache/v2",
        target_speaker: str = "",
        device: str | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.ckpt_dir = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.cache_root = Path(cache_root)
        speakers: dict[str, int] = json.loads((self.cache_root / "speakers.json").read_text())
        target_speaker = target_speaker or os.environ.get("PAWP_TARGET_SPEAKER", "")
        if target_speaker not in speakers:
            raise ValueError("target speaker not found in speakers.json (set PAWP_TARGET_SPEAKER)")
        self.target_idx = speakers[target_speaker]
        self.cfg = replace(load_student_config(self.config_path), num_speakers=len(speakers))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.prosody_free_prob = 0.0

    # ---------------------------------------------------------------- core step
    def _forward(
        self,
        model: PAWPBilingualTTS,
        aligner: AlignmentEncoder,
        b: dict[str, torch.Tensor],
        step: int,
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        n_max, t_max = b["phoneme_ids"].shape[1], b["log_mel"].shape[1]
        text_mask = _seq_mask(b["text_lens"], n_max)
        frame_mask = _seq_mask(b["mel_lens"], t_max)
        # Exposure to predicted prosody: on a share of training steps the acoustic network is
        # conditioned on its own predicted F0/energy (as at inference) instead of ground truth.
        free_prosody = model.training and random.random() < self.prosody_free_prob
        if "durations" in b:  # precomputed CTC forced-alignment durations
            durs = b["durations"]
            align_l: dict[str, torch.Tensor] = {}
        else:
            _, _, durs, align_l = self._align(aligner, b, text_mask)
        out = model(
            phoneme_ids=b["phoneme_ids"],
            lang_ids=b["lang_ids"],
            stress_ids=b["stress_ids"],
            boundary_flags=b["boundary_flags"],
            lang_masks=b["lang_masks"],
            speaker_ids=b["speaker"],
            target_durations=durs,
            max_frames=t_max,
            target_pitch_voicing=None if free_prosody else b["pitch_voicing"],
            target_energy=None if free_prosody else b["log_energy"],
        )
        t_out = out.mel.shape[1]
        fm = frame_mask[:, :t_out]
        losses = {
            "mel": _masked_mean((out.mel.float() - b["log_mel"][:, :t_out]).abs(), fm),
            "dur": _masked_mean(
                (out.durations.float() - torch.log1p(durs)).pow(2)
                if self.dur_is_log
                else (torch.log1p(out.durations.float().clamp(min=0)) - torch.log1p(durs)).pow(2),
                text_mask,
            ),
            "pitch": _masked_mean(
                (out.pitch_voicing[..., 0].float() - b["pitch_voicing"][:, :t_out, 0]).abs(),
                fm & (b["pitch_voicing"][:, :t_out, 1] > 0.5),
            ),
            "voicing": _masked_mean(
                F.binary_cross_entropy_with_logits(
                    out.pitch_voicing[..., 1].float(),
                    b["pitch_voicing"][:, :t_out, 1],
                    reduction="none",
                ),
                fm,
            ),
            "energy": _masked_mean((out.energy.float() - b["log_energy"][:, :t_out]).abs(), fm),
            **align_l,
        }
        return out, losses

    @staticmethod
    def _align(
        aligner: AlignmentEncoder, b: dict[str, torch.Tensor], text_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Aligner in full FP32 on normalized mel; returns hard durations + losses + health."""
        n_max, t_max = b["phoneme_ids"].shape[1], b["log_mel"].shape[1]
        with torch.autocast(b["log_mel"].device.type, enabled=False):
            prior = beta_binomial_log_prior(b["text_lens"], b["mel_lens"], n_max, t_max)
            mel_n = (b["log_mel"].float() - MEL_MEAN) / MEL_STD
            log_attn, _ = aligner(b["phoneme_ids"], b["lang_ids"], mel_n, text_mask, prior)
            hard = monotonic_alignment_search(log_attn.detach(), b["text_lens"], b["mel_lens"])
            durs = hard.sum(1)  # [B, N]
            losses = {
                "ctc": forward_sum_loss(log_attn, b["text_lens"], b["mel_lens"]),
                "bin": binarization_loss(hard, log_attn),
            }
        # health: share of real tokens squeezed to 1 frame (collapse -> close to 1.0)
        inner = text_mask.clone()
        inner[:, 0] = False
        inner[torch.arange(inner.shape[0]), b["text_lens"] - 1] = False
        losses["dur_one_frac"] = ((durs == 1) & inner).float().sum() / inner.float().sum().clamp(
            min=1.0
        )
        return log_attn, hard, durs, losses

    @property
    def dur_is_log(self) -> bool:
        return False  # DurationPredictor outputs linear frame counts (softplus) in this codebase

    def _wav_segments(
        self, pred: torch.Tensor, real: torch.Tensor, mel_lens: torch.Tensor, seg: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p, r = [], []
        # Discriminators see at most 16 random clips per step (bounded VRAM at any batch size).
        chosen = random.sample(range(pred.shape[0]), min(16, pred.shape[0]))
        for i in chosen:
            n = int(mel_lens[i]) * HOP
            n = min(n, pred.shape[-1], real.shape[-1])
            start = random.randint(0, max(n - seg, 0))
            pp, rr = pred[i, start : start + seg], real[i, start : start + seg]
            if pp.shape[0] < seg:
                pp, rr = F.pad(pp, (0, seg - pp.shape[0])), F.pad(rr, (0, seg - rr.shape[0]))
            p.append(pp)
            r.append(rr)
        return torch.stack(p).unsqueeze(1), torch.stack(r).unsqueeze(1)

    # ---------------------------------------------------------------- loop
    def train(
        self,
        steps: int = 200_000,
        max_batch_frames: int = 12_000,
        max_batch: int = 32,
        lr: float = 1e-3,
        seed: int = 42,
        resume: bool = True,
        gan_start: int = 30_000,
        bin_start: int = 5_000,
        eval_every: int = 2_000,
        log_every: int = 100,
        workers: int = 10,
        log_path: str | Path | None = None,
        run_name: str | None = None,
        train_splits: tuple[str, ...] = ("train", "train_extra"),
        replay_items: int = 0,
        init_from: str | Path | None = None,
        aligner_warmup: int = 0,
        fresh_aligner: bool = False,
        prosody_free_prob: float = 0.0,
    ) -> dict[str, float | int | str]:
        self.prosody_free_prob = prosody_free_prob
        torch.manual_seed(seed)
        random.seed(seed)
        dev = self.device
        train_ds = CachedFeatureDataset(self.cache_root, train_splits)
        if replay_items > 0:  # rehearsal of base data to prevent forgetting
            base = CachedFeatureDataset(self.cache_root, ("train", "train_extra"))
            train_ds.index += random.Random(seed).sample(
                base.index, min(replay_items, len(base.index))
            )
        val_ds = CachedFeatureDataset(self.cache_root, "val")
        sampler = FrameBucketSampler(
            [it["frames"] for it in train_ds.index], max_batch_frames, max_batch, seed
        )
        val_sampler = FrameBucketSampler(
            [it["frames"] for it in val_ds.index], max_batch_frames, max_batch, seed, shuffle=False
        )

        model = PAWPBilingualTTS(self.cfg).to(dev).train()
        aligner = AlignmentEncoder(self.cfg.vocab_size, self.cfg.n_mels).to(dev).train()
        mpd = MultiPeriodDiscriminator().to(dev).train()
        mrd = MultiResolutionDiscriminator().to(dev).train()
        mrstft = MultiResolutionSTFTLoss().to(dev)
        opt_g = torch.optim.AdamW(
            list(model.parameters()) + list(aligner.parameters()),
            lr=lr,
            betas=(0.8, 0.99),
            weight_decay=1e-2,
        )
        opt_d = torch.optim.AdamW(
            list(mpd.parameters()) + list(mrd.parameters()),
            lr=lr * 0.5,
            betas=(0.8, 0.99),
            weight_decay=1e-2,
        )
        warm = 2_000

        def sched_fn(s: int) -> float:
            cosine = 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)) * 0.95 + 0.05
            return min(1.0, (s + 1) / warm) * cosine

        sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, sched_fn)
        sched_d = torch.optim.lr_scheduler.LambdaLR(opt_d, sched_fn)

        state_path = self.ckpt_dir / "train_state.pt"
        step, best_val, epoch = 0, float("inf"), 0
        if resume and state_path.exists():
            st = torch.load(state_path, map_location=dev, weights_only=False)
            for obj, key in (
                (model, "model"),
                (aligner, "aligner"),
                (mpd, "mpd"),
                (mrd, "mrd"),
                (opt_g, "opt_g"),
                (opt_d, "opt_d"),
                (sched_g, "sched_g"),
                (sched_d, "sched_d"),
            ):
                obj.load_state_dict(st[key])
            step, best_val, epoch = st["step"], st["best_val"], st["epoch"]
            print(f"resumed from step {step} (best_val={best_val:.4f})", flush=True)
        elif init_from is not None:
            init_p = Path(init_from)
            if init_p.suffix == ".safetensors":
                from safetensors.torch import load_file

                src = load_file(str(init_p))
                old = src["speaker_emb.weight"]
                new = model.speaker_emb.weight.detach().clone()
                n = min(old.shape[0], new.shape[0])
                new[:n] = old[:n]
                new[n:] = old.mean(0, keepdim=True)
                src["speaker_emb.weight"] = new
                model.load_state_dict(src, strict=True)
                print(
                    f"initialized model from safetensors {init_from}; "
                    f"speakers {old.shape[0]} -> {new.shape[0]}",
                    flush=True,
                )
            else:
                st = torch.load(init_from, map_location=dev, weights_only=False)
                src = st["model"]
                old = src["speaker_emb.weight"]
                new = model.speaker_emb.weight.detach().clone()
                n = min(old.shape[0], new.shape[0])
                new[:n] = old[:n]
                new[n:] = old.mean(0, keepdim=True)  # new voices start at the mean speaker
                src = {**src, "speaker_emb.weight": new}
                model.load_state_dict(src, strict=True)
                if not fresh_aligner:
                    aligner.load_state_dict(st["aligner"])
                mpd.load_state_dict(st["mpd"])
                mrd.load_state_dict(st["mrd"])
                print(
                    f"initialized from {init_from} (base step {st['step']}); "
                    f"speakers {old.shape[0]} -> {new.shape[0]}",
                    flush=True,
                )

        use_bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()
        log_fh = open(log_path, "a", encoding="utf-8") if log_path else None  # noqa: SIM115
        wb: Any = None
        if os.environ.get("WANDB_API_KEY") and run_name:
            try:
                import wandb

                wb = wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "pawp-tts"),
                    name=run_name,
                    id=run_name,
                    resume="allow",
                    config={
                        "config_path": str(self.config_path),
                        "steps": steps,
                        "lr": lr,
                        "max_batch_frames": max_batch_frames,
                        "gan_start": gan_start,
                        "seed": seed,
                        "num_speakers": self.cfg.num_speakers,
                        "train_items": len(train_ds),
                        "train_splits": ",".join(train_splits),
                        "replay_items": replay_items,
                        "init_from": str(init_from),
                        "params_inference_graph": sum(
                            p.numel()
                            for n, p in model.named_parameters()
                            if not n.startswith("speaker_emb")
                        )
                        + model.cfg.d_model,
                    },
                )
            except Exception as exc:  # tracking must never break training
                print(f"wandb disabled: {exc}", flush=True)
                wb = None
        t0, last_t, run_steps = time.perf_counter(), time.perf_counter(), 0
        acc: dict[str, float] = {}

        def save_state() -> None:
            torch.save(
                {
                    "model": model.state_dict(),
                    "aligner": aligner.state_dict(),
                    "mpd": mpd.state_dict(),
                    "mrd": mrd.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "sched_g": sched_g.state_dict(),
                    "sched_d": sched_d.state_dict(),
                    "step": step,
                    "best_val": best_val,
                    "epoch": epoch,
                    "num_speakers": self.cfg.num_speakers,
                },
                state_path.with_suffix(".tmp"),
            )
            state_path.with_suffix(".tmp").replace(state_path)
            meta = {"step": str(step), "speaker_idx": str(self.target_idx)}
            save_file(
                collapse_to_speaker(model, self.target_idx),
                str(self.ckpt_dir / "latest.safetensors"),
                metadata=meta,
            )

        while step < steps:
            sampler.set_epoch(epoch)
            loader = DataLoader(
                train_ds,
                batch_sampler=sampler,
                collate_fn=collate,
                num_workers=workers,
                pin_memory=dev.type == "cuda",
                prefetch_factor=4 if workers > 0 else None,
            )
            for batch in loader:
                if step >= steps:
                    break
                b = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
                if step < aligner_warmup:  # aligner-only phase (fast; no acoustic model)
                    tmask = _seq_mask(b["text_lens"], b["phoneme_ids"].shape[1])
                    _, _, _, L = self._align(aligner, b, tmask)
                    a_total = L["ctc"] + (0.3 * L["bin"] if step >= bin_start else 0.0)
                    opt_g.zero_grad(set_to_none=True)
                    a_total.backward()  # type: ignore[no-untyped-call]
                    gn = torch.nn.utils.clip_grad_norm_(aligner.parameters(), 10.0)
                    opt_g.step()
                    sched_g.step()
                    step += 1
                    run_steps += 1
                    for k, v in L.items():
                        acc[k] = acc.get(k, 0.0) + float(v.detach())
                    acc["gnorm"] = acc.get("gnorm", 0.0) + float(gn)
                    if step % log_every == 0:
                        rec = {
                            "step": step,
                            "phase": "aligner_warmup",
                            **{k: round(v / log_every, 4) for k, v in acc.items()},
                        }
                        acc = {}
                        print(json.dumps(rec), flush=True)
                        if wb is not None:
                            wb.log(
                                {k: v for k, v in rec.items() if k not in ("step", "phase")},
                                step=step,
                            )
                        if log_fh:
                            log_fh.write(json.dumps(rec) + "\n")
                            log_fh.flush()
                    if step == aligner_warmup:
                        save_state()
                    continue
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_bf16):
                    out, L = self._forward(model, aligner, b, step)
                    wav_len = b["mel_lens"] * HOP
                    smask = _seq_mask(wav_len, out.waveform.shape[-1])
                    pred_w = out.waveform.float() * smask
                    real_w = b["waveform"][:, : pred_w.shape[-1]]
                    if real_w.shape[-1] < pred_w.shape[-1]:
                        real_w = F.pad(real_w, (0, pred_w.shape[-1] - real_w.shape[-1]))
                    L["stft"] = mrstft(pred_w, real_w)
                    g_total = (
                        15.0 * L["mel"]
                        + 2.5 * L["stft"]
                        + L["dur"]
                        + L["pitch"]
                        + 0.5 * L["voicing"]
                        + L["energy"]
                        + L.get("ctc", 0.0)
                        + (0.3 * L["bin"] if (step >= bin_start and "bin" in L) else 0.0)
                    )
                    if step >= gan_start:
                        fake_s, real_s = self._wav_segments(pred_w, real_w, b["mel_lens"], 16_384)
                        opt_d.zero_grad(set_to_none=True)
                        r1, f1, _, _ = mpd(real_s, fake_s.detach())
                        r2, f2, _, _ = mrd(real_s, fake_s.detach())
                        d_loss = discriminator_lsgan_loss(r1 + r2, f1 + f2)
                        d_loss.backward()  # type: ignore[no-untyped-call]
                        torch.nn.utils.clip_grad_norm_(
                            list(mpd.parameters()) + list(mrd.parameters()), 5.0
                        )
                        opt_d.step()
                        _, f1, rf1, ff1 = mpd(real_s, fake_s)
                        _, f2, rf2, ff2 = mrd(real_s, fake_s)
                        adv, fmatch = generator_adversarial_and_fm_loss(
                            f1 + f2, rf1 + rf2, ff1 + ff2
                        )
                        L["adv"], L["fm"], L["d"] = adv, fmatch, d_loss
                        g_total = g_total + adv + 2.0 * fmatch
                opt_g.zero_grad(set_to_none=True)
                g_total.backward()  # type: ignore[no-untyped-call]
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(aligner.parameters(), 10.0)
                opt_g.step()
                sched_g.step()
                sched_d.step()
                step += 1
                run_steps += 1
                for k, v in L.items():
                    acc[k] = acc.get(k, 0.0) + float(v.detach())
                acc["gnorm"] = acc.get("gnorm", 0.0) + float(gn)
                if step % log_every == 0:
                    now = time.perf_counter()
                    rec = {
                        "step": step,
                        "lr": sched_g.get_last_lr()[0],
                        "steps_per_s": round(log_every / (now - last_t), 2),
                        **{k: round(v / log_every, 4) for k, v in acc.items()},
                    }
                    last_t, acc = now, {}
                    print(json.dumps(rec), flush=True)
                    if wb is not None:
                        wb.log({k: v for k, v in rec.items() if k != "step"}, step=step)
                    if log_fh:
                        log_fh.write(json.dumps(rec) + "\n")
                        log_fh.flush()
                if step % eval_every == 0 or step == steps:
                    self._log_samples(model, step, wb, run_name)
                    val = self._validate(model, aligner, val_ds, val_sampler, use_bf16)
                    improved = val["val_mel"] < best_val
                    best_val = min(best_val, val["val_mel"])
                    save_state()
                    if improved:
                        save_file(
                            collapse_to_speaker(model, self.target_idx),
                            str(self.ckpt_dir / "best.safetensors"),
                            metadata={"step": str(step), "val_mel": f"{best_val:.5f}"},
                        )
                    rec = {"step": step, "eval": True, **val, "best_val_mel": best_val}
                    print(json.dumps(rec), flush=True)
                    if wb is not None:
                        wb.log({k: v for k, v in rec.items() if k != "step"}, step=step)
                    if log_fh:
                        log_fh.write(json.dumps(rec) + "\n")
                        log_fh.flush()
            epoch += 1

        if log_fh:
            log_fh.close()
        if wb is not None:
            wb.finish()
        peak = torch.cuda.max_memory_allocated(dev) / 1e6 if dev.type == "cuda" else 0.0
        return {
            "steps": step,
            "best_val_mel": round(best_val, 5),
            "epochs": epoch,
            "train_hours_run": round((time.perf_counter() - t0) / 3600, 3),
            "steps_per_sec": round(run_steps / max(time.perf_counter() - t0, 1e-6), 2),
            "peak_vram_mb": round(peak, 1),
            "bf16": use_bf16,
            "num_speakers_train": self.cfg.num_speakers,
            "target_speaker_idx": self.target_idx,
        }

    _SAMPLE_PROMPTS: dict[str, str] = {  # noqa: RUF012
        "en": "Good morning. Your appointment is confirmed for ten thirty tomorrow.",
        "ne": "नमस्ते, आज काठमाडौंको मौसम निकै रमाइलो छ।",
        "codemix": "आज मेरो office मा एउटा important meeting छ।",
    }

    @torch.no_grad()
    def _log_samples(
        self, model: PAWPBilingualTTS, step: int, wb: Any, run_name: str | None
    ) -> None:
        """Synthesize fixed EN/NE/code-mix prompts in the target voice; log to wandb + disk."""
        try:
            import soundfile as sf

            from pawp_tts.application.frontend_service import PhonemizeUtteranceUseCase
            from pawp_tts.infrastructure.evaluation.benchmark_suite import _synthesize_utterance
            from pawp_tts.infrastructure.frontend.english_adapter import EnglishEspeakG2PAdapter
            from pawp_tts.infrastructure.frontend.nepali_adapter import NepaliRuleG2PAdapter

            if not hasattr(self, "_phonemize"):
                self._phonemize = PhonemizeUtteranceUseCase(
                    english_g2p=EnglishEspeakG2PAdapter(), nepali_g2p=NepaliRuleG2PAdapter()
                )
            model.eval()
            out_dir = Path("artifacts/audio") / (run_name or "train") / "latest"
            out_dir.mkdir(parents=True, exist_ok=True)
            logs: dict[str, Any] = {}
            for key, text in self._SAMPLE_PROMPTS.items():
                _, _, wav, _ = _synthesize_utterance(
                    model, self._phonemize, text, self.device, speaker_id=self.target_idx
                )
                sf.write(out_dir / f"{key}.wav", wav, 24000)
                if wb is not None:
                    import wandb

                    logs[f"samples/{key}"] = wandb.Audio(wav, sample_rate=24000, caption=text)
            if logs:
                wb.log(logs, step=step)
        except Exception as exc:  # sampling must never break training
            print(f"sample logging failed: {exc}", flush=True)
        finally:
            model.train()

    @torch.no_grad()
    def _validate(
        self,
        model: PAWPBilingualTTS,
        aligner: AlignmentEncoder,
        ds: CachedFeatureDataset,
        sampler: FrameBucketSampler,
        use_bf16: bool,
    ) -> dict[str, float]:
        model.eval()
        aligner.eval()
        tot: dict[str, float] = {}
        n = 0
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=4)
        for batch in loader:
            b = {k: v.to(self.device) for k, v in batch.items()}
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=use_bf16):
                _, L = self._forward(model, aligner, b, 0)
            for k in ("mel", "dur", "pitch", "voicing", "energy"):
                tot[f"val_{k}"] = tot.get(f"val_{k}", 0.0) + float(L[k])
            n += 1
        model.train()
        aligner.train()
        return {k: round(v / max(n, 1), 5) for k, v in tot.items()}
