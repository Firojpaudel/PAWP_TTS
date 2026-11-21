"""Training-only CTC phoneme recognizer + Viterbi forced alignment for phoneme durations.

Replaces the jointly learned attention aligner, which did not converge on our mixed
multi-speaker bilingual data. A frame-level phoneme recognizer trained with CTC is a robust,
standard way to obtain alignments. It is never part of the inference graph.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

MEL_MEAN, MEL_STD = -5.0, 2.5


class CTCPhonemeRecognizer(nn.Module):
    """log-mel `[B, T, 80]` -> per-frame log-probs over `vocab + blank` `[B, T, V+1]`."""

    def __init__(self, vocab_size: int = 128, n_mels: int = 80, dim: int = 256) -> None:
        super().__init__()
        self.blank = vocab_size
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, dim, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(dim, dim, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(dim, dim, 5, padding=2),
            nn.GELU(),
        )
        self.rnn = nn.GRU(dim, dim, num_layers=2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * dim, vocab_size + 1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.conv(((mel - MEL_MEAN) / MEL_STD).transpose(1, 2)).transpose(1, 2)
        x, _ = self.rnn(x)
        return F.log_softmax(self.out(x).float(), dim=-1)


def ctc_forced_durations(logp: np.ndarray, tokens: np.ndarray, blank: int) -> np.ndarray:
    """Viterbi CTC alignment -> integer frames per token (sum == T, every token >= 1).

    Blank frames are attributed to the token emitted most recently, and leading blanks
    to the first token, so leading/trailing silence lands on <bos>/<eos>.
    """
    t_len, n = logp.shape[0], len(tokens)
    s_len = 2 * n + 1
    labels = np.full(s_len, blank, dtype=np.int64)
    labels[1::2] = tokens
    emit = logp[:, labels]  # [T, S]
    neg = -1e30
    alpha = np.full(s_len, neg)
    alpha[0], alpha[1] = emit[0, 0], emit[0, 1]
    # skip transition allowed into a label state whose label differs from two states back
    skip_ok = np.zeros(s_len, dtype=bool)
    skip_ok[3::2] = labels[3::2] != labels[1:-2:2]
    back = np.zeros((t_len, s_len), dtype=np.int8)  # 0 stay, 1 from s-1, 2 from s-2
    for t in range(1, t_len):
        a1 = np.concatenate(([neg], alpha[:-1]))
        a2 = np.where(skip_ok, np.concatenate(([neg, neg], alpha[:-2])), neg)
        stacked = np.stack([alpha, a1, a2])
        arg = stacked.argmax(0)
        alpha = stacked[arg, np.arange(s_len)] + emit[t]
        back[t] = arg
    s = s_len - 1 if alpha[-1] >= alpha[-2] else s_len - 2
    states = np.empty(t_len, dtype=np.int64)
    for t in range(t_len - 1, -1, -1):
        states[t] = s
        s -= int(back[t, s])
    durs = np.zeros(n, dtype=np.int64)
    current = 0
    for st in states:
        if st % 2 == 1:
            current = int((st - 1) // 2)
        durs[current] += 1
    # guarantee >= 1 frame per token by borrowing from the longest neighbour
    for k in np.where(durs == 0)[0]:
        donor = int(np.argmax(durs))
        durs[donor] -= 1
        durs[k] += 1
    return durs


def _align_job(args: tuple[np.ndarray, np.ndarray, int, str]) -> tuple[str, np.ndarray]:
    logp, tokens, blank, out_path = args
    durs = ctc_forced_durations(logp, tokens, blank)
    np.save(out_path, durs.astype(np.int16))
    return out_path, durs


def train_recognizer(
    cache_root: str | Path = "data/cache/v2",
    splits: tuple[str, ...] = ("train", "train_extra", "train_ft"),
    out_path: str | Path = "artifacts/checkpoints/ctc_aligner.pt",
    steps: int = 6000,
    max_batch_frames: int = 64_000,
    lr: float = 1e-3,
    workers: int = 10,
    log_every: int = 100,
) -> dict[str, Any]:
    from pawp_tts.infrastructure.training.trainer import (
        CachedFeatureDataset,
        FrameBucketSampler,
        collate,
    )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CachedFeatureDataset(cache_root, splits)
    model = CTCPhonemeRecognizer().to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    step, t0, acc, n_acc = 0, time.perf_counter(), 0.0, 0
    epoch = 0
    while step < steps:
        sampler = FrameBucketSampler(
            [it["frames"] for it in ds.index], max_batch_frames, 128, seed=epoch
        )
        for b in DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=workers):
            if step >= steps:
                break
            mel = b["log_mel"].to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logp = model(mel)
            loss = F.ctc_loss(
                logp.transpose(0, 1),
                b["phoneme_ids"].to(dev),
                b["mel_lens"].to(dev),
                b["text_lens"].to(dev),
                blank=model.blank,
                zero_infinity=True,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            acc += float(loss)
            n_acc += 1
            if step % log_every == 0:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "ctc": round(acc / n_acc, 4),
                            "elapsed_s": round(time.perf_counter() - t0),
                        }
                    ),
                    flush=True,
                )
                acc, n_acc = 0.0, 0
        epoch += 1
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    return {"steps": step, "minutes": round((time.perf_counter() - t0) / 60, 1)}


@torch.no_grad()
def extract_durations(
    cache_root: str | Path = "data/cache/v2",
    splits: tuple[str, ...] = ("train", "train_extra", "train_ft", "val"),
    model_path: str | Path = "artifacts/checkpoints/ctc_aligner.pt",
    procs: int = 16,
) -> dict[str, Any]:
    """Write `<cache_root>/durations/<split>/<file>.npy` (int16 frames per token)."""
    from pawp_tts.infrastructure.training.trainer import CachedFeatureDataset

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CTCPhonemeRecognizer().to(dev).eval()
    model.load_state_dict(torch.load(model_path, map_location=dev))
    stats: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=procs) as pool:
        for split in splits:
            ds = CachedFeatureDataset(cache_root, split)
            out_dir = Path(cache_root) / "durations" / split
            out_dir.mkdir(parents=True, exist_ok=True)
            jobs: list[Any] = []
            ones = total = 0
            for i in range(len(ds)):
                out = out_dir / (Path(ds.index[i]["file"]).stem + ".npy")
                if out.exists():
                    continue
                item = ds[i]
                logp = model(item["log_mel"].unsqueeze(0).to(dev))[0].float().cpu().numpy()
                jobs.append(
                    pool.submit(
                        _align_job, (logp, item["phoneme_ids"].numpy(), model.blank, str(out))
                    )
                )
                if len(jobs) >= 512:
                    for j in jobs:
                        _, d = j.result()
                        ones += int((d[1:-1] == 1).sum())
                        total += max(len(d) - 2, 1)
                    jobs = []
            for j in jobs:
                _, d = j.result()
                ones += int((d[1:-1] == 1).sum())
                total += max(len(d) - 2, 1)
            stats[split] = {"items": len(ds), "dur_one_frac": round(ones / max(total, 1), 4)}
            print(json.dumps({split: stats[split]}), flush=True)
    return stats
