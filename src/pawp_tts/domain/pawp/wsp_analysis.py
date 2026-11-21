"""Pure domain metrics for mechanistic weight-space programming (PER, confusions, bootstrap CIs)."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class BootstrapResult:
    """Paired bootstrap confidence interval for `edit - baseline`."""

    mean_baseline: float
    mean_edit: float
    mean_delta: float
    ci95_low: float
    ci95_high: float
    p_value_improved: float
    n_items: int


@dataclass(frozen=True)
class PhonemeConfusionReport:
    """Aggregate PER and top phoneme substitution confusions."""

    language: str
    num_utterances: int
    total_ref_phonemes: int
    substitutions: int
    deletions: int
    insertions: int
    per: float
    utterance_pers: tuple[float, ...]
    top_confusions: tuple[tuple[str, str, int], ...]


def ctc_greedy_decode(frame_ids: tuple[int, ...], blank_id: int) -> tuple[int, ...]:
    """Collapse consecutive duplicates and drop CTC blank tokens."""
    out: list[int] = []
    prev = -1
    for tok in frame_ids:
        if tok != prev and tok != blank_id:
            out.append(tok)
        prev = tok
    return tuple(out)


def align_phoneme_sequences(
    ref: tuple[int, ...],
    hyp: tuple[int, ...],
) -> tuple[int, int, int, tuple[tuple[int, int], ...]]:
    """Compute Levenshtein alignment returning (subs, dels, ins, substitution_pairs)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[0] * (m + 1) for _ in range(n + 1)]  # 0=match, 1=sub, 2=del, 3=ins
    for i in range(1, n + 1):
        dp[i][0] = i
        op[i][0] = 2
    for j in range(1, m + 1):
        dp[0][j] = j
        op[0][j] = 3

    for i in range(1, n + 1):
        ri = ref[i - 1]
        dp_i = dp[i]
        dp_im1 = dp[i - 1]
        op_i = op[i]
        for j in range(1, m + 1):
            cost_sub = dp_im1[j - 1] + (0 if ri == hyp[j - 1] else 1)
            cost_del = dp_im1[j] + 1
            cost_ins = dp_i[j - 1] + 1
            if cost_sub <= cost_del and cost_sub <= cost_ins:
                dp_i[j] = cost_sub
                op_i[j] = 0 if ri == hyp[j - 1] else 1
            elif cost_del <= cost_ins:
                dp_i[j] = cost_del
                op_i[j] = 2
            else:
                dp_i[j] = cost_ins
                op_i[j] = 3

    subs = dels = ins = 0
    sub_pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        code = op[i][j]
        if code == 0:
            i -= 1
            j -= 1
        elif code == 1:
            subs += 1
            sub_pairs.append((ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif code == 2:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1

    return subs, dels, ins, tuple(sub_pairs)


def compute_per_and_confusions(
    language: str,
    refs: list[tuple[int, ...]],
    hyps: list[tuple[int, ...]],
    id_to_symbol: dict[int, str],
    ignore_ids: frozenset[int] = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}),
    top_k: int = 15,
) -> PhonemeConfusionReport:
    """Compute corpus PER, per-utterance PERs, and top phoneme substitution confusions."""
    total_ref = total_sub = total_del = total_ins = 0
    utt_pers: list[float] = []
    conf: Counter[tuple[str, str]] = Counter()

    for r_raw, h_raw in zip(refs, hyps, strict=True):
        r = tuple(t for t in r_raw if t not in ignore_ids)
        h = tuple(t for t in h_raw if t not in ignore_ids)
        if not r:
            continue
        s, d, ins, pairs = align_phoneme_sequences(r, h)
        total_ref += len(r)
        total_sub += s
        total_del += d
        total_ins += ins
        utt_pers.append((s + d + ins) / len(r))
        for a, b in pairs:
            conf[(id_to_symbol.get(a, str(a)), id_to_symbol.get(b, str(b)))] += 1

    per = (total_sub + total_del + total_ins) / max(total_ref, 1)
    top = tuple((a, b, c) for (a, b), c in conf.most_common(top_k))
    return PhonemeConfusionReport(
        language=language,
        num_utterances=len(utt_pers),
        total_ref_phonemes=total_ref,
        substitutions=total_sub,
        deletions=total_del,
        insertions=total_ins,
        per=per,
        utterance_pers=tuple(utt_pers),
        top_confusions=top,
    )


def bootstrap_paired_ci(
    baseline_scores: tuple[float, ...],
    edit_scores: tuple[float, ...],
    n_resamples: int = 1000,
    seed: int = 42,
) -> BootstrapResult:
    """Paired bootstrap CI for `delta = edit - baseline` (negative delta = error reduction)."""
    if len(baseline_scores) != len(edit_scores) or not baseline_scores:
        raise ValueError("Score tuples must be non-empty and equal length.")

    n = len(baseline_scores)
    diffs = [e - b for b, e in zip(baseline_scores, edit_scores, strict=True)]
    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(n_resamples):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boot_means.append(s / n)

    boot_means.sort()
    lo_idx = max(0, int(0.025 * n_resamples))
    hi_idx = min(n_resamples - 1, int(0.975 * n_resamples))
    ge_zero = sum(1 for m in boot_means if m >= 0.0)
    return BootstrapResult(
        mean_baseline=sum(baseline_scores) / n,
        mean_edit=sum(edit_scores) / n,
        mean_delta=sum(diffs) / n,
        ci95_low=boot_means[lo_idx],
        ci95_high=boot_means[hi_idx],
        p_value_improved=ge_zero / n_resamples,
        n_items=n,
    )
