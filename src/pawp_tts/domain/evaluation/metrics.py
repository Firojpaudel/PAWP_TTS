"""Pure edit-distance alignment metrics for WER, CER, and PER auditing."""

from __future__ import annotations

import re
from collections.abc import Sequence

from pawp_tts.domain.entities.models import TextErrorStats

_PUNCT_STRIP_RE = re.compile(r"[,।\.?!:;\"'()\[\]—–\-]+")


def normalize_for_eval(text: str) -> str:
    """Normalize punctuation and whitespace while preserving Devanagari and Latin scripts."""
    cleaned = _PUNCT_STRIP_RE.sub(" ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def align_units(ref: Sequence[str], hyp: Sequence[str]) -> TextErrorStats:
    """Compute optimal Levenshtein alignment separating substitutions, deletions, insertions."""
    n = len(ref)
    m = len(hyp)
    if n == 0:
        return TextErrorStats(
            ref_units=0,
            hyp_units=m,
            substitutions=0,
            deletions=0,
            insertions=m,
            error_rate=float(m) if m > 0 else 0.0,
        )

    # DP table storing (total_ops, subs, dels, ins)
    dp: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)
    ]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub_t, sub_s, sub_d, sub_i = dp[i - 1][j - 1]
                del_t, del_s, del_d, del_i = dp[i - 1][j]
                ins_t, ins_s, ins_d, ins_i = dp[i][j - 1]

                cand_sub = (sub_t + 1, sub_s + 1, sub_d, sub_i)
                cand_del = (del_t + 1, del_s, del_d + 1, del_i)
                cand_ins = (ins_t + 1, ins_s, ins_d, ins_i + 1)
                dp[i][j] = min(cand_sub, cand_del, cand_ins, key=lambda x: x[0])

    total_ops, subs, dels, ins = dp[n][m]
    return TextErrorStats(
        ref_units=n,
        hyp_units=m,
        substitutions=subs,
        deletions=dels,
        insertions=ins,
        error_rate=total_ops / n,
    )


def compute_wer(reference: str, hypothesis: str) -> TextErrorStats:
    """Compute word error rate and I/D/S breakdown."""
    ref_words = normalize_for_eval(reference).split()
    hyp_words = normalize_for_eval(hypothesis).split()
    return align_units(ref_words, hyp_words)


def compute_cer(reference: str, hypothesis: str) -> TextErrorStats:
    """Compute character error rate (whitespace stripped) and I/D/S breakdown."""
    ref_chars = list(normalize_for_eval(reference).replace(" ", ""))
    hyp_chars = list(normalize_for_eval(hypothesis).replace(" ", ""))
    return align_units(ref_chars, hyp_chars)
