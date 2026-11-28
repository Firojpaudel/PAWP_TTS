"""EXP-018 & EXP-019: Independent local mixed-script ASR re-scoring + code-mix error breakdown."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
from scipy import stats as scipy_stats

from pawp_tts.domain.entities.models import LanguageCode
from pawp_tts.domain.evaluation.metrics import compute_cer, compute_wer, normalize_for_eval
from pawp_tts.domain.phonology.codemix import build_language_masks
from pawp_tts.infrastructure.data.bulk_cache import _read_audio
from pawp_tts.infrastructure.data.feature_extractor import resample_to_24k
from pawp_tts.infrastructure.evaluation.mixed_script_asr_adapter import MixedScriptLocalAsrAdapter
from pawp_tts.infrastructure.neural.bilingual_tts import PAWPBilingualTTS
from pawp_tts.infrastructure.persistence.config_loader import load_student_config
from pawp_tts.infrastructure.training.trainer import CheckpointManager
from pawp_tts.interfaces.container import build_container

load_dotenv()

CHECKPOINTS: list[tuple[str, str, str, int]] = [
    ("pawp_1_5m_v3", "configs/student_1_5m.yaml", "artifacts/checkpoints/exp015_1_5m_v3/best.safetensors", 42),
    ("shared_300k_s42", "configs/student_300k_pawp.yaml", "artifacts/checkpoints/exp015_300k_shared_v3/best.safetensors", 42),
    ("shared_300k_s101", "configs/student_300k_pawp.yaml", "artifacts/checkpoints/exp015_300k_shared_v3_s101/best.safetensors", 101),
    ("shared_300k_s202", "configs/student_300k_pawp.yaml", "artifacts/checkpoints/exp015_300k_shared_v3_s202/best.safetensors", 202),
    ("dense_300k_s42", "configs/student_300k_dense.yaml", "artifacts/checkpoints/exp015_300k_dense_v3/best.safetensors", 42),
    ("dense_300k_s101", "configs/student_300k_dense.yaml", "artifacts/checkpoints/exp015_300k_dense_v3_s101/best.safetensors", 101),
    ("dense_300k_s202", "configs/student_300k_dense.yaml", "artifacts/checkpoints/exp015_300k_dense_v3_s202/best.safetensors", 202),
]


def _word_lang(word: str) -> str:
    has_dev = any("\u0900" <= ch <= "\u097F" for ch in word)
    has_lat = any(("a" <= ch.lower() <= "z") or ch.isdigit() for ch in word)
    if has_dev and not has_lat:
        return "ne"
    if has_lat and not has_dev:
        return "en"
    if has_dev and has_lat:
        return "mixed"
    return "other"


def _normalize_words(text: str) -> list[str]:
    """Use the exact same punctuation-stripping normalizer as compute_wer()."""
    return normalize_for_eval(text).split()


def _bootstrap_micro_ci(pairs: list[tuple[int, int]], n_boot: int = 1000, seed: int = 42) -> dict[str, float]:
    """Micro error-rate bootstrap over per-utterance (ops, ref_units) pairs."""
    rng = np.random.default_rng(seed)
    ops = np.array([o for o, _ in pairs], dtype=np.float64)
    refs = np.array([n for _, n in pairs], dtype=np.float64)
    n = len(ops)
    idx = rng.integers(0, n, size=(n_boot, n))
    rates = ops[idx].sum(axis=1) / np.maximum(refs[idx].sum(axis=1), 1.0)
    return {
        "micro": round(float(ops.sum() / max(refs.sum(), 1.0)), 5),
        "ci95_low": round(float(np.percentile(rates, 2.5)), 5),
        "ci95_high": round(float(np.percentile(rates, 97.5)), 5),
    }


def _paired_bootstrap_micro(
    a_pairs: list[tuple[int, int]], b_pairs: list[tuple[int, int]], n_boot: int = 1000, seed: int = 42
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    oa = np.array([o for o, _ in a_pairs], dtype=np.float64)
    na = np.array([n for _, n in a_pairs], dtype=np.float64)
    ob = np.array([o for o, _ in b_pairs], dtype=np.float64)
    nb = np.array([n for _, n in b_pairs], dtype=np.float64)
    n = len(oa)
    idx = rng.integers(0, n, size=(n_boot, n))
    ra = oa[idx].sum(axis=1) / np.maximum(na[idx].sum(axis=1), 1.0)
    rb = ob[idx].sum(axis=1) / np.maximum(nb[idx].sum(axis=1), 1.0)
    diff = ra - rb
    ma = float(oa.sum() / max(na.sum(), 1.0))
    mb = float(ob.sum() / max(nb.sum(), 1.0))
    return {
        "micro_a": round(ma, 5),
        "micro_b": round(mb, 5),
        "diff_a_minus_b": round(ma - mb, 5),
        "ci95_low": round(float(np.percentile(diff, 2.5)), 5),
        "ci95_high": round(float(np.percentile(diff, 97.5)), 5),
        "p_value_a_lt_b": round(float(np.mean(diff >= 0.0)), 4),
    }


def _align_word_ops(ref: list[str], hyp: list[str]) -> list[tuple[str, int | None, int | None]]:
    """Levenshtein DP returning list of ('eq'|'sub'|'del'|'ins', ref_idx, hyp_idx)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    ops: list[tuple[str, int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("eq", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def _codemix_error_breakdown(refs: list[str], hyps: list[str]) -> dict[str, object]:
    ins = dels = subs = eqs = 0
    en_ref = en_err = ne_ref = ne_err = other_ref = other_err = 0

    for r_txt, h_txt in zip(refs, hyps, strict=True):
        r_words = _normalize_words(r_txt)
        h_words = _normalize_words(h_txt)
        langs = [_word_lang(w) for w in r_words]

        for op, ri, _hi in _align_word_ops(r_words, h_words):
            if op == "ins":
                ins += 1
                continue
            assert ri is not None
            lg = langs[ri]
            err = 0 if op == "eq" else 1
            if op == "eq":
                eqs += 1
            elif op == "sub":
                subs += 1
            elif op == "del":
                dels += 1

            if lg == "en":
                en_ref += 1
                en_err += err
            elif lg == "ne":
                ne_ref += 1
                ne_err += err
            else:
                other_ref += 1
                other_err += err

    tot_ref = eqs + subs + dels
    return {
        "normalizer": "pawp_tts.domain.evaluation.metrics.normalize_for_eval",
        "total_ref_words": tot_ref,
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "total_word_errors": subs + dels + ins,
        "micro_wer": round((subs + dels + ins) / max(tot_ref, 1), 5),
        "en_span_words": en_ref,
        "en_span_errors": en_err,
        "en_span_error_rate": round(en_err / max(en_ref, 1), 5),
        "ne_span_words": ne_ref,
        "ne_span_errors": ne_err,
        "ne_span_error_rate": round(ne_err / max(ne_ref, 1), 5),
        "other_span_words": other_ref,
        "other_span_errors": other_err,
        "switch_boundary_analysis": (
            "withdrawn: at 4.74 switches per 10-word sentence, 76% of words are adjacent to a "
            "switch, confounding boundary-vs-inside comparisons by word composition."
        ),
    }


def _bootstrap_ci(vals: list[float], n_boot: int = 1000, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=np.float64)
    n = len(arr)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    return {
        "mean": round(float(arr.mean()), 5),
        "ci95_low": round(float(np.percentile(means, 2.5)), 5),
        "ci95_high": round(float(np.percentile(means, 97.5)), 5),
    }


def _paired_bootstrap(a: list[float], b: list[float], n_boot: int = 1000, seed: int = 42) -> dict[str, float]:
    """Paired bootstrap of (a - b). Negative delta means a < b (a is better)."""
    rng = np.random.default_rng(seed)
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    diff = aa - bb
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    p_a_better = float(np.mean(means >= 0.0))
    return {
        "mean_a": round(float(aa.mean()), 5),
        "mean_b": round(float(bb.mean()), 5),
        "mean_diff_a_minus_b": round(float(diff.mean()), 5),
        "ci95_low": round(float(np.percentile(means, 2.5)), 5),
        "ci95_high": round(float(np.percentile(means, 97.5)), 5),
        "p_value_a_lt_b": round(p_a_better, 4),
    }


def _synth_wav(model: PAWPBilingualTTS, phonemize, text: str, device: torch.device) -> tuple[np.ndarray, int]:
    seq = phonemize.execute(text)
    tokens = seq.tokens
    lang_map = {LanguageCode.SHARED: 0, LanguageCode.EN: 1, LanguageCode.NE: 2, LanguageCode.MIXED: 3}
    p_ids = torch.tensor([[t.token_id for t in tokens]], dtype=torch.long, device=device)
    l_ids = torch.tensor([[lang_map.get(t.language, 0) for t in tokens]], dtype=torch.long, device=device)
    s_ids = torch.tensor([[min(t.stress, 3) for t in tokens]], dtype=torch.long, device=device)
    b_flags = torch.tensor(
        [
            [
                [
                    1.0 if t.is_word_start else 0.0,
                    1.0 if t.is_word_end else 0.0,
                    1.0 if t.is_syllable_start else 0.0,
                    1.0 if t.is_phrase_break else 0.0,
                ]
                for t in tokens
            ]
        ],
        dtype=torch.float32,
        device=device,
    )
    masks = torch.tensor([build_language_masks(tokens)], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(
            phoneme_ids=p_ids,
            lang_ids=l_ids,
            stress_ids=s_ids,
            boundary_flags=b_flags,
            lang_masks=masks,
        )
        wav = out.waveform[0].detach().cpu().numpy()
    return wav, model.cfg.sample_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codemix-only", action="store_true", help="Score only data/evaluation/codemix_v1.jsonl")
    args = parser.parse_args()

    t_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phonemize = build_container().phonemize
    asr = MixedScriptLocalAsrAdapter()

    try:
        # 1. Select exact existing held-out set (seed=0, 1.5..10.0s, 30 EN, 30 NE, 9 MIX)
        heldout: dict[str, list[dict[str, object]]] = {"en": [], "ne": [], "mixed": []}
        if not args.codemix_only:
            rows = [
                json.loads(x)
                for x in Path("data/manifests/test.jsonl").read_text(encoding="utf-8").splitlines()
                if x.strip()
            ]
            rows = [r for r in rows if 1.5 <= float(r["duration_sec"]) <= 10.0]
            random.Random(0).shuffle(rows)
            for r in rows:
                lg = str(r["language"])
                if lg in heldout and len(heldout[lg]) < 30:
                    heldout[lg].append(r)
            print(
                f"Held-out selected: en={len(heldout['en'])}, ne={len(heldout['ne'])}, mixed={len(heldout['mixed'])}",
                flush=True,
            )

        # 2. Load expanded codemix_v1.jsonl (n=160)
        cm_rows = [
            json.loads(x)
            for x in Path("data/evaluation/codemix_v1.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        print(f"Expanded code-mix loaded: n={len(cm_rows)}", flush=True)

        # 3. Score Human Floor on held-out (EN, NE, MIX)
        sources_dir = Path("data/sources")
        human_scores: dict[str, dict[str, object]] = {}
        human_per_item: dict[str, dict[str, list[float]]] = {}
        if not args.codemix_only:
            with tempfile.TemporaryDirectory() as td:
                for lg, items in heldout.items():
                    asr_lang = LanguageCode.EN if lg == "en" else LanguageCode.NE
                    wers, cers = [], []
                    for r in items:
                        gt_raw, gt_sr = _read_audio(dict(r), sources_dir)
                        wav_p = Path(td) / "gt.wav"
                        sf.write(wav_p, resample_to_24k(gt_raw, gt_sr), 24000)
                        hyp = asr.transcribe(str(wav_p), language=asr_lang).text
                        wers.append(compute_wer(str(r["text"]), hyp).error_rate)
                        cers.append(compute_cer(str(r["text"]), hyp).error_rate)
                    human_per_item[lg] = {"wer": wers, "cer": cers}
                    human_scores[lg] = {
                        "n": len(items),
                        "wer": _bootstrap_ci(wers),
                        "cer": _bootstrap_ci(cers),
                    }
                    print(
                        f"[HUMAN {lg}] WER={human_scores[lg]['wer']['mean']:.4f} CER={human_scores[lg]['cer']['mean']:.4f}",
                        flush=True,
                    )

        # 4. Score all 7 checkpoints on held-out AND codemix_v1.jsonl
        ckpt_heldout: dict[str, dict[str, object]] = {}
        ckpt_heldout_items: dict[str, dict[str, dict[str, list[float]]]] = {}
        ckpt_codemix: dict[str, dict[str, object]] = {}
        ckpt_codemix_pairs: dict[str, dict[str, list[tuple[int, int]]]] = {}

        for name, cfg_p, ckpt_p, seed in CHECKPOINTS:
            cfg = load_student_config(cfg_p)
            model = PAWPBilingualTTS(cfg).to(device).eval()
            CheckpointManager.load(model, ckpt_p)

            ckpt_heldout[name] = {"checkpoint": ckpt_p, "seed": seed}
            ckpt_heldout_items[name] = {}

            with tempfile.TemporaryDirectory() as td:
                if not args.codemix_only:
                    for lg, items in heldout.items():
                        asr_lang = LanguageCode.EN if lg == "en" else LanguageCode.NE
                        wers, cers = [], []
                        for r in items:
                            wav, sr = _synth_wav(model, phonemize, str(r["text"]), device)
                            wav_p = Path(td) / "syn.wav"
                            sf.write(wav_p, wav, sr)
                            hyp = asr.transcribe(str(wav_p), language=asr_lang).text
                            wers.append(compute_wer(str(r["text"]), hyp).error_rate)
                            cers.append(compute_cer(str(r["text"]), hyp).error_rate)
                        ckpt_heldout_items[name][lg] = {"wer": wers, "cer": cers}
                        ckpt_heldout[name][lg] = {
                            "n": len(items),
                            "wer": _bootstrap_ci(wers),
                            "cer": _bootstrap_ci(cers),
                            "vs_human_wer": _paired_bootstrap(wers, human_per_item[lg]["wer"]),
                            "vs_human_cer": _paired_bootstrap(cers, human_per_item[lg]["cer"]),
                        }
                        print(
                            f"[{name} heldout-{lg}] WER={ckpt_heldout[name][lg]['wer']['mean']:.4f} "
                            f"CER={ckpt_heldout[name][lg]['cer']['mean']:.4f}",
                            flush=True,
                        )

                # Expanded codemix_v1.jsonl (n=160)
                cm_refs, cm_hyps = [], []
                cm_wers_macro, cm_cers_macro = [], []
                cm_wer_pairs: list[tuple[int, int]] = []
                cm_cer_pairs: list[tuple[int, int]] = []
                for r in cm_rows:
                    txt = str(r["text"])
                    wav, sr = _synth_wav(model, phonemize, txt, device)
                    wav_p = Path(td) / "cm.wav"
                    sf.write(wav_p, wav, sr)
                    hyp = asr.transcribe(str(wav_p), language=LanguageCode.NE).text
                    cm_refs.append(txt)
                    cm_hyps.append(hyp)
                    ws = compute_wer(txt, hyp)
                    cs = compute_cer(txt, hyp)
                    cm_wers_macro.append(ws.error_rate)
                    cm_cers_macro.append(cs.error_rate)
                    cm_wer_pairs.append((ws.substitutions + ws.deletions + ws.insertions, ws.ref_units))
                    cm_cer_pairs.append((cs.substitutions + cs.deletions + cs.insertions, cs.ref_units))

                breakdown = _codemix_error_breakdown(cm_refs, cm_hyps)
                micro_w = _bootstrap_micro_ci(cm_wer_pairs)
                micro_c = _bootstrap_micro_ci(cm_cer_pairs)
                assert abs(micro_w["micro"] - float(breakdown["micro_wer"])) < 1e-4, (
                    f"Micro WER mismatch: {micro_w['micro']} vs {breakdown['micro_wer']}"
                )
                ckpt_codemix_pairs[name] = {"wer": cm_wer_pairs, "cer": cm_cer_pairs}
                ckpt_codemix[name] = {
                    "checkpoint": ckpt_p,
                    "seed": seed,
                    "n": len(cm_rows),
                    "human_floor": None,
                    "micro_wer": micro_w,
                    "micro_cer": micro_c,
                    "macro_wer": _bootstrap_ci(cm_wers_macro),
                    "macro_cer": _bootstrap_ci(cm_cers_macro),
                    "error_breakdown": breakdown,
                }
                print(
                    f"[{name} codemix_v1 n={len(cm_rows)}] micro_WER={micro_w['micro']:.4f} "
                    f"(macro={ckpt_codemix[name]['macro_wer']['mean']:.4f}) "
                    f"micro_CER={micro_c['micro']:.4f} "
                    f"S={breakdown['substitutions']} D={breakdown['deletions']} I={breakdown['insertions']} "
                    f"N={breakdown['total_ref_words']}",
                    flush=True,
                )

        seeds = [42, 101, 202]
        if not args.codemix_only:
            sv_d_heldout: dict[str, object] = {}
            all_diffs_cer_en_ne = []
            all_diffs_wer_en_ne = []
            for lg in ["en", "ne", "mixed"]:
                s_wers = [ckpt_heldout[f"shared_300k_s{s}"][lg]["wer"]["mean"] for s in seeds]
                d_wers = [ckpt_heldout[f"dense_300k_s{s}"][lg]["wer"]["mean"] for s in seeds]
                s_cers = [ckpt_heldout[f"shared_300k_s{s}"][lg]["cer"]["mean"] for s in seeds]
                d_cers = [ckpt_heldout[f"dense_300k_s{s}"][lg]["cer"]["mean"] for s in seeds]
                if lg in ("en", "ne"):
                    all_diffs_wer_en_ne.extend([sw - dw for sw, dw in zip(s_wers, d_wers, strict=True)])
                    all_diffs_cer_en_ne.extend([sc - dc for sc, dc in zip(s_cers, d_cers, strict=True)])
                sv_d_heldout[lg] = {
                    "shared_wer_mean_std": [round(float(np.mean(s_wers)), 5), round(float(np.std(s_wers, ddof=1)), 5)],
                    "dense_wer_mean_std": [round(float(np.mean(d_wers)), 5), round(float(np.std(d_wers, ddof=1)), 5)],
                    "shared_cer_mean_std": [round(float(np.mean(s_cers)), 5), round(float(np.std(s_cers, ddof=1)), 5)],
                    "dense_cer_mean_std": [round(float(np.mean(d_cers)), 5), round(float(np.std(d_cers, ddof=1)), 5)],
                    "per_seed_diff_shared_minus_dense_cer": [
                        {"seed": s, "diff": round(sc - dc, 5)} for s, sc, dc in zip(seeds, s_cers, d_cers, strict=True)
                    ],
                    "seed42_within_seed_paired_bootstrap_wer": _paired_bootstrap(
                        ckpt_heldout_items["shared_300k_s42"][lg]["wer"],
                        ckpt_heldout_items["dense_300k_s42"][lg]["wer"],
                    ),
                    "seed42_within_seed_paired_bootstrap_cer": _paired_bootstrap(
                        ckpt_heldout_items["shared_300k_s42"][lg]["cer"],
                        ckpt_heldout_items["dense_300k_s42"][lg]["cer"],
                    ),
                }

            t_cer, p_cer = scipy_stats.ttest_1samp(all_diffs_cer_en_ne, 0.0)
            t_wer, p_wer = scipy_stats.ttest_1samp(all_diffs_wer_en_ne, 0.0)
            try:
                w_cer_stat, w_cer_p = scipy_stats.wilcoxon(all_diffs_cer_en_ne)
            except ValueError:
                w_cer_stat, w_cer_p = float("nan"), float("nan")
            sv_d_heldout["pooled_3seeds_x_2langs_en_ne"] = {
                "diffs_shared_minus_dense_cer": [round(x, 5) for x in all_diffs_cer_en_ne],
                "diffs_shared_minus_dense_wer": [round(x, 5) for x in all_diffs_wer_en_ne],
                "paired_ttest_cer": {"t": round(float(t_cer), 4), "p": round(float(p_cer), 5)},
                "paired_ttest_wer": {"t": round(float(t_wer), 4), "p": round(float(p_wer), 5)},
                "wilcoxon_cer": {"stat": float(w_cer_stat), "p": round(float(w_cer_p), 5)},
            }
            out_exp018 = {
                "recognizer": "local-mixed-script-asr",
                "elapsed_sec": round(time.perf_counter() - t_start, 1),
                "human_floor_heldout": human_scores,
                "checkpoints_heldout": ckpt_heldout,
                "shared_vs_dense_heldout": sv_d_heldout,
            }
            Path("artifacts/reports/exp018_independent_asr.json").write_text(
                json.dumps(out_exp018, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        # Shared vs Dense on expanded codemix_v1 (n=160) using micro-WER / micro-CER
        s_cm_w = [ckpt_codemix[f"shared_300k_s{s}"]["micro_wer"]["micro"] for s in seeds]
        d_cm_w = [ckpt_codemix[f"dense_300k_s{s}"]["micro_wer"]["micro"] for s in seeds]
        s_cm_c = [ckpt_codemix[f"shared_300k_s{s}"]["micro_cer"]["micro"] for s in seeds]
        d_cm_c = [ckpt_codemix[f"dense_300k_s{s}"]["micro_cer"]["micro"] for s in seeds]
        sv_d_codemix = {
            "metric": "micro_wer_and_micro_cer_via_normalize_for_eval",
            "shared_micro_wer_mean_std": [round(float(np.mean(s_cm_w)), 5), round(float(np.std(s_cm_w, ddof=1)), 5)],
            "dense_micro_wer_mean_std": [round(float(np.mean(d_cm_w)), 5), round(float(np.std(d_cm_w, ddof=1)), 5)],
            "shared_micro_cer_mean_std": [round(float(np.mean(s_cm_c)), 5), round(float(np.std(s_cm_c, ddof=1)), 5)],
            "dense_micro_cer_mean_std": [round(float(np.mean(d_cm_c)), 5), round(float(np.std(d_cm_c, ddof=1)), 5)],
            "seed42_within_seed_paired_bootstrap_micro_wer": _paired_bootstrap_micro(
                ckpt_codemix_pairs["shared_300k_s42"]["wer"],
                ckpt_codemix_pairs["dense_300k_s42"]["wer"],
            ),
            "seed42_within_seed_paired_bootstrap_micro_cer": _paired_bootstrap_micro(
                ckpt_codemix_pairs["shared_300k_s42"]["cer"],
                ckpt_codemix_pairs["dense_300k_s42"]["cer"],
            ),
        }

        elapsed = round(time.perf_counter() - t_start, 1)
        out_exp019 = {
            "recognizer": "local-mixed-script-asr",
            "normalizer": "pawp_tts.domain.evaluation.metrics.normalize_for_eval",
            "averaging": "micro (sum(S+D+I)/sum(N)) as headline; macro (mean per-utterance) also logged",
            "dataset": "data/evaluation/codemix_v1.jsonl",
            "n_sentences": len(cm_rows),
            "human_floor": None,
            "note": "No human recordings exist for codemix_v1.jsonl; no substitute floor is reported.",
            "checkpoints": ckpt_codemix,
            "shared_vs_dense": sv_d_codemix,
        }
        Path("artifacts/reports/exp019_codemix_asr.json").write_text(
            json.dumps(out_exp019, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"DONE EXP-019 reconciled code-mix ASR in {elapsed}s!", flush=True)
    finally:
        asr.close()


if __name__ == "__main__":
    main()
