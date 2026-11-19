"""Deterministic prompt sanitization, mixed-script segmentation, and coarticulatory masks."""

from __future__ import annotations

import re
import unicodedata

from pawp_tts.domain.entities.models import LanguageCode, LanguageSpan, PhonemeToken

_META_TAG_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_COMMA_NUMBER_RE = re.compile(r"(?<=[0-9०-९]),(?=[0-9०-९]+)")
_HYPHENATED_CODE_RE = re.compile(r"\b([A-Za-z0-9]+(?:[-–—][A-Za-z0-9]+){2,})\b")
_ELLIPSES_RE = re.compile(r"\.{2,}|…+")
_TOKEN_RE = re.compile(
    r"[\u0900-\u0963\u0966-\u097F]+%?"
    r"|\$?(?:[A-Za-z0-9]+(?:\.[0-9]+|'[A-Za-z]+|/[0-9]+)*)%?"
    r"|[,।॥\.?!:\-–—]+"
)
_PURE_ASCII_NUM_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)?%?$")
_EN_UNIT_WORDS = frozenset({"dollar", "dollars", "usd", "percent", "am", "pm", "km", "kg", "gb"})


def sanitize_prompt(raw_text: str) -> str:
    """
    Strip literal meta-tags, normalize grouped numerals (`25,000` -> `25000`),
    expand hyphenated IBAN/digit codes, and normalize ellipses.
    """
    text = unicodedata.normalize("NFC", raw_text)
    text = _META_TAG_RE.sub(" ", text)
    text = _COMMA_NUMBER_RE.sub("", text)

    def _expand_hyphen_code(match: re.Match[str]) -> str:
        raw_code = match.group(1)
        chars = [ch for ch in raw_code if ch not in "-–—"]
        return " ".join(chars)

    text = _HYPHENATED_CODE_RE.sub(_expand_hyphen_code, text)
    text = _ELLIPSES_RE.sub(", ", text)
    return re.sub(r"\s+", " ", text).strip()


def segment_language_spans(text: str) -> tuple[LanguageSpan, ...]:
    """
    Segment normalized text deterministically into EN, NE, and SHARED spans.
    Resolves context-dependent pure ASCII numbers (`500`, `50%`) to NE or EN
    from surrounding linguistic cues and handles EN-NE compounds/clitics cleanly.
    """
    raw_items: list[tuple[str, int, int, str]] = []
    for match in _TOKEN_RE.finditer(text):
        tok = match.group(0)
        start, end = match.span()
        if re.fullmatch(r"[,।॥\.?!:\-–—]+", tok):
            kind = "punct"
        elif any(("\u0900" <= ch <= "\u0963") or ("\u0966" <= ch <= "\u097f") for ch in tok):
            kind = "ne"
        elif _PURE_ASCII_NUM_RE.fullmatch(tok):
            kind = "num"
        else:
            kind = "en"
        raw_items.append((tok, start, end, kind))

    # Drop tight inter-script hyphen connectors (`check-गरिदिनुहोस्` -> seamless join)
    filtered: list[tuple[str, int, int, str]] = []
    for idx, item in enumerate(raw_items):
        tok, start, end, kind = item
        if kind == "punct" and tok in {"-", "–", "—"}:
            has_left = (
                idx > 0 and raw_items[idx - 1][2] == start and raw_items[idx - 1][3] != "punct"
            )
            has_right = (
                idx + 1 < len(raw_items)
                and end == raw_items[idx + 1][1]
                and raw_items[idx + 1][3] != "punct"
            )
            if has_left and has_right:
                continue
        filtered.append(item)

    # Resolve pure ASCII numbers (`500`, `50%`) using neighboring language context
    resolved_langs: list[LanguageCode] = []
    for idx, (tok, _s, _e, kind) in enumerate(filtered):
        if kind == "punct":
            resolved_langs.append(LanguageCode.SHARED)
        elif kind == "ne":
            resolved_langs.append(LanguageCode.NE)
        elif kind == "en":
            resolved_langs.append(LanguageCode.EN)
        else:
            prev_item = next(
                (filtered[j] for j in range(idx - 1, -1, -1) if filtered[j][3] != "punct"),
                None,
            )
            next_item = next(
                (filtered[j] for j in range(idx + 1, len(filtered)) if filtered[j][3] != "punct"),
                None,
            )
            if next_item and next_item[0].lower() in _EN_UNIT_WORDS:
                resolved_langs.append(LanguageCode.EN)
            elif (
                (next_item and next_item[3] == "ne")
                or (prev_item and prev_item[3] == "ne" and (not next_item or next_item[3] == "ne"))
                or (prev_item and prev_item[3] == "ne" and next_item and not tok.endswith("%"))
            ):
                resolved_langs.append(LanguageCode.NE)
            else:
                resolved_langs.append(LanguageCode.EN)

    spans: list[LanguageSpan] = []
    cur_lang: LanguageCode | None = None
    cur_tokens: list[str] = []
    cur_start = 0
    cur_end = 0

    for (tok, start, end, _kind), lang in zip(filtered, resolved_langs, strict=True):
        if lang == LanguageCode.SHARED:
            if cur_tokens and cur_lang is not None:
                spans.append(LanguageSpan(" ".join(cur_tokens), cur_lang, cur_start, cur_end))
                cur_tokens = []
                cur_lang = None
            spans.append(LanguageSpan("." if tok == "।" else tok, LanguageCode.SHARED, start, end))
            continue

        if cur_lang == lang:
            cur_tokens.append(tok)
            cur_end = end
        else:
            if cur_tokens and cur_lang is not None:
                spans.append(LanguageSpan(" ".join(cur_tokens), cur_lang, cur_start, cur_end))
            cur_lang = lang
            cur_tokens = [tok]
            cur_start = start
            cur_end = end

    if cur_tokens and cur_lang is not None:
        spans.append(LanguageSpan(" ".join(cur_tokens), cur_lang, cur_start, cur_end))

    return tuple(spans)


def build_language_masks(
    tokens: tuple[PhonemeToken, ...],
    boundary_blend: float = 0.15,
) -> tuple[tuple[float, float], ...]:
    """
    Build per-token `(m_en, m_ne)` routing gates with smooth coarticulatory blending
    at direct EN <-> NE code-switching phoneme boundaries.
    """
    if not 0.0 <= boundary_blend <= 0.5:
        raise ValueError("boundary_blend must lie in [0.0, 0.5].")

    phoneme_indices = [
        i
        for i, tok in enumerate(tokens)
        if tok.language in {LanguageCode.EN, LanguageCode.NE} and tok.symbol != "<wb>"
    ]
    prev_phoneme_lang: dict[int, LanguageCode] = {}
    next_phoneme_lang: dict[int, LanguageCode] = {}
    for pos, tok_idx in enumerate(phoneme_indices):
        if pos > 0:
            prev_phoneme_lang[tok_idx] = tokens[phoneme_indices[pos - 1]].language
        if pos + 1 < len(phoneme_indices):
            next_phoneme_lang[tok_idx] = tokens[phoneme_indices[pos + 1]].language

    masks: list[tuple[float, float]] = []
    for idx, tok in enumerate(tokens):
        if tok.language == LanguageCode.SHARED:
            masks.append((0.0, 0.0))
            continue

        is_boundary = (idx in prev_phoneme_lang and prev_phoneme_lang[idx] != tok.language) or (
            idx in next_phoneme_lang and next_phoneme_lang[idx] != tok.language
        )
        blend = boundary_blend if is_boundary else 0.0
        if tok.language == LanguageCode.EN:
            masks.append((1.0 - blend, blend))
        else:
            masks.append((blend, 1.0 - blend))
    return tuple(masks)
