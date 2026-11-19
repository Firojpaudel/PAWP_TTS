"""Unified English-Nepali phonological inventory and articulatory feature table."""

from __future__ import annotations

ARTICULATORY_DIM = 28

SPECIAL_TOKENS: tuple[str, ...] = (
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<sp>",
    "<wb>",
    "<sb>",
    "<pb>",
)

PUNCTUATION_TOKENS: tuple[str, ...] = (",", ".", "?", "!", ":", "-")

SUPRASEGMENTALS: tuple[str, ...] = ("ˈ", "ˌ", "ː", "̃", "̤")

# Unified IPA symbols preserving distinct English and Nepali contrasts:
# - Nepali dental stops (t̪, t̪ʰ, d̪, d̪ʱ) vs. English alveolar stops (t, d) vs. retroflex (ʈ, ʈʰ, ɖ, ɖʱ)
# - Nepali alveolar affricates (ts, tsʰ, dz, dzʱ) vs. English postalveolar affricates (tʃ, dʒ)
# - Nepali retroflex flaps (ɽ, ɽʱ), glottal fricative (ɦ), inherent vowel (ʌ)
CONSONANT_SYMBOLS: tuple[str, ...] = (
    "k",
    "kʰ",
    "ɡ",
    "ɡʱ",
    "t̪",
    "t̪ʰ",
    "d̪",
    "d̪ʱ",
    "ʈ",
    "ʈʰ",
    "ɖ",
    "ɖʱ",
    "p",
    "pʰ",
    "b",
    "bʱ",
    "t",
    "d",
    "ts",
    "tsʰ",
    "dz",
    "dzʱ",
    "tʃ",
    "dʒ",
    "ŋ",
    "ɲ",
    "ɳ",
    "n̪",
    "n",
    "m",
    "ɽ",
    "ɽʱ",
    "r",
    "ɾ",
    "ɹ",
    "l",
    "ʋ",
    "w",
    "j",
    "s",
    "z",
    "ʃ",
    "ʒ",
    "f",
    "v",
    "θ",
    "ð",
    "ɦ",
    "h",
)

VOWEL_SYMBOLS: tuple[str, ...] = (
    "i",
    "ɪ",
    "e",
    "ɛ",
    "æ",
    "a",
    "ɑ",
    "ʌ",
    "ə",
    "ɔ",
    "o",
    "ʊ",
    "u",
    "ɜ",
    "aɪ",
    "aʊ",
    "eɪ",
    "oʊ",
    "ɔɪ",
    "ɪə",
    "eə",
    "ʊə",
    "ʌɪ",
    "ʌʊ",
)

UNIFIED_PHONEMES: tuple[str, ...] = (
    SPECIAL_TOKENS + PUNCTUATION_TOKENS + SUPRASEGMENTALS + CONSONANT_SYMBOLS + VOWEL_SYMBOLS
)

MAX_VOCAB_SIZE = 128
if len(UNIFIED_PHONEMES) > MAX_VOCAB_SIZE:
    raise RuntimeError(
        f"Unified phoneme inventory ({len(UNIFIED_PHONEMES)}) exceeds {MAX_VOCAB_SIZE}"
    )

SYMBOL_TO_ID: dict[str, int] = {sym: idx for idx, sym in enumerate(UNIFIED_PHONEMES)}
ID_TO_SYMBOL: dict[int, str] = {idx: sym for idx, sym in enumerate(UNIFIED_PHONEMES)}

# Longest-first greedy tokenizer order for multi-character IPA symbols (affricates, aspirates, diphthongs)
_SORTED_IPA_SYMBOLS: tuple[str, ...] = tuple(
    sorted(
        PUNCTUATION_TOKENS + SUPRASEGMENTALS + CONSONANT_SYMBOLS + VOWEL_SYMBOLS,
        key=len,
        reverse=True,
    )
)

# Articulatory feature indices (28 dims):
# 0:consonantal, 1:sonorant, 2:voiced, 3:aspirated, 4:breathy,
# 5:labial, 6:dental, 7:alveolar, 8:postalveolar, 9:retroflex, 10:palatal, 11:velar, 12:glottal,
# 13:stop, 14:fricative, 15:affricate, 16:nasal, 17:flap, 18:liquid, 19:glide,
# 20:vowel_high, 21:vowel_mid, 22:vowel_low, 23:vowel_front, 24:vowel_central, 25:vowel_back,
# 26:rounded, 27:suprasegmental_long_nasal
_FEATURE_INDEX: dict[str, int] = {
    "cons": 0,
    "son": 1,
    "voi": 2,
    "asp": 3,
    "breathy": 4,
    "lab": 5,
    "dent": 6,
    "alv": 7,
    "post": 8,
    "retr": 9,
    "pal": 10,
    "vel": 11,
    "glot": 12,
    "stop": 13,
    "fric": 14,
    "affr": 15,
    "nas": 16,
    "flap": 17,
    "liq": 18,
    "glide": 19,
    "v_high": 20,
    "v_mid": 21,
    "v_low": 22,
    "v_front": 23,
    "v_cent": 24,
    "v_back": 25,
    "round": 26,
    "supra": 27,
}

_PHONEME_FEATURES: dict[str, tuple[str, ...]] = {
    "k": ("cons", "vel", "stop"),
    "kʰ": ("cons", "vel", "stop", "asp"),
    "ɡ": ("cons", "vel", "stop", "voi"),
    "ɡʱ": ("cons", "vel", "stop", "voi", "asp", "breathy"),
    "t̪": ("cons", "dent", "stop"),
    "t̪ʰ": ("cons", "dent", "stop", "asp"),
    "d̪": ("cons", "dent", "stop", "voi"),
    "d̪ʱ": ("cons", "dent", "stop", "voi", "asp", "breathy"),
    "ʈ": ("cons", "retr", "stop"),
    "ʈʰ": ("cons", "retr", "stop", "asp"),
    "ɖ": ("cons", "retr", "stop", "voi"),
    "ɖʱ": ("cons", "retr", "stop", "voi", "asp", "breathy"),
    "p": ("cons", "lab", "stop"),
    "pʰ": ("cons", "lab", "stop", "asp"),
    "b": ("cons", "lab", "stop", "voi"),
    "bʱ": ("cons", "lab", "stop", "voi", "asp", "breathy"),
    "t": ("cons", "alv", "stop"),
    "d": ("cons", "alv", "stop", "voi"),
    "ts": ("cons", "alv", "affr"),
    "tsʰ": ("cons", "alv", "affr", "asp"),
    "dz": ("cons", "alv", "affr", "voi"),
    "dzʱ": ("cons", "alv", "affr", "voi", "asp", "breathy"),
    "tʃ": ("cons", "post", "affr"),
    "dʒ": ("cons", "post", "affr", "voi"),
    "ŋ": ("cons", "son", "voi", "vel", "nas"),
    "ɲ": ("cons", "son", "voi", "pal", "nas"),
    "ɳ": ("cons", "son", "voi", "retr", "nas"),
    "n̪": ("cons", "son", "voi", "dent", "nas"),
    "n": ("cons", "son", "voi", "alv", "nas"),
    "m": ("cons", "son", "voi", "lab", "nas"),
    "ɽ": ("cons", "son", "voi", "retr", "flap"),
    "ɽʱ": ("cons", "son", "voi", "retr", "flap", "asp", "breathy"),
    "r": ("cons", "son", "voi", "alv", "flap"),
    "ɾ": ("cons", "son", "voi", "alv", "flap"),
    "ɹ": ("cons", "son", "voi", "alv", "liq"),
    "l": ("cons", "son", "voi", "alv", "liq"),
    "ʋ": ("son", "voi", "lab", "glide"),
    "w": ("son", "voi", "lab", "vel", "glide", "round"),
    "j": ("son", "voi", "pal", "glide"),
    "s": ("cons", "alv", "fric"),
    "z": ("cons", "voi", "alv", "fric"),
    "ʃ": ("cons", "post", "fric"),
    "ʒ": ("cons", "voi", "post", "fric"),
    "f": ("cons", "lab", "fric"),
    "v": ("cons", "voi", "lab", "fric"),
    "θ": ("cons", "dent", "fric"),
    "ð": ("cons", "voi", "dent", "fric"),
    "ɦ": ("voi", "glot", "fric", "breathy"),
    "h": ("glot", "fric"),
    "i": ("son", "voi", "v_high", "v_front"),
    "ɪ": ("son", "voi", "v_high", "v_front"),
    "e": ("son", "voi", "v_mid", "v_front"),
    "ɛ": ("son", "voi", "v_mid", "v_front"),
    "æ": ("son", "voi", "v_low", "v_front"),
    "a": ("son", "voi", "v_low", "v_cent"),
    "ɑ": ("son", "voi", "v_low", "v_back"),
    "ʌ": ("son", "voi", "v_mid", "v_back"),
    "ə": ("son", "voi", "v_mid", "v_cent"),
    "ɔ": ("son", "voi", "v_mid", "v_back", "round"),
    "o": ("son", "voi", "v_mid", "v_back", "round"),
    "ʊ": ("son", "voi", "v_high", "v_back", "round"),
    "u": ("son", "voi", "v_high", "v_back", "round"),
    "ɜ": ("son", "voi", "v_mid", "v_cent"),
    "aɪ": ("son", "voi", "v_low", "v_front"),
    "aʊ": ("son", "voi", "v_low", "v_back", "round"),
    "eɪ": ("son", "voi", "v_mid", "v_front"),
    "oʊ": ("son", "voi", "v_mid", "v_back", "round"),
    "ɔɪ": ("son", "voi", "v_mid", "v_front", "round"),
    "ɪə": ("son", "voi", "v_high", "v_cent"),
    "eə": ("son", "voi", "v_mid", "v_cent"),
    "ʊə": ("son", "voi", "v_high", "v_cent", "round"),
    "ʌɪ": ("son", "voi", "v_mid", "v_front"),
    "ʌʊ": ("son", "voi", "v_mid", "v_back", "round"),
    "ː": ("supra",),
    "̃": ("supra",),
    "̤": ("supra", "breathy"),
    "ˈ": ("supra",),
    "ˌ": ("supra",),
}


def symbol_to_id(symbol: str) -> int:
    """Map an IPA or control symbol to its vocabulary ID, falling back to `<unk>`."""
    return SYMBOL_TO_ID.get(symbol, SYMBOL_TO_ID["<unk>"])


def parse_ipa_string(ipa: str) -> tuple[str, ...]:
    """Tokenize a raw IPA string greedily into unified inventory symbols."""
    tokens: list[str] = []
    i = 0
    n = len(ipa)
    while i < n:
        ch = ipa[i]
        if ch.isspace():
            i += 1
            continue
        matched: str | None = None
        for sym in _SORTED_IPA_SYMBOLS:
            if ipa.startswith(sym, i):
                matched = sym
                break
        if matched is not None:
            tokens.append(matched)
            i += len(matched)
        else:
            # Handle common IPA aliases cleanly
            if ch == "g":
                if i + 1 < n and ipa[i + 1] == "ʱ":
                    tokens.append("ɡʱ")
                    i += 2
                else:
                    tokens.append("ɡ")
                    i += 1
            elif ch == "ɾ":
                tokens.append("ɾ")
                i += 1
            else:
                i += 1
    return tuple(tokens)


def phoneme_to_articulatory_vector(symbol: str) -> tuple[int, ...]:
    """Return a deterministic 28-dimensional binary phonological feature vector."""
    vec = [0] * ARTICULATORY_DIM
    for feat in _PHONEME_FEATURES.get(symbol, ()):
        vec[_FEATURE_INDEX[feat]] = 1
    return tuple(vec)
