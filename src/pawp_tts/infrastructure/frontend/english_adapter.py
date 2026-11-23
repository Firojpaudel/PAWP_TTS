"""Zero-subprocess in-process libespeak-ng English normalizer & G2P adapter."""

from __future__ import annotations

import ctypes
import json
import re
import shutil
import subprocess
import threading
from pathlib import Path

from pawp_tts.domain.phonology.inventory import parse_ipa_string

_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")

_IPA_NORMALIZATION_RULES: tuple[tuple[str, str], ...] = (
    ("_", ""),
    ("\u200d", ""),
    ("\u0361", ""),  # tie bar if emitted
    ("ɚɹ", "əɹ"),
    ("ɚ", "əɹ"),
    ("ɐ", "ə"),
    ("ᵻ", "ɪ"),
    ("ʔ", "t"),
    ("n̩", "ən"),
    ("l̩", "əl"),
    ("m̩", "əm"),
    ("̩", ""),
    # Strip English tense-vowel length marks so `ː` stays exclusive to Nepali gemination / /ɦ/-deletion
    ("ɜː", "ɜ"),
    ("ɑː", "ɑ"),
    ("iː", "i"),
    ("uː", "u"),
    ("ɔː", "ɔ"),
    ("oː", "o"),
    ("æː", "æ"),
    ("ɛː", "ɛ"),
)


class _InProcessEspeakNg:
    """Thread-safe in-process wrapper around libespeak-ng C API with CLI fallback."""

    _lock = threading.Lock()
    _initialized = False
    _lib: ctypes.CDLL | None = None
    _lib_failed = False
    _current_voice: str | None = None
    _cli_bin: str | None = None

    @classmethod
    def _ensure_backend(cls) -> bool:
        """Return True if shared library is ready; False if CLI fallback must be used."""
        if cls._initialized:
            return cls._lib is not None

        for lib_name in ("libespeak-ng.so.1", "libespeak-ng.so"):
            try:
                lib = ctypes.CDLL(lib_name)
                lib.espeak_Initialize.argtypes = [
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_int,
                ]
                lib.espeak_Initialize.restype = ctypes.c_int
                lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
                lib.espeak_SetVoiceByName.restype = ctypes.c_int
                lib.espeak_TextToPhonemes.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_int,
                    ctypes.c_int,
                ]
                lib.espeak_TextToPhonemes.restype = ctypes.c_char_p

                sr = int(lib.espeak_Initialize(2, 0, None, 0))  # AUDIO_OUTPUT_SYNCHRONOUS = 2
                if sr >= 0:
                    cls._lib = lib
                    cls._initialized = True
                    return True
            except (OSError, AttributeError):
                continue

        cls._lib_failed = True
        cls._cli_bin = shutil.which("espeak-ng")
        if cls._cli_bin is None:
            raise RuntimeError(
                "Neither libespeak-ng shared library nor `espeak-ng` CLI binary was found."
            )
        cls._initialized = True
        return False

    @classmethod
    def text_to_ipa(cls, text: str, voice: str = "en-us") -> str:
        """Convert normalized English text to raw IPA in-process (< 0.1 ms)."""
        with cls._lock:
            use_c_api = cls._ensure_backend()
            if use_c_api and cls._lib is not None:
                if cls._current_voice != voice:
                    rc = int(cls._lib.espeak_SetVoiceByName(voice.encode("utf-8")))
                    if rc != 0:
                        raise RuntimeError(
                            f"libespeak-ng failed to select voice '{voice}' (rc={rc})"
                        )
                    cls._current_voice = voice

                buf = ctypes.create_string_buffer(text.encode("utf-8"))
                ptr = ctypes.c_void_p(ctypes.addressof(buf))
                chunks: list[str] = []
                while ptr.value:
                    res = cls._lib.espeak_TextToPhonemes(ctypes.byref(ptr), 1, 0x02)
                    if res:
                        chunks.append(bytes(res).decode("utf-8", errors="ignore"))
                    else:
                        break
                return " ".join(chunks)

            assert cls._cli_bin is not None
            proc = subprocess.run(
                [cls._cli_bin, "-q", "--ipa=3", "-v", voice, text],
                check=True,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            return proc.stdout


def _int_to_english(n: int) -> str:
    if n < 0:
        return f"minus {_int_to_english(-n)}"
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return f"{_TENS[tens]} {_ONES[ones]}" if ones else _TENS[tens]
    if n < 1000:
        hundreds, rem = divmod(n, 100)
        head = f"{_ONES[hundreds]} hundred"
        return f"{head} {_int_to_english(rem)}" if rem else head
    if n < 1_000_000:
        thousands, rem = divmod(n, 1000)
        head = f"{_int_to_english(thousands)} thousand"
        return f"{head} {_int_to_english(rem)}" if rem else head
    millions, rem = divmod(n, 1_000_000)
    head = f"{_int_to_english(millions)} million"
    return f"{head} {_int_to_english(rem)}" if rem else head


def _clean_espeak_ipa(raw: str) -> tuple[str, ...]:
    cleaned = raw.strip()
    for old, new in _IPA_NORMALIZATION_RULES:
        cleaned = cleaned.replace(old, new)
    return parse_ipa_string(cleaned)


class EnglishEspeakG2PAdapter:
    """Unbiased in-process libespeak-ng English normalizer & G2P adapter."""

    _MAX_CACHE_SIZE = 131_072

    def __init__(
        self,
        voice: str = "en-us",
        override_lexicon_path: str | Path | None = None,
    ) -> None:
        self._voice = voice
        self._overrides: dict[str, tuple[str, ...]] = {}
        self._cache: dict[str, tuple[str, ...]] = {}

        target_path: Path | None = None
        if override_lexicon_path is not None:
            target_path = Path(override_lexicon_path)
        else:
            package_lexicon = Path(__file__).parent / "data" / "en_fast_lexicon.json"
            repo_lexicon = (
                Path(__file__).resolve().parents[4] / "data" / "lexicon" / "en_fast_lexicon.json"
            )
            if package_lexicon.exists():
                target_path = package_lexicon
            elif repo_lexicon.exists():
                target_path = repo_lexicon

        if target_path is not None and target_path.exists():
            loaded = json.loads(target_path.read_text(encoding="utf-8"))
            for word, syms in loaded.items():
                self._overrides[str(word)] = tuple(str(s) for s in syms)

    def normalize(self, text: str) -> str:
        """Expand numbers, currency, percentages, and general orthographic initialisms."""
        out = re.sub(r"\b24/7\b", "twenty four seven", text)
        out = re.sub(
            r"(\d+(?:\.\d+)?)\s*%",
            lambda m: f"{self._expand_num(m.group(1))} percent",
            out,
        )
        out = re.sub(
            r"\$(\d+(?:\.\d+)?)",
            lambda m: f"{self._expand_num(m.group(1))} dollars",
            out,
        )
        out = re.sub(
            r"\b(\d+)([km])\b",
            lambda m: (
                f"{self._expand_num(m.group(1))} {'thousand' if m.group(2) == 'k' else 'million'}"
            ),
            out,
        )
        out = re.sub(
            r"\b(\d+)([A-Z])\b",
            lambda m: f"{self._expand_num(m.group(1))} {m.group(2)}",
            out,
        )
        out = re.sub(r"\b\d+(?:\.\d+)?\b", lambda m: self._expand_num(m.group(0)), out)
        # General consonant-only all-caps initialisms (SMS, KYC, GPU, CPU, TTS, etc.)
        out = re.sub(r"\b([BCDFGHJKLMNPQRSTVWXYZ]{2,5})\b", lambda m: " ".join(m.group(1)), out)
        return re.sub(r"\s+", " ", out).strip()

    def phonemize(self, text: str) -> tuple[str, ...]:
        """Convert normalized English span into unified IPA via in-process libespeak-ng."""
        clean = text.strip()
        if not clean:
            return ()

        if clean in self._overrides:
            return self._overrides[clean]
        lower_key = clean.lower()
        if lower_key in self._overrides:
            return self._overrides[lower_key]

        cached = self._cache.get(clean)
        if cached is not None:
            return cached

        raw_ipa = _InProcessEspeakNg.text_to_ipa(clean, voice=self._voice)
        parsed = _clean_espeak_ipa(raw_ipa)
        if len(self._cache) >= self._MAX_CACHE_SIZE:
            self._cache.pop(next(iter(self._cache)))
        self._cache[clean] = parsed
        return parsed

    @staticmethod
    def _expand_num(num_str: str) -> str:
        if "." in num_str:
            whole, frac = num_str.split(".", maxsplit=1)
            w_str = _int_to_english(int(whole)) if whole else "zero"
            f_str = " ".join(_ONES[int(d)] for d in frac if d.isdigit())
            return f"{w_str} point {f_str}".strip()
        return _int_to_english(int(num_str))
