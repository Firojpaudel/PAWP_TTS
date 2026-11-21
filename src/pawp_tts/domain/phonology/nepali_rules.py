"""Pure algorithmic Nepali Devanagari normalization and contrast-preserving G2P rules."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

DEVANAGARI_DIGITS: dict[str, str] = {
    "०": "0",
    "१": "1",
    "२": "2",
    "३": "3",
    "४": "4",
    "५": "5",
    "६": "6",
    "७": "7",
    "८": "8",
    "९": "9",
}

_UNITS_0_99: tuple[str, ...] = (
    "शून्य",
    "एक",
    "दुई",
    "तीन",
    "चार",
    "पाँच",
    "छ",
    "सात",
    "आठ",
    "नौ",
    "दस",
    "एघार",
    "बाह्र",
    "तेह्र",
    "चौध",
    "पन्ध्र",
    "सोह्र",
    "सत्र",
    "अठार",
    "उन्नाइस",
    "बीस",
    "एक्काइस",
    "बाइस",
    "तेइस",
    "चौबिस",
    "पच्चिस",
    "छब्बिस",
    "सत्ताइस",
    "अठ्ठाइस",
    "उनन्तिस",
    "तीस",
    "एकतिस",
    "बत्तिस",
    "तेत्तिस",
    "चौँतिस",
    "पैँतिस",
    "छत्तिस",
    "सैँतिस",
    "अठतिस",
    "उनन्चालिस",
    "चालिस",
    "एकचालिस",
    "बयालिस",
    "त्रिचालिस",
    "चवालिस",
    "पैँतालिस",
    "छयालिस",
    "सच्चालिस",
    "अठचालिस",
    "उनन्चास",
    "पचास",
    "एकाउन्न",
    "बाउन्न",
    "त्रिपन्न",
    "चवन्न",
    "पचपन्न",
    "छपन्न",
    "सन्ताउन्न",
    "अन्ठाउन्न",
    "उनन्साठी",
    "साठी",
    "एकसट्ठी",
    "बैसट्ठी",
    "त्रिसट्ठी",
    "चौसट्ठी",
    "पैँसट्ठी",
    "छैसट्ठी",
    "सतसट्ठी",
    "अठसट्ठी",
    "उनन्सत्तरी",
    "सत्तरी",
    "एकहत्तर",
    "बहत्तर",
    "त्रिहत्तर",
    "चौहत्तर",
    "पचहत्तर",
    "छहत्तर",
    "सतहत्तर",
    "अठहत्तर",
    "उनासी",
    "असी",
    "एकासी",
    "बयासी",
    "त्रियासी",
    "चौरासी",
    "पचासी",
    "छयासी",
    "सतासी",
    "अठासी",
    "उनान्नब्बे",
    "नब्बे",
    "एकानब्बे",
    "बयानब्बे",
    "त्रियानब्बे",
    "चौरानब्बे",
    "पन्चानब्बे",
    "छयानब्बे",
    "सन्तानब्बे",
    "अन्ठानब्बे",
    "उनान्सय",
)

DEV_INDEPENDENT_VOWELS: dict[str, tuple[str, ...]] = {
    "अ": ("ʌ",),
    "आ": ("a",),
    "इ": ("i",),
    "ई": ("i",),  # Modern spoken Nepali lacks phonemic length for इ/ई
    "उ": ("u",),
    "ऊ": ("u",),  # Modern spoken Nepali lacks phonemic length for उ/ऊ
    "ऋ": ("r", "i"),
    "ए": ("e",),
    "ऐ": ("ʌɪ",),
    "ओ": ("o",),
    "औ": ("ʌʊ",),
}

DEV_MATRA_TO_IPA: dict[str, tuple[str, ...]] = {
    "ा": ("a",),
    "ि": ("i",),
    "ी": ("i",),
    "ु": ("u",),
    "ू": ("u",),
    "ृ": ("r", "i"),
    "े": ("e",),
    "ै": ("ʌɪ",),
    "ो": ("o",),
    "ौ": ("ʌʊ",),
}

# Native Nepali consonant mappings:
# - Alveolar affricates: च छ ज झ -> ts tsʰ dz dzʱ
# - Sibilant merger: स श ष -> s (explicitly preventing Hindi /ʃ/ bleed on native Nepali)
# - Four-way laryngeal stop contrast across velar, retroflex, dental, bilabial places
DEV_CONSONANT_TO_IPA: dict[str, str] = {
    "क": "k",
    "ख": "kʰ",
    "ग": "ɡ",
    "घ": "ɡʱ",
    "ङ": "ŋ",
    "च": "ts",
    "छ": "tsʰ",
    "ज": "dz",
    "झ": "dzʱ",
    "ञ": "ɲ",
    "ट": "ʈ",
    "ठ": "ʈʰ",
    "ड": "ɖ",
    "ढ": "ɖʱ",
    "ण": "ɳ",
    "त": "t̪",
    "थ": "t̪ʰ",
    "द": "d̪",
    "ध": "d̪ʱ",
    "न": "n̪",
    "प": "p",
    "फ": "pʰ",
    "ब": "b",
    "भ": "bʱ",
    "म": "m",
    "य": "j",
    "र": "r",
    "ल": "l",
    "व": "w",
    "श": "s",
    "ष": "s",
    "स": "s",
    "ह": "ɦ",
    "क़": "k",
    "ख़": "kʰ",
    "ग़": "ɡ",
    "ज़": "z",
    "ड़": "ɽ",
    "ढ़": "ɽʱ",
    "फ़": "f",
}

DEV_VIRAMA = "्"
DEV_ANUSVARA = "ं"
DEV_CHANDRABINDU = "ँ"
DEV_VISARGA = "ः"
DEV_NUKTA = "़"

_STOP_HOMORGANIC_NASAL: dict[str, str] = {
    "k": "ŋ",
    "kʰ": "ŋ",
    "ɡ": "ŋ",
    "ɡʱ": "ŋ",
    "ts": "n",
    "tsʰ": "n",
    "dz": "n",
    "dzʱ": "n",
    "ʈ": "ɳ",
    "ʈʰ": "ɳ",
    "ɖ": "ɳ",
    "ɖʱ": "ɳ",
    "t̪": "n̪",
    "t̪ʰ": "n̪",
    "d̪": "n̪",
    "d̪ʱ": "n̪",
    "p": "m",
    "pʰ": "m",
    "b": "m",
    "bʱ": "m",
}

_AUXILIARY_STEM_PATTERNS = re.compile(
    r"^(नपा(ए|एर|उँ|इ)|नसक(े|एर|्|ि)|नहु(ने|ँ|न्)|नभ्या(ए|एर)|"
    r"पा(यो|ए|एर|उँ|एन|इ|उनु)|"
    r"सक्(नु|ने|छ|छन्|छु|दै|दिन|दैन|दिएला)|"
    r"सक(्यो|े|एर|्छ|िन्छ|ेन)|"
    r"चाह(न्छु|न्छ|न्छौँ|न्थे|न्थ्यो|नु|ने|िएला|े|्यो)|"
    r"चाहि(न्छ|न्थ्यो|एला|एको|नु)|"
    r"थाल्(नु|ने|छ|छन्|छु)|"
    r"थाल(्यो|े|एर|्छ|िन्)|"
    r"लाग्(नु|ने|छ|छन्|छु)|"
    r"लाग(्यो|े|एर|्छ|िन्)|"
    r"खोज्(नु|ने|छ|छन्|छु)|"
    r"खोज(्यो|े|एर|्छ|िन्)|"
    r"पर्(छ|यो|े|दैन|िएला|नु|ने|थ्यो)|"
    r"दि(यो|ए|एर|न्छ|इन्|एको|नु)|"
    r"गर्य(ो)|गरे(र|को)|"
    r"आ(यो|ए|एर|उँछ|नु)|"
    r"ग(यो|ए|एर|इन्|नु)|"
    r"हुँ(दैन|हुन्न)|"
    r"जानु(पर्छ|पर्ने|पर्यो)?|"
    r"छाड्(यो|े|एर)|छोड्(यो|े|एर)|हाल(्यो|े|एर))"
)
_CLOSED_FUNCTION_WORDS = frozenset({"तीन", "किन", "कुन", "पनि", "छैन", "होइन", "हैन", "अझै", "सधैँ"})
_NOUN_POSTPOSITIONS = re.compile(r"(सम्म|भर|हुँ|प्रतिदिन|देखि|बाट|मा|को|का|की|लाई|सित|सँग)$")
_DETERMINER_PATTERNS = re.compile(
    r"^(धेरै|थोरै|केही|कति|कुन|त्यो|यो|यी|ती|सबै|हरेक|प्रत्येक|पहिलो|दोस्रो|तेस्रो|विगत|आगामी|पछिल्लो|अघिल्लो)$"
)
_CARDINAL_WORDS = frozenset(
    {
        "एक",
        "दुई",
        "तीन",
        "चार",
        "पाँच",
        "छ",
        "सात",
        "आठ",
        "नौ",
        "दस",
        "दश",
        "एघार",
        "बाह्र",
        "तेह्र",
        "चौध",
        "पन्ध्र",
        "सोह्र",
        "सत्र",
        "अठार",
        "उन्नाइस",
        "बीस",
    }
)
_PURPOSIVE_PARTICLES = frozenset({"लागि", "निम्ति", "भनेर", "भन्दै", "खोजेर", "को", "का"})
_GENITIVE_MARKERS = ("को", "का", "की")
_POSSESSIVE_PRONOUNS = frozenset(
    {
        "मेरो",
        "मेरा",
        "मेरी",
        "हाम्रो",
        "हाम्रा",
        "हाम्री",
        "तिम्रो",
        "तिम्रा",
        "तिम्री",
        "आफ्नो",
        "आफ्ना",
        "आफ्नी",
    }
)

# Nouns that act as the experiencer/subject of modal constructions ("<inf> मन लाग्नु").
# They are never infinitives themselves; the word *before* them is.
# Source: infinitive -न with potential/desiderative constructions (Nepali grammar references).
_MODAL_NOUNS = frozenset({"मन", "इच्छा", "रहर", "डर", "लाज", "जाँगर"})
_LAAG_FORMS = re.compile(r"^(लाग(्यो|े|एर|्छ|िन्|ेको|्दैन|ेन|्ने|्थ्यो))$")

# Words that keep the final inherent vowel /ʌ/ despite the general noun deletion rule.
# Adverbs and postpositions keep it (अब /ʌbʌ/, तिर /t̪irʌ/), as do a few nouns (दुख /dukʰʌ/,
# सुख /sukʰʌ/). Source: Nepali phonology (Khatiwada 2009; Pokharel 1989), via secondary
# description; see docs/PHONOLOGY.md.
_KEEP_FINAL_SCHWA = frozenset({"अब", "जब", "तब", "तिर", "दुख", "सुख"})

# Negative past/finite verb endings keep the final /ʌ/ unless a halanta is written
# (e.g. गएन /ɡʌenʌ/). Source: finite verbs retain final schwa (Khatiwada 2009 via secondary).
_NEGATIVE_FINITE_ENDINGS = ("एन", "ेन", "ैन")


@dataclass(frozen=True)
class Akshara:
    """Single orthographic Devanagari syllable unit `(C)*V(M)`."""

    raw: str
    consonants: tuple[str, ...]
    vowel_ipa: tuple[str, ...]
    has_inherent_schwa: bool
    is_conjunct: bool
    has_virama: bool
    nasal_mark: str


def integer_to_nepali_words(value: int) -> str:
    """Expand an integer into standard Nepali number words using Indian/Nepali grouping."""
    if value < 0:
        return f"माइनस {integer_to_nepali_words(-value)}"
    if value < 100:
        return _UNITS_0_99[value]
    if value < 1000:
        hundreds, rem = divmod(value, 100)
        head = f"{_UNITS_0_99[hundreds]} सय"
        return f"{head} {integer_to_nepali_words(rem)}" if rem else head

    scales: tuple[tuple[int, str], ...] = (
        (10_000_000_000_00, "खर्ब"),
        (1_000_000_000, "अर्ब"),
        (10_000_000, "करोड"),
        (100_000, "लाख"),
        (1_000, "हजार"),
    )
    parts: list[str] = []
    rem = value
    for divisor, label in scales:
        if rem >= divisor:
            q, rem = divmod(rem, divisor)
            parts.append(f"{integer_to_nepali_words(q)} {label}")
    if rem:
        parts.append(integer_to_nepali_words(rem))
    return " ".join(parts)


def normalize_nepali_text(text: str) -> str:
    """Normalize Devanagari Unicode (NFC), currency, decimals, and numerals to Nepali words."""
    normalized = unicodedata.normalize("NFC", text)
    # Translate Devanagari digits inside numbers to ASCII temporarily for clean arithmetic
    ascii_digits = "".join(DEVANAGARI_DIGITS.get(ch, ch) for ch in normalized)
    # Replace `%` with Nepali word `प्रतिशत`
    ascii_digits = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"\1 प्रतिशत", ascii_digits)
    # Replace currency prefixes `रु.` / `रु` followed by digits
    ascii_digits = re.sub(
        r"(?:रु\.?|NPR)\s*(\d+(?:\.\d+)?)",
        lambda m: f"{_expand_number_str(m.group(1))} रुपैयाँ",
        ascii_digits,
        flags=re.IGNORECASE,
    )
    ascii_digits = re.sub(r"\d+(?:\.\d+)?", lambda m: _expand_number_str(m.group(0)), ascii_digits)
    ascii_digits = ascii_digits.replace("।", " . ")
    return re.sub(r"\s+", " ", ascii_digits).strip()


def _expand_number_str(num_str: str) -> str:
    if "." in num_str:
        whole_s, frac_s = num_str.split(".", maxsplit=1)
        whole = integer_to_nepali_words(int(whole_s)) if whole_s else "शून्य"
        frac_words = " ".join(_UNITS_0_99[int(d)] for d in frac_s if d.isdigit())
        return f"{whole} दशमलव {frac_words}".strip()
    return integer_to_nepali_words(int(num_str))


def segment_aksharas(word: str) -> list[Akshara]:
    """Segment a single Devanagari word into constituent Aksharas."""
    chars = list(unicodedata.normalize("NFC", word))
    n = len(chars)
    i = 0
    aksharas: list[Akshara] = []

    while i < n:
        ch = chars[i]
        if ch in DEV_INDEPENDENT_VOWELS:
            raw = ch
            vowel = DEV_INDEPENDENT_VOWELS[ch]
            i += 1
            nasal_mark = ""
            if i < n and chars[i] in {DEV_ANUSVARA, DEV_CHANDRABINDU, DEV_VISARGA}:
                nasal_mark = chars[i]
                raw += nasal_mark
                i += 1
            aksharas.append(
                Akshara(
                    raw=raw,
                    consonants=(),
                    vowel_ipa=vowel,
                    has_inherent_schwa=False,
                    is_conjunct=False,
                    has_virama=False,
                    nasal_mark=nasal_mark,
                )
            )
            continue

        if ch in DEV_CONSONANT_TO_IPA:
            cons = [ch]
            raw = ch
            i += 1
            if i < n and chars[i] == DEV_NUKTA:
                composed = unicodedata.normalize("NFC", ch + chars[i])
                if composed in DEV_CONSONANT_TO_IPA:
                    cons[-1] = composed
                raw += chars[i]
                i += 1

            is_conjunct = False
            while i < n - 1 and chars[i] == DEV_VIRAMA and chars[i + 1] in DEV_CONSONANT_TO_IPA:
                is_conjunct = True
                next_c = chars[i + 1]
                raw += chars[i] + next_c
                i += 2
                if i < n and chars[i] == DEV_NUKTA:
                    composed = unicodedata.normalize("NFC", next_c + chars[i])
                    if composed in DEV_CONSONANT_TO_IPA:
                        next_c = composed
                    raw += chars[i]
                    i += 1
                cons.append(next_c)

            has_virama = False
            has_inherent_schwa = False
            vowel_ipa: tuple[str, ...] = ("ʌ",)
            if i < n and chars[i] == DEV_VIRAMA:
                has_virama = True
                vowel_ipa = ()
                raw += chars[i]
                i += 1
            elif i < n and chars[i] in DEV_MATRA_TO_IPA:
                vowel_ipa = DEV_MATRA_TO_IPA[chars[i]]
                raw += chars[i]
                i += 1
            else:
                has_inherent_schwa = True

            nasal_mark = ""
            if i < n and chars[i] in {DEV_ANUSVARA, DEV_CHANDRABINDU, DEV_VISARGA}:
                nasal_mark = chars[i]
                raw += nasal_mark
                i += 1

            aksharas.append(
                Akshara(
                    raw=raw,
                    consonants=tuple(cons),
                    vowel_ipa=vowel_ipa,
                    has_inherent_schwa=has_inherent_schwa,
                    is_conjunct=is_conjunct,
                    has_virama=has_virama,
                    nasal_mark=nasal_mark,
                )
            )
            continue

        # Skip stray marks/characters outside Devanagari letters
        i += 1

    return aksharas


def _is_verbal_infinitive(
    word: str,
    aksharas: list[Akshara],
    prev_word: str | None,
    next_word: str | None,
    next_next_word: str | None = None,
) -> bool:
    if not aksharas or len(word) < 2 or word in _CLOSED_FUNCTION_WORDS:
        return False
    if word in _MODAL_NOUNS:
        return False
    last = aksharas[-1]
    if not (last.consonants == ("न",) and last.has_inherent_schwa and not last.has_virama):
        return False
    if prev_word:
        clean_prev = re.sub(r"[^\u0900-\u097F]", "", prev_word)
        if clean_prev in _CARDINAL_WORDS or _DETERMINER_PATTERNS.match(clean_prev):
            return False
        if clean_prev in _POSSESSIVE_PRONOUNS or any(
            clean_prev.endswith(g) for g in _GENITIVE_MARKERS
        ):
            return False
    if next_word:
        clean_next = re.sub(r"[^\u0900-\u097F]", "", next_word)
        # "<inf> मन लाग्यो": the infinitive precedes the modal noun
        if clean_next in _MODAL_NOUNS:
            clean_nn = re.sub(r"[^\u0900-\u097F]", "", next_next_word or "")
            return bool(_LAAG_FORMS.match(clean_nn))
        if _AUXILIARY_STEM_PATTERNS.match(clean_next) or clean_next in _PURPOSIVE_PARTICLES:
            return True
    return bool(word.endswith("ाउन") and len(word) >= 4 and next_word)


def word_to_nepali_ipa(
    word: str,
    prev_word: str | None = None,
    next_word: str | None = None,
    override_lexicon: dict[str, tuple[str, ...]] | None = None,
    next_next_word: str | None = None,
) -> tuple[str, ...]:
    """Convert a single Devanagari word in syntactic context to contrast-preserving IPA."""
    clean = re.sub(r"[^\u0900-\u097F]", "", unicodedata.normalize("NFC", word))
    if not clean:
        return ()
    if override_lexicon and clean in override_lexicon:
        return override_lexicon[clean]

    aksharas = segment_aksharas(clean)
    if not aksharas:
        return ()

    n_ak = len(aksharas)
    retain_final_schwa = False
    if aksharas[-1].has_inherent_schwa:
        if (
            n_ak == 1
            or clean in _KEEP_FINAL_SCHWA
            or (clean.endswith(_NEGATIVE_FINITE_ENDINGS) and n_ak >= 2)
        ):
            retain_final_schwa = True
        elif aksharas[-1].is_conjunct:
            # Tatsama cluster protection (e.g., मित्र -> /mit̪rʌ/)
            retain_final_schwa = True
        elif not _NOUN_POSTPOSITIONS.search(clean) and _is_verbal_infinitive(
            clean, aksharas, prev_word, next_word, next_next_word
        ):
            retain_final_schwa = True

    # Right-to-left schwa retention decision per Akshara
    emit_vowel = [True] * n_ak
    if aksharas[-1].has_inherent_schwa and not retain_final_schwa:
        emit_vowel[-1] = False
    for idx in range(n_ak - 2, 0, -1):
        ak = aksharas[idx]
        if (
            ak.has_inherent_schwa
            and not ak.is_conjunct
            and not aksharas[idx + 1].is_conjunct
            and not ak.nasal_mark
            and aksharas[idx + 1].vowel_ipa
            and emit_vowel[idx + 1]
            and aksharas[idx - 1].vowel_ipa
            and len(ak.consonants) == 1
        ):
            # Systematic medial syncope when flanked by retained vowels and non-conjunct onset
            emit_vowel[idx] = False

    ipa: list[str] = []
    skip_idx = -1
    for idx, ak in enumerate(aksharas):
        if idx == skip_idx:
            continue

        # Handle intervocalic /ɦ/-deletion with surface breathy vowel lengthening
        # e.g., पहाड /pʌɦaɖ/ -> [p a ̤ ː ɽ]
        if (
            0 < idx < n_ak - 1
            and ak.consonants == ("ह",)
            and ak.vowel_ipa in {("ʌ",), ("a",)}
            and emit_vowel[idx - 1]
            and emit_vowel[idx]
            and aksharas[idx - 1].vowel_ipa in {("ʌ",), ("a",)}
        ):
            if ipa and ipa[-1] == "ʌ":
                ipa[-1] = "a"
            ipa.extend(("̤", "ː"))
            continue

        # Emit consonants with gemination and intervocalic retroflex flapping
        c_list = ak.consonants
        c_idx = 0
        while c_idx < len(c_list):
            c_char = c_list[c_idx]
            mapped = DEV_CONSONANT_TO_IPA.get(c_char, "")
            # Phonemic gemination inside conjuncts (C + ् + C): e.g., त्त -> t̪ ː
            if c_idx + 1 < len(c_list) and c_list[c_idx + 1] == c_char:
                if mapped:
                    ipa.extend((mapped, "ː"))
                c_idx += 2
                continue
            # Post-vocalic retroflex flapping for singleton ड -> ɽ, ढ -> ɽʱ
            if (
                len(c_list) == 1
                and c_char in {"ड", "ढ"}
                and idx > 0
                and emit_vowel[idx - 1]
                and aksharas[idx - 1].vowel_ipa
                and (not aksharas[idx - 1].nasal_mark)
            ):
                mapped = "ɽ" if c_char == "ड" else "ɽʱ"
            # व -> /b/ before e, i, o, ʌi and before र/य/व (Nepali phonology, secondary source)
            if c_char == "व":
                nxt_c = c_list[c_idx + 1] if c_idx + 1 < len(c_list) else None
                vow = ak.vowel_ipa[0] if (nxt_c is None and ak.vowel_ipa) else ""
                if nxt_c in {"र", "य", "व"} or vow in {"e", "i", "o", "ʌi", "ʌɪ"}:
                    mapped = "b"
            if mapped:
                ipa.append(mapped)
            c_idx += 1

        # Special Nepali conjunct क्ष -> k s, ज्ञ -> ɡ j
        # (Handled cleanly via sequence above unless explicit override needed; adjust क्ष -> k s)
        if ak.consonants == ("क", "ष") and len(ipa) >= 2:
            ipa[-2:] = ["k", "s"]

        if emit_vowel[idx] and ak.vowel_ipa:
            ipa.extend(ak.vowel_ipa)

        if ak.nasal_mark == DEV_CHANDRABINDU:
            ipa.append("̃")
        elif ak.nasal_mark == DEV_ANUSVARA:
            next_c_ipa = (
                DEV_CONSONANT_TO_IPA.get(aksharas[idx + 1].consonants[0], "")
                if idx + 1 < n_ak and aksharas[idx + 1].consonants
                else ""
            )
            if next_c_ipa in _STOP_HOMORGANIC_NASAL:
                ipa.append(_STOP_HOMORGANIC_NASAL[next_c_ipa])
            else:
                ipa.append("̃")
        elif ak.nasal_mark == DEV_VISARGA:
            ipa.append("h")

    return tuple(ipa)
