"""Gold test suite for Nepali pronunciation frontend validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrontendGoldCase:
    case_id: int
    word: str
    expected_ipa: tuple[str, ...]
    pos: str
    confidence: str


GOLD_300_CASES: tuple[FrontendGoldCase, ...] = (
    FrontendGoldCase(1, "उखाड्छेस्", ("u", "kʰ", "a", "ɽ", "tsʰ", "e", "s"), "verb", "medium"),
    FrontendGoldCase(2, "पिराउन", ("p", "i", "r", "a", "u", "n̪", "ʌ"), "verb", "medium"),
    FrontendGoldCase(3, "झराउँदै", ("dzʱ", "ʌ", "r", "a", "u", "̃", "d̪", "ʌɪ"), "verb", "medium"),
    FrontendGoldCase(
        4,
        "सेमिनारहरूलाई",
        ("s", "e", "m", "i", "n̪", "a", "r", "ɦ", "ʌ", "r", "u", "l", "a", "i"),
        "noun",
        "low",
    ),
    FrontendGoldCase(5, "सुकुलबाट", ("s", "u", "k", "u", "l", "b", "a", "ʈ", "ʌ"), "noun", "medium"),
    FrontendGoldCase(6, "बिझाएकी", ("b", "i", "dzʱ", "a", "e", "k", "i"), "verb", "medium"),
    FrontendGoldCase(7, "लागेनि", ("l", "a", "ɡ", "e", "n̪", "i"), "verb", "medium"),
    FrontendGoldCase(8, "सुँघाएर", ("s", "u", "̃", "ɡʱ", "a", "e", "r", "ʌ"), "verb", "medium"),
    FrontendGoldCase(
        9,
        "केमिस्ट्रीसँग",
        ("k", "e", "m", "i", "s", "ʈ", "r", "i", "s", "ʌ", "̃", "ɡ", "ʌ"),
        "noun",
        "low",
    ),
    FrontendGoldCase(10, "चार्जर", ("ts", "a", "r", "dz", "ʌ", "r"), "noun", "low"),
    FrontendGoldCase(
        11, "केन्द्रसँग", ("k", "e", "n̪", "d̪", "r", "ʌ", "s", "ʌ", "̃", "ɡ", "ʌ"), "noun", "medium"
    ),
    FrontendGoldCase(
        12,
        "व्यवस्थाहरूले",
        ("b", "j", "ʌ", "w", "ʌ", "s", "t̪ʰ", "a", "ɦ", "ʌ", "r", "u", "l", "e"),
        "noun",
        "medium",
    ),
    FrontendGoldCase(
        13,
        "कालिमाटीसम्म",
        ("k", "a", "l", "i", "m", "a", "ʈ", "i", "s", "ʌ", "m", "ː", "ʌ"),
        "noun",
        "medium",
    ),
    FrontendGoldCase(14, "ओइलाए", ("o", "i", "l", "a", "e"), "verb", "medium"),
    FrontendGoldCase(
        15, "जलिएछैन", ("dz", "ʌ", "l", "i", "e", "tsʰ", "ʌɪ", "n̪", "ʌ"), "verb", "medium"
    ),
    FrontendGoldCase(16, "तोडिएपछि", ("t̪", "o", "ɽ", "i", "e", "p", "tsʰ", "i"), "verb", "medium"),
    FrontendGoldCase(
        17, "पुग्नुभयो", ("p", "u", "ɡ", "n̪", "u", "bʱ", "ʌ", "j", "o"), "verb", "medium"
    ),
    FrontendGoldCase(18, "बयानब्बे", ("b", "ʌ", "j", "a", "n̪", "ʌ", "b", "ː", "e"), "num", "high"),
    FrontendGoldCase(19, "पुछेन", ("p", "u", "tsʰ", "e", "n̪", "ʌ"), "verb", "medium"),
    FrontendGoldCase(20, "बगरहरू", ("b", "ʌ", "ɡ", "ʌ", "r", "ɦ", "ʌ", "r", "u"), "noun", "medium"),
    FrontendGoldCase(21, "चित्रको", ("ts", "i", "t̪", "r", "ʌ", "k", "o"), "noun", "medium"),
    FrontendGoldCase(
        22, "नकार्नेछैन", ("n̪", "ʌ", "k", "a", "r", "n̪", "e", "tsʰ", "ʌɪ", "n̪", "ʌ"), "verb", "medium"
    ),
    FrontendGoldCase(23, "रोबोटका", ("r", "o", "b", "o", "ʈ", "k", "a"), "noun", "low"),
    FrontendGoldCase(24, "बैसट्ठी", ("b", "ʌɪ", "s", "ʌ", "ʈ", "ʈʰ", "i"), "num", "high"),
    FrontendGoldCase(
        25,
        "हवाइजहाजहरूको",
        ("ɦ", "ʌ", "w", "a", "i", "dz", "ɦ", "a", "dz", "ɦ", "ʌ", "r", "u", "k", "o"),
        "noun",
        "medium",
    ),
    FrontendGoldCase(26, "पाकौँ", ("p", "a", "k", "ʌʊ", "̃"), "verb", "medium"),
    FrontendGoldCase(27, "म्याट्रेसकी", ("m", "j", "a", "ʈ", "r", "e", "s", "k", "i"), "noun", "low"),
    FrontendGoldCase(
        28, "डाडुहरूका", ("ɖ", "a", "ɽ", "u", "ɦ", "ʌ", "r", "u", "k", "a"), "noun", "medium"
    ),
    FrontendGoldCase(
        29,
        "अक्षरहरूलाई",
        ("ʌ", "k", "tsʰ", "ʌ", "r", "ɦ", "ʌ", "r", "u", "l", "a", "i"),
        "noun",
        "low",
    ),
    FrontendGoldCase(
        30,
        "ओखलढुङ्गातिर",
        ("o", "kʰ", "ʌ", "l", "ɖʱ", "u", "ŋ", "ɡ", "a", "t̪", "i", "r", "ʌ"),
        "noun",
        "medium",
    ),
    FrontendGoldCase(
        31, "डढेलोसँग", ("ɖ", "ʌ", "ɽʱ", "e", "l", "o", "s", "ʌ", "̃", "ɡ", "ʌ"), "noun", "medium"
    ),
    FrontendGoldCase(32, "बोकेका", ("b", "o", "k", "e", "k", "a"), "verb", "medium"),
    FrontendGoldCase(
        33, "हडतालहरू", ("ɦ", "ʌ", "ɽ", "t̪", "a", "l", "ɦ", "ʌ", "r", "u"), "noun", "medium"
    ),
    FrontendGoldCase(34, "चङ्गाकी", ("ts", "ʌ", "ŋ", "ɡ", "a", "k", "i"), "noun", "medium"),
    FrontendGoldCase(
        35, "कुट्नुभएन", ("k", "u", "ʈ", "n̪", "u", "bʱ", "ʌ", "e", "n̪", "ʌ"), "verb", "medium"
    ),
    FrontendGoldCase(36, "हे", ("ɦ", "e"), "interj", "high"),
    FrontendGoldCase(
        37, "गहुँभित्र", ("ɡ", "ʌ", "ɦ", "u", "̃", "bʱ", "i", "t̪", "r", "ʌ"), "noun", "medium"
    ),
    FrontendGoldCase(
        38,
        "आत्मनिर्भरतादेखि",
        ("a", "t̪", "m", "ʌ", "n̪", "i", "r", "bʱ", "ʌ", "r", "ʌ", "t̪", "a", "d̪", "e", "kʰ", "i"),
        "noun",
        "medium",
    ),
    FrontendGoldCase(
        39, "कुहाउँछन्", ("k", "u", "ɦ", "a", "u", "̃", "tsʰ", "ʌ", "n̪"), "verb", "medium"
    ),
    FrontendGoldCase(
        40, "साँचोहरूका", ("s", "a", "̃", "ts", "o", "ɦ", "ʌ", "r", "u", "k", "a"), "noun", "medium"
    ),
    FrontendGoldCase(41, "थुन्नुमा", ("t̪ʰ", "u", "n̪", "ː", "u", "m", "a"), "verb", "medium"),
    FrontendGoldCase(42, "गाड्यौ", ("ɡ", "a", "ɽ", "j", "ʌʊ"), "verb", "medium"),
    FrontendGoldCase(43, "काँक्रोमा", ("k", "a", "̃", "k", "r", "o", "m", "a"), "noun", "medium"),
    FrontendGoldCase(44, "लजाएछिन्", ("l", "ʌ", "dz", "a", "e", "tsʰ", "i", "n̪"), "verb", "medium"),
    FrontendGoldCase(45, "नपाओस्", ("n̪", "ʌ", "p", "a", "o", "s"), "verb", "medium"),
    FrontendGoldCase(
        46, "लुटिनेहरू", ("l", "u", "ʈ", "i", "n̪", "e", "ɦ", "ʌ", "r", "u"), "verb", "medium"
    ),
    FrontendGoldCase(47, "आदेशका", ("a", "d̪", "e", "s", "k", "a"), "noun", "medium"),
    FrontendGoldCase(48, "चम्क्न", ("ts", "ʌ", "m", "k", "n̪", "ʌ"), "verb", "medium"),
    FrontendGoldCase(49, "नासेन", ("n̪", "a", "s", "e", "n̪", "ʌ"), "verb", "medium"),
    FrontendGoldCase(
        50, "दबाएछैन", ("d̪", "ʌ", "b", "a", "e", "tsʰ", "ʌɪ", "n̪", "ʌ"), "verb", "medium"
    ),
    FrontendGoldCase(51, "लडाउन", ("l", "ʌ", "ɽ", "a", "u", "n̪", "ʌ"), "verb", "medium"),
    FrontendGoldCase(52, "मोडिएछ", ("m", "o", "ɽ", "i", "e", "tsʰ", "ʌ"), "verb", "medium"),
    FrontendGoldCase(53, "जाडोमा", ("dz", "a", "ɽ", "o", "m", "a"), "noun", "medium"),
    FrontendGoldCase(54, "सडाएनस्", ("s", "ʌ", "ɽ", "a", "e", "n̪", "ʌ", "s"), "verb", "medium"),
    FrontendGoldCase(55, "नुहाउँथे", ("n̪", "u", "ɦ", "a", "u", "̃", "t̪ʰ", "e"), "verb", "medium"),
    FrontendGoldCase(
        56,
        "बुद्धिमत्ताअनुसार",
        ("b", "u", "d̪", "d̪ʱ", "i", "m", "ʌ", "t̪", "ː", "a", "ʌ", "n̪", "u", "s", "a", "r", "ʌ"),
        "noun",
        "medium",
    ),
    FrontendGoldCase(57, "पारेछ", ("p", "a", "r", "e", "tsʰ", "ʌ"), "verb", "medium"),
    FrontendGoldCase(
        58, "बिर्साउँथिस्", ("b", "i", "r", "s", "a", "u", "̃", "t̪ʰ", "i", "s"), "verb", "medium"
    ),
    FrontendGoldCase(
        59, "चालकहरूले", ("ts", "a", "l", "ʌ", "k", "ɦ", "ʌ", "r", "u", "l", "e"), "noun", "medium"
    ),
    FrontendGoldCase(
        60, "बिझाउनेछिन्", ("b", "i", "dzʱ", "a", "u", "n̪", "e", "tsʰ", "i", "n̪"), "verb", "medium"
    ),
    FrontendGoldCase(61, "जमाउँदो", ("dz", "ʌ", "m", "a", "u", "̃", "d̪", "o"), "verb", "medium"),
    FrontendGoldCase(62, "तर्सेमा", ("t̪", "ʌ", "r", "s", "e", "m", "a"), "verb", "medium"),
    FrontendGoldCase(
        63, "भेट्नुहुन्न", ("bʱ", "e", "ʈ", "n̪", "u", "̤", "ː", "n̪", "ː", "ʌ"), "verb", "medium"
    ),
    FrontendGoldCase(
        64,
        "मैनबत्तीदेखि",
        ("m", "ʌɪ", "n̪", "ʌ", "b", "ʌ", "t̪", "ː", "i", "d̪", "e", "kʰ", "i"),
        "noun",
        "medium",
    ),
    FrontendGoldCase(65, "सहाउँथ्यो", ("s", "a", "̤", "ː", "u", "̃", "t̪ʰ", "j", "o"), "verb", "medium"),
    FrontendGoldCase(66, "भेटाएनस्", ("bʱ", "e", "ʈ", "a", "e", "n̪", "ʌ", "s"), "verb", "medium"),
    FrontendGoldCase(
        67, "तबलासम्म", ("t̪", "ʌ", "b", "l", "a", "s", "ʌ", "m", "ː", "ʌ"), "noun", "medium"
    ),
    FrontendGoldCase(68, "साटिएन", ("s", "a", "ʈ", "i", "e", "n̪", "ʌ"), "verb", "medium"),
    FrontendGoldCase(69, "थुनाकी", ("t̪ʰ", "u", "n̪", "a", "k", "i"), "noun", "medium"),
    FrontendGoldCase(70, "उठेछौ", ("u", "ʈʰ", "e", "tsʰ", "ʌʊ"), "verb", "medium"),
    FrontendGoldCase(
        71, "कोर्सहरूलाई", ("k", "o", "r", "s", "ʌ", "ɦ", "ʌ", "r", "u", "l", "a", "i"), "noun", "low"
    ),
    FrontendGoldCase(72, "छात्रावास", ("tsʰ", "a", "t̪", "r", "a", "w", "a", "s"), "noun", "medium"),
    FrontendGoldCase(73, "सुरुङ", ("s", "u", "r", "u", "ŋ"), "noun", "medium"),
    FrontendGoldCase(
        74, "भालुबाहिर", ("bʱ", "a", "l", "u", "b", "a", "ɦ", "i", "r", "ʌ"), "noun", "medium"
    ),
    FrontendGoldCase(75, "ओढ्दिनँ", ("o", "ɽʱ", "d̪", "i", "n̪", "ʌ", "̃"), "verb", "medium"),
    FrontendGoldCase(
        76, "शक्तिहरूका", ("s", "ʌ", "k", "t̪", "i", "ɦ", "ʌ", "r", "u", "k", "a"), "noun", "medium"
    ),
    FrontendGoldCase(
        77, "बुहारीबाट", ("b", "u", "ɦ", "a", "r", "i", "b", "a", "ʈ", "ʌ"), "noun", "medium"
    ),
    FrontendGoldCase(
        78,
        "नफुटाउनुहोस्",
        ("n̪", "ʌ", "pʰ", "u", "ʈ", "a", "u", "n̪", "u", "ɦ", "o", "s"),
        "verb",
        "medium",
    ),
    FrontendGoldCase(79, "चिन्यौँ", ("ts", "i", "n̪", "j", "ʌʊ", "̃"), "verb", "medium"),
    FrontendGoldCase(80, "गन्छस्", ("ɡ", "ʌ", "n̪", "tsʰ", "ʌ", "s"), "verb", "medium"),
    FrontendGoldCase(81, "दलाएनौँ", ("d̪", "ʌ", "l", "a", "e", "n̪", "ʌʊ", "̃"), "verb", "medium"),
    FrontendGoldCase(82, "काट्नेलाई", ("k", "a", "ʈ", "n̪", "e", "l", "a", "i"), "verb", "medium"),
    FrontendGoldCase(
        83, "ओखरहरूलाई", ("o", "kʰ", "ʌ", "r", "ɦ", "ʌ", "r", "u", "l", "a", "i"), "noun", "medium"
    ),
    FrontendGoldCase(84, "सुल्झेछौ", ("s", "u", "l", "dzʱ", "e", "tsʰ", "ʌʊ"), "verb", "medium"),
    FrontendGoldCase(85, "उधारेमा", ("u", "d̪ʱ", "a", "r", "e", "m", "a"), "verb", "medium"),
    FrontendGoldCase(
        86, "डराएजस्तो", ("ɖ", "ʌ", "r", "a", "e", "dz", "ʌ", "s", "t̪", "o"), "verb", "medium"
    ),
    FrontendGoldCase(87, "घरहरूले", ("ɡʱ", "ʌ", "r", "ɦ", "ʌ", "r", "u", "l", "e"), "noun", "medium"),
    FrontendGoldCase(88, "बचिनन्", ("b", "ʌ", "ts", "i", "n̪", "ʌ", "n̪"), "verb", "medium"),
    FrontendGoldCase(89, "छानिने", ("tsʰ", "a", "n̪", "i", "n̪", "e"), "verb", "medium"),
    FrontendGoldCase(
        90, "सूचीहरूले", ("s", "u", "ts", "i", "ɦ", "ʌ", "r", "u", "l", "e"), "noun", "medium"
    ),
    FrontendGoldCase(91, "फल्दैनौँ", ("pʰ", "ʌ", "l", "d̪", "ʌɪ", "n̪", "ʌʊ", "̃"), "verb", "medium"),
    FrontendGoldCase(
        92,
        "अनुभूतिसम्म",
        ("ʌ", "n̪", "u", "bʱ", "u", "t̪", "i", "s", "ʌ", "m", "ː", "ʌ"),
        "noun",
        "medium",
    ),
    FrontendGoldCase(
        93, "तिराउनेहरू", ("t̪", "i", "r", "a", "u", "n̪", "e", "ɦ", "ʌ", "r", "u"), "verb", "medium"
    ),
    FrontendGoldCase(
        94, "जोडिन्थेन", ("dz", "o", "ɽ", "i", "n̪", "t̪ʰ", "e", "n̪", "ʌ"), "verb", "medium"
    ),
    FrontendGoldCase(95, "फर्क्छिन्", ("pʰ", "ʌ", "r", "k", "tsʰ", "i", "n̪"), "verb", "medium"),
    FrontendGoldCase(96, "ओसारेझैँ", ("o", "s", "a", "r", "e", "dzʱ", "ʌɪ", "̃"), "verb", "medium"),
    FrontendGoldCase(97, "सिक्नेको", ("s", "i", "k", "n̪", "e", "k", "o"), "verb", "medium"),
    FrontendGoldCase(98, "सोच्थ्यौँ", ("s", "o", "ts", "t̪ʰ", "j", "ʌʊ", "̃"), "verb", "medium"),
    FrontendGoldCase(
        99, "पेलाउँथेनन्", ("p", "e", "l", "a", "u", "̃", "t̪ʰ", "e", "n̪", "ʌ", "n̪"), "verb", "medium"
    ),
    FrontendGoldCase(
        100, "फोक्सोद्वारा", ("pʰ", "o", "k", "s", "o", "d̪", "w", "a", "r", "a"), "noun", "medium"
    ),
    FrontendGoldCase(101, "बिकेपछि", ("b", "i", "k", "e", "p", "ʌ", "tsʰ", "i"), "verb", "medium"),
    FrontendGoldCase(
        102,
        "थमाउनेछैन",
        ("t̪ʰ", "ʌ", "m", "a", "u", "n̪", "e", "tsʰ", "ʌɪ", "n̪", "ʌ"),
        "verb",
        "medium",
    ),
    FrontendGoldCase(
        103, "झरनाहरूले", ("dzʱ", "ʌ", "r", "n̪", "a", "ɦ", "ʌ", "r", "u", "l", "e"), "noun", "medium"
    ),
    FrontendGoldCase(104, "थाकेछैन", ("t̪ʰ", "a", "k", "e", "tsʰ", "ʌɪ", "n̪", "ʌ"), "verb", "medium"),
    FrontendGoldCase(105, "भण्डारका", ("bʱ", "ʌ", "ɳ", "ɖ", "a", "r", "k", "a"), "noun", "medium"),
    FrontendGoldCase(106, "उफ्रिनेछु", ("u", "pʰ", "r", "i", "n̪", "e", "tsʰ", "u"), "verb", "medium"),
    FrontendGoldCase(
        107, "प्राथमिकले", ("p", "r", "a", "t̪ʰ", "ʌ", "m", "i", "k", "l", "e"), "adj", "medium"
    ),
    FrontendGoldCase(
        108,
        "टल्काउनेछैनन्",
        ("ʈ", "ʌ", "l", "k", "a", "u", "n̪", "e", "tsʰ", "ʌɪ", "n̪", "ʌ", "n̪"),
        "verb",
        "medium",
    ),
    FrontendGoldCase(109, "बालकमा", ("b", "a", "l", "ʌ", "k", "m", "a"), "noun", "medium"),
    FrontendGoldCase(
        110, "नझुन्डिएको", ("n̪", "ʌ", "dzʱ", "u", "n̪", "ɖ", "i", "e", "k", "o"), "verb", "medium"
    ),
    FrontendGoldCase(
        111,
        "काँप्नुहुन्थ्यो",
        ("k", "a", "̃", "p", "n̪", "u", "̤", "ː", "n̪", "t̪ʰ", "j", "o"),
        "verb",
        "medium",
    ),
    FrontendGoldCase(
        112,
        "जीवनहरूसँग",
        ("dz", "i", "w", "ʌ", "n̪", "ɦ", "ʌ", "r", "u", "s", "ʌ", "̃", "ɡ", "ʌ"),
        "noun",
        "medium",
    ),
    FrontendGoldCase(
        113,
        "उदाहरणबाहिर",
        ("u", "d̪", "a", "ɦ", "ʌ", "r", "ʌ", "ɳ", "b", "a", "ɦ", "i", "r", "ʌ"),
        "noun",
        "medium",
    ),
    FrontendGoldCase(114, "गर्छिन्", ("ɡ", "ʌ", "r", "tsʰ", "i", "n̪"), "verb", "medium"),
    FrontendGoldCase(115, "कुटिएको", ("k", "u", "ʈ", "i", "e", "k", "o"), "verb", "medium"),
    FrontendGoldCase(
        116, "मिलाउँदैन", ("m", "i", "l", "a", "u", "̃", "d̪", "ʌɪ", "n̪", "ʌ"), "verb", "medium"
    ),
    FrontendGoldCase(
        117,
        "प्रशिक्षणहरूबाट",
        ("p", "r", "ʌ", "s", "i", "k", "tsʰ", "ʌ", "ɳ", "ɦ", "ʌ", "r", "u", "b", "a", "ʈ", "ʌ"),
        "noun",
        "low",
    ),
    FrontendGoldCase(
        118,
        "हवाइजहाजलाई",
        ("ɦ", "ʌ", "w", "a", "i", "dz", "ɦ", "a", "dz", "l", "a", "i"),
        "noun",
        "medium",
    ),
    FrontendGoldCase(119, "साँध्छे", ("s", "a", "̃", "d̪ʱ", "tsʰ", "e"), "verb", "medium"),
    FrontendGoldCase(120, "बङ्ग्याऊ", ("b", "ʌ", "ŋ", "ɡ", "j", "a", "u"), "verb", "medium"),
    FrontendGoldCase(121, "बानेश्वर", ("b", "a", "n̪", "e", "s", "w", "ʌ", "r"), "noun", "medium"),
    FrontendGoldCase(122, "झल्काई", ("dzʱ", "ʌ", "l", "k", "a", "i"), "verb", "medium"),
    FrontendGoldCase(
        123,
        "लक्ष्मीप्रसादबाहिर",
        (
            "l",
            "ʌ",
            "k",
            "tsʰ",
            "m",
            "i",
            "p",
            "r",
            "ʌ",
            "s",
            "a",
            "d̪",
            "b",
            "a",
            "ɦ",
            "i",
            "r",
            "ʌ",
        ),
        "noun",
        "low",
    ),
    FrontendGoldCase(
        124,
        "शनिबारभित्र",
        ("s", "ʌ", "n̪", "i", "b", "a", "r", "bʱ", "i", "t̪", "r", "ʌ"),
        "noun",
        "medium",
    ),
    FrontendGoldCase(
        125,
        "सम्झाउनेछौँ",
        ("s", "ʌ", "m", "dzʱ", "a", "u", "n̪", "e", "tsʰ", "ʌʊ", "̃"),
        "verb",
        "medium",
    ),
    FrontendGoldCase(126, "डुलाऔँ", ("ɖ", "u", "l", "a", "ʌʊ", "̃"), "verb", "medium"),
    FrontendGoldCase(
        127, "पारिन्थेन", ("p", "a", "r", "i", "n̪", "t̪ʰ", "e", "n̪", "ʌ"), "verb", "medium"
    ),
    FrontendGoldCase(
        128,
        "चरिकोटदेखि",
        ("ts", "ʌ", "r", "i", "k", "o", "ʈ", "d̪", "e", "kʰ", "i"),
        "noun",
        "medium",
    ),
    FrontendGoldCase(129, "माघदेखि", ("m", "a", "ɡʱ", "d̪", "e", "kʰ", "i"), "noun", "medium"),
    FrontendGoldCase(130, "क्लिनिकको", ("k", "l", "i", "n̪", "i", "k", "ː", "o"), "noun", "low"),
    FrontendGoldCase(
        131, "चमत्कारमा", ("ts", "ʌ", "m", "ʌ", "t̪", "k", "a", "r", "m", "a"), "noun", "medium"
    ),
    FrontendGoldCase(132, "खेल्थ्यौ", ("kʰ", "e", "l", "t̪ʰ", "j", "ʌʊ"), "verb", "medium"),
    FrontendGoldCase(133, "पछुताइन्", ("p", "ʌ", "tsʰ", "u", "t̪", "a", "i", "n̪"), "verb", "medium"),
    FrontendGoldCase(134, "सुकाएको", ("s", "u", "k", "a", "e", "k", "o"), "verb", "medium"),
    FrontendGoldCase(
        135, "मगाउनेछ", ("m", "ʌ", "ɡ", "a", "u", "n̪", "e", "tsʰ", "ʌ"), "verb", "medium"
    ),
    FrontendGoldCase(
        136,
        "मङ्गलबारहरूबाट",
        ("m", "ʌ", "ŋ", "ɡ", "ʌ", "l", "ʌ", "b", "a", "r", "ɦ", "ʌ", "r", "u", "b", "a", "ʈ", "ʌ"),
        "noun",
        "medium",
    ),
    FrontendGoldCase(
        137,
        "अन्टार्कटिकाकी",
        ("ʌ", "n̪", "ʈ", "a", "r", "k", "ʌ", "ʈ", "i", "k", "a", "k", "i"),
        "noun",
        "medium",
    ),
    FrontendGoldCase(
        138,
        "रत्नपार्ककी",
        ("r", "ʌ", "t̪", "n̪", "ʌ", "p", "a", "r", "k", "ʌ", "k", "i"),
        "noun",
        "medium",
    ),
    FrontendGoldCase(139, "तौलाउँछौ", ("t̪", "ʌʊ", "l", "a", "u", "̃", "tsʰ", "ʌʊ"), "verb", "medium"),
    FrontendGoldCase(
        140,
        "खुम्च्याउँदैनस्",
        ("kʰ", "u", "m", "ts", "j", "a", "u", "̃", "d̪", "ʌɪ", "n̪", "ʌ", "s"),
        "verb",
        "medium",
    ),
    FrontendGoldCase(141, "अड्नेले", ("ʌ", "ɽ", "n̪", "e", "l", "e"), "verb", "medium"),
    FrontendGoldCase(142, "नास्नाले", ("n̪", "a", "s", "n̪", "a", "l", "e"), "verb", "medium"),
    FrontendGoldCase(143, "सीताका", ("s", "i", "t̪", "a", "k", "a"), "noun", "medium"),
    FrontendGoldCase(144, "शय्याको", ("s", "ʌ", "j", "ː", "a", "k", "o"), "noun", "medium"),
    FrontendGoldCase(145, "सुकिएन", ("s", "u", "k", "i", "e", "n̪", "ʌ"), "verb", "medium"),
    FrontendGoldCase(146, "बाँधौँ", ("b", "a", "̃", "d̪ʱ", "ʌʊ", "̃"), "verb", "medium"),
    FrontendGoldCase(147, "लेप्छन्", ("l", "e", "p", "tsʰ", "ʌ", "n̪"), "verb", "medium"),
    FrontendGoldCase(
        148, "पुर्याएनौँ", ("p", "u", "r", "j", "a", "e", "n̪", "ʌʊ", "̃"), "verb", "medium"
    ),
    FrontendGoldCase(
        149, "असोजहरूलाई", ("ʌ", "s", "o", "dz", "ɦ", "ʌ", "r", "u", "l", "a", "i"), "noun", "medium"
    ),
    FrontendGoldCase(150, "ठेलाए", ("ʈʰ", "e", "l", "a", "e"), "verb", "medium"),
    FrontendGoldCase(
        151, "भयअनुसार", ("bʱ", "ʌ", "j", "ʌ", "n̪", "u", "s", "a", "r", "ʌ"), "noun", "medium"
    ),
    FrontendGoldCase(152, "खस्यौँ", ("kʰ", "ʌ", "s", "j", "ʌʊ", "̃"), "verb", "medium"),
    FrontendGoldCase(153, "नसुतेर", ("n̪", "ʌ", "s", "u", "t̪", "e", "r", "ʌ"), "verb", "medium"),
    FrontendGoldCase(154, "बाँचिनन्", ("b", "a", "̃", "ts", "i", "n̪", "ʌ", "n̪"), "verb", "medium"),
    FrontendGoldCase(
        155, "बेल्टसँग", ("b", "e", "l", "ʈ", "ʌ", "s", "ʌ", "̃", "ɡ", "ʌ"), "noun", "low"
    ),
    FrontendGoldCase(
        156, "खुवाउँदिन", ("kʰ", "u", "w", "a", "u", "̃", "d̪", "i", "n̪", "ʌ"), "verb", "medium"
    ),
    FrontendGoldCase(
        157,
        "बेलाउनेहरूको",
        ("b", "e", "l", "a", "u", "n̪", "e", "ɦ", "ʌ", "r", "u", "k", "o"),
        "verb",
        "medium",
    ),
    FrontendGoldCase(
        158,
        "जनताहरूसँग",
        ("dz", "ʌ", "n̪", "t̪", "a", "ɦ", "ʌ", "r", "u", "s", "ʌ", "̃", "ɡ", "ʌ"),
        "noun",
        "medium",
    ),
    FrontendGoldCase(159, "थाक्दैनौ", ("t̪ʰ", "a", "k", "d̪", "ʌɪ", "n̪", "ʌʊ"), "verb", "medium"),
    FrontendGoldCase(160, "पोस्टको", ("p", "o", "s", "ʈ", "ʌ", "k", "o"), "noun", "low"),
    FrontendGoldCase(
        161, "गणतन्त्र", ("ɡ", "ʌ", "ɳ", "ʌ", "t̪", "ʌ", "n̪", "t̪", "r", "ʌ"), "noun", "medium"
    ),
    FrontendGoldCase(
        162, "नबल्झेर", ("n̪", "ʌ", "b", "ʌ", "l", "dzʱ", "e", "r", "ʌ"), "verb", "medium"
    ),
    FrontendGoldCase(163, "नउर्लेर", ("n̪", "ʌ", "u", "r", "l", "e", "r", "ʌ"), "verb", "medium"),
    FrontendGoldCase(164, "पार्कको", ("p", "a", "r", "k", "ʌ", "k", "o"), "noun", "low"),
    FrontendGoldCase(165, "धोक्थ्यौँ", ("d̪ʱ", "o", "k", "t̪ʰ", "j", "ʌʊ", "̃"), "verb", "medium"),
    FrontendGoldCase(
        166, "टल्क्नेछिन्", ("ʈ", "ʌ", "l", "k", "n̪", "e", "tsʰ", "i", "n̪"), "verb", "medium"
    ),
    FrontendGoldCase(167, "सुझ्यौँ", ("s", "u", "dzʱ", "j", "ʌʊ", "̃"), "verb", "medium"),
    FrontendGoldCase(
        168, "सँभाल्नेछिन्", ("s", "ʌ", "̃", "bʱ", "a", "l", "n̪", "e", "tsʰ", "i", "n̪"), "verb", "medium"
    ),
    FrontendGoldCase(169, "कैँची", ("k", "ʌɪ", "̃", "ts", "i"), "noun", "medium"),
    FrontendGoldCase(
        170, "छेकाउनेहरू", ("tsʰ", "e", "k", "a", "u", "n̪", "e", "ɦ", "ʌ", "r", "u"), "verb", "medium"
    ),
    FrontendGoldCase(
        171,
        "सेनिटाइजरका",
        ("s", "e", "n̪", "i", "ʈ", "a", "i", "dz", "ʌ", "r", "k", "a"),
        "noun",
        "low",
    ),
    FrontendGoldCase(172, "बोक्दै", ("b", "o", "k", "d̪", "ʌɪ"), "verb", "medium"),
    FrontendGoldCase(
        173, "टिभीहरूलाई", ("ʈ", "i", "bʱ", "i", "ɦ", "ʌ", "r", "u", "l", "a", "i"), "noun", "low"
    ),
    FrontendGoldCase(
        174, "ब्युँझाइनन्", ("b", "j", "u", "̃", "dzʱ", "a", "i", "n̪", "ʌ", "n̪"), "verb", "medium"
    ),
    FrontendGoldCase(
        175,
        "स्वास्थ्यचौकीहरूबाट",
        (
            "s",
            "w",
            "a",
            "s",
            "t̪ʰ",
            "j",
            "ʌ",
            "ts",
            "ʌʊ",
            "k",
            "i",
            "ɦ",
            "ʌ",
            "r",
            "u",
            "b",
            "a",
            "ʈ",
            "ʌ",
        ),
        "noun",
        "medium",
    ),
    FrontendGoldCase(176, "प्रहरीले", ("p", "r", "a", "̤", "ː", "r", "i", "l", "e"), "noun", "medium"),
    FrontendGoldCase(
        177,
        "टोलीहरूबाट",
        ("ʈ", "o", "l", "i", "ɦ", "ʌ", "r", "u", "b", "a", "ʈ", "ʌ"),
        "noun",
        "medium",
    ),
    FrontendGoldCase(
        178, "कन्सर्टकी", ("k", "ʌ", "n̪", "s", "ʌ", "r", "ʈ", "ʌ", "k", "i"), "noun", "low"
    ),
    FrontendGoldCase(179, "गाजलले", ("ɡ", "a", "dz", "ʌ", "l", "ː", "e"), "noun", "medium"),
    FrontendGoldCase(180, "पुकारे", ("p", "u", "k", "a", "r", "e"), "verb", "medium"),
    FrontendGoldCase(
        181, "सिपाहीबाट", ("s", "i", "p", "a", "ɦ", "i", "b", "a", "ʈ", "ʌ"), "noun", "medium"
    ),
    FrontendGoldCase(
        182, "बचाउनाले", ("b", "ʌ", "ts", "a", "u", "n̪", "a", "l", "e"), "verb", "medium"
    ),
    FrontendGoldCase(
        183, "हटाउँथेन", ("ɦ", "ʌ", "ʈ", "a", "u", "̃", "t̪ʰ", "e", "n̪", "ʌ"), "verb", "medium"
    ),
    FrontendGoldCase(
        184, "कपालहरूको", ("k", "ʌ", "p", "a", "l", "ɦ", "ʌ", "r", "u", "k", "o"), "noun", "medium"
    ),
    FrontendGoldCase(185, "थियौँ", ("t̪ʰ", "i", "j", "ʌʊ", "̃"), "verb", "high"),
    FrontendGoldCase(
        186, "होच्याउनेले", ("ɦ", "o", "ts", "j", "a", "u", "n̪", "e", "l", "e"), "verb", "medium"
    ),
    FrontendGoldCase(
        187, "सहिष्णुताकी", ("s", "ʌ", "ɦ", "i", "s", "ɳ", "u", "t̪", "a", "k", "i"), "noun", "medium"
    ),
    FrontendGoldCase(188, "दुखाएपछि", ("d̪", "u", "kʰ", "a", "e", "p", "tsʰ", "i"), "verb", "medium"),
    FrontendGoldCase(
        189, "जर्मनीदेखि", ("dz", "ʌ", "r", "m", "ʌ", "n̪", "i", "d̪", "e", "kʰ", "i"), "noun", "medium"
    ),
    FrontendGoldCase(
        190, "बाँधहरूका", ("b", "a", "̃", "d̪ʱ", "ɦ", "ʌ", "r", "u", "k", "a"), "noun", "medium"
    ),
    FrontendGoldCase(
        191, "सोचिँदैनौँ", ("s", "o", "ts", "i", "̃", "d̪", "ʌɪ", "n̪", "ʌʊ", "̃"), "verb", "medium"
    ),
    FrontendGoldCase(
        192, "पत्याउँथेन", ("p", "ʌ", "t̪", "j", "a", "u", "̃", "t̪ʰ", "e", "n̪", "ʌ"), "verb", "medium"
    ),
    FrontendGoldCase(193, "तताउनु", ("t̪", "ʌ", "t̪", "a", "u", "n̪", "u"), "verb", "medium"),
    FrontendGoldCase(194, "फिट्थिनँ", ("pʰ", "i", "ʈ", "t̪ʰ", "i", "n̪", "ʌ", "̃"), "verb", "medium"),
    FrontendGoldCase(
        195, "झुलाउनेछौँ", ("dzʱ", "u", "l", "a", "u", "n̪", "e", "tsʰ", "ʌʊ", "̃"), "verb", "medium"
    ),
    FrontendGoldCase(
        196, "आयातहरूले", ("a", "j", "a", "t̪", "ɦ", "ʌ", "r", "u", "l", "e"), "noun", "medium"
    ),
    FrontendGoldCase(
        197, "पत्याउँछस्", ("p", "ʌ", "t̪", "j", "a", "u", "̃", "tsʰ", "ʌ", "s"), "verb", "medium"
    ),
    FrontendGoldCase(198, "ऐनाभित्र", ("ʌɪ", "n̪", "a", "bʱ", "i", "t̪", "r", "ʌ"), "noun", "medium"),
    FrontendGoldCase(199, "पालाऊ", ("p", "a", "l", "a", "u"), "verb", "medium"),
    FrontendGoldCase(
        200, "बस्नेछैन", ("b", "ʌ", "s", "n̪", "e", "tsʰ", "ʌɪ", "n̪", "ʌ"), "verb", "medium"
    ),
    FrontendGoldCase(201, "जर्नलका", ("dz", "ʌ", "r", "n̪", "ʌ", "l", "k", "a"), "noun", "medium"),
    FrontendGoldCase(
        202,
        "मतपत्रतिर",
        ("m", "ʌ", "t̪", "ʌ", "p", "ʌ", "t̪", "r", "ʌ", "t̪", "i", "r", "ʌ"),
        "noun",
        "medium",
    ),
    FrontendGoldCase(
        203,
        "जमाउनेछैन",
        ("dz", "ʌ", "m", "a", "u", "n̪", "e", "tsʰ", "ʌɪ", "n̪", "ʌ"),
        "verb",
        "medium",
    ),
    FrontendGoldCase(
        204, "रोपाउँथेनन्", ("r", "o", "p", "a", "u", "̃", "t̪ʰ", "e", "n̪", "ʌ", "n̪"), "verb", "medium"
    ),
    FrontendGoldCase(
        205,
        "ज्ञानहरूबाट",
        ("ɡ", "j", "a", "n̪", "ɦ", "ʌ", "r", "u", "b", "a", "ʈ", "ʌ"),
        "noun",
        "low",
    ),
    FrontendGoldCase(
        206, "तिलोत्तमाकी", ("t̪", "i", "l", "o", "t̪", "ː", "ʌ", "m", "a", "k", "i"), "noun", "medium"
    ),
    FrontendGoldCase(207, "छस्", ("tsʰ", "ʌ", "s"), "verb", "medium"),
    FrontendGoldCase(208, "भज्थिनँ", ("bʱ", "ʌ", "dz", "t̪ʰ", "i", "n̪", "ʌ", "̃"), "verb", "medium"),
    FrontendGoldCase(
        209,
        "सोझ्याउनुहुन्न",
        ("s", "o", "dzʱ", "j", "a", "u", "n̪", "u", "̤", "ː", "n̪", "ː", "ʌ"),
        "verb",
        "medium",
    ),
    FrontendGoldCase(210, "सम्हाल्छे", ("s", "ʌ", "m", "ɦ", "a", "l", "tsʰ", "e"), "verb", "medium"),
    FrontendGoldCase(211, "थाल्ने", ("t̪ʰ", "a", "l", "n̪", "e"), "verb", "medium"),
    FrontendGoldCase(212, "पोत्छु", ("p", "o", "t̪", "tsʰ", "u"), "verb", "medium"),
    FrontendGoldCase(213, "अछामबाट", ("ʌ", "tsʰ", "a", "m", "b", "a", "ʈ", "ʌ"), "noun", "medium"),
    FrontendGoldCase(
        214,
        "हस्तान्तरणअनुसार",
        ("ɦ", "ʌ", "s", "t̪", "a", "n̪", "t̪", "ʌ", "r", "ʌ", "ɳ", "ʌ", "n̪", "u", "s", "a", "r", "ʌ"),
        "noun",
        "medium",
    ),
    FrontendGoldCase(
        215, "खोसाउँदैनौँ", ("kʰ", "o", "s", "a", "u", "̃", "d̪", "ʌɪ", "n̪", "ʌʊ", "̃"), "verb", "medium"
    ),
    FrontendGoldCase(216, "नाचेझैँ", ("n̪", "a", "ts", "e", "dzʱ", "ʌɪ", "̃"), "verb", "medium"),
    FrontendGoldCase(217, "नापेपछि", ("n̪", "a", "p", "e", "p", "ʌ", "tsʰ", "i"), "verb", "medium"),
    FrontendGoldCase(218, "पसारोस्", ("p", "ʌ", "s", "a", "r", "o", "s"), "verb", "medium"),
    FrontendGoldCase(
        219,
        "स्विट्जरल्यान्डदेखि",
        (
            "s",
            "b",
            "i",
            "ʈ",
            "dz",
            "ʌ",
            "r",
            "ʌ",
            "l",
            "j",
            "a",
            "n̪",
            "ɖ",
            "ʌ",
            "d̪",
            "e",
            "kʰ",
            "i",
        ),
        "noun",
        "medium",
    ),
    FrontendGoldCase(
        220, "बोलिनुहुन्न", ("b", "o", "l", "i", "n̪", "u", "̤", "ː", "n̪", "ː", "ʌ"), "verb", "medium"
    ),
    FrontendGoldCase(221, "रोका", ("r", "o", "k", "a"), "noun", "medium"),
    FrontendGoldCase(
        222,
        "वर्तमानहरूमा",
        ("w", "ʌ", "r", "t̪", "ʌ", "m", "a", "n̪", "ɦ", "ʌ", "r", "u", "m", "a"),
        "noun",
        "low",
    ),
    FrontendGoldCase(223, "विश्वासमा", ("b", "i", "s", "w", "a", "s", "m", "a"), "noun", "medium"),
    FrontendGoldCase(
        224, "गणेशहिमाल", ("ɡ", "ʌ", "ɳ", "e", "s", "ɦ", "i", "m", "a", "l"), "noun", "medium"
    ),
    FrontendGoldCase(
        225, "दाबहरूबाट", ("d̪", "a", "b", "ɦ", "ʌ", "r", "u", "b", "a", "ʈ", "ʌ"), "noun", "medium"
    ),
    FrontendGoldCase(
        226, "पगलाउँथ्यौँ", ("p", "ʌ", "ɡ", "l", "a", "u", "̃", "t̪ʰ", "j", "ʌʊ", "̃"), "verb", "medium"
    ),
    FrontendGoldCase(
        227, "उफ्रिनेहरू", ("u", "pʰ", "r", "i", "n̪", "e", "ɦ", "ʌ", "r", "u"), "verb", "medium"
    ),
    FrontendGoldCase(228, "झुलाउनु", ("dzʱ", "u", "l", "a", "u", "n̪", "u"), "verb", "medium"),
    FrontendGoldCase(229, "धपाउँदा", ("d̪ʱ", "ʌ", "p", "a", "u", "̃", "d̪", "a"), "verb", "medium"),
    FrontendGoldCase(
        230, "बिराउनेले", ("b", "i", "r", "a", "u", "n̪", "e", "l", "e"), "verb", "medium"
    ),
    FrontendGoldCase(231, "पोल्दिनन्", ("p", "o", "l", "d̪", "i", "n̪", "ʌ", "n̪"), "verb", "medium"),
    FrontendGoldCase(
        232,
        "प्राथमिकताको",
        ("p", "r", "a", "t̪ʰ", "ʌ", "m", "i", "k", "t̪", "a", "k", "o"),
        "noun",
        "medium",
    ),
    FrontendGoldCase(
        233, "पगलाउँदैन", ("p", "ʌ", "ɡ", "l", "a", "u", "̃", "d̪", "ʌɪ", "n̪", "ʌ"), "verb", "medium"
    ),
    FrontendGoldCase(
        234, "पन्युबाट", ("p", "ʌ", "n̪", "j", "u", "b", "a", "ʈ", "ʌ"), "noun", "medium"
    ),
    FrontendGoldCase(
        235, "चिर्नेछैनन्", ("ts", "i", "r", "n̪", "e", "tsʰ", "ʌɪ", "n̪", "ʌ", "n̪"), "verb", "medium"
    ),
    FrontendGoldCase(
        236, "अपडेटसँग", ("ʌ", "p", "ɖ", "e", "ʈ", "s", "ʌ", "̃", "ɡ", "ʌ"), "noun", "low"
    ),
    FrontendGoldCase(237, "दाजुदेखि", ("d̪", "a", "dz", "u", "d̪", "e", "kʰ", "i"), "noun", "medium"),
    FrontendGoldCase(238, "टेकोस्", ("ʈ", "e", "k", "o", "s"), "verb", "medium"),
    FrontendGoldCase(
        239, "रूपन्देहीले", ("r", "u", "p", "ʌ", "n̪", "d̪", "e", "ɦ", "i", "l", "e"), "noun", "medium"
    ),
    FrontendGoldCase(240, "ध्मेयतिर", ("d̪ʱ", "m", "e", "j", "t̪", "i", "r", "ʌ"), "noun", "medium"),
    FrontendGoldCase(
        241, "बुहारीद्वारा", ("b", "u", "ɦ", "a", "r", "i", "d̪", "w", "a", "r", "a"), "noun", "medium"
    ),
    FrontendGoldCase(
        242, "बानीहरूलाई", ("b", "a", "n̪", "i", "ɦ", "ʌ", "r", "u", "l", "a", "i"), "noun", "medium"
    ),
    FrontendGoldCase(243, "भुलौँ", ("bʱ", "u", "l", "ʌʊ", "̃"), "verb", "medium"),
    FrontendGoldCase(
        244,
        "बटाराउनेलाई",
        ("b", "ʌ", "ʈ", "a", "r", "a", "u", "n̪", "e", "l", "a", "i"),
        "verb",
        "medium",
    ),
    FrontendGoldCase(
        245, "पारिनुहुन्छ", ("p", "a", "r", "i", "n̪", "u", "̤", "ː", "n̪", "tsʰ", "ʌ"), "verb", "medium"
    ),
    FrontendGoldCase(246, "विभागको", ("b", "i", "bʱ", "a", "ɡ", "k", "o"), "noun", "medium"),
    FrontendGoldCase(247, "हेल्मेटमा", ("ɦ", "e", "l", "m", "e", "ʈ", "m", "a"), "noun", "low"),
    FrontendGoldCase(
        248, "डाडुभित्र", ("ɖ", "a", "ɽ", "u", "bʱ", "i", "t̪", "r", "ʌ"), "noun", "medium"
    ),
    FrontendGoldCase(249, "छोडिइन्", ("tsʰ", "o", "ɽ", "i", "i", "n̪"), "verb", "medium"),
    FrontendGoldCase(250, "भन्सारको", ("bʱ", "ʌ", "n̪", "s", "a", "r", "k", "o"), "noun", "medium"),
    FrontendGoldCase(
        251, "ल्होसारलाई", ("l", "ɦ", "o", "s", "a", "r", "l", "a", "i"), "noun", "medium"
    ),
    FrontendGoldCase(
        252, "आँपहरूलाई", ("a", "̃", "p", "ɦ", "ʌ", "r", "u", "l", "a", "i"), "noun", "medium"
    ),
    FrontendGoldCase(253, "ओर्ल्नू", ("o", "r", "l", "n̪", "u"), "verb", "medium"),
    FrontendGoldCase(254, "रङ्गाएँ", ("r", "ʌ", "ŋ", "ɡ", "a", "e", "̃"), "verb", "medium"),
    FrontendGoldCase(255, "साँचौँ", ("s", "a", "̃", "ts", "ʌʊ", "̃"), "verb", "medium"),
    FrontendGoldCase(
        256, "छुटाएपछि", ("tsʰ", "u", "ʈ", "a", "e", "p", "tsʰ", "i"), "verb", "medium"
    ),
    FrontendGoldCase(257, "किनाऊँ", ("k", "i", "n̪", "a", "u", "̃"), "verb", "medium"),
    FrontendGoldCase(258, "बाँडेर", ("b", "a", "̃", "ɽ", "e", "r", "ʌ"), "verb", "medium"),
    FrontendGoldCase(259, "गनिनुको", ("ɡ", "ʌ", "n̪", "i", "n̪", "u", "k", "o"), "verb", "medium"),
    FrontendGoldCase(
        260,
        "सामाखुसीसम्म",
        ("s", "a", "m", "a", "kʰ", "u", "s", "i", "s", "ʌ", "m", "ː", "ʌ"),
        "noun",
        "medium",
    ),
    FrontendGoldCase(261, "खेलिँदा", ("kʰ", "e", "l", "i", "̃", "d̪", "a"), "verb", "medium"),
    FrontendGoldCase(262, "गल्नेको", ("ɡ", "ʌ", "l", "n̪", "e", "k", "o"), "verb", "medium"),
    FrontendGoldCase(263, "बेचेछौ", ("b", "e", "ts", "e", "tsʰ", "ʌʊ"), "verb", "medium"),
    FrontendGoldCase(264, "पसे", ("p", "ʌ", "s", "e"), "verb", "medium"),
    FrontendGoldCase(
        265, "पढाउँथ्यो", ("p", "ʌ", "ɽʱ", "a", "u", "̃", "t̪ʰ", "j", "o"), "verb", "medium"
    ),
    FrontendGoldCase(
        266, "ट्विटरहरूले", ("ʈ", "b", "i", "ʈ", "ʌ", "r", "ɦ", "ʌ", "r", "u", "l", "e"), "noun", "low"
    ),
    FrontendGoldCase(
        267, "रुघासम्म", ("r", "u", "ɡʱ", "a", "s", "ʌ", "m", "ː", "ʌ"), "noun", "medium"
    ),
    FrontendGoldCase(268, "फुटाओस्", ("pʰ", "u", "ʈ", "a", "o", "s"), "verb", "medium"),
    FrontendGoldCase(269, "कीरालाई", ("k", "i", "r", "a", "l", "a", "i"), "noun", "medium"),
    FrontendGoldCase(270, "लेखाएछन्", ("l", "e", "kʰ", "a", "e", "tsʰ", "ʌ", "n̪"), "verb", "medium"),
    FrontendGoldCase(
        271, "टिप्नुभयो", ("ʈ", "i", "p", "n̪", "u", "bʱ", "ʌ", "j", "o"), "verb", "medium"
    ),
    FrontendGoldCase(272, "टोलकी", ("ʈ", "o", "l", "k", "i"), "noun", "medium"),
    FrontendGoldCase(
        273, "निस्केपछि", ("n̪", "i", "s", "k", "e", "p", "ʌ", "tsʰ", "i"), "verb", "medium"
    ),
    FrontendGoldCase(274, "भिड्नेछिन्", ("bʱ", "i", "ɽ", "n̪", "e", "tsʰ", "i", "n̪"), "verb", "medium"),
    FrontendGoldCase(275, "थाक्यौँ", ("t̪ʰ", "a", "k", "j", "ʌʊ", "̃"), "verb", "medium"),
    FrontendGoldCase(
        276, "खँगालेनन्", ("kʰ", "ʌ", "̃", "ɡ", "a", "l", "e", "n̪", "ʌ", "n̪"), "verb", "medium"
    ),
    FrontendGoldCase(
        277, "सामान्यमा", ("s", "a", "m", "a", "n̪", "j", "ʌ", "m", "a"), "adj", "medium"
    ),
    FrontendGoldCase(
        278, "लुटाउँदिन", ("l", "u", "ʈ", "a", "u", "̃", "d̪", "i", "n̪", "ʌ"), "verb", "medium"
    ),
    FrontendGoldCase(279, "दबाइनँ", ("d̪", "ʌ", "b", "a", "i", "n̪", "ʌ", "̃"), "verb", "medium"),
    FrontendGoldCase(
        280, "रुपन्देहीमा", ("r", "u", "p", "ʌ", "n̪", "d̪", "e", "ɦ", "i", "m", "a"), "noun", "medium"
    ),
    FrontendGoldCase(281, "पाएछिन्", ("p", "a", "e", "tsʰ", "i", "n̪"), "verb", "medium"),
    FrontendGoldCase(
        282,
        "इमान्दारीसँग",
        ("i", "m", "a", "n̪", "d̪", "a", "r", "i", "s", "ʌ", "̃", "ɡ", "ʌ"),
        "noun",
        "medium",
    ),
    FrontendGoldCase(283, "रविका", ("r", "ʌ", "b", "i", "k", "a"), "noun", "medium"),
    FrontendGoldCase(284, "टिकटको", ("ʈ", "i", "k", "ʌ", "ʈ", "k", "o"), "noun", "low"),
    FrontendGoldCase(
        285, "पदार्थलाई", ("p", "ʌ", "d̪", "a", "r", "t̪ʰ", "ʌ", "l", "a", "i"), "noun", "medium"
    ),
    FrontendGoldCase(286, "फुटिएको", ("pʰ", "u", "ʈ", "i", "e", "k", "o"), "verb", "medium"),
    FrontendGoldCase(
        287,
        "सख्खरहरूबाट",
        ("s", "ʌ", "kʰ", "ː", "ʌ", "r", "ɦ", "ʌ", "r", "u", "b", "a", "ʈ", "ʌ"),
        "noun",
        "medium",
    ),
    FrontendGoldCase(
        288, "परीक्षणले", ("p", "ʌ", "r", "i", "k", "tsʰ", "ʌ", "ɳ", "l", "e"), "noun", "low"
    ),
    FrontendGoldCase(
        289,
        "उपमहानगरपालिकाहरूका",
        (
            "u",
            "p",
            "ʌ",
            "m",
            "ɦ",
            "a",
            "n̪",
            "ʌ",
            "ɡ",
            "ʌ",
            "r",
            "p",
            "a",
            "l",
            "i",
            "k",
            "a",
            "ɦ",
            "ʌ",
            "r",
            "u",
            "k",
            "a",
        ),
        "noun",
        "medium",
    ),
    FrontendGoldCase(
        290, "कक्षाबाट", ("k", "ʌ", "k", "tsʰ", "a", "b", "a", "ʈ", "ʌ"), "noun", "low"
    ),
    FrontendGoldCase(291, "नपाऊ", ("n̪", "ʌ", "p", "a", "u"), "verb", "medium"),
    FrontendGoldCase(292, "फुल्न", ("pʰ", "u", "l", "n̪", "ʌ"), "verb", "medium"),
    FrontendGoldCase(293, "लुटेन", ("l", "u", "ʈ", "e", "n̪", "ʌ"), "verb", "medium"),
    FrontendGoldCase(
        294, "बाँधिनाले", ("b", "a", "̃", "d̪ʱ", "i", "n̪", "a", "l", "e"), "verb", "medium"
    ),
    FrontendGoldCase(295, "पाक्नुमा", ("p", "a", "k", "n̪", "u", "m", "a"), "verb", "medium"),
    FrontendGoldCase(296, "फाटिएछ", ("pʰ", "a", "ʈ", "i", "e", "tsʰ", "ʌ"), "verb", "medium"),
    FrontendGoldCase(297, "दुल्नुको", ("d̪", "u", "l", "n̪", "u", "k", "o"), "verb", "medium"),
    FrontendGoldCase(298, "गाग्रोमा", ("ɡ", "a", "ɡ", "r", "o", "m", "a"), "noun", "medium"),
    FrontendGoldCase(
        299, "दायित्वका", ("d̪", "a", "j", "i", "t̪", "w", "ʌ", "k", "a"), "noun", "medium"
    ),
    FrontendGoldCase(
        300,
        "फुपाजुद्वारा",
        ("pʰ", "u", "p", "a", "dz", "u", "d̪", "w", "a", "r", "a"),
        "noun",
        "medium",
    ),
)
