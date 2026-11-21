"""Unit tests for the Nepali G2P frontend and 300-word native gold validation."""

from __future__ import annotations

from pawp_tts.domain.evaluation.frontend_validation import GOLD_300_CASES
from pawp_tts.infrastructure.frontend.nepali_adapter import NepaliRuleG2PAdapter


def test_gold_300_sample_accuracy() -> None:
    adapter = NepaliRuleG2PAdapter()
    mismatches: list[str] = []
    for case in GOLD_300_CASES:
        ipa = adapter.phonemize(case.word)
        if ipa != case.expected_ipa:
            mismatches.append(
                f"{case.word}: expected {' '.join(case.expected_ipa)} but got {' '.join(ipa)}"
            )

    assert not mismatches, f"Mismatches in 300 gold sample ({len(mismatches)}):\n" + "\n".join(
        mismatches[:10]
    )


def test_heteronym_context_resolution_infinitive_vs_noun() -> None:
    adapter = NepaliRuleG2PAdapter()

    # दिन (noun /d̪ i n̪/ vs infinitive verb /d̪ i n̪ ʌ/)
    din_noun = adapter.phonemize("आज राम्रो दिन छ")
    assert ("d̪", "i", "n̪") in [tuple(din_noun[i : i + 3]) for i in range(len(din_noun) - 2)]
    assert ("d̪", "i", "n̪", "ʌ") not in [
        tuple(din_noun[i : i + 4]) for i in range(len(din_noun) - 3)
    ]

    # Before modal verb "चाहन्छु", दिन should retain final schwa as an infinitive
    din_verb = adapter.phonemize("म किताब दिन चाहन्छु")
    assert ("d̪", "i", "n̪", "ʌ") in [tuple(din_verb[i : i + 4]) for i in range(len(din_verb) - 3)]

    # लिन (infinitive verb /l i n̪ ʌ/)
    lin_verb = adapter.phonemize("तरकारी लिन जानुपर्छ")
    assert ("l", "i", "n̪", "ʌ") in [tuple(lin_verb[i : i + 4]) for i in range(len(lin_verb) - 3)]

    # खान (noun /kʰ a n̪/ vs infinitive verb /kʰ a n̪ ʌ/)
    khan_noun = adapter.phonemize("त्यो कोइलाको खान हो")
    khan_verb = adapter.phonemize("दालभात खान मन लाग्यो")
    assert ("kʰ", "a", "n̪") in [tuple(khan_noun[i : i + 3]) for i in range(len(khan_noun) - 2)]
    assert ("kʰ", "a", "n̪", "ʌ") in [tuple(khan_verb[i : i + 4]) for i in range(len(khan_verb) - 3)]
