# Nepali Phonology Reference for the G2P Frontend

Every rule in `src/pawp_tts/domain/phonology/nepali_rules.py` should trace to an entry here.
**Status** marks how each rule is supported:

- **primary:** taken from the primary text;
- **secondary:** taken from a secondary description that cites the primary literature (primary text not yet checked);
- **to-verify:** not yet confirmed against native recordings.

Variant choices follow standard Kathmandu Nepali.

## Implemented rules

| Rule | Examples | Status | Context needed |
|---|---|---|---|
| No phonemic vowel length: इ/ई → /i/, उ/ऊ → /u/ | — | secondary [1,2] | no |
| Inherent vowel is /ʌ/ (not Hindi /ə/) | — | secondary [1,2] | no |
| Alveolar affricates च छ ज झ → /ts tsʰ dz dzʱ/ (not Hindi /tʃ dʒ/) | चिया /tsija/, छ /tsʰʌ/ | secondary [1,2] | no |
| Sibilant merger स श ष → /s/ | शहर /sʌɦʌr/ | secondary [2] | no |
| Contrastive gemination | पाता /pat̪a/ vs पत्ता /pʌt̪ːa/ | secondary [2] | no |
| Intervocalic /ɦ/ deletion with breathy long vowel | पहाड /pʌɦaɖ/ → [pa̤ːɽ] | secondary [2] | no |
| Post-vocalic flapping of ड ढ → [ɽ ɽʱ] | सडक [sʌɽʌk] | secondary [2] | no |
| Final /ʌ/ kept after a final conjunct | मित्र /mit̪rʌ/, अन्त /ʌnt̪ʌ/ | secondary [2] | no |
| Final /ʌ/ kept on finite verbs (deleted when a halanta is written) | हुन्छ /ɦuntsʰʌ/, गएन /ɡʌenʌ/ vs छन् /tsʰʌn/ | secondary [2] | suffix |
| Final /ʌ/ kept on adverbs and postpositions | अब /ʌbʌ/, तिर /t̪irʌ/, जब, तब | secondary [2] | closed list |
| Lexical exceptions keeping final /ʌ/ | दुख /dukʰʌ/, सुख /sukʰʌ/ | secondary [2] | lexicon |
| Infinitive -न keeps final /ʌ/ before modal/auxiliary verbs | किताब दिन सक्छ → /dinʌ/ | secondary [3] | next word |
| In "*inf* मन लाग्नु", the modal noun (मन, इच्छा, रहर…) is a noun, and the preceding word is the infinitive | पैसा दिन मन लाग्यो → /dinʌ mʌn/ | secondary [3] | two-word lookahead |
| Noun reading after numerals, determiners and genitives | राम्रो दिन /din/, तीन दिन /din/ | grammar | previous word |
| व → /b/ before e, i, o, ai and before र य व | विद्यालय /bid̪jaljʌ/, व्यक्ति /bjʌkt̪i/ | secondary [2] | no |

## Candidate rules (not yet enabled)

| Rule | Example | Why not yet |
|---|---|---|
| Flapping of ट ठ after vowels | पेट [peɽ] | Optional variant [2]; to-verify on native recordings |
| Weakening of च/ज → [s]/[z] between vowels | आजै [azʱʌi̯] | Optional, gradient [2] |
| Spirantisation of aspirated stops (/pʰ/ → [ɸ]) | सफा [sʌɸa] | Optional, gradient [2] |
| Medial schwa syncope beyond the current heuristic | — | Not covered by the sources accessed |
| Realisation of ज्ञ, क्ष, त्र, श्र, ऋ | — | Current choices unverified; check against native recordings |

## Evaluation & Validation Suites

- **Unit Regression Tests:** `tests/unit/test_nepali_phonology.py` holds regression tests for every implemented phonological rule (sibilant merger, alveolar affricates, 4-way laryngeal stops, retroflex flapping, gemination, breathy lengthening, and clitic code-mixing).
- **Gold Native Evaluation (`make frontend-validate`):** `domain/evaluation/frontend_validation.py` and `tests/unit/test_frontend_validation.py` hold a 300-word native Kathmandu gold evaluation suite (covering core closed-class, inflected verbs, nouns, and gazetteer terms), validated at **100.0% accuracy** (300/300).
- **Pronunciation Lexicon (`ne_lexicon_v1`):** Integrated 105,868-word permissive Kathmandu lexicon (`data/lexicon/ne_lexicon_v1/`) with 100% symbol compliance with our unified IPA inventory (0 missing tokens across 936,987 symbols).
- **Context Heteronym Disambiguation:** Integrated 152 context-dependent readings across 76 homographs (e.g., verbal infinitive in `-न` like `दिन/लिन/खान/जान` vs nominal readings) mediated through domain syntactic lookahead.
- **Planned:**
  - native-recording acoustic cross-check using the training-only CTC phoneme recognizer.

## Sources

1. Khatiwada, R. (2009). *Nepali*. Journal of the International Phonetic Association 39(3), 373–380. doi:10.1017/S0025100309990181. [link](https://www.cambridge.org/core/journals/journal-of-the-international-phonetic-association/article/nepali/A887820E3B1293EFD43FBD3259D41F73)
2. *Nepali phonology*, Wikipedia (secondary summary citing Khatiwada 2009, Pokharel 1989, Acharya 1991, Bandhu et al. 1971). [link](https://en.wikipedia.org/wiki/Nepali_phonology)
3. Nepali grammar references on the infinitive in -न with potential/desiderative constructions, e.g. [BCcampus Nepali textbook, ch. 13](https://pressbooks.bccampus.ca/nepali/chapter/chapter-13-unit-4-grammar-focus/) and [nepalgo: infinitive form of verbs](https://nepalgo.de/grammar/infinitive-form-of-verbs-in-nepali/).
4. Method references (not phonology): Arora, Gessler & Schneider (2020), supervised schwa deletion, [arXiv:2004.10353](https://arxiv.org/abs/2004.10353); Johny & Jansche (2018), Brahmic schwa deletion, [SLTU 2018](https://www.isca-archive.org/sltu_2018/johny18_sltu.pdf).
5. Other open Nepali frontends were consulted for design comparison (e.g. [nepa-newa-text-frontend](https://github.com/Ampixa/nepa-newa-text-frontend)); no code or data was reused.
