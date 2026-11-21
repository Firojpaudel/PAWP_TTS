"""NepTTS-Bench 77 minimal-pair bilingual phonological benchmark suite."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MinimalPairCase:
    """Single contrastive or sibilant-merger pair in NepTTS-Bench."""

    pair_id: str
    category: str
    word_a: str
    word_b: str
    contrast_description: str
    expect_distinct_phonemes: bool = True


# 77 minimal pairs (11 per category across 7 bilingual phonological dimensions)
NEPTTS_MINIMAL_PAIRS: tuple[MinimalPairCase, ...] = (
    # 1. Dental vs. Retroflex (11 pairs)
    MinimalPairCase("dr_01", "dental_vs_retroflex", "तामा", "टामा", "/t̪/ vs /ʈ/ onset"),
    MinimalPairCase("dr_02", "dental_vs_retroflex", "ताल", "टाल", "/t̪/ vs /ʈ/ onset"),
    MinimalPairCase("dr_03", "dental_vs_retroflex", "थाल", "ठाल", "/t̪ʰ/ vs /ʈʰ/ aspirated onset"),
    MinimalPairCase("dr_04", "dental_vs_retroflex", "दाल", "डाल", "/d̪/ vs /ɖ/ voiced onset"),
    MinimalPairCase("dr_05", "dental_vs_retroflex", "धान", "ढान", "/d̪ʱ/ vs /ɖʱ/ breathy onset"),
    MinimalPairCase("dr_06", "dental_vs_retroflex", "तीन", "टिन", "/t̪/ vs /ʈ/ + high vowel"),
    MinimalPairCase("dr_07", "dental_vs_retroflex", "सात", "साट", "/t̪/ vs /ʈ/ coda contrast"),
    MinimalPairCase("dr_08", "dental_vs_retroflex", "पाती", "पाटी", "medial dental vs retroflex"),
    MinimalPairCase("dr_09", "dental_vs_retroflex", "कोत", "कोट", "final dental vs retroflex stop"),
    MinimalPairCase("dr_10", "dental_vs_retroflex", "बात", "बाट", "final /t̪/ vs /ʈ/"),
    MinimalPairCase("dr_11", "dental_vs_retroflex", "कात", "काट", "final /t̪/ vs /ʈ/"),
    # 2. Four-way Laryngeal Stop Contrast (11 pairs)
    MinimalPairCase("la_01", "four_way_laryngeal", "काल", "खाल", "voiceless /k/ vs aspirated /kʰ/"),
    MinimalPairCase("la_02", "four_way_laryngeal", "गाल", "घाम", "voiced /ɡ/ vs breathy /ɡʱ/"),
    MinimalPairCase("la_03", "four_way_laryngeal", "पल", "फल", "bilabial /p/ vs /pʰ/"),
    MinimalPairCase("la_04", "four_way_laryngeal", "बल", "भल", "voiced /b/ vs breathy /bʱ/"),
    MinimalPairCase("la_05", "four_way_laryngeal", "तान", "थान", "dental /t̪/ vs aspirated /t̪ʰ/"),
    MinimalPairCase("la_06", "four_way_laryngeal", "दान", "धान", "dental /d̪/ vs breathy /d̪ʱ/"),
    MinimalPairCase("la_07", "four_way_laryngeal", "टाल", "ठाल", "retroflex /ʈ/ vs /ʈʰ/"),
    MinimalPairCase("la_08", "four_way_laryngeal", "डाल", "ढाल", "retroflex /ɖ/ vs /ɖʱ/"),
    MinimalPairCase("la_09", "four_way_laryngeal", "काम", "खाम", "velar /k/ vs /kʰ/"),
    MinimalPairCase("la_10", "four_way_laryngeal", "बात", "भात", "bilabial /b/ vs /bʱ/"),
    MinimalPairCase("la_11", "four_way_laryngeal", "कोर", "खोर", "velar /k/ vs /kʰ/"),
    # 3. Active Consonant Gemination (11 pairs)
    MinimalPairCase("gm_01", "gemination", "पाता", "पत्ता", "singleton /t̪/ vs geminate /t̪ː/"),
    MinimalPairCase("gm_02", "gemination", "पना", "पन्ना", "singleton /n̪/ vs geminate /n̪ː/"),
    MinimalPairCase("gm_03", "gemination", "सता", "सत्ता", "singleton vs geminate /t̪ː/"),
    MinimalPairCase("gm_04", "gemination", "बचा", "बच्चा", "singleton affricate /ts/ vs /tsː/"),
    MinimalPairCase("gm_05", "gemination", "कचा", "कच्चा", "singleton /ts/ vs geminate /tsː/"),
    MinimalPairCase("gm_06", "gemination", "माता", "मत्ता", "singleton vs geminate /t̪ː/"),
    MinimalPairCase("gm_07", "gemination", "सुता", "सुत्ता", "singleton vs geminate /t̪ː/"),
    MinimalPairCase("gm_08", "gemination", "पका", "पक्का", "singleton /k/ vs geminate /kː/"),
    MinimalPairCase("gm_09", "gemination", "चका", "चक्का", "singleton /k/ vs geminate /kː/"),
    MinimalPairCase("gm_10", "gemination", "गला", "गल्ला", "singleton /l/ vs geminate /lː/"),
    MinimalPairCase("gm_11", "gemination", "भता", "भत्ता", "singleton vs geminate /t̪ː/"),
    # 4. Alveolar Affricates (11 pairs)
    MinimalPairCase("af_01", "alveolar_affricates", "चार", "छार", "alveolar /ts/ vs /tsʰ/"),
    MinimalPairCase("af_02", "alveolar_affricates", "जाल", "झाल", "voiced /dz/ vs breathy /dzʱ/"),
    MinimalPairCase("af_03", "alveolar_affricates", "चोखो", "छोरो", "/ts/ vs /tsʰ/ contrast"),
    MinimalPairCase("af_04", "alveolar_affricates", "जारी", "झारी", "/dz/ vs /dzʱ/ contrast"),
    MinimalPairCase("af_05", "alveolar_affricates", "चुरो", "छुरो", "/ts/ vs /tsʰ/"),
    MinimalPairCase("af_06", "alveolar_affricates", "जोड", "झोड", "/dz/ vs /dzʱ/"),
    MinimalPairCase("af_07", "alveolar_affricates", "चाल", "छाल", "/ts/ vs /tsʰ/"),
    MinimalPairCase("af_08", "alveolar_affricates", "जाम", "झाम", "/dz/ vs /dzʱ/"),
    MinimalPairCase("af_09", "alveolar_affricates", "चीर", "छीर", "/ts/ vs /tsʰ/"),
    MinimalPairCase("af_10", "alveolar_affricates", "जोल", "झोल", "/dz/ vs /dzʱ/"),
    MinimalPairCase("af_11", "alveolar_affricates", "चप", "छप", "/ts/ vs /tsʰ/"),
    # 5. Sibilant Merger (श/ष/स -> /s/) & Vowel Nasalization (11 pairs)
    MinimalPairCase(
        "sn_01",
        "sibilant_merger_and_nasality",
        "शहर",
        "सहर",
        "sibilant merger: श/स -> /s/ (zero /ʃ/ bleed)",
        expect_distinct_phonemes=False,
    ),
    MinimalPairCase(
        "sn_02",
        "sibilant_merger_and_nasality",
        "शान्ति",
        "सान्ति",
        "sibilant merger: शान्ति/सान्ति -> /s/",
        expect_distinct_phonemes=False,
    ),
    MinimalPairCase(
        "sn_03", "sibilant_merger_and_nasality", "बास", "बाँस", "oral vs nasalized vowel"
    ),
    MinimalPairCase("sn_04", "sibilant_merger_and_nasality", "सास", "साँस", "oral vs nasalized /ã/"),
    MinimalPairCase(
        "sn_05", "sibilant_merger_and_nasality", "काध", "काँध", "oral vs nasalized vowel"
    ),
    MinimalPairCase(
        "sn_06", "sibilant_merger_and_nasality", "गाउ", "गाउँ", "oral vs nasalized diphthong"
    ),
    MinimalPairCase(
        "sn_07", "sibilant_merger_and_nasality", "नाउ", "नाउँ", "oral vs nasalized vowel"
    ),
    MinimalPairCase(
        "sn_08", "sibilant_merger_and_nasality", "पाच", "पाँच", "oral vs nasalized vowel"
    ),
    MinimalPairCase(
        "sn_09", "sibilant_merger_and_nasality", "सात", "साँत", "oral vs nasalized vowel"
    ),
    MinimalPairCase(
        "sn_10", "sibilant_merger_and_nasality", "आखा", "आँखा", "oral vs nasalized vowel"
    ),
    MinimalPairCase(
        "sn_11", "sibilant_merger_and_nasality", "चाद", "चाँद", "oral vs nasalized vowel"
    ),
    # 6. Schwa Syncope, Tatsama Retention & Flapping /ɦ/-Deletion (11 pairs)
    MinimalPairCase(
        "sf_01", "schwa_and_flapping", "घर", "मित्र", "final schwa deletion vs cluster retention"
    ),
    MinimalPairCase(
        "sf_02",
        "schwa_and_flapping",
        "दिन नपाएर",
        "तीन दिन",
        "verbal infinitive /ʌ/ vs nominal syncope",
    ),
    MinimalPairCase(
        "sf_03", "schwa_and_flapping", "सडक", "पहाड", "flapping vs breathy /ɦ/-deletion lengthening"
    ),
    MinimalPairCase(
        "sf_04", "schwa_and_flapping", "पत्र", "कलम", "cluster schwa retention vs syncope"
    ),
    MinimalPairCase(
        "sf_05", "schwa_and_flapping", "चक्र", "नमक", "tatsama final /ʌ/ retention vs deletion"
    ),
    MinimalPairCase(
        "sf_06", "schwa_and_flapping", "मन्त्र", "सडक", "conjunct schwa retention vs flapping"
    ),
    MinimalPairCase(
        "sf_07", "schwa_and_flapping", "यन्त्र", "बहर", "conjunct schwa retention vs /ɦ/-deletion"
    ),
    MinimalPairCase(
        "sf_08",
        "schwa_and_flapping",
        "तन्त्र",
        "सहर",
        "conjunct retention vs intervocalic breathy vowel",
    ),
    MinimalPairCase("sf_09", "schwa_and_flapping", "वस्त्र", "कमल", "tatsama retention vs syncope"),
    MinimalPairCase(
        "sf_10",
        "schwa_and_flapping",
        "शास्त्र",
        "रहर",
        "conjunct retention vs intervocalic breathiness",
    ),
    MinimalPairCase(
        "sf_11",
        "schwa_and_flapping",
        "पात्र",
        "नहर",
        "tatsama retention vs intervocalic breathiness",
    ),
    # 7. English & Code-Mix Contrasts (11 pairs)
    MinimalPairCase(
        "ec_01",
        "english_and_codemix_contrasts",
        "she",
        "see",
        "English /ʃ/ vs /s/ sibilant contrast",
    ),
    MinimalPairCase(
        "ec_02",
        "english_and_codemix_contrasts",
        "thin",
        "tin",
        "English dental fricative /θ/ vs stop /t/",
    ),
    MinimalPairCase(
        "ec_03",
        "english_and_codemix_contrasts",
        "vine",
        "wine",
        "English labiodental /v/ vs glide /w/",
    ),
    MinimalPairCase(
        "ec_04", "english_and_codemix_contrasts", "ship", "sip", "English /ʃ/ vs /s/ contrast"
    ),
    MinimalPairCase(
        "ec_05", "english_and_codemix_contrasts", "thank", "tank", "English /θ/ vs /t/ contrast"
    ),
    MinimalPairCase(
        "ec_06", "english_and_codemix_contrasts", "vest", "west", "English /v/ vs /w/ contrast"
    ),
    MinimalPairCase(
        "ec_07",
        "english_and_codemix_contrasts",
        "officeमा",
        "office मा",
        "tight clitic attachment vs word boundary",
    ),
    MinimalPairCase(
        "ec_08",
        "english_and_codemix_contrasts",
        "check-गरिदिनुहोस्",
        "check लिनुहोस्",
        "hyphenated code-mix verb boundary",
    ),
    MinimalPairCase(
        "ec_09",
        "english_and_codemix_contrasts",
        "500 रुपैयाँ",
        "500 dollars",
        "context-aware Nepali vs English numeral expansion",
    ),
    MinimalPairCase(
        "ec_10",
        "english_and_codemix_contrasts",
        "SMS",
        "4G",
        "initialism vs digit+letter code-mix normalization",
    ),
    MinimalPairCase(
        "ec_11",
        "english_and_codemix_contrasts",
        "narrative",
        "different",
        "code-mixed English lexical stress realization",
    ),
)
