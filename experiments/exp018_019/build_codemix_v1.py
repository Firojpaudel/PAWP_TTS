"""Build and audit data/evaluation/codemix_v1.jsonl (EXP-019)."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from pawp_tts.domain.entities.models import LanguageCode
from pawp_tts.interfaces.container import build_container

SENTENCES: list[tuple[str, str]] = [
    # --- Everyday & Office (40) ---
    ("office", "आज मेरो office मा important meeting छ।"),
    ("office", "कृपया यो document आजै email गरिदिनुहोस्।"),
    ("office", "Team को weekly standup बिहान दस बजे start हुन्छ।"),
    ("office", "मेरो laptop को battery चाँडै drain हुन्छ, charger सँगै बोक्नुपर्छ।"),
    ("office", "यो report को final draft review गरेर feedback दिनुहोस् न।"),
    ("office", "आज traffic धेरै जाम थियो, त्यसैले office पुग्न अलि late भयो।"),
    ("office", "Manager ले project को deadline अर्को शुक्रबारसम्म extend गर्नुभएको छ।"),
    ("office", "हामीले client सँग Zoom मा आधा घण्टा call गर्यौँ।"),
    ("office", "तपाईँको presentation slides एकदमै clear र well structured देखिन्छ।"),
    ("office", "मेरो desk मा राखेको notebook र blue pen कता हरायो?"),
    ("office", "Office को canteen मा आज lunch को menu के छ?"),
    ("office", "नयाँ intern हरूको onboarding session दिउँसो दुई बजे सुरु हुन्छ।"),
    ("office", "कृपया यो spreadsheet मा सबै sales figures update गरिदिनुहोस्।"),
    ("office", "हाम्रो HR department ले वार्षिक holiday calendar पठाइसकेको छ।"),
    ("office", "तपाईँको leave request approve भइसक्यो, आरामसँग छुट्टी मनाउनुहोस्।"),
    ("office", "आजको workshop मा time management र productivity बारे छलफल भयो।"),
    ("office", "मेरो phone silent mode मा थियो, त्यसैले तपाईँको missed call देखिनँ।"),
    ("office", "कृपया meeting room को projector र microphone एकपटक check गर्नुहोस्।"),
    ("office", "यो proposal submit गर्नुअघि एकचोटि grammar र formatting हेरौँ।"),
    ("office", "बिहानै coffee पिएर काम सुरु गर्दा mood एकदमै fresh हुन्छ।"),
    ("everyday", "बाहिर weather एकदमै cold छ, न्यानो jacket लगाएर मात्र निस्कनुहोस्।"),
    ("everyday", "Weekend मा साथीहरूसँग movie हेर्न जाने plan बनाएका छौँ।"),
    ("everyday", "मेरो favourite playlist shuffle मा चलिरहेको छ।"),
    ("everyday", "बजारबाट fresh vegetables र milk लिएर फर्कनुहोला।"),
    ("everyday", "आज बेलुका gym जाने कि घरमै workout गर्ने सोचिरहेको छु।"),
    ("everyday", "तपाईँको नयाँ haircut साँच्चै smart देखिएको छ।"),
    ("everyday", "यो restaurant को momo र cold coffee दुवै एकदमै tasty छ।"),
    ("everyday", "मेरो bike को servicing गराउन आज workshop जानुपर्छ।"),
    ("everyday", "घरको wifi अलि slow चलिरहेको छ, router एकपटक restart गरौँ।"),
    ("everyday", "राति अबेरसम्म mobile scroll गर्दा बिहान उठ्न गाह्रो हुन्छ।"),
    ("everyday", "दिदीले मेरो birthday मा surprise party arrange गर्नुभएको रहेछ।"),
    ("everyday", "पसलमा त्यो jacket को medium size out of stock रहेछ।"),
    ("everyday", "मेरो shoes को sole उक्किएछ, नयाँ pair किन्नुपर्ने भयो।"),
    ("everyday", "आजको football match को second half एकदमै exciting थियो।"),
    ("everyday", "तपाईँले पठाउनुभएको photo gallery मा save भइसकेको छ।"),
    ("everyday", "यो novel को ending यति emotional थियो कि आँसु नै आयो।"),
    ("everyday", "बिहानको alarm snooze गर्दागर्दै आज फेरि ढिलो भयो।"),
    ("everyday", "मेरो backpack को zipper बिग्रियो, बाटोमै सामान झर्न लाग्यो।"),
    ("everyday", "तिमीलाई त्यो recipe को YouTube link अहिले नै WhatsApp गरिदिन्छु।"),
    ("everyday", "चिसो मौसममा तातो soup र garlic bread खानुको मज्जा नै बेग्लै छ।"),

    # --- Tech, Software & Acronyms (30) ---
    ("tech", "हाम्रो backend को REST API अहिले 200 OK response दिइरहेको छ।"),
    ("tech", "नयाँ TTS model ले CPU मा पनि real-time audio generate गर्छ।"),
    ("tech", "कृपया यो JSON file को schema validate गरेर मात्र deploy गर्नुहोस्।"),
    ("tech", "आज server को RAM र SSD usage अचानक धेरै बढ्यो।"),
    ("tech", "GitHub मा नयाँ pull request open गरेको छु, एकपटक code review गरिदिनुस्।"),
    ("tech", "तपाईँको laptop मा GPU driver र CUDA version कुन install छ?"),
    ("tech", "यो Docker container build हुन लगभग तीन मिनेट लाग्यो।"),
    ("tech", "Database को SQL query optimize गरेपछि latency आधा घट्यो।"),
    ("tech", "कृपया यो PDF file को download URL मलाई Slack मा पठाइदिनुहोस्।"),
    ("tech", "Security को लागि सधैँ VPN र SSH key प्रयोग गर्नु सुरक्षित हुन्छ।"),
    ("tech", "हाम्रो mobile app को latest update Google Play Store मा live भइसक्यो।"),
    ("tech", "Browser मा WebAssembly र ONNX Runtime प्रयोग गरेर local inference चलायौँ।"),
    ("tech", "यो Python script मा memory leak देखियो, profiler चलाएर debug गरौँ।"),
    ("tech", "Cloud server को DNS record update हुन कहिलेकाहीँ एक घण्टा लाग्छ।"),
    ("tech", "हाम्रो AI team ले नयाँ ASR र NLP pipeline benchmark गर्दैछ।"),
    ("tech", "USB cable खराब भएको रहेछ, त्यसैले data transfer fail भइरह्यो।"),
    ("tech", "यो frontend component को CSS layout mobile screen मा अलि बिग्रिएको छ।"),
    ("tech", "कृपया दुई चरणको OTP verification पूरा गरेर account login गर्नुहोस्।"),
    ("tech", "नयाँ SDK documentation पढेर हामीले CLI tool सजिलै integrate गर्यौँ।"),
    ("tech", "Production मा deploy गर्नुअघि सबै unit test र integration test pass हुनुपर्छ।"),
    ("tech", "HTTP request timeout भइरहेकाले retry logic थप्नुपर्ने देखियो।"),
    ("tech", "यो HTML template मा dark mode toggle थप्दा UI झन् सफा देखियो।"),
    ("tech", "Model को checkpoint size जम्मा पाँच megabyte मात्र छ।"),
    ("tech", "WiFi router को firmware update गरेपछि connection stable भयो।"),
    ("tech", "Linux terminal मा bash script चलाएर दैनिक backup automate गरेका छौँ।"),
    ("tech", "यो dataset को train, validation र test split पूर्ण रूपमा अलग छन्।"),
    ("tech", "Streaming audio मा first chunk latency दस millisecond भन्दा कम छ।"),
    ("tech", "Machine learning pipeline मा feature caching गर्दा training समय धेरै बच्यो।"),
    ("tech", "यो microservice ले प्रति सेकेन्ड पाँच सय request सजिलै handle गर्छ।"),
    ("tech", "API key कहिल्यै पनि public Git repository मा commit गर्नु हुँदैन।"),

    # --- Finance, Currency, Numbers & Percentages (30) ---
    ("finance", "मेरो बैंक खातामा अहिले रु. 25,000 मात्र बाँकी छ।"),
    ("finance", "यो अनलाइन sale मा सबै सामानमा 50% discount पाइन्छ।"),
    ("finance", "कृपया रु. 1,500 को bill QR code scan गरेर तिर्नुहोस्।"),
    ("finance", "विदेशी website बाट subscription लिँदा महिनाको $12.50 लाग्छ।"),
    ("finance", "बैंकको ATM बाट पैसा निकाल्दा मेरो PIN तीन पटक गलत भयो।"),
    ("finance", "यो घरको मासिक EMI रु. 45,000 पन्ध्र तारिखभित्र तिर्नुपर्छ।"),
    ("finance", "कम्पनीको वार्षिक revenue यो वर्ष 18.5% ले वृद्धि भयो।"),
    ("finance", "तपाईँको PAN number र KYC form बैंकमा update गराउनुभयो?"),
    ("finance", "यो invoice मा 13% VAT जोडेर कुल रकम रु. 11,300 पुगेको छ।"),
    ("finance", "Online payment fail भयो, तर मेरो account बाट रु. 2,000 कटिसक्यो।"),
    ("finance", "यो क्रेडिट कार्डको limit रु. 150,000 सम्म बढाउन मिल्छ।"),
    ("finance", "हाम्रो customer support service 24/7 खुला रहन्छ।"),
    ("finance", "नयाँ laptop किन्न मैले $999.99 budget छुट्याएको छु।"),
    ("finance", "शेयर बजारमा आज NEPSE index 2.4% ले उकालो लाग्यो।"),
    ("finance", "कृपया तपाईँको SMS मा आएको 6-digit OTP कसैलाई नदिनुहोस्।"),
    ("finance", "यो startup ले seed round मा $500,000 funding उठाउन सफल भयो।"),
    ("finance", "पसलमा cash छैन भने mobile banking वा wallet बाट transfer गर्नुहोस्।"),
    ("finance", "बिजुली र खानेपानीको bill मिलाएर यो महिना रु. 3,450 आयो।"),
    ("finance", "यो कोर्समा 100% scholarship पाउन entrance exam मा top 10 भित्र पर्नुपर्छ।"),
    ("finance", "होटलको कोठा प्रति रात रु. 4,500 पर्छ, जसमा breakfast समावेश छ।"),
    ("finance", "तपाईँको salary slip र tax statement HR portal बाट download गर्नुहोस्।"),
    ("finance", "यो बीमा योजनामा वार्षिक रु. 18,000 premium तिर्नुपर्छ।"),
    ("finance", "डलरको exchange rate आज रु. 134.50 पुगेको छ।"),
    ("finance", "यो पसलमा दुई हजारभन्दा माथिको किनमेलमा 10% cashback अफर छ।"),
    ("finance", "मेरो wallet मा अहिले जम्मा रु. 750 छ, थप top-up गर्नुपर्छ।"),
    ("finance", "अन्तर्राष्ट्रिय courier पठाउँदा तौल अनुसार $35 charge लाग्छ।"),
    ("finance", "यो परियोजनाको कुल budget रु. 12,50,000 स्वीकृत भएको छ।"),
    ("finance", "हामीले खर्च 25% घटाएर नाफा बढाउने लक्ष्य राखेका छौँ।"),
    ("finance", "कृपया receipt मा लेखिएको reference number सुरक्षित राख्नुहोस्।"),
    ("finance", "नयाँ bike को down payment रु. 80,000 बुझाइसकेको छु।"),

    # --- Dates, Times, Travel & Logistics (30) ---
    ("travel", "हाम्रो Kathmandu देखि Pokhara को flight भोलि बिहान 7:30 AM मा छ।"),
    ("travel", "Tribhuvan International Airport मा दुई घण्टा अगाडि नै check-in गर्नुपर्छ।"),
    ("travel", "कृपया तपाईँको boarding pass र passport हातमा तयार राख्नुहोस्।"),
    ("travel", "यो conference 2026 को November 15 मा Lalitpur मा आयोजना हुँदैछ।"),
    ("travel", "२०८१ साल असार १५ गते बिहान 9 बजे हाम्रो office उद्घाटन छ।"),
    ("travel", "Pokhara को Lakeside मा हामीले तीन रातको hotel book गरेका छौँ।"),
    ("travel", "Bus park बाट Chitwan जाने tourist bus बिहान 7 बजे छुट्छ।"),
    ("travel", "खराब मौसमका कारण Biratnagar को उडान 45 minutes delay भयो।"),
    ("travel", "Google Maps मा हेर्दा Bhaktapur पुग्न अहिले 35 minutes लाग्छ।"),
    ("travel", "मेरो flight ticket confirm भइसक्यो, seat number 14A परेको छ।"),
    ("travel", "आगामी Friday बेलुका 5:30 PM भित्र सबै documents submit गर्नुपर्छ।"),
    ("travel", "यो trekking route समुद्री सतहबाट 3,800 meters उचाइमा पर्छ।"),
    ("travel", "Airport बाट hotel सम्मको taxi fare लगभग रु. 800 पर्छ।"),
    ("travel", "हाम्रो return flight Sunday दिउँसो 2:15 PM मा land हुन्छ।"),
    ("travel", "तपाईँको parcel अहिले delivery hub मा आइपुगेको छ, भोलि बिहान पुग्छ।"),
    ("travel", "यो पासपोर्ट 2030 को December 31 सम्म valid छ।"),
    ("travel", "Lumbini भ्रमणको itinerary र hotel voucher email मा पठाइसकेको छु।"),
    ("travel", "आज बेलुका 6 PM देखि 8 PM सम्म server maintenance हुँदैछ।"),
    ("travel", "टिकट cancel गर्दा 20% cancellation fee काटिने नियम रहेछ।"),
    ("travel", "हामीले Janakpur जाने train ticket online बाटै book गर्यौँ।"),
    ("travel", "मेरो visa interview अर्को महिनाको 10th March मा तोकिएको छ।"),
    ("travel", "यात्राको बेला power bank सधैँ cabin baggage मा राख्नुपर्छ।"),
    ("travel", "यो highway मर्मत भइरहेकोले यात्रा गर्न थप 1.5 hours लाग्न सक्छ।"),
    ("travel", "Hotel को check-out time बिहान 11:00 AM सम्म मात्र छ।"),
    ("travel", "मेरो luggage को तौल 22 kg पुग्यो, limit जम्मा 20 kg मात्र छ।"),
    ("travel", "आइतबार बिहान 10 AM मा Patan Durbar Square मा भेटौँला।"),
    ("travel", "यो offer 2026-12-31 राति बाह्र बजेसम्म मात्र मान्य छ।"),
    ("travel", "नागढुङ्गा tunnel खुलेपछि यात्रा समय निकै छोटिने अपेक्षा छ।"),
    ("travel", "हाम्रो team अर्को हप्ता Butwal र Bhairahawa को field visit मा जाँदैछ।"),
    ("travel", "Flight को departure gate परिवर्तन भएर Gate 4 मा सरेको छ।"),

    # --- Names, Education, Healthcare & Daily Life (30) ---
    ("names_health", "Dr. Sharma ले भोलि बिहान 10 बजे OPD मा follow-up को लागि बोलाउनुभएको छ।"),
    ("names_health", "Ramesh र Sita दुवैजना Kathmandu University मा computer science पढ्छन्।"),
    ("names_health", "अस्पतालमा blood test र ECG report आउन दुई घण्टा लाग्यो।"),
    ("names_health", "Anita ले LinkedIn बाट नयाँ software engineer को job offer पाइन्।"),
    ("names_health", "डाक्टरको सल्लाह अनुसार यो antibiotic tablet पाँच दिनसम्म खानुपर्छ।"),
    ("names_health", "Bikash ले TU को entrance exam मा राम्रो rank ल्याएर नाम निकाल्यो।"),
    ("names_health", "मेरो हजुरबुबाको MRI र CT scan report आज बेलुका आउँदैछ।"),
    ("names_health", "Sunita ले आफ्नो research paper arXiv मा upload गरिसकेकी छिन्।"),
    ("names_health", "यो विद्यालयमा SEE को तयारी कक्षा बिहान 6 बजेदेखि सुरु हुन्छ।"),
    ("names_health", "Emergency ward बाट बिरामीलाई अहिले ICU मा shift गरिएको छ।"),
    ("names_health", "Prakash दाइले अमेरिकाबाट पठाएको parcel आज बल्ल आइपुग्यो।"),
    ("names_health", "यो online course मा 12 वटा video lecture र 4 वटा assignment छन्।"),
    ("names_health", "मेरो blood pressure अहिले 120/80 छ, एकदमै normal देखियो।"),
    ("names_health", "Nisha ले Google Scholar मा धेरै राम्रा reference paper हरू भेट्टाइन्।"),
    ("names_health", "बच्चाहरूलाई समयमै polio र BCG vaccine लगाउनु अत्यन्त जरुरी छ।"),
    ("names_health", "Suman ले आफ्नो final year project मा voice assistant बनाएको छ।"),
    ("names_health", "यो pharmacy मा digital thermometer र oximeter दुवै उपलब्ध छ।"),
    ("names_health", "Kiran ले IELTS मा 7.5 band score ल्याएर scholarship पायो।"),
    ("names_health", "आजको lecture मा Professor Joshi ले neural network को math बुझाउनुभयो।"),
    ("names_health", "नियमित yoga र meditation गर्दा mental health धेरै राम्रो हुन्छ।"),
    ("names_health", "Dipak ले YouTube मा नेपाली coding tutorial channel सुरु गरेको छ।"),
    ("names_health", "यो अस्पतालमा 24-hour emergency र ambulance सेवा उपलब्ध छ।"),
    ("names_health", "Puja ले आफ्नो thesis defense सफलतापूर्वक सम्पन्न गरिन्।"),
    ("names_health", "डाक्टरले दिनको कम्तीमा 3 liters पानी पिउन र पर्याप्त sleep लिन भन्नुभयो।"),
    ("names_health", "Arjun ले नयाँ startup को pitch deck investors हरूलाई पठाइसक्यो।"),
    ("names_health", "यो lab मा X-ray र ultrasound को सुविधा एउटै तलामा छ।"),
    ("names_health", "Manisha ले Python र PyTorch सिकेर डेटा विज्ञानमा काम सुरु गरिन्।"),
    ("names_health", "आज school को science exhibition मा robotics model प्रथम भयो।"),
    ("names_health", "Sagar ले मेरो guitar को string फेरिदियो, अहिले sound एकदमै मिठो छ।"),
    ("names_health", "स्वास्थ्य बीमा card देखाउँदा अस्पतालको धेरैजसो शुल्कमा छुट पाइन्छ।"),
]


def _norm(text: str) -> str:
    s = unicodedata.normalize("NFC", text).lower().strip()
    s = re.sub(r"[^\w\s\u0900-\u097F]", "", s)
    return re.sub(r"\s+", " ", s)


def main() -> None:
    # 1. Load all existing manifest texts for strict dedup check
    manifest_dir = Path("data/manifests")
    existing_exact: set[str] = set()
    existing_norm: set[str] = set()
    manifest_counts: dict[str, int] = {}
    for mp in sorted(manifest_dir.glob("*.jsonl")):
        cnt = 0
        with mp.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                t = str(rec.get("text", "")).strip()
                if t:
                    existing_exact.add(t)
                    existing_norm.add(_norm(t))
                    cnt += 1
        manifest_counts[mp.name] = cnt

    # 2. Check internal & external overlap
    seen_norm: set[str] = set()
    exact_overlaps: list[str] = []
    norm_overlaps: list[str] = []
    internal_dups: list[str] = []
    cleaned: list[tuple[str, str]] = []
    for cat, text in SENTENCES:
        nt = _norm(text)
        if text in existing_exact:
            exact_overlaps.append(text)
            continue
        if nt in existing_norm:
            norm_overlaps.append(text)
            continue
        if nt in seen_norm:
            internal_dups.append(text)
            continue
        seen_norm.add(nt)
        cleaned.append((cat, text))

    # 3. Run frontend on every sentence; fix nothing silently
    phonemize = build_container().phonemize
    out_dir = Path("data/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "codemix_v1.jsonl"

    unk_sentences: list[dict[str, object]] = []
    failed_sentences: list[dict[str, object]] = []
    total_en_spans = 0
    total_ne_spans = 0
    total_switches = 0
    records: list[dict[str, object]] = []

    for idx, (cat, text) in enumerate(cleaned):
        utt_id = f"cm_{idx + 1:03d}"
        try:
            seq = phonemize.execute(text)
            spans: list[str] = []
            unks: list[str] = []
            for tok in seq.tokens:
                if tok.symbol == "<unk>":
                    unks.append(tok.symbol)
                if tok.language in (LanguageCode.EN, LanguageCode.NE):
                    lang_tag = tok.language.value
                    if not spans or spans[-1] != lang_tag:
                        spans.append(lang_tag)
            en_spans = sum(1 for s in spans if s == "en")
            ne_spans = sum(1 for s in spans if s == "ne")
            switches = max(len(spans) - 1, 0)
            total_en_spans += en_spans
            total_ne_spans += ne_spans
            total_switches += switches
            if unks:
                unk_sentences.append({"id": utt_id, "text": text, "unk_count": len(unks)})
            records.append(
                {
                    "id": utt_id,
                    "language": "mixed",
                    "category": cat,
                    "text": text,
                    "word_count": len(text.split()),
                    "token_count": len(seq.tokens),
                    "en_spans": en_spans,
                    "ne_spans": ne_spans,
                    "switches": switches,
                    "unk_count": len(unks),
                }
            )
        except Exception as exc:
            failed_sentences.append({"id": utt_id, "text": text, "error": str(exc)})

    with out_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(records)
    audit = {
        "output_path": str(out_path),
        "total_sentences": n,
        "categories": {
            cat: sum(1 for r in records if r["category"] == cat)
            for cat in sorted({str(r["category"]) for r in records})
        },
        "word_count_min": min(int(r["word_count"]) for r in records),
        "word_count_median": sorted(int(r["word_count"]) for r in records)[n // 2],
        "word_count_max": max(int(r["word_count"]) for r in records),
        "dedup_checked_manifests": manifest_counts,
        "exact_overlaps_with_manifests": len(exact_overlaps),
        "normalized_overlaps_with_manifests": len(norm_overlaps),
        "internal_duplicates": len(internal_dups),
        "frontend_segmentation": {
            "total_en_spans": total_en_spans,
            "total_ne_spans": total_ne_spans,
            "total_switches": total_switches,
            "mean_switches_per_sentence": round(total_switches / max(n, 1), 2),
            "sentences_with_both_en_and_ne_spans": sum(
                1 for r in records if int(r["en_spans"]) > 0 and int(r["ne_spans"]) > 0
            ),
            "sentences_with_unk": len(unk_sentences),
            "unk_details": unk_sentences,
            "frontend_failures": len(failed_sentences),
            "failure_details": failed_sentences,
        },
    }
    audit_path = Path("artifacts/reports/exp019_codemix_v1_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
