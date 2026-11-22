# Benchmark & Pareto Frontier Tracker

## 1. Parameter & Storage Footprint Frontier (4-Metric Standard)

| Model / Variant | `total_params` | `unique_stored_params` | `effective_compute_params` | FP32 / FP16 / INT8 Size | Target Budget | Notes |
|---|---:|---:|---:|---|---:|---|
| `PAWP-1.5M` (uncompressed baseline) | `1,490,390` | `1,490,390` | `1,293,750` | `5.96 MB / 2.98 MB / 1.49 MB` | `1,500,000` | Initial bilingual 24 kHz ConvNeXt1D+iSTFT architecture (`~106.1 MMAC/s`) |
| `PAWP-1M` (`configs/student_1m_pawp.yaml`) | `1,223,734` | **`928,854`** | `1,125,398` | `3.72 MB / 1.86 MB / 0.93 MB` | `1,000,000` | Cross-depth `K=2` shared basis pools (`r_layer=16`, `r_lang=8`) + `28`-dim articulatory conditioning (`~63.3 MMAC/s`) |
| `PAWP-300K` (`configs/student_300k_pawp.yaml`) | `328,158` | **`281,078`** | `309,702` | `1.12 MB / 0.56 MB / 0.28 MB` | `300,000` | Decoder-centric (`52.6%`), `n_fft=512`, `K=2` shared bases + `28`-dim articulatory conditioning (`20.96 MMAC/s`, `EXP-011` completed) |
| `OmniVoice` Teacher (LM + Audio Tokenizer) | `1,015,378,394` | `1,015,378,394` | `1,015,378,394` | `~2.84 GB` (BF16+F32) | — | Zero-shot 24 kHz teacher baseline (`813.98M` LM + `201.40M` tokenizer) |
| `PiVo-1.4M` (external reference) | `1,401,562` | `1,401,562` | `1,401,562` | `5.6 MB` FP32 | — | English-only, 16 kHz FARGAN vocoder |

## 2. Bilingual Pronunciation & Linguistic Fidelity (`WER` / `CER` / Contrast Checks)

### Held-Out Test Set ASR Evaluation (Live Speech Recognizer Floor Comparison)

*Scored on held-out sentences (`n=30` English, `n=30` Nepali, `n=9` Code-Mixed) against the live ASR pipeline alongside the human ground-truth recording floor. The 77-pair column is an **internal G2P + non-silent/mel-differs phonological smoke test** (NOT the published NepTTS-Bench listening/ASR benchmark; see `AUDIT-001` in `docs/EXPERIMENTS.md`).*

| Model & Checkpoint | English ASR WER / CER (`n=30`) | Nepali ASR WER / CER (`n=30`) | Code-Mixed ASR WER / CER (`n=9`) | Internal Contrast Smoke Test (77 pairs) | Status |
|---|---|---|---|---|---|
| **Human Ground-Truth Audio Floor** | `2.63% / 0.75%` | `11.60% / 3.82%` | `25.91% / 12.88%` | — | Ground truth human studio speech |
| **`PAWP-1.5M` (Pre-FT Base, Step 92k, `EXP-007`)** | **`3.14% / 1.01%`** | **`12.02% / 3.12%`** | **`16.18% / 11.02%`** | `100.0%` (`0%` `/ʃ/` symbol bleed) | Base multi-speaker checkpoint |
| **`PAWP-1.5M` (Target-Voice FT, Step 20k, `EXP-008-FT`)** | **`5.03% / 2.28%`** | **`9.93% / 1.97%`** | **`11.99% / 4.92%`** | `100.0%` (`0%` `/ʃ/` symbol bleed) | Target single-speaker (`exp008-ft`) |
| **`PAWP-1.5M` Polished Default (`EXP-012`, +10k)** | **`4.86% / 2.79%`** | **`11.41% / 2.66%`** | **`10.99% / 4.10%`** | `100.0%` (`0%` `/ʃ/` symbol bleed) | Pure-target phase/timbre polish; default HF release |
| **`PAWP-1.5M` + WSP `E3` (Zero-Training Edit, `EXP-009`)** | **`5.03% / 2.20%`** | **`9.93% / 1.97%`** | **`9.93% / 3.97%`** | `100.0%` (`0%` `/ʃ/` symbol bleed) | Surgical `acoustic u_en/v_en` (`3.3%` edited; `pawp-tts-1.5m-wsp`) |
| **`PAWP-1.5M` + WSP `Null-64` (Zero-Training Edit, `EXP-009`)** | **`4.73% / 2.02%`** | **`10.09% / 1.93%`** | **`11.65% / 4.90%`** | `100.0%` (`0%` `/ʃ/` symbol bleed) | Nepali-orthogonal null-space (`4.7%` edited) |
| **`PAWP-1.0M` Decoder-Centric (Zero-Shot, `EXP-010b`)** | **`9.56% / 6.91%`** | **`22.76% / 9.73%`** | **`19.09% / 9.66%`** | `100.0%` (`0%` `/ʃ/` symbol bleed) | `998.8K` params; zero-shot post-hoc compression (no retraining) |
| **`PAWP-300K` Base (`EXP-011`, Step 80k)** | `8.40% / 5.32%` | `20.49% / 7.49%` | `16.95% / 7.74%` | `100.0%` (`0%` `/ʃ/` symbol bleed) | `281,078` unique / `328,158` total; trained end-to-end on v2 cache (pre-lexicon) |
| **`PAWP-300K` Target Voice (`EXP-011-FT`, +20k)** | `8.80% / 3.92%` | `19.42% / 6.93%` | `16.28% / 8.74%` | `100.0%` (`0%` `/ʃ/` symbol bleed) | `281,078` unique / `328,158` total; Mix CER (`7.74->8.74%`) & EN WER (`8.40->8.80%`) regressed slightly; pre-lexicon v2 cache |

### Phonological Smoke Checks (Internal G2P + Non-Silent Audio)

| Model | EN Suite Valid / `0-<unk>` | NE Suite Valid / `0-<unk>` | Internal Contrast Smoke Test (77 pairs) | Code-Mix Valid / `0-<unk>` | Validation `val_mel` | Notes |
|---|---:|---:|---:|---:|---:|---|
| `PAWP-1.5M` (`EXP-007` Step 92k) | `100% / 100%` (`10/10`) | `100% / 100%` (`10/10`) | `100.0%` (`77/77`) | `100% / 100%` (`10/10`) | **`0.8009`** | Free-running prosody (`p=0.5`), `0%` Hindi `/ʃ/` symbol bleed |
| `PAWP-1.5M` (`EXP-008-FT` Step 20k) | `100% / 100%` (`10/10`) | `100% / 100%` (`10/10`) | `100.0%` (`77/77`) | `100% / 100%` (`10/10`) | **`0.8113`** | Target voice timbre, 1.27 h fine-tune |
| `PAWP-1.5M` (`EXP-012` Polish +10k) | `100% / 100%` (`10/10`) | `100% / 100%` (`10/10`) | `100.0%` (`77/77`) | `100% / 100%` (`10/10`) | **`0.8212`** | Pure-target phase & timbre polish (default HF checkpoint) |

## 3. Real-Hardware Latency & Resource Efficiency (`p50` / `p95` / `p99`)

*Protocol aligned with `hf-audio/open_tts_leaderboard`: batch size `1`, `3` warmup utterances discarded, `30` timed utterances, streaming first-chunk size `6` frames (`60.0 ms` at `24 kHz` / `hop=240`). Note: Milestone 2 entries measure the structural `PAWP-1.5M` folded inference graph prior to acoustic training (untrained duration predictions); end-to-end quality (`WER` / `UTMOS` / `SIM`) is evaluated starting Stage 3.*

| Model & Backend | Hardware & Threads | Chunk Size | Frontend `p50 / p95` | Streaming TTFA `p50 / p95 / p99` | RTFx / RTF | Peak RSS |
|---|---|---:|---:|---:|---:|---:|
| `PAWP-1.5M` (`onnx_cpu_folded`, structural) | CPU (`4` threads) | `60.0 ms` | `0.159 / 0.330 ms` | **`1.11 / 1.42 / 1.52 ms`** | **`173.36x`** (`0.0058`) | `652.6 MB` |
| `PAWP-1.5M` (`pytorch_cpu_folded`, structural) | CPU (`4` threads) | `60.0 ms` | `0.171 / 0.198 ms` | `3.26 / 3.86 / 4.21 ms` | `74.49x` (`0.0134`) | `661.3 MB` |
| `PAWP-1.5M` (`pytorch_cuda_folded`, structural) | NVIDIA GeForce RTX 4090 (`1` thread) | `60.0 ms` | `0.165 / 0.192 ms` | `3.30 / 3.67 / 3.72 ms` | `96.36x` (`0.0104`) | `1313.6 MB` |

## 4. Linguistic Frontend & Pronunciation Lexicon Benchmarks

| Metric / Evaluation Component | Measurement | Standard / Reference | Status |
|---|---:|---|---|
| Full Nepali Lexicon Coverage (`ne_lexicon.tsv`) | **105,868 words** | NFC-normalized standard Kathmandu Nepali; ≈99% rule/paradigm-generated (≈995 cross-checked) | Symbol-checked (0 missing symbols); not native-verified |
| Unified Phoneme Inventory Compliance | **100.0%** (936,987 / 936,987 tokens) | Unified 128-token phonological inventory | Verified (0 unknown tokens) |
| Context Heteronym Entries (`ne_heteronyms.tsv`) | **152 readings** across 76 words | Grammatical lookahead (infinitives vs nominals) | Verified |
| Lexicon self-consistency check (`make frontend-validate`) | **100.0%** (300 / 300) | 300 entries sampled from the same lexicon (seed=42); not native-verified | Regression check only |
| In-Process English G2P (`libespeak-ng.so.1` via ctypes) | `< 0.20 ms` / sentence | Zero-word hardcoding, thread-safe | Verified |
| Nepali Lexicon Startup & Cache Latency | `~0.20 s` (one-time) | Module-level cached dict (~10.1 MB RAM) | Sub-millisecond lookup |
