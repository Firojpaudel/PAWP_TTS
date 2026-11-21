# Benchmark & Pareto Frontier Tracker

## 1. Parameter & Storage Footprint Frontier (4-Metric Standard)

| Model / Variant | `total_params` | `unique_stored_params` | `effective_compute_params` | FP32 / FP16 / INT8 Size | Target Budget | Notes |
|---|---:|---:|---:|---|---:|---|
| `PAWP-1.5M` (uncompressed baseline) | `1,490,390` | `1,490,390` | `1,293,750` | `5.96 MB / 2.98 MB / 1.49 MB` | `1,500,000` | Initial bilingual 24 kHz ConvNeXt1D+iSTFT architecture (`~106.1 MMAC/s`) |
| `PAWP-1M` (`configs/student_1m_pawp.yaml`) | `1,223,734` | **`928,854`** | `1,125,398` | `3.72 MB / 1.86 MB / 0.93 MB` | `1,000,000` | Cross-depth `K=2` shared basis pools (`r_layer=16`, `r_lang=8`) + `28`-dim articulatory conditioning (`~63.3 MMAC/s`) |
| `OmniVoice` Teacher (LM + Audio Tokenizer) | `1,015,378,394` | `1,015,378,394` | `1,015,378,394` | `~2.84 GB` (BF16+F32) | — | Zero-shot 24 kHz teacher baseline (`813.98M` LM + `201.40M` tokenizer) |
| `PiVo-1.4M` (external reference) | `1,401,562` | `1,401,562` | `1,401,562` | `5.6 MB` FP32 | — | English-only, 16 kHz FARGAN vocoder |
| `sanoTTS` `heart-nano` (external reference) | `294,279` | `294,279` | `294,279` | `337 KB` INT8 | — | English-only, 24 kHz ConvNeXt1D+iSTFT |

## 2. Bilingual Pronunciation & Linguistic Fidelity (`WER` / `CER` / `NepTTS-Bench`)

### Held-Out Test Set ASR Evaluation (Live Speech Recognizer Floor Comparison)

*Scored across 30 held-out sentences per language against the live ASR pipeline with dual ground-truth transcription floor.*

| Model & Checkpoint | English ASR WER / CER | Nepali ASR WER / CER | Code-Mixed ASR WER / CER | NepTTS-Bench (77 pairs) | Status |
|---|---|---|---|---|---|
| **Human Ground-Truth Audio Floor** | `2.6% / 0.8%` | `11.6% / 3.8%` | `25.9% / 12.9%` | — | Ground truth human studio speech |
| **`PAWP-1.5M` (Pre-FT Base, Step 92k)** | **`3.1% / 1.0%`** | **`12.0% / 3.1%`** | **`16.2% / 11.0%`** | **`100.0%`** (0% Hindi bleed) | Base multi-speaker centroid checkpoint |
| **`PAWP-1.5M` (Target-Voice Fine-Tuned, Step 20k)** | **`5.0% / 2.3%`** | **`9.9% / 2.0%`** *(beats Human GT!)* | **`12.0% / 4.9%`** *(beats Human GT!)* | **`100.0%`** (0% Hindi bleed) | Target single-speaker (`exp008-ft`) |
| **`PAWP-1.5M` Polished Default (`EXP-012`, +10k)** | **`4.9% / 2.8%`** | **`11.4% / 2.7%`** *(beats Human GT!)* | **`11.0% / 4.1%`** *(beats Human GT!)* | **`100.0%`** (0% Hindi bleed) | Pure-target phase/timbre polish; default HF release |
| **`PAWP-1.5M` + WSP `E3` (Zero-Training Edit)** | **`5.0% / 2.2%`** | **`9.9% / 2.0%`** | **`9.9% / 4.0%`** | **`100.0%`** (0% Hindi bleed) | `EXP-009` surgical `acoustic u_en/v_en` (`3.3%` edited) |
| **`PAWP-1.5M` + WSP `Null-64` (Zero-Training Edit)** | **`4.7% / 2.0%`** | **`10.1% / 1.9%`** | **`11.7% / 4.9%`** | **`100.0%`** (0% Hindi bleed) | `EXP-009` Nepali-orthogonal null-space (`4.7%` edited) |
| **`PAWP-1.0M` Decoder-Centric (Zero-Shot)** | **`9.6% / 6.9%`** | **`22.8% / 9.7%`** | **`19.1% / 9.7%`** | **`100.0%`** (0% Hindi bleed) | `EXP-010b` (`998.8K` params; beats Uniform SVD `14.3% / 23.1% / 20.8%`) |

### Phonological & Contrast Benchmarks

| Model | EN Suite Valid / `0-<unk>` | NE Suite Valid / `0-<unk>` | NepTTS-Bench (77 minimal pairs) | Code-Mix Valid / `0-<unk>` | Validation `val_mel` | Notes |
|---|---:|---:|---:|---:|---:|---|
| `PAWP-1.5M` (Step 92k Pre-FT) | `100% / 100%` (`10/10`) | `100% / 100%` (`10/10`) | **`100.0%` (`77/77`)** | `100% / 100%` (`10/10`) | **`0.8009`** | Free-running prosody (`p=0.5`), 0% Hindi `/ʃ/` bleed |
| `PAWP-1.5M` (Step 20k Post-FT) | `100% / 100%` (`10/10`) | `100% / 100%` (`10/10`) | **`100.0%` (`77/77`)** | `100% / 100%` (`10/10`) | **`0.8113`** | Target voice timbre, 1.27 hrs fine-tune, beats human GT WER/CER |

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
| Full Nepali Lexicon Coverage (`ne_lexicon.tsv`) | **105,868 words** | NFC-normalized standard Kathmandu Nepali | Verified (0 missing symbols) |
| Unified Phoneme Inventory Compliance | **100.0%** (936,987 / 936,987 tokens) | Unified 128-token phonological inventory | Verified (0 unknown tokens) |
| Context Heteronym Entries (`ne_heteronyms.tsv`) | **152 readings** across 76 words | Grammatical lookahead (infinitives vs nominals) | Verified |
| Gold 300 Native Kathmandu Validation (`make frontend-validate`) | **100.0%** (300 / 300 passed) | Native-verified benchmark sample (seed=42) | Verified clean pass |
| In-Process English G2P (`libespeak-ng.so.1` via ctypes) | `< 0.20 ms` / sentence | Zero-word hardcoding, thread-safe | Verified |
| Nepali Lexicon Startup & Cache Latency | `~0.20 s` (one-time) | Module-level cached dict (~10.1 MB RAM) | Sub-millisecond lookup |
