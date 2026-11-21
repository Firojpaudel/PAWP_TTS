# Benchmark & Pareto Frontier Tracker

## 1. Parameter & Storage Footprint Frontier (4-Metric Standard)

| Model / Variant | `total_params` | `unique_stored_params` | `effective_compute_params` | FP32 / FP16 / INT8 Size | Target Budget | Notes |
|---|---:|---:|---:|---|---:|---|
| `PAWP-1.5M` (uncompressed baseline) | `1,490,390` | `1,490,390` | `1,293,750` | `5.96 MB / 2.98 MB / 1.49 MB` | `1,500,000` | Initial bilingual 24 kHz ConvNeXt1D+iSTFT architecture (`~106.1 MMAC/s`) |
| `PAWP-1M` (`configs/student_1m_pawp.yaml`) | `1,223,734` | **`928,854`** | `1,125,398` | `3.72 MB / 1.86 MB / 0.93 MB` | `1,000,000` | Cross-depth `K=2` shared basis pools (`r_layer=16`, `r_lang=8`) + `28`-dim articulatory conditioning (`~63.3 MMAC/s`) |
| `OmniVoice` Teacher (LM + Audio Tokenizer) | `1,015,378,394` | `1,015,378,394` | `1,015,378,394` | `~2.84 GB` (BF16+F32) | — | Zero-shot 24 kHz teacher baseline (`813.98M` LM + `201.40M` tokenizer) |
| `PiVo-1.4M` (external reference) | `1,401,562` | `1,401,562` | `1,401,562` | `5.6 MB` FP32 | — | English-only, 16 kHz FARGAN vocoder |
| `sanoTTS` `heart-nano` (external reference) | `294,279` | `294,279` | `294,279` | `337 KB` INT8 | — | English-only, 24 kHz ConvNeXt1D+iSTFT |

## 2. Bilingual Pronunciation & Linguistic Fidelity (`WER` / `CER` / `PER` / `NepTTS-Bench`)

| Model | EN Suite Valid / `0-<unk>` | NE Suite Valid / `0-<unk>` | NepTTS-Bench (77 minimal pairs) | Code-Mix Valid / `0-<unk>` | Training Loss (`init -> best`) | Notes |
|---|---:|---:|---:|---:|---:|---|
| `PAWP-1.5M` (Stage-3 trained checkpoint) | `100% / 100%` (`10/10`) | `100% / 100%` (`10/10`) | **`100.0%` (`77/77`)** phonological & acoustic contrast (`0%` Hindi `/ʃ/` bleed) | `100% / 100%` (`10/10`) | `92.71 -> 34.54` (`mel L1=1.51`) | Trained on RTX 4090 `BF16` (`40+30` steps, `1.53 GB` VRAM) |

## 3. Real-Hardware Latency & Resource Efficiency (`p50` / `p95` / `p99`)

*Protocol aligned with `hf-audio/open_tts_leaderboard`: batch size `1`, `3` warmup utterances discarded, `30` timed utterances, streaming first-chunk size `6` frames (`60.0 ms` at `24 kHz` / `hop=240`). Note: Milestone 2 entries measure the structural `PAWP-1.5M` folded inference graph prior to acoustic training (untrained duration predictions); end-to-end quality (`WER` / `UTMOS` / `SIM`) is evaluated starting Stage 3.*

| Model & Backend | Hardware & Threads | Chunk Size | Frontend `p50 / p95` | Streaming TTFA `p50 / p95 / p99` | RTFx / RTF | Peak RSS |
|---|---|---:|---:|---:|---:|---:|
| `PAWP-1.5M` (`onnx_cpu_folded`, structural) | CPU (`4` threads) | `60.0 ms` | `0.159 / 0.330 ms` | **`1.11 / 1.42 / 1.52 ms`** | **`173.36x`** (`0.0058`) | `652.6 MB` |
| `PAWP-1.5M` (`pytorch_cpu_folded`, structural) | CPU (`4` threads) | `60.0 ms` | `0.171 / 0.198 ms` | `3.26 / 3.86 / 4.21 ms` | `74.49x` (`0.0134`) | `661.3 MB` |
| `PAWP-1.5M` (`pytorch_cuda_folded`, structural) | NVIDIA GeForce RTX 4090 (`1` thread) | `60.0 ms` | `0.165 / 0.192 ms` | `3.30 / 3.67 / 3.72 ms` | `96.36x` (`0.0104`) | `1313.6 MB` |
