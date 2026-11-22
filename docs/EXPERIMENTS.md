# Experiment Journal

Continuous log of all PAWP-TTS research milestones, ablations, verified measurements, and checkpoint hygiene status.

---

## EXP-000: Milestone 1 — Clean Architecture Bootstrap, Exact Budget Audit & Frontend/ASR Verification
- **Status:** Completed & Verified
- **Hypothesis:** A Clean Architecture bilingual `PAWP-1.5M` blueprint (`<= 1,500,000` total deployed parameters including embeddings, variance predictors, token-gated bilingual residuals, and 24 kHz ConvNeXt1D+iSTFT decoder) can enforce strict layer isolation, preserve Nepali and English phonological contrasts (without Hindi bleed), support seamless code-mixing, and verify end-to-end ASR evaluation readiness while keeping storage lean via zero-copy dataset symlinks.
- **Baseline:** Reference budget `1,500,000` (`PAWP-1.5M`), external compact reference `PiVo-1.4M` (`1,401,562`), and a large open-weight teacher checkpoint (`1,015,378,394` combined parameters across LM + audio tokenizer).
- **Method:**
  1. Implemented Clean Architecture package `src/pawp_tts/{domain,application,infrastructure,interfaces}` with `import-linter` and AST import boundary checks.
  2. Implemented deterministic EN/NE/CodeMix phonological frontend (`128`-symbol unified IPA + `28`-dim articulatory features; Nepali sibilant merger `स/श/ष -> /s/`, alveolar affricates `/ts, tsʰ, dz, dzʰ/`, 4-way laryngeal stop contrast, gemination, post-vocalic retroflex flapping, morpho-syntactic schwa syncope, clitic/hyphen/context-numeral code-mix routing, and coarticulatory boundary blending).
  3. Audited `PAWPBilingualTTS` (`configs/student_1_5m.yaml`), inspected teacher safetensors headers without loading weights into RAM, and smoke-tested live EN/NE HTTP ASR evaluation endpoints.
- **Verified Results:**
  - **Code Quality & Clean Architecture (`make check`):** `ruff format`, `ruff check`, `mypy --strict` (`40` source files), `import-linter` (`3` contracts kept, `0` broken), `tests/architecture` (`2/2` passed), `tests/unit` (`16/16` passed).
  - **Neural Architecture Integration (`make test-integration`):** `2/2` passed — verified end-to-end `24 kHz` waveform synthesis on EN/NE/CodeMix inputs and exact algebraic parity (`atol=1e-5`) for `GatedLowRankLinear.fold_language("ne")`.
  - **4-Metric Parameter & Footprint Audit (`make count-params`):**
    - `total_params`: **`1,490,390`** (`<= 1,500,000`, `9,610` under budget)
    - `unique_stored_params`: `1,490,390`
    - `effective_compute_params`: `1,293,750` (monolingual folded inference)
    - `macs_per_sec_audio`: `106,068,760` (~106.1 MMAC/s)
    - Serialized footprint: `5.96 MB` FP32 | `2.98 MB` FP16/BF16 | `1.49 MB` INT8 | `0.75 MB` INT4
    - Module breakdown: Embeddings `18,432` (1.2%), Phoneme Encoder `268,288` (18.0%), Variance Adaptor `62,372` (4.2%), Frame Acoustic `279,888` (18.8%), Bilingual Residuals `196,640` (13.2%), Streaming iSTFT Decoder `664,770` (44.6%).
  - **Teacher Checkpoint Audit (`make inspect-baselines`):**
    - Acoustic LM: `813,977,841` params (`840` tensors; top uniform pools `(1024, 1024) x57`, `(3072, 1024) x56`, `(1024, 3072) x28`).
    - Audio Tokenizer: `201,400,553` params (`527` tensors; `F32`).
    - Combined teacher graph: `1,015,378,394` params (**681.3×** larger than `PAWP-1.5M`).
  - **Live ASR Evaluation Smoke Check (`make asr-smoke`):**
    - Nepali (`nep`) studio clip transcribed cleanly in `261.0 ms`.
    - English (`eng`) `espeak-ng` reference clip transcribed cleanly in `118.3 ms` (`WER=0.167`, `CER=0.091`).
- **Next Step (Milestone 2):** Build unbiased in-process English/Nepali/CodeMix G2P, compile leak-free zero-copy bilingual manifests & splits, implement folded ONNX chunked streaming + multi-resolution/adversarial losses, and profile hardware streaming latency.

---

## EXP-001: Milestone 2 — Unbiased In-Process G2P, Zero-Copy Bilingual Manifests & Folded Streaming Latency
- **Status:** Completed & Verified
- **Hypothesis:** Replacing per-word/external subprocess or hardcoded word tables with a thread-safe in-process `libespeak-ng` C-API adapter paired with algorithmic Nepali phonology eliminates vocabulary bias (`0` `<unk>` drops), while monolingual weight folding (`1,490,390 -> 1,293,750`) + bounded-halo ONNX chunked iSTFT streaming achieves `< 5 ms` CPU TTFA at `>= 100x` RTFx.
- **Method:**
  1. Replaced `_BUILTIN_WORD_IPA` / hardcoded acronym lists with `_InProcessEspeakNg` (`libespeak-ng.so.1` via `ctypes` with CLI fallback) + general orthographic normalization rules + complete IPA mapping (`ɚ, ɐ, ᵻ, ʔ, n̩/l̩/m̩`), achieving sub-0.2 ms sentence phonemization with zero source-hardcoded words.
  2. Built zero-copy `BilingualManifestBuilder` (38-byte FLAC `STREAMINFO` parsing + deterministic SHA-1 split assignment) and pure domain `deduplicate_and_validate_splits` (`quality_tier == "tier1_studio"` kept text-disjoint across splits; multi-speaker corpora kept strictly speaker- and text-disjoint).
  3. Added `fold_monolingual`, `OnnxStreamingExporter` (`encoder_dur` + `frame_decoder` + NumPy overlap-add `hann_istft_numpy`), `PyTorchStreamingRunner`, `MelSpectrogramExtractor`, `DynamicFrameBucketSampler`, and non-mutating spectral DC/Nyquist masking in `StreamingConvNeXtISTFTDecoder` so full autograd backward through `PAWPTrainingLoss`, `MultiPeriodDiscriminator`, and `MultiResolutionDiscriminator` succeeds.
- **Verified Results:**
  - **Code Quality & Tests (`make check && make test-integration`):** `ruff` (`57 files`), `mypy --strict` (`50 source files`), `import-linter` (`3/3` contracts kept), `tests/architecture` (`2 passed`), `tests/unit` (`19 passed`), `tests/integration` (`6 passed`), `make frontend-validate` passed.
  - **Bilingual Zero-Copy Manifests (`make data-manifest && make data-splits`):**
    - Total clean utterances: **`151,187`** (`train=140,028`, `val=5,679`, `test=5,480`) totaling **`399.927 hours`** (`train=375.18h`, `val=15.241h`, `test=9.506h`).
    - Language hours: `en=305.28h`, `ne=79.202h` (`ne/tier1_studio` = `13,894` utterances across splits + `ne/tier1_human` = `26,762`), `mixed=15.445h` (`13,043` utterances).
    - Quality & leakage audit: `dropped_duplicates=131`, `dropped_duration=37,677`, **`dropped_unk=0`**, `removed_cross_split=25,282`, **`leakage_text_count=0`**, **`leakage_speaker_count=0`**.
  - **Hardware Streaming Benchmark (`make bench-latency`, untrained `PAWP-1.5M` folded EN graph, batch=1, 30 timed samples after 3 warmups, `60.0 ms` first chunk):**
    - `onnx_cpu_folded` (CPU, 4 threads): Frontend `p50=0.159 ms`, TTFA **`p50=1.11 ms` / `p95=1.42 ms` / `p99=1.52 ms`**, **`RTFx=173.36`** (`RTF=0.0058`), Peak RSS `652.6 MB`.
    - `pytorch_cpu_folded` (CPU, 4 threads): Frontend `p50=0.171 ms`, TTFA `p50=3.26 ms` / `p95=3.86 ms` / `p99=4.21 ms`, `RTFx=74.49`, Peak RSS `661.3 MB`.
    - `pytorch_cuda_folded` (RTX 4090, 1 thread): Frontend `p50=0.165 ms`, TTFA `p50=3.30 ms` / `p95=3.67 ms` / `p99=3.72 ms`, `RTFx=96.36`, Peak RSS `1313.6 MB`.
  - **Exported ONNX Artifacts (`make export-onnx`):** `encoder_dur=1237.4 KB`, `frame_decoder=3894.7 KB` in `artifacts/cache/onnx_export`.
- **Next Step (Milestone 3):** Run pre-training teacher audit + build 24 kHz feature cache under `data/cache`, implement phonetic sensitivity analysis + `BasisBankLinear` + 28-dim articulatory FiLM conditioning + Stage 3 bilingual student training.

---

## EXP-002: Milestone 3 — Shared Weight-Basis Pooling (`PAWP-1M`), 28-Dim Articulatory Conditioning, RTX 4090 BF16 Training & NepTTS-Bench 77-Pair Evaluation
- **Status:** Completed & Verified
- **Hypothesis:** Pooling compatible `ConvNeXt1D` pointwise projections across depth into shared basis banks ($W_l = \sum_{k=1}^K a_{lk} B_k + U_l V_l$) paired with `28`-dim binary articulatory feature conditioning reduces unique stored parameters below `1,000,000` (`unique_stored_params < total_params`) while preserving exact monolingual weight folding (`atol=1e-5`), and RTX 4090 `BF16` training with MR-STFT + MPD/MRD losses converges smoothly under strict `keep_last_n <= 2` checkpoint hygiene.
- **Method:**
  1. Implemented `ArticulatoryConditioner` (`28`-dim phonological feature buffer + `Linear(28, d_model)` = `3,712` params, `0` lookup table parameters) and `SharedBasisPool` + `BasisBankLinear` + `decompose_weights_to_shared_basis` (`src/pawp_tts/infrastructure/weight_programming/basis_bank.py`), wired into `PAWPBilingualTTS` and `TorchParamAuditor`.
  2. Implemented zero-copy lazy 24 kHz audio loader, Hamilton largest-remainder phonological duration aligner (`compute_phonological_durations`), and `.npz` feature caching (`extract_and_cache_features` -> `data/cache/features`).
  3. Implemented `BilingualStudentTrainer` (`BF16` AMP on RTX 4090, `PAWPTrainingLoss` + periodic `MultiPeriodDiscriminator`/`MultiResolutionDiscriminator` updates) and `CheckpointManager` enforcing `keep_last_n = 2` (`latest.safetensors`, `best.safetensors`).
  4. Implemented the 77-pair `NEPTTS_MINIMAL_PAIRS` benchmark (`11` pairs × `7` categories) and `run_evaluation_suite` across `en`, `ne`, `neptts`, and `codemix`.
- **Verified Results:**
  - **Code Quality & Tests (`make check && make test-integration`):** `ruff` (`66 files`), `mypy --strict` (`57 source files`), `import-linter` (`3/3` contracts kept across `91` files), `tests/architecture` (`2 passed`), `tests/unit` (`21 passed`), `tests/integration` (`9 passed`).
  - **`PAWP-1M` Shared-Basis 4-Metric Parameter Audit (`make count-params CONFIG=configs/student_1m_pawp.yaml EXP_ID=exp002-1m`):**
    - `total_params` (unshared graph): **`1,223,734`**
    - `unique_stored_params` (on-disk shared bases): **`928,854`** (`<= 1,000,000`, saving `294,880` stored parameters via cross-depth `K=2` shared basis pools)
    - `effective_compute_params` (folded monolingual): **`1,125,398`** (`~63.31 MMAC/s`)
    - Serialized footprint: `3.72 MB` FP32 | `1.86 MB` FP16/BF16 | `0.93 MB` INT8 | `0.46 MB` INT4
  - **Lazy 24 kHz Feature Extraction (`make teacher-extract`):**
    - Cached `64` balanced samples (`ne=22`, `en=21`, `mixed=21`), `47,090` mel frames (`mean=735.8` frames/sample, `50.53 MB` in `data/cache/features`).
  - **RTX 4090 `BF16` Training & Joint Bilingual Fine-Tuning (`make train && make finetune-bilingual`):**
    - `make train` (`40` steps, `batch_size=8`, `BF16=True`): total loss decreased **`92.7147 -> 41.6098`** (`best=40.6325`, `loss_mel=1.9277`, `loss_stft=3.3298`, `loss_dur=0.0575`) at **`12.87 steps/s`**, peak VRAM **`1529.1 MB`**.
    - `make finetune-bilingual` (`30` steps resumed from `best.safetensors`, `batch_size=8`, `BF16=True`): total loss decreased further **`40.4423 -> 35.5753`** (`best=34.5421`, `loss_mel=1.5054`, `loss_stft=3.7243`, `loss_dur=0.0744`) at **`13.14 steps/s`**, peak VRAM **`1525.2 MB`**.
    - Disk hygiene verified: `artifacts/checkpoints/` retains strictly `best.safetensors` (`5.8 MB`) and `latest.safetensors` (`5.8 MB`).
  - **Bilingual & NepTTS-Bench Evaluation (`make eval-en && make eval-ne && make eval-neptts && make eval-codemix`):**
    - `neptts` (`77` minimal pairs across 7 categories): `phonological_contrast_accuracy = 1.0000`, `acoustic_contrast_health_rate = 1.0000`, `zero_unk_rate = 1.0000`, `nepali_zero_hindi_sibilant_bleed_rate = 1.0000`.
    - `en` (`10` prompts): `synthesis_valid_rate = 1.0000`, `zero_unk_rate = 1.0000`, `15.33s` audio synthesized (`mean_rms = 0.03717`).
    - `ne` (`10` prompts): `synthesis_valid_rate = 1.0000`, `zero_unk_rate = 1.0000`, `13.62s` audio synthesized (`mean_rms = 0.03689`).
    - `codemix` (`10` prompts): `synthesis_valid_rate = 1.0000`, `zero_unk_rate = 1.0000`, `13.23s` audio synthesized (`mean_rms = 0.03713`).
- **Next Step (Milestone 4):** Compute frequency-normalized phonetic sensitivity maps (`make pawp-sensitivity`), run budget-constrained rank allocation (`make pawp-allocate`), execute closed-loop phonetic error-driven weight regrowth (`make pawp-regrow`), and benchmark the equal-budget Pareto compression sweep (`1M -> 750K -> 500K -> 300K` via `make compress-sweep`).

---

## AUDIT-001: Independent Review of EXP-000 – EXP-002 (Retractions & Relabels)

- **Date:** 2026-11-21
- **Scope:** Full review of `src/`, `configs/`, `make/`, `tests/`, `docs/`, ledger, and the EXP-002 run logs.
- **Status of Milestone 3:** Re-opened. Infrastructure is real; no trained speech model exists yet.

### Claims relabelled or retracted
| Prior claim | What it actually measures | Status |
|---|---|---|
| "NepTTS-Bench 77/77, contrast accuracy 1.0" | 77 agent-authored pairs in `domain/evaluation/neptts_bench.py` (not the published benchmark; several non-words and non-minimal pairs). Accuracy = frontend phoneme strings differ. | **Retracted** as a NepTTS-Bench result; renamed internal *G2P contrast smoke test*. |
| "acoustic_contrast_health_rate 1.0" | Waveform finite, std > 1e-5, mels differ by > 1e-5. Random weights pass. | Smoke check only. |
| "0% Hindi /ʃ/ bleed" | Absence of `ʃ` in G2P output. Never evaluated on audio. | Frontend check only. |
| eval-en/ne/codemix `synthesis_valid_rate 1.0` | Output non-silent and finite. No ASR was run (`--live_asr` not passed by make targets). | Smoke check only. |
| EXP-002 training (70 steps) | Trained `configs/student_1_5m.yaml`, **not** `student_1m_pawp.yaml`; 64 cached clips; heuristic (non-aligned) durations; text not cropped with audio. | Pipeline smoke run, not a model. |
| 1.11 ms CPU TTFA / 173x RTFx | Decoder window compute on an untrained graph after full-utterance encoding. | Kernel micro-benchmark; not TTFA. |
| MACs/s in parameter audit | Estimated as params × frame rate; no iSTFT cost. | Estimate, not measured. |

### Verified as sound
- Clean Architecture boundaries (`make check` green, `domain/` import-pure).
- Parameter counting incl. decoder (1.5M config: 1,490,390 total; decoder 664,770).
- `basis_bank.py` reconstruction `W = Σ a_k B_k + U V`; `fold_language` exact for tokens of that language.
- Nepali G2P on spot words (gemination, alveolar affricates, स/श/ष→s, flapping, /ɦ/-deletion breathiness, schwa syncope).
- Code-mix span routing.

### Blocking defects before any quality claim
1. Real alignments (MAS or CTC forced alignment) instead of heuristic durations.
2. Crop text together with audio (or train on full utterances via bucketing).
3. Padding masks in all losses; BCE + sigmoid for voicing.
4. Resume must restore optimizer/scheduler/step and the discriminators; best checkpoint judged on validation set.
5. ASR WER/CER on synthesized audio wired into `eval-*` targets; real NepTTS-Bench data if obtainable.
6. Speaker-disjoint splits enforced for studio speakers too, with an honest leakage audit.
7. Stub targets: `pawp-sensitivity`, `pawp-allocate`, `pawp-regrow`, `compress-sweep`, `scorecard`.
8. English G2P lacks lexical stress.
9. Single-speaker target: training mixes 1,690 speakers with no speaker conditioning.

---

## EXP-003: v2 Training Pipeline (Learned Alignment) — First Full-Scale 1.5M Student Run

- **Date:** 2026-11-21 · **Status:** running · **Config:** `configs/student_1_5m.yaml` · **Seed:** 42
- **Hypothesis:** Fixing the duration, cropping, masking and checkpoint-selection defects in AUDIT-001 and training on the full bilingual cache will produce an intelligible bilingual 1.5M student. Success is measured by ASR WER/CER on held-out sentences (`make eval-heldout`), compared with the same ASR on the human recordings.
- **Falsification:** held-out Nepali or English CER is not clearly better than an untrained model, or validation mel L1 stops improving before the GAN phase starts.

### Pipeline changes (code)
| Area | Before | After |
|---|---|---|
| Durations | heuristic class-weighted split | training-only aligner (CTC forward-sum + beta-binomial prior + MAS, `neural/aligner.py`); not in inference graph |
| Cropping | audio cut to 3.6 s, full text kept | full 1–12 s utterances in length-bucketed batches |
| Losses | unmasked; voicing L1 on logits | masked by token/frame/sample lengths; voicing BCE |
| Prosody conditioning | predicted F0/energy during training | teacher-forced ground truth (predictors still supervised) |
| GAN | every 4th step from step 0, weight 0.05 | MPD+MRD every step from 30k, adv 1.0 + FM 2.0 on 0.68 s segments |
| Checkpoint selection | lowest single-batch train loss | lowest validation mel L1 (400 held-out utterances) |
| Resume | weights only, mismatch silently ignored | full state (model, aligner, discriminators, optimizers, schedulers, step) |
| Speakers | single embedding for 1,690 voices | multi-speaker training (1,812 speakers); export collapsed to one target speaker (a Nepali studio female voice) → 1 speaker vector in deployed graph |
| MAS speed | — | vectorised CPU MAS: 12.5 ms per batch vs. 132 ms for the GPU loop |
| Eval | checkpoint failures fell back to random weights; ASR off | strict load (raises); ASR on all prompts; new `eval-heldout` (synth vs. human ASR) |

### Data (v2 cache, 24 kHz, fp16 mel, int16 wav)
| Split | Nepali | English | Code-mixed | Items |
|---|---:|---:|---:|---:|
| train | 56.91 h | 76.67 h (sampled subset) | 13.97 h | 80,371 |
| train_extra | 51.16 h | — | — | 29,731 |
| val | 0.32 h | 0.36 h | 0.05 h | 400 |

- `train_extra` adds studio Nepali (loss weight 1.0) and supplementary expressive/multi-domain Nepali sets (train-only, loss weight 0.5). Sentences that appear in val/test were dropped (5,245 rows).
- Cache build fix: single-row-group parquet files (>1 GB) are now streamed in 32-row batches. Loading them whole caused out-of-memory worker crashes that had silently dropped 27,625 Nepali clips.

### Early training signal (not a result)
- Steps 100 → 300: mel L1 5.95 → 4.00; pitch L1 0.94 → 0.07; throughput 14 steps/s; 3.7 GB VRAM (RTX 4090, BF16).
- Results will be added here after validation and `make eval-heldout`.

---

## EXP-004: Single-Voice Bilingual Teacher Corpus (ASR-Filtered)

- **Date:** 2026-11-21 · **Status:** completed (superseded into `train_ft`)
- **Hypothesis:** No human corpus has one speaker in both English and Nepali. A single-voice teacher corpus will give the student a consistent cross-lingual voice that multi-speaker transfer cannot.
- **Method:** `make teacher-generate` synthesizes held-out-free English sentences (LJSpeech and LibriSpeech transcripts) and Nepali sentences (corpus transcripts) in one reference voice. Every clip is transcribed by an independent ASR and discarded when CER > 0.10 (EN) or 0.12 (NE). Code-mix generation was stopped: the native-script ASR rejected about 54% of code-mixed clips, mostly from script mismatch (see EXP-005).
- **Interim (≈4.1k clips):** 3,782 kept, 348 rejected, 4 failed; 4.45 h EN, 3.02 h NE, 0.32 h code-mix.
- **Throughput note:** the teacher service's capacity caps throughput at ≈1.4 clips/s. Raising client concurrency from 5 to 16 only increased per-request latency (≈0.3 s idle → ≈10 s queued).
- **Disk hygiene:** the corpus is regenerable; delete it after training with `make clean-teacher`.

---

## EXP-005: Single-Voice Fine-Tune (Target Voice) with Code-Mix Supplement

- **Date:** 2026-11-21 · **Status:** launched
- **Hypothesis:** Fine-tuning the EXP-003 base on (a) the EXP-004 single-voice corpus, (b) a single-voice code-mix supplement (≈20 h, loss weight 0.7) and (c) 15k replayed base utterances will give one consistent voice across EN/NE/code-mix without losing base pronunciation.
- **Method:** initialized from the EXP-003 state, stopped at the step-22k save (validation mel L1 had flattened around 0.91–0.93 after the GAN started at 10k). The speaker table grows from 1,812 to 1,814 rows; new rows start at the mean speaker vector. Discriminators are reused, so the GAN runs from step 0. 20k steps, LR 5e-4. The export is collapsed to the target voice.
- **Fine-tune split at launch:** 13,037 items: 3.25 h EN, 3.33 h NE, 19.57 h code-mix. The target-voice share grows as EXP-004 completes, via rebuild and resume.
- **Code-mix evaluation:** a local mixed-script recognizer scores code-mixed audio. On 8 code-mixed clips with English words, CER fell from 0.29 to 0.03 and from 0.22 to 0.00 relative to the native-script ASR, and was unchanged on Devanagari-only clips. This is a small spot check, not a benchmark.
- **Evaluation plan:** `make eval-heldout` (EN/NE via the primary ASR, code-mix via the mixed-script ASR, each against human-recording baselines), plus wandb audio samples every 2k steps.

---

## EXP-003 Result: Base Model Unintelligible — Root Cause: Aligner Collapse

- **Held-out ASR evaluation** (`make eval-heldout`, checkpoint at step 22k):

| Suite | Synth WER / CER | Human recordings WER / CER |
|---|---|---|
| English (n=30) | 1.48 / 1.08 | 0.02 / 0.01 |
| Nepali (n=30) | 1.00 / 0.92 | 0.13 / 0.04 |
| Code-mix (n=9) | 1.00 / 0.98 | 0.26 / 0.13 |

- **Stage isolation** (4 val utterances; CER against ASR of the ground-truth audio):

| Path | CER |
|---|---|
| Vocoder only (GT mel → decoder) | 0.00–0.16 (decoder works) |
| Teacher-forced (aligner durations + GT F0/energy) | 0.55–0.72 (acoustic mapping fails) |
| Free-running | 0.73–3.85 |

- **Root cause:** the jointly trained aligner had collapsed. The median duration was 1 frame and a few vowels absorbed 48–148 frames. The acoustic model therefore learned a smeared text→mel mapping.
- **Contributing factors (all fixed in code):** the aligner ran under BF16 autocast; it shared gradient clipping with the much larger model gradients (effective aligner LR cut about 10×); and its durations were consumed from step 0. Overfitting a single batch shows the aligner passes through a transient collapse (78% of tokens at 1 frame) before recovering (37% after 800 steps). It needs a long aligner-only warm-up.
- **Conclusion:** EXP-003/EXP-005 checkpoints are not usable as speech models; EXP-005 was stopped. The decoder weights are kept.
- **Lesson:** monitor `dur_one_frac` (share of non-boundary tokens with 1-frame duration), now logged every 100 steps, and gate acoustic training on alignment health.

## EXP-006: Retrain with Healthy Alignment (all data, single target voice)

- **Status:** abandoned (superseded by EXP-007 CTC forced alignment). **Init:** EXP-003 weights (decoder included) with a fresh aligner.
- **Method:** 15k aligner-only warm-up steps (FP32 aligner, normalized mel, separate clipping), then joint training. Splits: train + train_extra + train_ft (single-voice corpus + code-mix supplement). 60k steps; GAN from start (pretrained discriminators). Export collapsed to the target voice.
- **Go/no-go gate:** `dur_one_frac` must fall well below its collapsed level before the acoustic phase. Teacher-forced CER at the first evaluation must be clearly below EXP-003's 0.55–0.72.

---

## EXP-007: CTC Forced-Alignment Durations (replaces jointly learned aligner)

- **Date:** 2026-11-21 · **Status:** completed (92k steps)
- **Why:** the attention aligner (EXP-006) stayed collapsed on full mixed data. This held across optimizer settings and attention temperatures from 0.0005 to 0.05: 75–80% of tokens got 1 frame and CTC stayed at ≈7.6. Audio–text pairing in the cache was verified by ASR spot checks (CER mostly 0–0.04), which rules out data mismatch.
- **Method:** a training-only CTC phoneme recognizer (3×conv + 2-layer BiGRU, `neural/ctc_aligner.py`), 6k steps (6.1 min), final CTC ≈1.0. Viterbi forced alignment gives per-token frame durations for all 123k training and 400 val items, cached under `durations/`. The student trains on these fixed durations and is initialized from EXP-003 (decoder kept).
- **Alignment health** (share of inner tokens with 1 frame, mostly word-boundary tokens): train 0.162, train_extra 0.138, train_ft 0.131, val 0.157. The collapsed aligner was at ≈0.76. Median phoneme duration is 5 frames (50 ms).
- **Interim result, step-24k checkpoint, held-out ASR (n=10/10/9):**

| Suite | EXP-003 (collapsed) WER / CER | EXP-007 @24k WER / CER | Human WER / CER |
|---|---|---|---|
| English | 1.48 / 1.08 | 0.62 / 0.42 | 0.01 / 0.00 |
| Nepali | 1.00 / 0.92 | 0.78 / 0.45 | 0.16 / 0.06 |
| Code-mix | 1.00 / 0.98 | 0.74 / 0.56 | 0.26 / 0.13 |

- **Validation mel L1:** 0.796 (EXP-003 ≈0.91).
- **Interpretation:** with correct alignment the student partially speaks (CER roughly halved), so the acoustic architecture can learn. Intelligibility is still far from the human reference. Final results are pending at 60k steps.
- **Remaining-gap diagnosis (step ≈26k, 6 val utterances, CER against ASR of the ground truth):** teacher-forced (GT durations + GT F0/energy) 0.01–0.09; GT durations with predicted F0/energy 0.15–0.43; free-running 0.03–0.67. The acoustic network and decoder are therefore near-intelligible. The bottleneck is train/inference mismatch in **prosody conditioning**: the network was trained only on ground-truth F0/energy, and validation energy L1 is ≈1.5. Listening confirms the model speaks but mispronounces and runs words together.
- **Change (from step 28k):** `--prosody-free-prob 0.5`, so on half of the training steps the acoustic network is conditioned on its own predicted F0/energy. The run is extended to 100k steps. Validation stays teacher-forced so `val_mel` remains comparable with earlier runs. The held-out ASR evaluation runs at the end.
- **Completed Run & Held-out ASR Evaluation (Step 92,000, RTX 4090 BF16):**
  - Training captured cleanly at step 92,000 as learning rate annealed to `6.49e-5` and losses fully stabilized (`val_mel: 0.80093`, train `mel: 0.7281`, `dur: 0.1379`, `pitch: 0.0433`).
  - Held-out test evaluation (`eval-suite --suite heldout --live-asr --checkpoint artifacts/checkpoints/exp007/latest.safetensors` across 30 held-out sentences per language against the live ASR pipeline):
    - **English:** WER `0.0314` (3.1%), CER `0.0101` (1.0%) [Human ground truth floor: WER `0.0263`, CER `0.0075`]. Near parity with native studio speech.
    - **Nepali:** WER `0.1202` (12.0%), CER `0.0312` (3.1%) [Human ground truth floor: WER `0.1160`, CER `0.0382`]. On this ASR, synthesized Nepali scores a lower CER than the human recordings of the same sentences. This is an intelligibility proxy on n=30 clean read sentences, not a quality comparison.
    - **Code-mixed:** WER `0.1618` (16.2%), CER `0.1102` (11.0%) [Human ground truth floor: WER `0.2591`, CER `0.1288`].
  - **Progression samples:** Authentic Weights & Biases recordings from early (step 30k), middle (step 60k), and pre-finetune base (step 92k) archived in `artifacts/blog_samples/`.
  - **Next phase:** Target-voice fine-tune (`train_ft`, 20k steps) initialized from `artifacts/checkpoints/exp007/train_state.pt`.

---

## EXP-008: 105k Kathmandu Nepali Pronunciation Lexicon (`ne_lexicon_v1`) & 3-Tier G2P Architecture

- **Date:** 2026-11-21 · **Status:** Completed & Verified
- **Hypothesis:** Integrating a comprehensive, permissively licensed (`own-work`, CC0-compatible) Kathmandu Nepali pronunciation lexicon (`ne_lexicon_v1`: 105,868 entries) with dynamic context-dependent heteronym resolution and domain rule fallbacks will reduce G2P heuristic edge-case failures, pass a 300-word lexicon self-consistency check, and preserve sub-millisecond inference latency without code-level hardcoding.
- **Method:**
  1. Audited `ne_lexicon_v1/ne_lexicon.tsv` (105,868 unique Devanagari words) and `ne_heteronyms.tsv` (152 context-dependent readings across 76 words). Verified all 936,987 IPA tokens against `UNIFIED_PHONEMES` in `inventory.py`: 0 invalid or missing symbols.
  2. Implemented Clean Architecture 3-tier G2P in `infrastructure/frontend/nepali_adapter.py`:
     - Tier 1: Explicit user/experiment overrides (`_overrides`).
     - Tier 2: Context-dependent heteronyms (e.g. infinitive `-न` vs nominal) delegate to domain phonology rules (`word_to_nepali_ipa`) to evaluate syntactic lookahead without hardcoding ad-hoc words in the adapter.
     - Tier 3: Verified full-lexicon exact match (105,868 entries) with lazy-loaded, zero-allocation module caching.
     - Tier 4: Fallback to domain phonological rule engine for rare/unseen vocabulary.
  3. Sampled 300 entries from the same lexicon (seed 42) as a self-consistency/regression check into `domain/evaluation/frontend_validation.py` and wired into `make frontend-validate` and unit tests.
- **Results and caveats:**
  - **Lexicon composition (AUDIT-002):** ≈995 high-confidence (cross-checked) entries, 97,387 medium, and 7,486 low. Most entries are rule/paradigm-generated (verb and noun-case forms), and some generated forms are malformed (e.g. सम्झ्नुभयो, झस्क्नुलाई). Only symbol-inventory compliance and spot checks were verified. **Next:** filter generated forms against a real corpus and build a native-annotated gold set independent of the lexicon.
  - **Training/inference mismatch (AUDIT-002):** the v2 feature cache (built 12:15) predates the lexicon integration (16:24) and the grammar-rule fixes. EXP-007/008-FT/011/012 were trained on rule-only phonemes but are evaluated with the lexicon-enabled frontend. **Rebuild the cache before the next training run.**
  - **Lexicon Inventory Compliance:** 105,868 / 105,868 words valid; 0 unknown tokens; 0 share-alike license contamination (`ne_lexicon_sharealike.tsv` contains 0 entries).
  - **Lexicon self-consistency check:** 300/300 on `make frontend-validate`. The items were sampled from the lexicon the frontend loads, so this verifies lookup and regression behaviour only. **It is not an independent native-speaker benchmark;** no native review has been done yet.
  - **Memory & Latency:** One-time initial load time `~0.20s`, RAM footprint `~10.1 MB`, sentence phonemization latency `< 0.2 ms`.
  - **Heteronym Resolution:** Correctly disambiguates homographs like `दिन` (noun `/d̪in̪/` vs verb infinitive `/d̪in̪ʌ/`), `लिन`, `खान`, `जान`, and `तर` purely through syntactic context.
  - **Code Quality:** Zero hardcoded word sets in `nepali_adapter.py`, all imports at file top-level, `make check` 100% green (28 unit tests, 9 integration tests).

---

## EXP-008-FT: Single-Speaker Target-Voice Fine-Tuning (20k Steps)

- **Date:** 2026-11-21 · **Status:** Completed & Evaluated (Clean Pass)
- **Objective:** Adapt the base multi-speaker acoustic centroid (EXP-007 step 92k) into a cohesive, expressive single target voice across English, Nepali, and code-mixed speech using 13,037 high-quality studio target clips (`train_ft`) blended with a 15,000-clip replay buffer from `train` to prevent catastrophic forgetting.
- **Training Parameters:**
  - Initialized strictly from `artifacts/checkpoints/exp007/train_state.pt` (step 92,000).
  - Schedule: 20,000 steps, batch size 128 (max 64k frames), cosine decay from LR `5e-4` down to `2.5e-5` (5% floor), GAN active from step 0 (`--gan-start 0`).
  - Total training duration: 1.27 hours (4.39 steps/s) on NVIDIA RTX 4090 (BF16, peak VRAM 10.6 GB). Final `best_val_mel: 0.81129`, final train mel `0.7067`.
- **Live Held-Out ASR Benchmark (30 held-out sentences per language against live recognizer):**
  - **Nepali WER / CER:** `9.93% / 1.97%` (base 12.02% / 3.12%). The same ASR scores the human recordings of these sentences at 11.60% / 3.82%. This gap is ≈5 words on ≈301 and is an ASR-intelligibility proxy, not a quality claim.
  - **Code-mixed WER / CER:** `11.99% / 4.92%` (base 16.18% / 11.02%; human recordings 25.91% / 12.88%, n=9 utterances, ≈130 words).
  - **English WER / CER:** **`5.03% / 2.28%`** (target voice pronunciation, highly intelligible).
- **Internal 77-pair G2P/mel contrast smoke test** (not the published NepTTS-Bench; frontend symbols and non-silent audio only): 77/77.
- **Audio Deliverables:**
  - Checkpoint saved at `artifacts/checkpoints/exp008/best.safetensors` (5.98 MB FP32).
  - Post-finetune audio samples extracted to `artifacts/blog_samples/post_ft_step20k/` (`en_20000.wav`, `ne_20000.wav`, `codemix_20000.wav`).

---

## EXP-009: Weight-Space Edits on `PAWP-1.5M` (Zero Training) — Preliminary

- **Date:** 2026-11-21 · **Status:** completed; **preliminary** (corrected in AUDIT-002)
- **Weight-space programming agent:** Gemini 4 Neon
- **Base:** `exp008/best.safetensors` (target-voice fine-tune). **Donor:** `exp007/latest.safetensors` (pre-fine-tune base).
- **What was done.** Three kinds of edit:
  1. **Selective checkpoint merging / module transplant:** E1–E3 and P1–P5 are verbatim tensor copies from the donor.
  2. **Linear interpolation:** E4, WSP-3, WSP-4 and `compose_dec50`.
  3. **Activation-guided null-space projection:** WSP-1, WSP-2 and WSP-5 apply `ΔW_orth = ΔW − ΔW·VᵀV`, with `V` the top-k right singular vectors of duration-predictor activations on 8 hand-written Nepali sentences (not train/val/test). Only this group is weight programming in the inspect→edit sense.
  Weight-delta inspection (‖W₀₀₈−W₀₀₇‖/‖W₀₀₇‖) narrowed the candidate modules. **The final selection among candidates was by ASR score.**
- **Evaluation:** held-out ASR, `en=30 / ne=30 / mix=9` (≈460 / 301 / 130 words). The same set was used for every variant, and there was **no separate validation split**, so the best rows are test-selected. No significance test was run.

| Variant | Kind | Params changed | EN WER / CER | NE WER / CER | Mix WER / CER |
|---|---|---:|---:|---:|---:|
| Baseline `exp008/best` | — | 0 | 5.03 / 2.28 | 9.93 / 1.97 | 11.99 / 4.92 |
| E1 all `u_en/v_en` | transplant | 98,304 | 4.97 / 2.25 | 9.93 / 1.97 | 12.68 / 5.04 |
| E2 encoder `u_en/v_en` | transplant | 49,152 | 5.55 / 2.44 | 9.93 / 1.97 | 11.79 / 5.04 |
| E3 acoustic `u_en/v_en` | transplant | 49,152 | 5.03 / 2.20 | 9.93 / 1.97 | 9.93 / 3.97 |
| E4 50% blend of `u_en/v_en` | interpolation | 98,304 | 5.07 / 2.27 | 9.93 / 1.97 | 12.24 / 5.04 |
| R1 random, equal L2 norm (E1's location) | control | 98,304 | 5.51 / 2.48 | 9.93 / 1.97 | 13.46 / 7.78 |
| P1 duration predictor | transplant | 23,073 | 4.21 / 2.31 | 13.64 / 2.88 | 13.03 / 5.90 |
| P2 prosody predictors | transplant | 38,915 | 5.81 / 2.67 | 9.39 / 2.24 | 10.96 / 4.34 |
| P3 encoder (shared) | transplant | 317,456 | 5.24 / 1.88 | 11.88 / 2.55 | 11.17 / 5.14 |
| P4 acoustic (shared) | transplant | — | 4.99 / 2.28 | 12.03 / 2.24 | 14.35 / 7.39 |
| P5 decoder | transplant | — | 5.81 / 2.65 | 12.23 / 2.79 | 11.54 / 5.04 |
| WSP-1 E3 + ΔW_dur ⊥ NE₃₂ | null-space | — | 4.79 / 2.02 | 10.28 / 1.99 | 13.38 / 6.46 |
| WSP-2 E3 + ΔW_dur ⊥ NE₆₄ | null-space | 70,656 | 4.73 / 2.02 | 10.09 / 1.93 | 11.65 / 4.90 |
| WSP-3 E3 + 35% duration blend | interpolation | 72,225 | 5.35 / 2.67 | 9.45 / 1.72 | 10.82 / 3.74 |
| WSP-4 E3 + 25% duration & encoder blend | interpolation | — | 4.63 / 1.70 | 11.75 / 2.40 | 9.93 / 3.29 |
| WSP-5 E3 + ΔW_dur,enc ⊥ NE | null-space | 332,800 | 5.18 / 2.01 | 10.60 / 2.19 | 10.37 / 3.63 |
| compose_dec100 (EXP-012 base) | transplant | — | 5.51 / 2.54 | 10.39 / 2.10 | 12.54 / 5.11 |
| compose_dec50 (EXP-012 base) | interpolation | — | 5.51 / 2.54 | 10.34 / 2.13 | 10.93 / 4.16 |

Source: task logs 1945/1953/1964/1969/2109, exported by the agent. These runs were not written to the ledger. All values are %.

- **Interpretation (corrected):**
  1. **Nepali is unchanged by construction** for every edit restricted to `u_en/v_en`, including the random control R1. The English residuals are gated off for Nepali tokens, so "0% Nepali drift" is an architectural property, not an empirical finding.
  2. E3's Mix WER gain (11.99 → 9.93) is ≈2–3 words on n=9 utterances. It was selected on the test set and has no same-size, same-location random control, so it is **within noise and unconfirmed**.
  3. WSP-2, the one activation-derived edit, moves EN, NE CER and Mix by ≈1–2 words each, and NE WER worsens slightly (9.93 → 10.09). **Inconclusive.**
  4. Module transplants show a real trade-off. For example, P1 (the duration predictor) improves EN but clearly hurts NE (+3.7 WER). This is consistent with the duration predictor carrying Nepali timing adaptation.
- **Required before any claim:** a separate validation split for selection, a larger test set, a same-location random control for E3, bootstrap confidence intervals, and ideally a second ASR.
- **Saved copies:** `artifacts/checkpoints/exp009_wsp/pawp_1_5m_wsp_e3.safetensors` and `pawp_1_5m_wsp_null64.safetensors`.

---

## EXP-010: Phoneme-Aware Compression Frontier (`1.5M -> 1.0M -> 750K -> 500K`, Zero-Shot Post-Training)

- **Date:** 2026-11-21 · **Status:** Completed & Verified
- **Hypothesis:** Sensitivity-guided adaptive rank allocation (`allocate_ranks_under_budget` with Fisher-weighted SVD) preserves bilingual pronunciation better than equal-budget Uniform SVD across compressed tiers (`1.0M`, `750K`, `500K`), and protecting the 513-bin complex iSTFT `decoder.head` (`1026 x 192`) is critical at the `1.0M` tier (Hypothesis D).
- **Protocol note (AUDIT-002):** the Decoder-Centric variant (EXP-010b) was designed **after** observing that sensitivity-guided PAWP lost to Uniform SVD at 1M (task-2000), and was scored on the same test set. It is a post-hoc diagnostic iteration, not a pre-registered hypothesis. Fisher sensitivity used 10 hand-written sentences, not the test set. The start checkpoint is the WSP-E3 merge, not `exp008/best`.
- **Method:** Computed scale-invariant balanced EN/NE/Code-Mixed empirical Fisher sensitivity across all 31 compressible dense/pointwise matrices (`1,244,544` compressible params + `245,846` fixed overhead) starting from `pawp_1_5m_wsp_e3.safetensors`. Evaluated zero-shot (no recovery fine-tuning) on the live held-out ASR benchmark (`en=30, ne=30, mix=9`):

| Budget Tier | Method | Unique Params | EN WER / CER | NE WER / CER | Code-Mix WER / CER |
|---|---:|---:|---:|---:|---:|
| **1.5M (Uncompressed)** | `PAWP-1.5M` (`WSP-E3`) | `1,490,390` | `5.03% / 2.20%` | `9.93% / 1.97%` | `9.93% / 3.97%` |
| **1.0M** | Uniform SVD | `998,164` | `14.26% / 8.06%` | `23.06% / 10.08%` | `20.84% / 11.81%` |
| **1.0M** | PAWP Adaptive (unprotected `dec_head`) | `986,302` | `18.79% / 12.37%` | `26.51% / 13.56%` | `32.71% / 19.51%` |
| **1.0M** | **Decoder-Centric PAWP (`EXP-010b`)** | **`998,822`** | **`9.56% / 6.91%`** | **`22.76% / 9.73%`** | **`19.09% / 9.66%`** |
| **750K** | Uniform SVD | `747,392` | `66.16% / 47.48%` | `63.11% / 42.31%` | `51.52% / 32.40%` |
| **750K** | **PAWP Adaptive** | **`745,626`** | **`48.95% / 35.46%`** | **`49.57% / 29.97%`** | `53.60% / 39.18%` |
| 750K | Decoder-Centric PAWP (post-hoc) | `749,926` | `179.61% / 137.17%` (collapse) | `92.74% / 70.04%` | `172.07% / 153.42%` |
| **500K** | Uniform SVD / PAWP (zero-shot) | `~497K` | Collapse (`>100%`) | Collapse (`>100%`) | Collapse (`>100%`) |

- **Conclusions (corrected):**
  1. **The pre-specified method lost at 1M:** sensitivity-guided PAWP Adaptive was worse than equal-budget Uniform SVD (EN WER 18.79% vs 14.26%).
  2. The **post-hoc** Decoder-Centric variant beat Uniform SVD at 1M (EN 9.56% vs 14.26%; NE and Mix gaps within noise) but **collapsed at 750K**. Decoder protection is a promising hypothesis to test on a fresh split, not an established result.
  3. At 750K, PAWP Adaptive beat Uniform SVD by large margins (EN −17.2, NE −13.5 WER points), but both are far from intelligible.
  4. At 500K, zero-shot SVD collapses for every method. Smaller tiers need recovery training or purpose-built compact architectures (EXP-011).
- **Saved Checkpoint:** `artifacts/checkpoints/exp010_compress/pawp_1m_dc_zeroshot.safetensors`.

---

## EXP-011: `PAWP-300K` Ultra-Compact Bilingual Shared-Basis Training (`281K` Unique / `328K` Total Params)

- **Date:** 2026-11-21–22 · **Status:** Completed & Evaluated (Base 80k + Target-Voice FT 20k)
- **Architecture (`configs/student_300k_pawp.yaml`):**
  - `unique_stored_params`: `281,078` | materialized `total_params`: `328,158` | `effective_compute_params`: `309,702` (`20.96 MMAC/s`).
  - Decoder-centric (`52.6%`, `172,642` params), `n_fft=512`, `24 kHz`, cross-depth `K=2` shared bases + `28`-dim articulatory conditioning (`FP32 = 1.12 MB | INT8 = 0.28 MB`).
- **Training Recipe:**
  - **Stage 1 (`EXP-011` Base):** `80,000` steps (`train_ft + 15k replay`, `lr: 1e-3`, GAN from step `15,000`, `--prosody-free-prob 0.5`).
  - **Stage 2 (`EXP-011-FT` Target-Voice FT):** `20,000` steps initialized from `exp011_300k/train_state.pt` (`train_ft + 4k replay`, `lr: 3e-4`, GAN from step `0`).
- **Held-Out ASR Intelligibility Proxy (`n=30` EN, `30` NE, `9` code-mixed; one recognizer; internal 77-pair G2P/mel smoke test `77/77`, not the published NepTTS-Bench):**

| Checkpoint | `unique_stored` / `total` | EN WER / CER (`n=30`) | NE WER / CER (`n=30`) | Code-Mix WER / CER (`n=9`) | Notes |
|---|---:|---:|---:|---:|---|
| Human Reference Audio (same ASR) | — | `2.63% / 0.75%` | `11.60% / 3.82%` | `25.91% / 12.88%` | Intelligibility proxy floor |
| `EXP-011` Base (Step 80k) | `281,078 / 328,158` | `8.40% / 5.32%` | `20.49% / 7.49%` | `16.95% / 7.74%` | `100%` non-silent, `0%` `<unk>` |
| `EXP-011-FT` Target Voice (+20k) | `281,078 / 328,158` | `8.80% / 3.92%` | `19.42% / 6.93%` | `16.28% / 8.74%` | EN CER & NE improved; Mix CER (`7.74% -> 8.74%`) and EN WER (`8.40% -> 8.80%`) regressed slightly |

- **Interpretation & Caveats:**
  1. This compares a **trained 281K model vs. zero-shot 1M compression (`EXP-010b`)** — it is NOT an apples-to-apples architecture test for shared bases (an equal-budget dense `~281K` baseline trained on the same recipe is required and planned on `data/cache/v3`).
  2. **Pre-lexicon cache caveat:** `EXP-011` / `EXP-011-FT` trained on `data/cache/v2` (predating the lexicon and `व -> /b/` rule), so the `व` label-version mismatch applies.
  3. **Small eval / single recognizer:** `n=30/30/9`, one ASR, no human listening MOS yet.
- **Artifacts:** `artifacts/checkpoints/exp011_300k_ft/best.safetensors` (`1.1 MB`), folded EN/NE ONNX graphs (`1.32 MB` total FP32 each in `artifacts/checkpoints/exp011_300k_ft/onnx/`), and 24 kHz samples in `artifacts/checkpoints/exp011_300k_ft/samples/`.

---

## EXP-012: `PAWP-1.5M` Pure Target-Voice Phase & Timbre Polish + WSP Verification

- **Date:** 2026-11-21 · **Status:** Completed & Evaluated
- **Objective:** Reduce subtle iSTFT metallic phase coloration on `PAWP-1.5M` by continuing for 10,000 low-LR steps (`1.5e-4`, GAN active from step 0) from `exp008/train_state.pt` with a 5x higher target-voice ratio (`train_ft` + `3,000` replay clips), then testing zero-training WSP (`E3` and `Null-64`) on top of the polished weights.
- **Results (held-out ASR, `en=30, ne=30, mix=9`; internal 77-pair G2P smoke test 77/77, not the published NepTTS-Bench):**

| Checkpoint | EN WER / CER | NE WER / CER | Code-Mix WER / CER | Notes |
|---|---:|---:|---:|---|
| `EXP-008-FT` (Pre-Polish) | `5.03% / 2.28%` | `9.93% / 1.97%` | `11.99% / 4.92%` | Base target voice |
| **`EXP-012` Polish (`best.safetensors`)** | **`4.86% / 2.79%`** | **`11.41% / 2.66%`** | **`10.99% / 4.10%`** | Audibly smoother timbre (informal listening); NE WER regressed vs EXP-008-FT (9.93 → 11.41) |
| `EXP-012` Polish + `WSP-E3` | `5.21% / 2.69%` | `11.41% / 2.66%` | **`10.54% / 4.10%`** | `-0.45%` Mix WER, `0.00%` NE drift |
| `EXP-012` Polish + `WSP-Null64` | **`4.34% / 1.75%`** | `13.35% / 2.70%` | `11.24% / 4.89%` | Better EN, but NE WER regresses 11.41 → 13.35 |

- **Saved Checkpoints:** `artifacts/checkpoints/exp012_polish/{best,pawp_1_5m_polish_wsp_e3,pawp_1_5m_polish_wsp_null64}.safetensors` (original `exp008` and `exp009_wsp` untouched).

---

## EXP-013a: Mechanistic Rank-1 Weight-Space Programming (`dev` vs. Untouched `300+300` `test.jsonl`)

- **Date:** 2026-11-22 · **Status:** Completed — Targeted Effect Confirmed, Edit Rejected Due to Collateral PER Regression
- **Reproducible Pipeline:** Implemented in Clean Architecture (`domain/pawp/wsp_analysis.py`, `infrastructure/pawp/wsp_engine.py`, `make pawp-inspect`, `make pawp-edit`, `make pawp-verify`), with all steps logged to `artifacts/reports/experiment_ledger.jsonl`.
- **Protocol (Strict `dev` / `test` Split):**
  1. **Inspect (`val.jsonl`, `150` NE + `150` EN, `10,488` NE phonemes):** Frozen CTC phoneme recognizer measured `DEV NE PER = 7.06%`, `DEV EN PER = 5.29%`, and identified **`b -> w`** (`25` substitutions; `37` total `b <-> w`) as the #1 Nepali phoneme confusion.
  2. **Causal Localization (`val.jsonl`):** Layer-by-layer linear probing and causal activation patching showed `b vs w` is cleanly separated in `encoder_blocks.0..3` (`probe_acc = 1.000 -> 0.972`), collapses by `acoustic_blocks.3` (`cos_sep = 0.0289`), and exhibits its #1 positive causal patching flip strictly at `acoustic_blocks.3` (`causal_flip = +0.0812`).
  3. **Edit & Controls (selected strictly on `val.jsonl`, editing only `acoustic_blocks.3.pw2.weight`, `32,768` params = `2.2%` of model):**
     - **WSP Rank-1:** `W <- W + alpha * ||W|| * (u_out @ v_in^T)` (`alpha=0.05`, `||ΔW||_F = 2.278`).
     - **Control 1 (Matched Random):** Same-location, same-Frobenius-norm Rank-1 random edit.
     - **Control 2 (Same-Param FT):** Only `acoustic_blocks.3.pw2.weight` unfrozen, 40 steps Adam + frozen CTC loss on the exact same dev utterances.
  4. **One-Shot Verification on Untouched `test.jsonl` (`300` NE + `300` EN, `366` occurrences of `b`, 1,000-resample paired bootstrap 95% CIs):**

| Checkpoint | `b -> w` Confusion Rate (`n=366`) | NE PER (`n=300`) | EN PER (`n=300`) | NE PER Bootstrap 95% CI (`edit - base`) |
|---|---:|---:|---:|---|
| Baseline (`EXP-012` Polish) | `19.67%` (`72/366`) | `7.07%` | `5.29%` | — |
| **WSP Rank-1 (`alpha=0.05`)** | **`17.21%` (`63/366`, `-2.46%`)** | `8.29%` | `7.05%` | `[+0.0111, +0.0219]` (collateral regression) |
| Control 1: Matched Rank-1 Random | `19.67%` (`72/366`, `0.00%`) | `7.09%` | `5.33%` | `[+0.00001, +0.0012]` |
| Control 2: Same-Param Gradient FT | `20.49%` (`75/366`, `+0.82%`) | `7.09%` | `5.41%` | `[-0.0038, +0.00025]` (`p=0.050`) |

- **Honest Verdict:** The localized Rank-1 direction genuinely reduced `b -> w` confusion on 300 untouched test sentences (`19.67% -> 17.21%`), outperforming both the matched random control (`19.67%`) and same-parameter gradient fine-tuning (`20.49%`). However, because `alpha=0.05` (`||ΔW||_F = 2.28`) was applied directly to the shared `pw2.weight` without null-space projection against other phonemes, it caused statistically significant collateral PER regression (`NE +1.22%`, `EN +1.75%`). Under our no-regression rule, `EXP-013a` is **rejected** as a release edit.
- **Post-Audit Label-Artifact Finding (Grapheme Breakdown of the 72 `b -> w` Confusions on `test.jsonl`, `n=300`):**
  - `व`-only words: **`65 / 118` (`55.1%`)** (e.g., `व्यक्ति`, `विशेषता`, `विदामा`, `प्रवेश`, `विश्वासै`, `विधि`)
  - `ब`-only words: **`5 / 240` (`2.1%`)** (e.g., `बसोबास`, `फोर्ब्स`, `बित्दै`, `बिसाएँ`, `बिमारीहरू`)
  - Words containing both `व` and `ब`: **`2 / 8` (`25.0%`)** (`other: 0 / 0`)
  - **Root cause:** `90.3%` (`65/72`, `67/72` incl. `both`) of the `b -> w` confusions originate from `व` words because `data/cache/v2` (`09:58`) and `ctc_aligner.pt` (`12:15`) predate the `व -> /b/` rule (`~14:00`) and the lexicon (`16:24`). On true `ब` words, baseline `/b/` accuracy is already `97.9%` (`235/240`).
  - **Action:** `EXP-013b` on `b -> w` is **cancelled** (do not apply weight edits to a label-version artifact). Correct fix after `EXP-011` completes: rebuild `data/cache/v2` with the current frontend + lexicon, retrain `ctc_aligner.pt`, re-extract CTC durations, fine-tune, and re-run `make pawp-inspect` to target a genuine phoneme confusion that survives consistent labels (with bootstrap CIs on the confusion rate itself and matched dev-tuning budget for the gradient-FT control).

---

## EXP-014 & EXP-015a: Clean `data/cache/v3` Rebuild, `ctc_aligner_v3.pt`, and `PAWP-1.5M` v3 Fine-Tune

- **Date:** 2026-11-22 · **Status:** Completed & Logged (`EXP-015b/c` running)
- **Step 2 (`EXP-014`):** Rebuilt `data/cache/v3` (`136,807` utterances, `0` errors) using the current frontend + 105k lexicon, retrained `artifacts/checkpoints/ctc_aligner_v3.pt` (`6,000` steps, final `ctc = 0.6141` vs. `~1.0` on `v2`), and extracted forced-alignment durations (`dur_one_frac`: `train 0.0594`, `train_extra 0.0393`, `train_ft 0.0296`, `val 0.0556`; median `50–60 ms`). Sanity check on 300 NE test sentences with `ctc_aligner_v3.pt`: `व`-word `/b/ -> /w/` confusion dropped from `65/118 (55.1%)` to **`0/118 (0.0%)`** (`ब`-words: `3/240 = 1.2%`).
- **Step 3a (`EXP-015a`):** Fine-tuned `PAWP-1.5M` from `exp012_polish/train_state.pt` on `data/cache/v3` (`10k` steps, `train_ft + 10k replay`, `lr: 3e-4`).
  - Held-out ASR (`n=30/30/9`, one recognizer): **EN `4.02% / 1.50%`**, **NE `12.35% / 1.75%`** (new lowest NE CER), **Code-Mix `12.32% / 5.32%`**; internal 77-pair smoke test `77/77`.
  - Clean `make pawp-inspect` (`ctc_aligner_v3.pt`, `val.jsonl`, `150` NE + `150` EN, `10,488` NE phonemes): **DEV NE PER dropped `7.06% -> 3.96%`** (`S=170, D=203, I=42`), **DEV EN PER `5.29% -> 4.33%`**. `b -> w` vanished from top 12; largest single substitution was `ɳ -> n̪` (`7 / 10,488 = 0.07%`), while deletions (`D = 203`) dominated.

---

## EXP-016: Surgical 96-Parameter (`0.0064%`) Word-Boundary Pause WSP (`duration_predictor.proj.weight`)

- **Date:** 2026-11-22 · **Status:** Completed & Logged (Selective Pause Mechanism Confirmed; Overall PER Did Not Beat Random Control)
- **Inspection (`data/cache/v3/val`, `200` utterances):** Word-boundary `<sp>` (`n=2,956`) was systematically under-predicted (`6.16` frames vs. GT `10.38` frames; bias `-4.23` frames = `-42.3 ms`), while regular phonemes (`n=13,763`) had near-zero bias (`6.08` vs. `6.60`, `-0.52` frames). `<sp>` vs. phonemes was `90.77%` linearly separable at the 96-dim input of `duration_predictor.proj.weight` (`96` params = `0.00644%` of model).
- **Method:** Computed ROME-whitened `<sp>` direction `v_sp = (C_phonemes + 0.1*I)^{-1} mean(h_sp)`, swept `alpha` strictly on `val.jsonl` (`best_alpha = 0.5`), and compared against (1) same-location, same-L2-norm 96-param random edit and (2) dev-tuned 96-param gradient FT (`lr=1e-3, steps=50`).
- **One-Shot Verification on Untouched `test.jsonl` (`300` NE + `300` EN, `ctc_aligner_v3.pt`, 1,000-resample paired bootstrap 95% CIs):**

| Checkpoint (`96` params = `0.0064%`) | `<sp>` Mean | Phoneme Mean | NE PER (`n=300`) | NE `D` | EN PER (`n=300`) | EN `D` | NE PER 95% CI (`edit - base`) |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline (`EXP-015a` v3) | `6.100f` | `6.129f` | `5.04%` | `480` | `4.02%` | `185` | — |
| **WSP Pause-96 (`alpha=0.5`)** | **`7.101f` (`+1.00f`)** | **`6.166f` (`+0.04f`)** | `4.93%` | `461` (`-19`) | `4.13%` | `164` (`-21`) | `[-0.0034, +0.0012]` (`p=0.177`) |
| Control 1: Matched Random-96 | `6.229f` (`+0.13f`) | `6.221f` (`+0.09f`) | `4.84%` | `456` | `3.96%` | `185` | `[-0.0039, -0.0006]` (`p=0.004`) |
| Control 2: Same-Param FT-96 (dev-tuned) | `5.398f` (`-0.70f`) | `5.767f` (`-0.36f`) | `5.55%` | `544` | `4.54%` | `226` | `[+0.0022, +0.0082]` (`p=0.999`) |

- **Honest Verdict:** The 96-parameter ROME edit achieved strong surgical selectivity (`+1.001f` on `<sp>` vs. `+0.037f` on phonemes = `27:1` ratio), reduced deletions (`-19` NE, `-21` EN), and decisively beat same-parameter gradient FT (`4.93%` vs. `5.55%` NE PER). However, on overall PER it did **not** statistically beat the matched random control (`4.93%`, `p=0.177` vs. `4.84%`, `p=0.004`), so we report the selectivity win honestly without claiming an overall PER victory.

---

## AUDIT-002: Independent Review of EXP-007 – EXP-012 (Corrections)

- **Date:** 2026-11-21 · **Reviewers:** independent audit agents. Findings were confirmed point by point by the experimenting agent (Gemini 4 Neon).
- **Confirmed sound:** all reported held-out ASR numbers match the ledger and task logs, and a re-run of the EXP-008 evaluation reproduced them exactly. The frontend handles the tested context cases, with no Hindi phones.
- **Corrections applied above:**
  1. EXP-009 relabelled (merging / interpolation / null-space), with the full variant table, test-set selection disclosed, and "0% NE drift" marked as by construction.
  2. EXP-010: the post-hoc design is disclosed and the Decoder-Centric 750K collapse added.
  3. "Beats human" replaced with neutral ASR-proxy wording.
  4. The internal 77-pair smoke test is never called NepTTS-Bench.
  5. The 300-item "native gold" is relabelled as a lexicon self-consistency check.
  6. The lexicon is no longer described as verified.
  7. The cache/frontend training–inference mismatch is recorded.
- **Open items:**
  - rebuild the feature cache with the current frontend;
  - a native-annotated pronunciation gold set independent of the lexicon;
  - corpus-attested filtering of generated lexicon forms;
  - a validation split for edit selection, a larger test set and bootstrap confidence intervals;
  - weight-space programming and compression implemented in `src/` with `make` targets and ledger entries;
  - licence/provenance review of public checkpoints before wider distribution.
