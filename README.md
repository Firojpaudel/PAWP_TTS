# PAWP-TTS

**Phoneme-Aware Weight Programming for Ultra-Compact Bilingual (English–Nepali) Neural Speech Synthesis**

> **Research repository.** This is an active research codebase, not a production TTS
> package. APIs, configs and checkpoints change between experiments, and every
> reported number is tied to a logged experiment in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).
> Numbers without a logged experiment should not be cited.

PAWP-TTS asks one question: *how few neural parameters does natural, streaming,
bilingual English–Nepali speech need, if those parameters are shared, decomposed and
allocated according to phonetic sensitivity?* The target is a complete deployed
inference graph, decoder/vocoder included, of **≤ 1.5M parameters**, followed by
controlled compression toward **300K**.

---

## Status

| Milestone | Scope | State |
|---|---|---|
| M1 | Clean Architecture bootstrap, unified EN/NE phoneme inventory, rule-based Nepali + code-mix G2P, 1.5M blueprint | done |
| M2 | Bilingual manifests and splits, eSpeak English G2P, ONNX chunked streaming path, MR-STFT + MPD/MRD losses | done |
| M3 | Shared-basis PAWP modules, full-scale training pipeline, ASR-based evaluation | **in progress**: first full-scale training run ([EXP-003](docs/EXPERIMENTS.md)) |
| M4 | Phonetic sensitivity maps, budget-constrained rank allocation, error-driven regrowth, 1M→300K sweep | planned |
| M5 | ONNX FP32/INT8 export and parity, multi-axis Pareto scorecard, release | planned |

No speech-quality results have been published yet. Intelligibility (ASR WER/CER), naturalness and latency
results will appear in the findings table below once they are measured.

## Findings so far

Only verified, reproducible observations are listed. Details and caveats are in the linked entries.

| # | Finding | Evidence |
|---|---|---|
| 1 | The full 1.5M baseline graph, **vocoder included**, has 1,490,390 parameters, of which the iSTFT decoder is 664,770 (45%). | `make count-params`, [AUDIT-001](docs/EXPERIMENTS.md) |
| 2 | Earlier evaluation metrics measured frontend symbol strings and non-silent audio, not speech quality. They were relabelled, and evaluation now uses ASR on synthesized vs. human audio. | [AUDIT-001](docs/EXPERIMENTS.md) |
| 3 | Heuristic phoneme durations were replaced by a training-only learned aligner (CTC forward-sum + beta-binomial prior + MAS). It adds **0 inference parameters**. | [EXP-003](docs/EXPERIMENTS.md) |
| 4 | MAS (monotonic alignment search) as a per-frame CUDA loop is launch-bound (132 ms/batch). A vectorised CPU version takes 12.5 ms (≈10×) with identical paths. | [EXP-003](docs/EXPERIMENTS.md) |
| 5 | Large single-row-group Parquet shards (>1 GB) silently OOM-killed feature workers and dropped ~27k Nepali clips. Streaming row batches fixed it. Data audits must count *outputs*, not just exit codes. | [EXP-003](docs/EXPERIMENTS.md) |
| 6 | On code-mixed speech, a native-script ASR inflates CER because it writes English words in Devanagari. A mixed-script recognizer is needed for fair code-mix scoring (small spot check, n=8). | [EXP-005](docs/EXPERIMENTS.md) |

## How it works

```text
text (EN / NE / code-mixed)
  └─ deterministic frontend: normalization → EN eSpeak G2P | NE rule G2P → code-mix span router
       └─ unified phonological tokens (phoneme, language, stress, boundaries)
            └─ phoneme + 28-dim articulatory embeddings
                 └─ ConvNeXt1D encoder with token-gated EN/NE low-rank residuals (optional shared bases)
                      └─ duration → length regulator → pitch/energy → frame acoustic network
                           └─ ConvNeXt1D + iSTFT decoder (24 kHz) → streaming PCM
```

- **Frontend** ([domain/phonology](src/pawp_tts/domain/phonology/)): Nepali rules keep native contrasts: alveolar affricates /ts tsʰ dz dzʱ/, स/श/ष → /s/, four-way laryngeal stops, dental vs. retroflex, gemination, intervocalic flapping, /ɦ/-deletion breathiness, and schwa syncope.
- **Acoustic model + decoder** ([infrastructure/neural](src/pawp_tts/infrastructure/neural/)): non-autoregressive, bounded lookahead, BF16 training.
- **Weight programming** ([infrastructure/weight_programming](src/pawp_tts/infrastructure/weight_programming/)): shared weight bases `W_l = Σ_k a_lk B_k + U_l V_l`, language-gated residuals with exact monolingual folding, and 4-metric parameter audits.
- **Training** ([infrastructure/training](src/pawp_tts/infrastructure/training/trainer.py)): learned alignment, masked losses, teacher-forced prosody, validation-selected checkpoints, and full resume. Training is multi-speaker; the exported graph keeps **one** speaker vector.
- **Evaluation** ([infrastructure/evaluation](src/pawp_tts/infrastructure/evaluation/)): ASR WER/CER on held-out sentences for synthesized vs. human audio, so recognizer errors are separated from TTS errors. Code-mixed audio is scored by a mixed-script recognizer.

## Code layout (Clean Architecture)

`interfaces → application → domain ← infrastructure`, enforced by `import-linter` in `make check`.

| Layer | Contents |
|---|---|
| [`domain/`](src/pawp_tts/domain/) | Entities, phonology rules, PAWP budget solvers, metrics, ports. No ML or I/O imports. |
| [`application/`](src/pawp_tts/application/) | Framework-free use cases (phonemization, audits, evaluation orchestration) |
| [`infrastructure/`](src/pawp_tts/infrastructure/) | PyTorch models, data/feature pipelines, trainer, ASR adapters, ONNX streaming, ledger |
| [`interfaces/`](src/pawp_tts/interfaces/) | DI container and the thin `pawp` CLI used by every `make` target |

## Quick start

```bash
cp .env.example .env     # fill in local paths/endpoints (never committed)
make setup               # uv environment (Python 3.12, PyTorch, dev tooling)
make check               # format, lint, mypy --strict, import-linter, unit tests
make test-integration    # model, training-loop and streaming integration tests
make help                # every workflow target with a one-line description
```

Typical research loop:

```bash
make teacher-extract EXP_ID=expNNN     # build the 24 kHz feature cache from manifests
make train EXP_ID=expNNN               # train (resumable; TRAIN_ARGS='...' for overrides)
make eval-heldout EXP_ID=expNNN        # ASR WER/CER: synthesized vs. human recordings
make eval-en eval-ne eval-codemix EXP_ID=expNNN
make count-params CONFIG=configs/student_1_5m.yaml EXP_ID=expNNN
```

Every run appends to `artifacts/reports/experiment_ledger.jsonl` (config hash, seed,
metrics, exact command). Datasets are symlinked under the git-ignored `data/sources/`,
never copied.

## Documentation

| Document | Purpose |
|---|---|
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Chronological experiment log: hypothesis, method, results, conclusions |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | Latency, footprint and quality benchmarks |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Tensor-level model and PAWP specification |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Architecture decision records |
| [docs/LITERATURE.md](docs/LITERATURE.md) | Related work |
| [docs/NOVELTY.md](docs/NOVELTY.md) | Closest prior work and claimed deltas for each contribution |

## Data and licensing notes

Training combines public corpora (e.g. LibriSpeech, LJSpeech, OpenSLR Nepali) with
additional Nepali and code-mixed material prepared for this research. Any released
checkpoint will document exactly which data it was trained on and its license terms.

## Scientific integrity

- Parameter counts always include the decoder/vocoder.
- TTFA, RTF and throughput are reported separately.
- No novelty claim is made until it is checked against the closest prior work (FWSVD, Basis Sharing, AdaLoRA, RigL, SPADE, ResidualTransformer); see [docs/NOVELTY.md](docs/NOVELTY.md).
- No "zero error" claim is made outside a defined evaluation set.
