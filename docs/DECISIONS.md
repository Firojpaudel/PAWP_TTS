# Architecture & Engineering Decision Records

## ADR-001: Clean Architecture with Automated Boundary Enforcement
- **Context:** Research codebases rapidly accumulate tight coupling between domain logic, PyTorch modules, dataset formats, and external APIs, preventing CPU unit testing and clean component swaps (e.g., iSTFT vs. FARGAN decoders, multiple G2P/ASR backends).
- **Decision:** Structure `src/pawp_tts/` into `domain`, `application`, `infrastructure`, and `interfaces`. `domain/` has zero third-party ML/I/O imports; `application/` depends solely on `domain/`; external integrations sit behind `typing.Protocol` ports wired in `interfaces/container.py`. Enforce automatically via `import-linter` and AST tests in `make arch-check`.
- **Consequences:** All phonological rules, error metrics, and PAWP knapsack allocation solvers run in milliseconds on CPU with zero GPU or PyTorch dependency.

## ADR-002: `24 kHz` ConvNeXt1D + iSTFT Primary Decoder with `16 kHz` FARGAN Stage-4 Ablation
- **Context:** Clean Nepali studio speech is natively `24 kHz`, whereas FARGAN (`PiVo-1.4M`) operates natively at `16 kHz`. Mixing `16 kHz` and `24 kHz` training audio without isolation risks acoustic bandwidth bias across languages.
- **Decision:** Target `24 kHz` (`hop=240`, `n_fft=1024`, `ConvNeXt1D + iSTFT`) as the primary Stage-3 student (`1,490,374` params) while preserving `16 kHz` FARGAN-style framewise AR decoding as an explicit matched-budget Stage-4 ablation, and auditing `16 kHz` vs. `24 kHz` English data alignment before bilingual joint training.

## ADR-003: Zero-Duplication Symlinks, Ephemeral Checkpoints & Open-Source Data Hygiene
- **Context:** Host primary storage has tight capacity limits (`~101 GB` free on primary volume) and the final model is intended for public open-source release.
- **Decision:** Never copy datasets into the repo—symlink raw/processed human corpora under git-ignored `data/sources/`. Keep at most latest + best checkpoints during active runs, prune obsolete artifacts via `make prune-artifacts` after logging results, and exclude proprietary/private synthetic datasets from any public training/release documentation.

## ADR-004: Unified EN/NE Phonology & Externalized ASR Ports
- **Context:** Hindi phonological assumptions (`ə`, postalveolar `tʃ/dʒ`, `ʃ` for `श/ष`) severely degrade native Nepali pronunciation, and hardcoded endpoints leak local environment config.
- **Decision:** Implement Nepali phonology directly in `domain/phonology/nepali_rules.py` (sibilant merger to `/s/`, alveolar affricates `/ts, tsʰ, dz, dzʰ/`, 4-way laryngeal stop contrast, gemination, intervocalic retroflex flapping, morpho-syntactic schwa syncope). Externalize ASR evaluation endpoints exclusively through `.env` variables consumed by `HttpMultipartAsrAdapter`.
