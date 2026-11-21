# Novelty Analysis & Falsification Matrix

## Explicit Separation: Established vs. Novel

### Established Techniques (No Novelty Claimed)
- Non-autoregressive duration/pitch/energy variance adaptors (`FastSpeech 2`, `FastPitch`).
- ConvNeXt1D + iSTFT Fourier-domain vocoding (`Vocos`) and FARGAN linear-prediction vocoding (`PiVo-1.4M`).
- Global Fisher-weighted SVD (`FWSVD`), unweighted cross-layer SVD basis sharing (`Basis Sharing`), uniform shared+low-rank reparameterization (`ResidualTransformer`), adaptive LoRA rank pruning (`AdaLoRA`), unstructured dynamic sparse training (`RigL`), and articulatory phonological embeddings (`IMS Toucan`).

- Weight-space programming as a concept: an AI agent reads a network's weights like a program and edits them without gradients ([Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), 2026; demonstrated on LLMs, RL policies and diffusion models). Related: knowledge editing (ROME/MEMIT), activation steering, and mechanistic interpretability.

### Core Hypothesis: Agent-Driven Weight Programming for Compact Bilingual TTS

| Proposed Contribution | Closest Prior Work | Delta | Evidence Required to Support vs. Falsify |
|---|---|---|---|
| **0. Agent-driven, measurement-guided weight edits that repair phonetic failures in a compact TTS model** | Weight-space programming (LLM/RL/diffusion), ROME/MEMIT, activation steering | Applies inspect → diagnose → edit → verify to *speech* failures: dental/retroflex confusion, Hindi /ʃ/ bleed, code-mix timbre seams, duration errors. Uses structured edit operators (direction removal, rank-1 updates, residual-subspace transplant, basis-coefficient edits) on a fixed forward-pass budget | Edits must improve the targeted metric (minimal-pair accuracy, PER, speaker similarity) on **held-out** data without regressions. They must beat random edits of equal size and match or beat equal-budget gradient fine-tuning. Falsified if gains vanish on held-out data or fail to beat gradient baselines. Note: this applies only to a model that already speaks intelligibly; editing cannot add missing ability |

### Proposed PAWP Contributions & Experimental Validation Criteria

| Proposed PAWP Contribution | Closest Prior Work | Mathematical / Experimental Delta | Evidence Required to Support vs. Falsify |
|---|---|---|---|
| **1. Frequency-Normalized Phonetic-Sensitivity Rank Allocation** | `FWSVD` (ICLR 2022), `AdaLoRA` (ICLR 2023), `SPADE` (2025) | Computes category-normalized diagonal Fisher/Taylor sensitivity over frame-aligned phonetic contrasts $\mathcal{L}_p = \frac{1}{\|T_p\|} \sum_{t \in T_p} \ell_t$ instead of global scalar loss (where frequent vowels drown out rare stops/fricatives) or coarse whole-layer WER dropping | Must outperform equal-budget uniform SVD, global FWSVD, and AdaLoRA on NepTTS-Bench minimal-pair accuracy & EN `/s/-/ʃ/, /t/-/θ/` PER at matched total params (`1M`, `750K`, `500K`, `300K`) |
| **2. Shape-Grouped Bilingual Shared Bases + Token/Frame-Gated Residuals** | `Basis Sharing` (ICLR 2025), `ResidualTransformer` (2023), multilingual adapters | Pools shape-compatible ConvNeXt1D projections across depth into shared bases $\sum_k a_{lk} B_k$ paired with deterministic frontend-gated EN/NE low-rank residuals $m_{\text{en}}(t) \Delta W_{\text{en}} + m_{\text{ne}}(t) \Delta W_{\text{ne}}$ (statically foldable for zero-overhead monolingual inference) | Must beat two half-budget monolingual models, a fully shared bilingual model, and bottleneck adapters on cross-lingual speaker cosine similarity and code-mix boundary smoothness at matched params |
| **3. Structured Phonetic Error-Driven Regrowth under Fixed Budget** | `RigL` (ICML 2020), dynamic sparse training | Replaces unstructured weight-magnitude/gradient masks with pronunciation-error-targeted singular-subspace restoration in failing layers balanced by trimming equivalent capacity from over-provisioned layers so $\sum_l P_l \le P_{\text{budget}}$ stays invariant | Must beat one-shot FWSVD, static pruning, and unstructured RigL on post-compression minimal-pair PER recovery and measured CPU ONNX latency |
| **4. Cross-Lingual Sensitivity Transfer (EN $\leftrightarrow$ NE)** | Multilingual warm-starting & Fisher transfer | Tests whether English phonetic sensitivity maps predict structurally critical acoustic/decoder layers for Nepali (and vice versa) under constrained Nepali data | Must beat random/unweighted rank initialization when fine-tuned on 10% and 25% Nepali data subsets |
| **5. Joint Decomposition + Pronunciation-Constrained Distillation** | `SPADE`, Spotify TTS KD, MIT `prune-tts` | Co-optimizes shared bases, adaptive ranks, multi-resolution STFT, and frozen phoneme-CTC alignment together rather than post-hoc truncation | Must beat compress-before-distill and distill-then-compress pipelines across the `1M → 300K` Pareto frontier |

---

## Current Empirical Validation Status (Logged in `docs/EXPERIMENTS.md`)

| Contribution | Status | Verified Empirical Evidence | Remaining Gaps / Caveats |
| :--- | :--- | :--- | :--- |
| **0. Agent-Driven Weight-Space Programming** | **Demonstrated (`EXP-009`, Gemini 4 Neon)** | Zero-training surgical edit `WSP-E3` (`3.3%` params, `acoustic_blocks.*.u_en/v_en`) lowered held-out Code-Mixed WER from `11.99% -> 9.93%` (`-2.06%`) and CER `4.92% -> 3.97%` with mathematically `0.00%` Nepali drift (`9.93% / 1.97%`). Nepali-orthogonal null-space projection `WSP-Null64` (`4.7%` params) improved EN WER (`5.03% -> 4.73%`), EN CER (`2.28% -> 2.02%`), NE CER (`1.97% -> 1.93%`), and Mix WER (`11.99% -> 11.65%`) simultaneously, decisively beating equal-norm random control `R1` (`5.51%` EN, `13.46%` Mix) and naive blending `WSP-3`. | Held-out evaluation set is currently `n=30` EN, `n=30` NE, `n=9` Code-Mix; larger held-out eval + human listening MOS still pending. |
| **1. Phonetic-Sensitivity Rank Allocation** | **Partially Validated (`EXP-010`, `EXP-010b`)** | Zero-shot Decoder-Centric PAWP (`998.8K` params) beat equal-budget Uniform SVD across all three suites (`9.56% / 22.76% / 19.09%` vs. `14.26% / 23.06% / 20.84%`). At `750K`, PAWP Adaptive beat Uniform SVD by `-17.21%` EN WER and `-13.54%` NE WER. | Zero-shot one-shot SVD collapses at `500K` and sounds thin at `1.0M` without recovery training; AdaLoRA/global-FWSVD head-to-head still pending. |
| **2. Shape-Grouped Shared Bases + Gated Residuals** | **In Training (`EXP-011`)** | Language-gated residuals verified on `PAWP-1.5M` (`EXP-007`–`EXP-012`), enabling code-mixing and zero-drift monolingual ONNX folding (`cosine 0.987–0.99998`). Cross-depth `K=2` shared-basis `PAWP-300K` (`281,078` unique params, `0.28 MB` INT8) currently training (`80k` base + `20k` FT). | 5-way architecture ablation (monolingual halves vs. shared vs. adapters vs. PAWP) still pending. |
| **3–5. Regrowth, Cross-Lingual Transfer, Joint Distillation** | **Planned / Next** | `EXP-010` confirmed one-shot SVD cannot reach `<= 500K`, empirically motivating joint shared-basis training (`EXP-011`) and recovery distillation. | Experimental comparison pending `EXP-011` completion. |
