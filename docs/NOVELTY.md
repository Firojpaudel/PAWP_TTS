# Novelty Analysis & Falsification Matrix

## Explicit Separation: Established vs. Novel

### Established Techniques (No Novelty Claimed)
- Non-autoregressive duration/pitch/energy variance adaptors (`FastSpeech 2`, `FastPitch`).
- ConvNeXt1D + iSTFT Fourier-domain vocoding (`Vocos`, `sanoTTS`) and FARGAN linear-prediction vocoding (`PiVo-1.4M`).
- Global Fisher-weighted SVD (`FWSVD`), unweighted cross-layer SVD basis sharing (`Basis Sharing`), uniform shared+low-rank reparameterization (`ResidualTransformer`), adaptive LoRA rank pruning (`AdaLoRA`), unstructured dynamic sparse training (`RigL`), and articulatory phonological embeddings (`IMS Toucan`).

### Proposed PAWP Contributions & Experimental Validation Criteria

| Proposed PAWP Contribution | Closest Prior Work | Mathematical / Experimental Delta | Evidence Required to Support vs. Falsify |
|---|---|---|---|
| **1. Frequency-Normalized Phonetic-Sensitivity Rank Allocation** | `FWSVD` (ICLR 2022), `AdaLoRA` (ICLR 2023), `SPADE` (2025) | Computes category-normalized diagonal Fisher/Taylor sensitivity over frame-aligned phonetic contrasts $\mathcal{L}_p = \frac{1}{\|T_p\|} \sum_{t \in T_p} \ell_t$ instead of global scalar loss (where frequent vowels drown out rare stops/fricatives) or coarse whole-layer WER dropping | Must outperform equal-budget uniform SVD, global FWSVD, and AdaLoRA on NepTTS-Bench minimal-pair accuracy & EN `/s/-/ʃ/, /t/-/θ/` PER at matched total params (`1M`, `750K`, `500K`, `300K`) |
| **2. Shape-Grouped Bilingual Shared Bases + Token/Frame-Gated Residuals** | `Basis Sharing` (ICLR 2025), `ResidualTransformer` (2023), multilingual adapters | Pools shape-compatible ConvNeXt1D projections across depth into shared bases $\sum_k a_{lk} B_k$ paired with deterministic frontend-gated EN/NE low-rank residuals $m_{\text{en}}(t) \Delta W_{\text{en}} + m_{\text{ne}}(t) \Delta W_{\text{ne}}$ (statically foldable for zero-overhead monolingual inference) | Must beat two half-budget monolingual models, a fully shared bilingual model, and bottleneck adapters on cross-lingual speaker cosine similarity and code-mix boundary smoothness at matched params |
| **3. Structured Phonetic Error-Driven Regrowth under Fixed Budget** | `RigL` (ICML 2020), dynamic sparse training | Replaces unstructured weight-magnitude/gradient masks with pronunciation-error-targeted singular-subspace restoration in failing layers balanced by trimming equivalent capacity from over-provisioned layers so $\sum_l P_l \le P_{\text{budget}}$ stays invariant | Must beat one-shot FWSVD, static pruning, and unstructured RigL on post-compression minimal-pair PER recovery and measured CPU ONNX latency |
| **4. Cross-Lingual Sensitivity Transfer (EN $\leftrightarrow$ NE)** | Multilingual warm-starting & Fisher transfer | Tests whether English phonetic sensitivity maps predict structurally critical acoustic/decoder layers for Nepali (and vice versa) under constrained Nepali data | Must beat random/unweighted rank initialization when fine-tuned on 10% and 25% Nepali data subsets |
| **5. Joint Decomposition + Pronunciation-Constrained Distillation** | `SPADE`, Spotify TTS KD, MIT `prune-tts` | Co-optimizes shared bases, adaptive ranks, multi-resolution STFT, and frozen phoneme-CTC alignment together rather than post-hoc truncation | Must beat compress-before-distill and distill-then-compress pipelines across the `1M → 300K` Pareto frontier |
