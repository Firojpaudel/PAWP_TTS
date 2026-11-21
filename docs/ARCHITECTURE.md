# Architecture & Weight-Space Programming Specification

## 1. End-to-End Signal Flow

```text
TEXT (EN / NE / Code-Mixed)
  │
  ▼
Deterministic Prompt Sanitizer & Script-Span Router (domain/phonology/codemix.py)
  ├──► English Normalization + G2P (CMUdict / rules / eSpeak NG)
  └──► Nepali Akshara Parser + Algorithmic Rules (sibilant merger, 4-way stops, gemination, schwa syncope)
  │
  ▼
Unified Phonological Sequence (<= 128 IPA symbols + lang_id + stress + word/syl/phrase boundaries)
  │
  ▼
Compact Embeddings (18,432 params)
  │
  ▼
Shared Phoneme Encoder: 4x Bounded-Lookahead ConvNeXt1D (C=128, FF=256, K=5; 268,288 params)
  │  ▲  Token-Gated Low-Rank Bilingual Residuals (ΔW_en, ΔW_ne, r=16)
  ▼
Variance Adaptor (Duration + Pitch/UV + Energy + Frame Progress; 62,372 params)
  │
  ▼
Shared Frame-Acoustic Net: 4x Bounded-Lookahead ConvNeXt1D (C=128, FF=256, K=7) + 80-Mel Head (279,888 params)
  │  ▲  Frame-Gated Low-Rank Bilingual Residuals (ΔW_en, ΔW_ne, r=16; 196,624 total residual params)
  ▼
Streaming Waveform Decoder (24 kHz ConvNeXt1D + iSTFT: hop=240, n_fft=1024, 513 bins; 664,770 params)
  │
  ▼
24 kHz Mono PCM Audio
```

## 2. Exact `PAWP-1.5M` Parameter Budget Verification

Every learned parameter in the complete deployed inference graph (including embeddings, language gates, duration/pitch/energy predictors, acoustic backbone, and waveform decoder):

| Module | Layer Tensor Breakdown | Exact Params | Share |
|---|---|---:|---:|
| **1. Embeddings & Conditioning** | Phoneme `128×128` (`16,384`) + Lang `4×128` (`512`) + Stress `4×128` (`512`) + Boundary `Linear(4,128)` (`640`) + Speaker `1×128` (`128`) + `LayerNorm(128)` (`256`) | **18,432** | 1.2% |
| **2. Shared Phoneme Encoder** | `4×` ConvNeXt1D (`C=128, FF=256, K=5`: DW `768` + LN `256` + PW1 `33,024` + PW2 `32,896` + $\gamma$ `128` = `67,072` / block) | **268,288** | 18.0% |
| **3. Variance Adaptor** | Duration (`23,073`) + Pitch/UV (`24,514`) + Energy (`14,401`) + Frame-Progress `Linear(2,128)` (`384`) | **62,372** | 4.2% |
| **4. Shared Frame-Acoustic Net** | `4×` ConvNeXt1D (`C=128, FF=256, K=7`: `67,328` / block = `269,312`) + `LayerNorm(128)` (`256`) + Mel Head `Linear(128,80)` (`10,320`) | **279,888** | 18.8% |
| **5. Token/Frame-Gated Residuals** | EN & NE low-rank ($r=16$) adapters on `pw1`/`pw2` across 8 shared blocks (`8 × 4 × 6,144 = 196,608`) + `32` learnable gate scalars *(folds to 0 extra MACs in monolingual inference)* | **196,640** | 13.2% |
| **6. Streaming Waveform Decoder** | Stem `Conv1d(80,192,1)` + LN (`15,936`) + `3×` ConvNeXt1D (`C=192, FF=384, K=7`: `150,144` / block = `450,432`) + LN (`384`) + iSTFT Head `Linear(192,1026)` (`198,018`) | **664,770** | 44.6% |
| **Complete Inference Graph Total** | **All modules combined (strictly `<= 1,500,000`)** | **1,490,390** | **100.0%** |

## 3. Weight-Space Programming Algorithms (Deliverable 4)

1. **Shape-Compatible Layer Pools ($\mathcal{G}$):**
   - Pool `pw1`: 8 matrices of shape `(256, 128)` across encoder & acoustic ConvNeXt1D blocks.
   - Pool `pw2`: 8 matrices of shape `(128, 256)` across encoder & acoustic ConvNeXt1D blocks.
   - Decoder pool (tested separately for phase sensitivity): 3 `pw1` `(384, 192)` and 3 `pw2` `(192, 384)` matrices.
   - Shared representation: $W_l = \sum_{k=1}^{K} a_{lk} B_k + U_l V_l$ for $l \in \mathcal{G}$.
2. **MAC Breakeven Guard:**
   - Low-rank factorization $U_l V_l$ reduces runtime MACs strictly when $r_l < \left\lfloor \frac{d_{\text{in}} d_{\text{out}}}{d_{\text{in}} + d_{\text{out}}} \right\rfloor$ (e.g., for `128×256`, breakeven rank is `85`). Layers above breakeven are stored factored on disk for `unique_stored_params` savings and materialized into dense matrices at load time for maximum CPU GEMM throughput.
3. **Budget-Constrained Knapsack & Fixed-Budget Regrowth:**
   - Pure solvers in `pawp_tts.domain.pawp.allocator` allocate singular-value ranks $r_l$ greedily by marginal phonetic sensitivity per parameter ($\Delta I / \Delta P$) under $\sum_l P_l(r_l) \le P_{\text{budget}}$, and reallocate rank from low-error layers to high-pronunciation-error layers during closed-loop regrowth without ever violating $P_{\text{budget}}$.
4. **Monolingual Residual Folding:**
   - During monolingual deployment ($m_{\text{lang}} \equiv 1$, $m_{\text{other}} \equiv 0$), `GatedLowRankLinear.fold_language(lang)` merges $g_{\text{lang}} U_{\text{lang}} V_{\text{lang}}$ directly into $W_{\text{shared}}$, eliminating all runtime routing overhead.
