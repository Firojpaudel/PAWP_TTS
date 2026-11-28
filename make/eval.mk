.PHONY: inspect-baselines count-params asr-smoke eval-en eval-ne eval-neptts eval-heldout eval-codemix eval-independent bench-latency export-onnx scorecard

TEACHER_DIR ?= $(shell grep -E "^PAWP_TEACHER_DIR=" .env 2>/dev/null | cut -d= -f2)

inspect-baselines: ## Inspect local teacher/baseline checkpoints (PAWP_TEACHER_DIR in .env) via safetensors headers
	@for f in $$(find $(TEACHER_DIR)/ -name "*.safetensors" 2>/dev/null); do $(PAWP_CLI) inspect-ckpt --path $$f; done

count-params: ## Audit 4-metric parameter count & footprint for $(CONFIG)
	$(PAWP_CLI) count-params --config $(CONFIG) --exp-id $(EXP_ID) --log-ledger

asr-smoke: ## Run live ASR API smoke test against one Nepali and one English audio sample
	$(PAWP_CLI) asr-smoke

eval-en: ## Evaluate English synthesis suite
	$(PAWP_CLI) eval-suite --suite en --live-asr --config $(CONFIG) --exp-id $(EXP_ID)-eval-en --log-ledger

eval-ne: ## Evaluate Nepali synthesis suite
	$(PAWP_CLI) eval-suite --suite ne --live-asr --config $(CONFIG) --exp-id $(EXP_ID)-eval-ne --log-ledger

eval-neptts: ## Internal 77-pair G2P/audio smoke test (NOT the published NepTTS-Bench)
	$(PAWP_CLI) eval-suite --suite neptts --config $(CONFIG) --exp-id $(EXP_ID)-eval-neptts --log-ledger

eval-heldout: ## Held-out test sentences: ASR WER/CER on synthesized vs. human audio
	$(PAWP_CLI) eval-suite --suite heldout --live-asr --config $(CONFIG) --exp-id $(EXP_ID)-eval-heldout --log-ledger

eval-independent: ## Run EXP-018 independent local mixed-script ASR and XLSR-eSpeak PER re-scoring
	$(PYTHON) experiments/exp018_019/run_exp018_independent_asr.py
	$(PYTHON) experiments/exp018_019/run_exp018_independent_per.py

eval-codemix: ## Build & evaluate EXP-019 expanded code-mixed benchmark (n=160) and unfolded latency
	$(PYTHON) experiments/exp018_019/build_codemix_v1.py
	$(PYTHON) experiments/exp018_019/run_exp018_independent_asr.py --codemix-only
	$(PYTHON) experiments/exp018_019/run_exp019_codemix_latency.py

bench-latency: ## Benchmark ONNX CPU, PyTorch CPU, and RTX 4090 streaming TTFA and RTFx
	$(PAWP_CLI) bench-latency --config $(CONFIG) --exp-id $(EXP_ID)-latency --log-ledger

export-onnx: ## Export folded monolingual encoder + frame decoder ONNX graphs
	$(PAWP_CLI) export-onnx --config $(CONFIG) --lang en --output-dir artifacts/cache/onnx_export

scorecard: ## Aggregate experiment ledger and Pareto scorecard (Milestone 3+)
	@echo "Scheduled for Milestone 3: generate Pareto scorecard from ledger."
