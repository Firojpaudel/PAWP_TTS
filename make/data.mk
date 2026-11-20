.PHONY: frontend-validate data-manifest data-splits teacher-extract teacher-generate clean-teacher

frontend-validate: ## Run bilingual EN/NE/CodeMix phonological frontend validation suite
	$(PAWP_CLI) frontend --validate

data-manifest: ## Build zero-copy bilingual dataset manifests via in-process G2P
	$(PAWP_CLI) build-manifests --exp-id $(EXP_ID)-data

data-splits: ## Verify leak-free train/val/test bilingual dataset splits
	$(PAWP_CLI) build-manifests --verify-only

teacher-extract: ## Extract language-balanced 24 kHz acoustic & duration targets into data/cache
	$(PAWP_CLI) teacher-extract --exp-id $(EXP_ID)-extract --log-ledger

VOICE ?= $(shell grep -E "^PAWP_TTS_VOICE=" .env 2>/dev/null | cut -d= -f2)

teacher-generate: ## Generate ASR-filtered single-voice EN/NE corpus from the configured teacher
	$(PAWP_CLI) teacher-generate --exp-id $(EXP_ID)-teacher --log-ledger

clean-teacher: ## Delete generated teacher audio + its feature cache (regenerable)
	rm -rf data/cache/teacher_$(VOICE) data/cache/v2/train_teacher data/cache/v2/train_teacher_index.json data/manifests/teacher_$(VOICE).jsonl
