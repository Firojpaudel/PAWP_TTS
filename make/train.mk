.PHONY: train finetune-bilingual pawp-sensitivity pawp-allocate pawp-regrow compress-sweep

train: ## Train bilingual student checkpoint on RTX 4090 BF16 (CONFIG=$(CONFIG) SEED=$(SEED))
	$(PAWP_CLI) train --config $(CONFIG) --seed $(SEED) --exp-id $(EXP_ID)-train --log-ledger $(TRAIN_ARGS)

finetune-bilingual: ## Run bilingual and code-mixed joint fine-tuning (CONFIG=$(CONFIG) SEED=$(SEED))
	$(PAWP_CLI) finetune-bilingual --config $(CONFIG) --seed $(SEED) --exp-id $(EXP_ID)-finetune --log-ledger

pawp-sensitivity: ## Compute frequency-normalized phonetic sensitivity maps (Milestone 4)
	@echo "Scheduled for Milestone 4: compute phonetic Fisher/Taylor sensitivity."

pawp-allocate: ## Allocate per-layer rank budget under $(BUDGET) parameter constraint (Milestone 4)
	@echo "Scheduled for Milestone 4: solve budget-constrained rank allocation ($(BUDGET))."

pawp-regrow: ## Run closed-loop phonetic error-driven capacity regrowth (Milestone 4)
	@echo "Scheduled for Milestone 4: run error-driven weight regrowth."

compress-sweep: ## Run controlled equal-budget compression ablation sweep (Milestone 4)
	@echo "Scheduled for Milestone 4: run Pareto compression sweep."
