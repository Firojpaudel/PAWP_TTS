.PHONY: setup sync format lint typecheck arch-check test-unit test-integration test check prune-artifacts finalize-release clean

setup: ## Sync full development and training environment via uv
	$(UV) sync --python 3.12 --extra dev
	$(UV) pip install --offline --python $(PYTHON) torch==2.14.1

sync: ## Sync core dependencies only
	$(UV) sync --python 3.12 --extra dev

format: ## Format source and test code with ruff
	$(RUFF) format $(SRC_DIR) $(TESTS_DIR)

lint: ## Check formatting and lint rules with ruff
	$(RUFF) format --check $(SRC_DIR) $(TESTS_DIR)
	$(RUFF) check $(SRC_DIR) $(TESTS_DIR)

typecheck: ## Run strict static type checking with mypy
	$(MYPY) $(SRC_DIR)

arch-check: ## Enforce Clean Architecture layer dependency boundaries
	$(LINT_IMPORTS)
	$(PYTEST) $(TESTS_DIR)/architecture -v

test-unit: ## Run pure domain and application unit tests (CPU-only, fast)
	$(PYTEST) $(TESTS_DIR)/unit -v

test-integration: ## Run neural architecture and end-to-end integration tests
	$(PYTEST) $(TESTS_DIR)/integration -v

test: ## Run complete test suite
	$(PYTEST) $(TESTS_DIR) -v

check: lint typecheck arch-check test-unit ## Run lint, typecheck, Clean Architecture checks, and unit tests

prune-artifacts: ## Prune intermediate checkpoints, temp audio, and caches (keeps final & reports)
	$(PAWP_CLI) prune-artifacts

finalize-release: ## Clean all non-final artifacts for lean open-source release
	$(PAWP_CLI) prune-artifacts --finalize-release

clean: ## Remove Python/tooling cache directories
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
