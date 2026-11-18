# Root Makefile for PAWP-TTS
UV := uv
VENV_BIN := .venv/bin
PYTHON := $(VENV_BIN)/python
PYTEST := $(VENV_BIN)/pytest
MYPY := $(VENV_BIN)/mypy
RUFF := $(VENV_BIN)/ruff
LINT_IMPORTS := $(VENV_BIN)/lint-imports
PAWP_CLI := $(VENV_BIN)/pawp

SRC_DIR := src
TESTS_DIR := tests
CONFIG ?= configs/student_1_5m.yaml
SEED ?= 42
CKPT ?=
BUDGET ?= 1000000
EXP_ID ?= exp001

.DEFAULT_GOAL := help

include make/dev.mk
include make/data.mk
include make/train.mk
include make/eval.mk

.PHONY: help
help: ## Show all available Makefile targets
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf " \033[36m%-24s\033[0m %s\n", $$1, $$2}'
