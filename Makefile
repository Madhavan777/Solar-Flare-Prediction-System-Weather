# Convenience wrapper around `python -m solarflare`.
#
# Every target works identically as `python -m solarflare <name>`, which is the
# supported path on Windows where make is usually absent. Nothing here can write
# to a frozen artefact: the CLI refuses.

PYTHON ?= python
SOLARFLARE = $(PYTHON) -m solarflare

.DEFAULT_GOAL := help
.PHONY: help env install install-dev audit features train evaluate analysis figures \
        demo-data dashboard screenshots test test-fast check lint format all clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the runtime dependencies and the package
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

install-dev: install  ## Also install the test, lint and browser tooling
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m playwright install chromium

env:  ## Report the runtime environment
	$(SOLARFLARE) env

features:  ## Extract features from raw_data/ (hours; verifies the archives first)
	$(SOLARFLARE) features

audit:  ## Merge the feature chunks and verify the data audit
	$(SOLARFLARE) audit

evaluate:  ## Recompute every frozen metric and compare
	$(SOLARFLARE) evaluate

train:  ## Clean-room retrain into build/retrain (frozen models untouched)
	$(SOLARFLARE) train

analysis:  ## Run the post-hoc analyses into results/extra/
	$(SOLARFLARE) analysis

figures:  ## Build figures/extra/
	$(SOLARFLARE) figures

demo-data:  ## Regenerate the dashboard JSON
	$(SOLARFLARE) demo-data

dashboard:  ## Serve the dashboard on http://localhost:8791
	$(SOLARFLARE) dashboard

screenshots:  ## Capture every dashboard view to docs/screenshots/
	$(SOLARFLARE) screenshots

test:  ## Run the full test suite
	$(SOLARFLARE) test

test-fast:  ## Run only the fast tests
	$(SOLARFLARE) test --fast

check:  ## Verify every documented number against results/
	$(SOLARFLARE) check

lint:  ## ruff and black, check only
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .

format:  ## Apply black and ruff --fix
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m black .

all:  ## The whole verification and build pipeline
	$(SOLARFLARE) all

clean:  ## Remove caches and scratch output (never results/, models/ or figures/)
	rm -rf build .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
