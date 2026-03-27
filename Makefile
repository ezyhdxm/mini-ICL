PYTHON ?= python
RESULTS_DIR := results
PORT := 8000

.PHONY: serve test lint typecheck

## Start the experiment browser at http://localhost:$(PORT)
serve:
	$(PYTHON) server.py $(PORT)

## Run tests
test:
	$(PYTHON) -m pytest tests/ -v

## Run linter (ruff)
lint:
	$(PYTHON) -m ruff check src/ tests/

## Run type checker
typecheck:
	$(PYTHON) -m mypy src/icl/ --ignore-missing-imports
