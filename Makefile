PYTHON ?= uv run python
RESULTS_DIR := results
FIGS_DIR := paper_figs
PORT := 8000
K_LIST ?= 0 1 2 3 4 5 6 7 8 9
RECOMPUTE ?= 0
COIN_VOCAB_SIZE ?= 6
COIN_INJS_LAYER ?= 3

.PHONY: train train-test train-latent train-coin train-linear index serve browse test lint typecheck help figs

## Show available targets
help:
	@echo "Training:"
	@echo "  make train            Run full pipeline (latent + coin + linear, K_LIST=$(K_LIST))"
	@echo "  make train-test       Quick test run (k=0,1 only)"
	@echo "  make train-latent     Run latent training only"
	@echo "  make train-coin       Run coin training only"
	@echo "  make train-linear     Run linear training only"
	@echo "  make train-dyck       Run Dyck training only"
	@echo ""
	@echo "Figures:"
	@echo "  make figs                    Generate all combined paper figures (skip existing)"
	@echo "  make figs RECOMPUTE=1        Regenerate all figures unconditionally"
	@echo "  make fig-task-vector-r2      Task-vector R² (combined E1/E2/E3)"
	@echo "  make fig-averaging-r2        Averaging R² (combined E1/E2/E3)"
	@echo "  make fig-beta-alpha-traj     Beta/alpha trajectories (combined E1/E2/E3)"
	@echo "  make fig-kl-transition       KL phase-transition heatmaps (combined E1/E2/E3)"
	@echo "  make fig-ood-r2              OOD R² projection curves (combined E1/E2/E3)"
	@echo "  make fig-injection-simplex   Injection simplex (combined E1/E2/E3)"
	@echo "  make fig-dyck                Dyck E4 combined (variance R² + prefix scatter)"
	@echo ""
	@echo "Experiment browser:"
	@echo "  make browse           Index experiments and start browser"
	@echo "  make index            Re-index experiments"
	@echo "  make serve            Start browser server only"
	@echo ""
	@echo "Development:"
	@echo "  make test             Run tests"
	@echo "  make lint             Run linter"
	@echo "  make typecheck        Run type checker"
	@echo ""
	@echo "Override K_LIST:         make train K_LIST='0 1 2 3'"
	@echo "Override COIN_VOCAB_SIZE: make figs COIN_VOCAB_SIZE=8"

## Run full training pipeline (latent + coin + linear)
train:
	$(PYTHON) scripts/run_pipeline.py -k $(K_LIST)

## Quick test run with k=0,1
train-test:
	$(PYTHON) scripts/run_pipeline.py -k 0 1

## Run latent training only
train-latent:
	$(PYTHON) -c "from scripts.run_pipeline import run_latent; run_latent([$(subst $(eval ) $(eval ),$(comma),$(K_LIST))])"

## Run coin training only
train-coin:
	$(PYTHON) -c "from scripts.run_pipeline import run_coin; run_coin([$(subst $(eval ) $(eval ),$(comma),$(K_LIST))])"

## Run linear training only
train-linear:
	$(PYTHON) -c "from scripts.run_pipeline import run_linear; run_linear([$(subst $(eval ) $(eval ),$(comma),$(K_LIST))])"

## Run Dyck training only
train-dyck:
	$(PYTHON) scripts/run_dyck.py -k $(K_LIST)

## Index all experiments
index:
	$(PYTHON) src/icl/utils/experiment_index.py $(RESULTS_DIR)

## Start the experiment browser at http://localhost:$(PORT)
serve:
	$(PYTHON) server.py $(PORT)

## Index experiments then start browser
browse: index serve

## Generate all combined paper figures (skips existing ones unless RECOMPUTE=1)
figs: fig-task-vector-r2 fig-averaging-r2 fig-beta-alpha-traj fig-kl-transition fig-ood-r2 fig-injection-simplex fig-dyck

# ── figure rules ──────────────────────────────────────────────────────────────
# Each fig-* target is a .PHONY alias for the real PNG file target.
# The PNG recipe skips generation when the file already exists, unless RECOMPUTE=1.
#
#   make fig-averaging-r2            # skip if paper_figs/averaging_r2_combined.png exists
#   make fig-averaging-r2 RECOMPUTE=1  # always regenerate
#
# $(call fig_rule, PHONY_SUFFIX, FILE_STEM[, SCRIPT_STEM[, EXTRA_ARGS]])
# FILE_STEM  : output PNG is $(FIGS_DIR)/FILE_STEM.png
# SCRIPT_STEM: script is scripts/fig_SCRIPT_STEM.py (defaults to FILE_STEM)
# EXTRA_ARGS : extra CLI flags forwarded to the script (optional)
# The phony recipe always runs so that RECOMPUTE=1 is honoured even when
# the PNG already exists (a pure file target would be silently skipped).
define fig_rule
.PHONY: fig-$(1)
fig-$(1):
	@if [ "$$(RECOMPUTE)" = "1" ] || [ ! -f "$(FIGS_DIR)/$(2).png" ]; then \
		echo "[fig] generating $(FIGS_DIR)/$(2).png"; \
		$$(PYTHON) scripts/fig_$(if $(3),$(3),$(2)).py $(4); \
	else \
		echo "[fig] skip  $(FIGS_DIR)/$(2).png  (already exists — use RECOMPUTE=1 to force)"; \
	fi
endef

_COIN_VS  := --coin-vocab-size $(COIN_VOCAB_SIZE)
_COIN_INJ := $(_COIN_VS) --coin-layer $(COIN_INJS_LAYER)

$(eval $(call fig_rule,task-vector-r2,task_vector_r2_combined,,$(_COIN_VS)))
$(eval $(call fig_rule,averaging-r2,averaging_r2_combined,,$(_COIN_VS)))
$(eval $(call fig_rule,beta-alpha-traj,beta_alpha_traj_c3_l9_t3_simplex,beta_alpha_traj_combined,$(_COIN_VS)))
$(eval $(call fig_rule,kl-transition,kl_transition_combined,,$(_COIN_VS)))
$(eval $(call fig_rule,ood-r2,ood_r2_c4_l10_t4,ood_r2_combined,$(_COIN_VS)))
$(eval $(call fig_rule,injection-simplex,injection_simplex_combined,,$(_COIN_INJ)))
$(eval $(call fig_rule,dyck,dyck_combined,))

## Run tests
test:
	$(PYTHON) -m pytest tests/ -v

## Run linter (ruff)
lint:
	$(PYTHON) -m ruff check src/ tests/

## Run type checker
typecheck:
	$(PYTHON) -m mypy src/icl/ --ignore-missing-imports

comma := ,
