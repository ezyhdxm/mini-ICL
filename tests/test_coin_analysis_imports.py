"""
Verify that icl.coin.analysis.__all__ is correct after cleanup.

These tests use static analysis (AST + grep) to avoid triggering the
currently-broken import chain in the rest of the icl package.
"""

import ast
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
INIT_PY = REPO_ROOT / "src/icl/coin/analysis/__init__.py"


def _parse_all(path: Path) -> list[str]:
    """Return the list of strings in __all__ from a Python source file."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return [elt.s for elt in node.value.elts]  # type: ignore
    return []


def _rg(pattern: str, exclude_dirs=(".venv", "__pycache__")) -> list[str]:
    """Return lines matching pattern across the repo (excluding .venv)."""
    excludes = []
    for d in exclude_dirs:
        excludes += ["--glob", f"!**/{d}/**"]
    result = subprocess.run(
        ["grep", "-r", "--include=*.py", "--include=*.ipynb", "-l", pattern,
         str(REPO_ROOT)] + [f"--exclude-dir={d}" for d in exclude_dirs],
        capture_output=True, text=True,
    )
    return result.stdout.strip().splitlines()


class TestRemovedFromAll:
    """Names that were removed from __all__ must no longer be present."""

    def test_private_stable_rank_not_in_all(self):
        assert "_stable_rank_from_gram" not in _parse_all(INIT_PY)

    def test_unused_helpers_not_in_all(self):
        all_ = _parse_all(INIT_PY)
        assert "compute_hiddens_multi_coin" not in all_
        assert "compute_stable_rank_at_real_positions" not in all_

    def test_unused_plots_not_in_all(self):
        all_ = _parse_all(INIT_PY)
        assert "plot_kl_lambda_vs_posterior_coin" not in all_
        assert "plot_probe_fit_r2_coin" not in all_
        assert "plot_p1_variance_coin" not in all_
        assert "traj_projection_plot_coin" not in all_

    def test_inject_posterior_per_position_not_in_all(self):
        assert "plot_inject_posterior_per_position" not in _parse_all(INIT_PY)


class TestActiveNamesStillInAll:
    """Names actively used in notebooks must remain in __all__."""

    def test_helpers_in_all(self):
        all_ = _parse_all(INIT_PY)
        assert "compute_hiddens_token_conditioned_coin" in all_
        assert "get_token_conditioned_hiddens_coin" in all_

    def test_posterior_in_all(self):
        all_ = _parse_all(INIT_PY)
        assert "plot_coin_task_posterior" in all_
        assert "plot_id_ood_loss_coin" in all_
        assert "plot_lambda_posterior_agreement_coin" in all_

    def test_probes_in_all(self):
        all_ = _parse_all(INIT_PY)
        assert "train_linear_hidden_predictor_coin" in all_
        assert "plot_val_r2_across_layers_coin" in all_
        assert "plot_averaging_r2_coin" in all_

    def test_variance_in_all(self):
        all_ = _parse_all(INIT_PY)
        assert "plot_task_vector_r2_coin" in all_
        assert "plot_anova_separability_coin" in all_

    def test_trajectory_in_all(self):
        all_ = _parse_all(INIT_PY)
        assert "traj_posterior_projection_plot_coin" in all_
        assert "traj_post_posterior_projection_plot_coin" in all_
        assert "traj_cellmean_projection_plot_coin" in all_
        assert "traj_averaging_projection_plot_coin" in all_

    def test_interventions_in_all(self):
        all_ = _parse_all(INIT_PY)
        for name in [
            "intervene_remove_task_subspace_coin",
            "plot_intervention_remove_task_across_layers_coin",
            "plot_optimal_orth_direction_across_layers_coin",
            "intervene_inject_task_vector_coin",
            "plot_inject_task_vector_across_layers_coin",
            "intervene_inject_posterior_coin",
            "intervene_direct_injection_coin",
            "intervene_averaging_injection_coin",
            "plot_inject_posterior_across_layers_coin",
        ]:
            assert name in all_, f"{name} missing from __all__"

    def test_ood_in_all(self):
        assert "plot_maj_r2_ood_across_steps_coin" in _parse_all(INIT_PY)


class TestNoExternalCallersForRemovedNames:
    """
    Removed functions have no external callers — confirm they appear only in
    their own definition files and re-export files.
    """

    def _callers_outside_definition(self, fn_name: str, definition_files: list[str]) -> list[str]:
        files = _rg(fn_name)
        return [
            f for f in files
            if not any(d in f for d in definition_files)
            and "tests/" not in f
        ]

    def test_compute_hiddens_multi_coin_unused(self):
        external = self._callers_outside_definition(
            "compute_hiddens_multi_coin",
            ["analysis/_helpers.py", "coin/analysis/__init__"],
        )
        assert external == [], f"Unexpected callers: {external}"

    def test_compute_stable_rank_unused(self):
        external = self._callers_outside_definition(
            "compute_stable_rank_at_real_positions",
            ["analysis/_helpers.py", "coin/analysis/__init__"],
        )
        assert external == [], f"Unexpected callers: {external}"

    def test_plot_kl_lambda_unused(self):
        external = self._callers_outside_definition(
            "plot_kl_lambda_vs_posterior_coin",
            ["analysis/posterior.py", "coin/analysis/__init__"],
        )
        assert external == [], f"Unexpected callers: {external}"

    def test_plot_probe_fit_r2_unused(self):
        external = self._callers_outside_definition(
            "plot_probe_fit_r2_coin",
            ["analysis/variance.py", "coin/analysis/__init__"],
        )
        assert external == [], f"Unexpected callers: {external}"

    def test_plot_p1_variance_unused(self):
        external = self._callers_outside_definition(
            "plot_p1_variance_coin",
            ["analysis/variance.py", "coin/analysis/__init__"],
        )
        assert external == [], f"Unexpected callers: {external}"

    def test_traj_projection_unused(self):
        external = self._callers_outside_definition(
            "traj_projection_plot_coin",
            ["analysis/trajectory.py", "coin/analysis/__init__"],
        )
        assert external == [], f"Unexpected callers: {external}"
