#!/usr/bin/env python3
"""Generate combined task-vector R² figure across E1, E2, E3.

Produces a single 1×3 subplot figure instead of three separate PNGs,
eliminating redundant y-axis labels and inter-panel gaps from LaTeX
subfigure composition.

Usage (from project root):
    python scripts/fig_task_vector_r2_combined.py
"""

import argparse
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_PATH = PROJECT_ROOT / "paper_figs" / "task_vector_r2_combined.png"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from icl.utils.plot_config import apply_paper_style
from icl.utils.separability import (
    TaskVectorR2Result,
    _layer_style,
    plot_task_vector_r2_on_ax,
)
from icl.utils.unified_interface import get_exp_name

apply_paper_style()


def _compute_coin(layers, vocab_size: int = 6):
    from icl.coin.analysis.variance import plot_task_vector_r2_coin
    exp_name = get_exp_name("coin", -1, vocab_size=vocab_size)
    out = plot_task_vector_r2_coin(
        exp_name, layers=layers, batch_size=1024,
        print_summary=False, log_x=False, show=False,
        extraction_point="post_mlp",
    )
    plt.close("all")
    return out["r2_results"]


def _compute_latent(layers):
    from icl.latent_markov.analysis import plot_task_vector_r2_latent
    exp_name = get_exp_name("latent", -1, total_steps=30_000, warmup_steps=15_000)
    out = plot_task_vector_r2_latent(
        exp_name, layers=layers, batch_size=1024,
        print_summary=False, log_x=False, show=False,
        extraction_point="post_mlp",
    )
    plt.close("all")
    return out["r2_results"]


def _compute_linear(layers):
    """Linear uses ANCOVA; convert r2_full → TaskVectorR2Result for compatibility."""
    from icl.linear.analysis import plot_task_vector_r2_linear
    exp_name = get_exp_name(
        "linear", -1,
        n_layer=16, total_steps=30_000,
        warmup_steps=15_000, batch_size=256,
        max_grad_norm=1.0,
    )
    out = plot_task_vector_r2_linear(
        exp_name, layers=layers, n_minor=0, batch_size=1024,
        print_summary=False, log_x=False, show=False,
        extraction_point="post_mlp",
    )
    plt.close("all")

    ancova = out["ancova_results"]
    converted = {}
    for l_num, pos_dict in ancova.items():
        converted[l_num] = {}
        for pos, res in pos_dict.items():
            converted[l_num][pos] = TaskVectorR2Result(
                r2=res.r2_full,
                ss_total=res.ss_total,
                ss_between=res.ss_total - res.ss_res_full,
                ss_within=res.ss_res_full,
                n_tasks=res.n_tasks,
                n_tokens=res.n_covariate_dims,
                n_batch=res.n_samples,
                layer_num=res.layer_num,
                position=res.position,
            )
    return converted


def parse_args():
    p = argparse.ArgumentParser(description="Generate combined task-vector R² figure (E1/E2/E3).")
    p.add_argument("--coin-vocab-size", type=int, default=6, metavar="V",
                   help="Vocabulary size for E1 Coins experiments (default: 16)")
    return p.parse_args()


def _compute_anova_coin(layers, vocab_size: int = 6):
    from icl.coin.analysis import plot_anova_separability_coin
    exp_name = get_exp_name("coin", -1, vocab_size=vocab_size)
    out = plot_anova_separability_coin(
        exp_name, layers=list(layers), batch_size=1024,
        print_summary=False, log_x=False, show=False,
        extraction_point="post_mlp",
    )
    plt.close("all")
    return out["anova_results"]


def _compute_anova_latent(layers):
    from icl.latent_markov.analysis import plot_anova_separability_latent
    exp_name = get_exp_name("latent", -1, total_steps=30_000, warmup_steps=15_000)
    out = plot_anova_separability_latent(
        exp_name, layers=list(layers), batch_size=1024,
        print_summary=False, log_x=False, show=False,
        extraction_point="post_mlp",
    )
    plt.close("all")
    return out["anova_results"]


def _compute_ancova_linear(layers):
    from icl.linear.analysis import plot_ancova_separability_linear
    exp_name = get_exp_name(
        "linear", -1,
        n_layer=16, total_steps=30_000,
        warmup_steps=15_000, batch_size=256,
        max_grad_norm=1.0,
    )
    out = plot_ancova_separability_linear(
        exp_name, layers=list(layers), n_minor=0, batch_size=1024,
        print_summary=False, log_x=False, show=False,
        extraction_point="post_mlp",
    )
    plt.close("all")
    return out["ancova_results"]


def _eta2_from_ancova(res) -> float:
    """η²_interaction for ANCOVA: fraction of R²_full due to task-specific slopes."""
    if res.r2_full <= 0:
        return 0.0
    return res.separability_gap / res.r2_full


def _fmt_residual(v: float) -> str:
    """Format a residual variance ratio or η²_interaction value for a LaTeX table cell."""
    if v < 0.001:
        return r"${<}0.001$"
    return f"{v:.3f}"


def _print_latex_table(panels, results) -> None:
    """Print a LaTeX table of residual variance ratios at the last context position.

    Layout mirrors tab:eta2_final: E1 (layers 0–5) and E2 (layers 4,6,…,14)
    share one double-column row; E3 (layers 0–5) occupies the row below.
    """
    # Collect per-experiment data: (short_label, ordered_layers, residuals_at_last_pos)
    exp_data = []
    for (label, _, layer_range), r2_dict in zip(panels, results):
        layers = list(layer_range)
        last_pos = max(pos for ld in r2_dict.values() for pos in ld)
        residuals = [1.0 - r2_dict[l][last_pos].r2 for l in layers]
        short = label.split()[0]          # "E1", "E2", "E3"
        exp_data.append((short, layers, residuals))

    e1_label, e1_layers, e1_vals = exp_data[0]
    e2_label, e2_layers, e2_vals = exp_data[1]
    e3_label, e3_layers, e3_vals = exp_data[2]

    n = len(e1_layers)      # 6 for E1/E3
    m = len(e2_layers)      # 6 for E2

    # Header row layer numbers
    e1_hdr = " & ".join(str(l) for l in e1_layers)
    e2_hdr = " & ".join(str(l) for l in e2_layers)
    e3_hdr = " & ".join(str(l) for l in e3_layers)

    # Data rows
    e1_row = " & ".join(_fmt_residual(v) for v in e1_vals)
    e2_row = " & ".join(_fmt_residual(v) for v in e2_vals)
    e3_row = " & ".join(_fmt_residual(v) for v in e3_vals)
    e3_pad = " & ".join([""] * m)        # empty cells to fill E2 column

    print("\n" + "=" * 72)
    print("  LaTeX table: residual variance ratio (1 − R²) at last position")
    print("=" * 72)
    print(r"""\begin{table}[t]
\centering
\scriptsize
\renewcommand{\arraystretch}{0.9}
\setlength{\tabcolsep}{4pt}
\caption{Residual variance ratio $\mathrm{SS}_{\mathrm{within}} / \mathrm{SS}_{\mathrm{total}}$"""
          r""" at the last context position.
Small values indicate that latent $z$ and last token $s_t$ together explain most hidden-state variance.}"""
          f"""
\\begin{{tabular}}{{@{{}}lcccccc@{{\\hspace{{10pt}}}}lcccccc@{{}}}}
\\toprule
& \\multicolumn{{{n}}}{{c}}{{\\textbf{{Layer}}}} && \\multicolumn{{{m}}}{{c}}{{\\textbf{{Layer}}}} \\\\
\\cmidrule(lr){{2-{n+1}}} \\cmidrule(lr){{{n+3}-{n+m+2}}}
& {e1_hdr} && {e2_hdr} \\\\
\\midrule
\\texttt{{{e1_label}}} (Dice)
& {e1_row}
& \\texttt{{{e2_label}}} (Linear)
& {e2_row} \\\\
\\texttt{{{e3_label}}} (Markov)
& {e3_row}
& {e3_pad} \\\\
\\bottomrule
\\end{{tabular}}
\\label{{tab:residual_variance_ratio}}
\\end{{table}}""")
    print("=" * 72 + "\n")


def _print_eta2_latex_table(panels, anova_results_list) -> None:
    """Print a LaTeX table of η²_interaction at the last context position.

    For E1/E3 (discrete tokens) this is ANOVAResult.eta2_interaction.
    For E2 (continuous covariates) this is ANCOVAResult.separability_gap / r2_full.
    """
    exp_data = []
    for (label, _, layer_range), anova_dict in zip(panels, anova_results_list):
        layers = list(layer_range)
        last_pos = max(pos for ld in anova_dict.values() for pos in ld)
        short = label.split()[0]
        if "Linear" in label:
            eta2_vals = [_eta2_from_ancova(anova_dict[l][last_pos]) for l in layers]
        else:
            eta2_vals = [anova_dict[l][last_pos].eta2_interaction for l in layers]
        exp_data.append((short, layers, eta2_vals))

    e1_label, e1_layers, e1_vals = exp_data[0]
    e2_label, e2_layers, e2_vals = exp_data[1]
    e3_label, e3_layers, e3_vals = exp_data[2]

    n = len(e1_layers)
    m = len(e2_layers)
    e1_hdr = " & ".join(str(l) for l in e1_layers)
    e2_hdr = " & ".join(str(l) for l in e2_layers)
    e1_row = " & ".join(_fmt_residual(v) for v in e1_vals)
    e2_row = " & ".join(_fmt_residual(v) for v in e2_vals)
    e3_row = " & ".join(_fmt_residual(v) for v in e3_vals)
    e3_pad = " & ".join([""] * m)

    print("\n" + "=" * 72)
    print("  LaTeX table: η²_interaction at last position  (tab:eta2_final)")
    print("=" * 72)
    print(r"""\begin{table}[t]
\centering
\scriptsize
\renewcommand{\arraystretch}{0.9}
\setlength{\tabcolsep}{4pt}
\caption{Interaction proportion $\eta^2_{\mathrm{interaction}}$ at the last context position."""
          r""" Small values indicate that the additive model
$\hat\vmu_{t,z,a} \approx \hat\vmu_t + \hat\vtheta_z + \hat\vnu_a$ is a good approximation.}"""
          f"""
\\begin{{tabular}}{{@{{}}lcccccc@{{\\hspace{{10pt}}}}lcccccc@{{}}}}
\\toprule
& \\multicolumn{{{n}}}{{c}}{{\\textbf{{Layer}}}} && \\multicolumn{{{m}}}{{c}}{{\\textbf{{Layer}}}} \\\\
\\cmidrule(lr){{2-{n+1}}} \\cmidrule(lr){{{n+3}-{n+m+2}}}
& {e1_hdr} && {e2_hdr} \\\\
\\midrule
\\texttt{{{e1_label}}} (Dice)
& {e1_row}
& \\texttt{{{e2_label}}} (Linear)
& {e2_row} \\\\
\\texttt{{{e3_label}}} (Markov)
& {e3_row}
& {e3_pad} \\\\
\\bottomrule
\\end{{tabular}}
\\label{{tab:eta2_final}}
\\end{{table}}""")
    print("=" * 72 + "\n")


def main():
    args = parse_args()
    panels = [
        ("E1 (Coins)",   lambda layers, _v=args.coin_vocab_size: _compute_coin(layers, _v),   range(6)),
        ("E2 (Linear)",  _compute_linear, range(4, 16, 2)),
        ("E3 (Latent)",  _compute_latent, range(6)),
    ]

    t_total = time.time()
    results = []
    for label, compute_fn, layers in panels:
        log.info(f"[{label}] computing task-vector R² …")
        t0 = time.time()
        results.append(compute_fn(layers))
        log.info(f"[{label}] done in {time.time() - t0:.1f}s")

    log.info("Composing figure …")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)

    for idx, (ax, r2) in enumerate(zip(axes, results)):
        plot_task_vector_r2_on_ax(
            ax, r2,
            log_x=False,
            show_ylabel=(idx == 0),
        )
        if idx > 0:
            ax.tick_params(labelleft=False)

    fig.tight_layout(w_pad=1.0)
    fig.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
    log.info(f"Saved → {SAVE_PATH}  (total {time.time() - t_total:.1f}s)")
    plt.close(fig)

    _print_latex_table(panels, results)

    # ---- ANOVA / ANCOVA interaction tables (reuses cached hidden states) ----
    anova_panels = [
        ("E1 (Coins)",  lambda layers, _v=args.coin_vocab_size: _compute_anova_coin(layers, _v),  range(6)),
        ("E2 (Linear)", _compute_ancova_linear, range(4, 16, 2)),
        ("E3 (Latent)", _compute_anova_latent, range(6)),
    ]
    anova_results = []
    for label, compute_fn, layers in anova_panels:
        log.info(f"[{label}] computing η²_interaction …")
        t0 = time.time()
        anova_results.append(compute_fn(layers))
        log.info(f"[{label}] done in {time.time() - t0:.1f}s")

    _print_eta2_latex_table(anova_panels, anova_results)


if __name__ == "__main__":
    main()
