#!/usr/bin/env python3
"""Generate combined ANOVA/ANCOVA interaction figure and LaTeX table for E1, E2, E3.

Produces a single 1×3 subplot figure (η²_interaction vs position) and prints
a ready-to-paste LaTeX table (tab:p0_p1_combined part b).

Usage (from project root):
    python scripts/fig_anova_interaction_combined.py
"""

import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_PATH = PROJECT_ROOT / "paper_figs" / "anova_interaction_combined.png"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

import argparse

from icl.utils.plot_config import apply_paper_style
from icl.utils.separability import plot_anova_interaction_on_ax, plot_ancova_interaction_on_ax
from icl.utils.unified_interface import get_exp_name

apply_paper_style()


def parse_args():
    p = argparse.ArgumentParser(description="Generate ANOVA/ANCOVA interaction figures (E1/E2/E3).")
    p.add_argument("--coin-vocab-size", type=int, default=6, metavar="V",
                   help="Vocabulary size for E1 Coins experiment (default: 6)")
    return p.parse_args()


# ── compute functions (parameters match the analysis notebooks) ───────────────

def _compute_coin(vocab_size: int = 6):
    from icl.coin.analysis import plot_anova_separability_coin
    exp_name = get_exp_name("coin", -1, vocab_size=vocab_size)
    out = plot_anova_separability_coin(
        exp_name,
        layers=range(6),
        batch_size=1024,
        log_x=False,
        show=False,
        print_summary=False,
        extraction_point="post_mlp",
    )
    plt.close("all")
    return out["anova_results"]


def _compute_latent():
    from icl.latent_markov.analysis import plot_anova_separability_latent
    exp_name = get_exp_name("latent", -1, total_steps=30_000, warmup_steps=15_000)
    out = plot_anova_separability_latent(
        exp_name,
        layers=range(6),
        batch_size=1024,
        log_x=False,
        show=False,
        print_summary=False,
        extraction_point="post_mlp",
    )
    plt.close("all")
    return out["anova_results"]


def _compute_linear():
    from icl.linear.analysis import plot_ancova_separability_linear
    exp_name = get_exp_name(
        "linear", -1,
        n_layer=16,
        total_steps=30_000,
        warmup_steps=15_000,
        batch_size=256,
        max_grad_norm=1.0,
    )
    out = plot_ancova_separability_linear(
        exp_name,
        layers=range(4, 16, 2),
        n_minor=0,
        batch_size=1024,
        log_x=False,
        show=False,
        print_summary=False,
        extraction_point="post_mlp",
    )
    plt.close("all")
    return out["ancova_results"]


# ── table helpers ─────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    if v < 0.001:
        return r"${<}0.001$"
    return f"{v:.3f}"


def _eta2_from_ancova(res) -> float:
    """η²_interaction for ANCOVA: separability_gap / r2_full."""
    if res.r2_full <= 0:
        return 0.0
    return res.separability_gap / res.r2_full


def _print_latex_table(coin_results, linear_results, latent_results) -> None:
    """Print the η²_interaction LaTeX table at the last context position."""
    panels = [
        ("E1", list(range(6)),         coin_results,    False),
        ("E2", list(range(4, 16, 2)),  linear_results,  True),   # ANCOVA
        ("E3", list(range(6)),         latent_results,  False),
    ]

    exp_data = []
    for label, layers, r2_dict, is_ancova in panels:
        last_pos = max(pos for ld in r2_dict.values() for pos in ld)
        if is_ancova:
            vals = [_eta2_from_ancova(r2_dict[l][last_pos]) for l in layers]
        else:
            vals = [r2_dict[l][last_pos].eta2_interaction for l in layers]
        exp_data.append((label, layers, vals))

    e1_label, e1_layers, e1_vals = exp_data[0]
    e2_label, e2_layers, e2_vals = exp_data[1]
    e3_label, e3_layers, e3_vals = exp_data[2]

    n, m = len(e1_layers), len(e2_layers)
    e1_hdr = " & ".join(str(l) for l in e1_layers)
    e2_hdr = " & ".join(str(l) for l in e2_layers)
    e1_row = " & ".join(_fmt(v) for v in e1_vals)
    e2_row = " & ".join(_fmt(v) for v in e2_vals)
    e3_row = " & ".join(_fmt(v) for v in e3_vals)
    e3_pad = " & ".join([""] * m)

    print("\n" + "=" * 72)
    print("  LaTeX table: η²_interaction at last position  (tab:p0_p1_combined b)")
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
\\label{{tab:p0_p1_combined}}
\\end{{table}}""")
    print("=" * 72 + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    panels = [
        ("E1 (Coins)",  lambda _v=args.coin_vocab_size: _compute_coin(_v),  range(6),       False),
        ("E2 (Linear)", _compute_linear,                                     range(4, 16, 2), True),
        ("E3 (Latent)", _compute_latent,                                     range(6),       False),
    ]

    t_total = time.time()
    results = []
    for label, compute_fn, _layers, _is_ancova in panels:
        log.info(f"[{label}] computing interaction …")
        t0 = time.time()
        r = compute_fn()
        results.append(r)
        log.info(f"[{label}] done in {time.time() - t0:.1f}s")

    log.info("Composing figure …")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=False)

    coin_results, linear_results, latent_results = results
    plot_anova_interaction_on_ax(axes[0], coin_results,    log_x=False, show_ylabel=True)
    plot_ancova_interaction_on_ax(axes[1], linear_results, log_x=False, show_ylabel=True)
    plot_anova_interaction_on_ax(axes[2], latent_results,  log_x=False, show_ylabel=True)

    for ax, (label, *_) in zip(axes, panels):
        ax.set_title(label)

    fig.tight_layout(w_pad=1.0)
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
    log.info(f"Saved → {SAVE_PATH}  (total {time.time() - t_total:.1f}s)")
    plt.close(fig)

    # ── print LaTeX table ─────────────────────────────────────────────────────
    _print_latex_table(coin_results, linear_results, latent_results)


if __name__ == "__main__":
    main()
