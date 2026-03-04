"""
Central plotting configuration for paper-ready figures.
- Default title is empty (figures go in papers without redundant titles).
- Font sizes increased for readability in print.
"""
import matplotlib

# Paper-friendly font sizes (increase from matplotlib defaults for readability)
_PAPER_RC = {
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
}


def apply_paper_style():
    """Apply paper-friendly matplotlib defaults. Call at start of plotting modules."""
    matplotlib.rcParams.update(_PAPER_RC)


# Apply when this module is imported (so all icl plots use it)
apply_paper_style()
