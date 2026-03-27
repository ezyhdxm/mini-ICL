"""
Backward-compatibility shim for ``icl.coin.coin_analysis``.

All analysis functions have been moved to sub-modules under
``icl.coin.analysis.*``.  This file re-exports every public name so
that existing imports such as::

    from icl.coin.coin_analysis import plot_coin_task_posterior

continue to work unchanged.
"""

from icl.coin.analysis import *  # noqa: F401,F403
