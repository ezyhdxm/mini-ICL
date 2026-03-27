"""
Pytest configuration.

Stubs out icl.latent_markov.legacy, which is a work-in-progress migration
that hasn't landed yet.  The stub provides the two classes that
markov_latent.py re-exports for backward compatibility so that the rest of
the icl package can be imported in tests.
"""

import sys
import types


def _install_legacy_stub() -> None:
    """Insert a minimal stub for icl.latent_markov.legacy before any icl import."""
    # Only needed if the real module isn't already present.
    if "icl.latent_markov.legacy" in sys.modules:
        return

    class _Stub:
        pass

    stub_pkg = types.ModuleType("icl.latent_markov.legacy")
    stub_latent_bayes = types.ModuleType("icl.latent_markov.legacy.latent_bayes")
    stub_latent_bayes.LatentIDBayes = _Stub
    stub_latent_bayes.LatentOODBayes = _Stub
    stub_latent_bayes.LatentIDBayes.__name__ = "LatentIDBayes"
    stub_latent_bayes.LatentOODBayes.__name__ = "LatentOODBayes"

    stub_pkg.latent_bayes = stub_latent_bayes  # type: ignore[attr-defined]
    sys.modules["icl.latent_markov.legacy"] = stub_pkg
    sys.modules["icl.latent_markov.legacy.latent_bayes"] = stub_latent_bayes

    # Also stub kv_task_vec, used by test_kv_latent_task_vec.py
    stub_kv = types.ModuleType("icl.latent_markov.legacy.kv_task_vec")
    for name in (
        "_get_n_emb",
        "_infer_n_head_head_dim",
        "_make_kv_caches_like_model",
        "_HookLastTokenAllLayers",
        "compute_hiddens_onepos_all_layers_kvcache",
    ):
        setattr(stub_kv, name, _Stub)
    sys.modules["icl.latent_markov.legacy.kv_task_vec"] = stub_kv


_install_legacy_stub()
