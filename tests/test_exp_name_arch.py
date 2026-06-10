"""Experiment-name identity must include the architecture.

Checkpoints are saved under ``train_{hash(config)}``; both training
(``train.py``) and analysis (``get_exp_name``) derive that hash via
``canonicalize_config_for_exp`` + ``get_hash``. Since ``config.model.arch`` is
part of the config, runs of different architectures must map to distinct names
(no overwrite) while the Transformer default stays unchanged.
"""

from icl.utils.basic import get_config_hash_for_exp
from icl.utils.unified_path_finder import unified_get_config, get_exp_name

ARCHS = ["transformer", "rnn", "lstm"]


def test_exp_names_distinct_across_archs():
    names = {a: get_exp_name("latent", k=-1, n_tasks=4, arch=a) for a in ARCHS}
    assert len(set(names.values())) == 3, names


def test_arch_none_defaults_to_transformer():
    # Backward compatibility: omitting arch == the Transformer default.
    assert get_exp_name("latent", k=-1, n_tasks=4) == \
        get_exp_name("latent", k=-1, n_tasks=4, arch="transformer")


def test_arch_is_part_of_experiment_identity():
    # Configs differing only in arch hash differently under the exact mechanism
    # training uses, so analysis locates the right checkpoint directory.
    hashes = set()
    for a in ARCHS:
        cfg = unified_get_config("latent")
        cfg.model.arch = a
        hashes.add(get_config_hash_for_exp(cfg))
    assert len(hashes) == 3
