import copy
import hashlib
from ml_collections import ConfigDict


def canonicalize_config_for_exp(config: ConfigDict) -> None:
    """Canonicalize config in-place for experiment identity (hashing).

    Device is an infrastructure detail, not an experimental parameter:
    experiments trained on cuda should be identifiable from a cpu-only
    machine. We therefore fix device to the constant "cuda" before hashing
    so that exp_name is the same regardless of runtime GPU availability.
    """
    config.device = "cuda"


def get_config_hash_for_exp(config: ConfigDict) -> str:
    """
    Return hash of config for experiment identity, without mutating the caller's config.
    Uses a canonical device so cuda:0 vs cuda does not change the hash.
    """
    config_copy = copy.deepcopy(config)
    canonicalize_config_for_exp(config_copy)
    return get_hash(config_copy)


def get_hash(config: ConfigDict) -> str:
    """
    Generate a hash for the given configuration.
    This is used to identify the experiment uniquely.
    Args:
        config (ConfigDict): The configuration object.
    Returns:
        str: The hash of the configuration.
    """
    return hashlib.md5(config.to_json(sort_keys=True).encode("utf-8")).hexdigest()