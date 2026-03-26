import copy
import hashlib
from ml_collections import ConfigDict


def canonicalize_config_for_exp(config: ConfigDict) -> None:
    """
    Canonicalize config.device in place for experiment identity (hashing/saving).
    Any cuda device (e.g. "cuda", "cuda:0") becomes "cuda" so exp_name does not
    depend on the runtime GPU index.
    """
    device = config.get("device", "")
    if isinstance(device, str) and device.startswith("cuda"):
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