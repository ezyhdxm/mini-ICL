import hashlib
from ml_collections import ConfigDict

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