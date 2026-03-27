"""Loader for external ICL task datasets.

Loads word-pair datasets from Todd et al. (2024) "Function Vectors in
Large Language Models" (ICLR 2024).

Source: https://github.com/ericwtodd/function_vectors
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "function_vectors")


def _load_pairs(filename: str) -> List[Tuple[str, str]]:
    """Load de-duplicated (input, output) pairs from a JSON file."""
    path = os.path.join(_DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    seen: set = set()
    pairs: List[Tuple[str, str]] = []
    for item in data:
        inp = item["input"]
        if inp not in seen:
            seen.add(inp)
            pairs.append((inp, item["output"]))
    return pairs


def load_all() -> Dict[str, List[Tuple[str, str]]]:
    """Load all available external datasets.

    Returns a dict mapping task name to list of (input, output) pairs.
    """
    return {name: _load_pairs(fname) for name, fname in _REGISTRY.items()}


# Maps a task name -> JSON filename.
_REGISTRY: Dict[str, str] = {
    "antonyms":             "antonym.json",
    "synonyms":             "synonym.json",
    "english_to_french":    "english-french.json",
    "english_to_spanish":   "english-spanish.json",
    "english_to_german":    "english-german.json",
    "country_to_capital":   "country-capital.json",
    "person_to_occupation": "person-occupation.json",
    "landmark_to_country":  "landmark-country.json",
    "product_to_company":   "product-company.json",
}

# Pre-load all datasets at import time (they're small JSON files).
_ALL = load_all()

ANTONYMS:             List[Tuple[str, str]] = _ALL["antonyms"]
SYNONYMS:             List[Tuple[str, str]] = _ALL["synonyms"]
ENGLISH_TO_FRENCH:    List[Tuple[str, str]] = _ALL["english_to_french"]
ENGLISH_TO_SPANISH:   List[Tuple[str, str]] = _ALL["english_to_spanish"]
ENGLISH_TO_GERMAN:    List[Tuple[str, str]] = _ALL["english_to_german"]
COUNTRY_TO_CAPITAL:   List[Tuple[str, str]] = _ALL["country_to_capital"]
PERSON_TO_OCCUPATION: List[Tuple[str, str]] = _ALL["person_to_occupation"]
LANDMARK_TO_COUNTRY:  List[Tuple[str, str]] = _ALL["landmark_to_country"]
PRODUCT_TO_COMPANY:   List[Tuple[str, str]] = _ALL["product_to_company"]
