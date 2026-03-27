"""
Standard ICL task definitions for real-LLM subspace experiments.

Twelve ID tasks spanning morphology, translation, semantic, classification,
and factual categories.  Hand-curated pairs are used where they are larger;
otherwise data comes from Todd et al. (2024) "Function Vectors in Large
Language Models" (ICLR 2024).

  ─── Morphology (hand-curated) ────────────────────────────────────────────
  present_to_past    : walk     → walked       (519 pairs)
  singular_to_plural : cat      → cats         (337 pairs)

  ─── Translation (Todd et al.) ────────────────────────────────────────────
  english_to_french  : cat      → chat         (4 698 pairs)
  english_to_spanish : girl     → chica        (5 199 pairs)
  english_to_german  : journal  → Tagebuch     (5 145 pairs)

  ─── Semantic (Todd et al.) ───────────────────────────────────────────────
  antonyms           : hot      → cold         (2 398 pairs)
  synonyms           : begin    → start        (2 880 pairs)

  ─── Classification (hand-curated) ────────────────────────────────────────
  word_to_category   : dog      → animal       (401 pairs, 19 categories)

  ─── Factual (Todd et al.) ────────────────────────────────────────────────
  country_to_capital   : France   → Paris      (197 pairs)
  person_to_occupation : Einstein → physicist  (821 pairs)
  landmark_to_country  : Colosseum → Italy     (836 pairs)
  product_to_company   : iPhone  → Apple       (522 pairs)

Each ID task has a matched OOD task (apply deterministic g to the output).
Use ``make_id_tasks(task_names=[...])`` to select a subset.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from icl.real_llm._external_data import (
    ANTONYMS as _EXT_ANTONYMS,
    COUNTRY_TO_CAPITAL as _EXT_COUNTRY_TO_CAPITAL,
    ENGLISH_TO_FRENCH as _EXT_ENGLISH_TO_FRENCH,
    ENGLISH_TO_GERMAN as _EXT_ENGLISH_TO_GERMAN,
    ENGLISH_TO_SPANISH as _EXT_ENGLISH_TO_SPANISH,
    LANDMARK_TO_COUNTRY as _EXT_LANDMARK_TO_COUNTRY,
    PERSON_TO_OCCUPATION as _EXT_PERSON_TO_OCCUPATION,
    PRODUCT_TO_COMPANY as _EXT_PRODUCT_TO_COMPANY,
    SYNONYMS as _EXT_SYNONYMS,
)
from icl.real_llm._task_vocab import (
    _PRESENT_TO_PAST,
    _SINGULAR_TO_PLURAL,
    _WORD_TO_CATEGORY,
)


# ---------------------------------------------------------------------------
# OOD transforms  g : str -> str
# ---------------------------------------------------------------------------

def _reverse(y: str) -> str:
    """Reverse the string character by character."""
    return y[::-1]


def _len_str(y: str) -> str:
    """Return the number of non-space characters as a string."""
    return str(len(y.replace(" ", "")))


# Maps OOD task name → (base_id_task, g_func, g_description)
OOD_SPEC: Dict[str, Tuple[str, Callable[[str], str], str]] = {
    "past_reversed":       ("present_to_past",      _reverse,  "reversed past tense"),
    "plural_len":          ("singular_to_plural",    _len_str,  "length of plural"),
    "french_reversed":     ("english_to_french",     _reverse,  "reversed French"),
    "spanish_reversed":    ("english_to_spanish",    _reverse,  "reversed Spanish"),
    "german_reversed":     ("english_to_german",     _reverse,  "reversed German"),
    "antonym_reversed":    ("antonyms",              _reverse,  "reversed antonym"),
    "synonym_reversed":    ("synonyms",              _reverse,  "reversed synonym"),
    "category_len":        ("word_to_category",      _len_str,  "length of category"),
    "capital_reversed":    ("country_to_capital",     _reverse,  "reversed capital"),
    "occupation_len":      ("person_to_occupation",   _len_str,  "length of occupation"),
    "country_reversed":    ("landmark_to_country",    _reverse,  "reversed country"),
    "company_len":         ("product_to_company",     _len_str,  "length of company"),
}


# ---------------------------------------------------------------------------
# Task dataset class
# ---------------------------------------------------------------------------

@dataclass
class ICLTask:
    """A single ICL task with (x, y) demonstration pairs."""

    name: str
    pairs: List[Tuple[str, str]]
    separator: str = ": "
    ood_transform: Optional[Callable[[str], str]] = None
    ood_name: Optional[str] = None

    def __post_init__(self):
        if len(self.pairs) < 10:
            raise ValueError(f"Task {self.name!r} needs ≥ 10 pairs, got {len(self.pairs)}")

    def format_demo(self, x: str, y: str) -> str:
        return f"{x}{self.separator}{y}"

    def format_query(self, x: str) -> str:
        return f"{x}{self.separator}"

    def build_prompt(self, demo_pairs: List[Tuple[str, str]], query_x: str) -> str:
        """Build ICL prompt ending with '<query_x>: ' (no trailing newline)."""
        lines = [self.format_demo(x, y) for x, y in demo_pairs]
        lines.append(self.format_query(query_x))
        return "\n".join(lines)

    def build_ood_prompt(self, demo_pairs: List[Tuple[str, str]], query_x: str) -> str:
        if self.ood_transform is None:
            raise ValueError(f"Task {self.name!r} has no OOD transform set.")
        return self.build_prompt([(x, self.ood_transform(y)) for x, y in demo_pairs], query_x)

    def support_eval_split(
        self,
        n_support: int = 100,
        seed: int = 0,
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Non-overlapping split into support and eval pools."""
        rng = random.Random(seed)
        shuffled = list(self.pairs)
        rng.shuffle(shuffled)
        if n_support >= len(shuffled):
            raise ValueError(
                f"n_support={n_support} ≥ len(pairs)={len(shuffled)} for task {self.name!r}"
            )
        return shuffled[:n_support], shuffled[n_support:]

    def sample_prompts(
        self,
        pool: List[Tuple[str, str]],
        n_prompts: int,
        n_shots: int,
        seed: int = 0,
        ood: bool = False,
    ) -> List[str]:
        """Sample n_prompts ICL prompts, each with n_shots demos + 1 query."""
        if len(pool) < n_shots + 1:
            raise ValueError(
                f"pool size {len(pool)} < n_shots+1={n_shots + 1} for task {self.name!r}"
            )
        rng = random.Random(seed)
        prompts = []
        for _ in range(n_prompts):
            sample = rng.sample(pool, n_shots + 1)
            demos, query = sample[:n_shots], sample[n_shots]
            query_x = query[0]
            prompt = self.build_ood_prompt(demos, query_x) if ood else self.build_prompt(demos, query_x)
            prompts.append(prompt)
        return prompts

    def sample_prompts_with_answers(
        self,
        pool: List[Tuple[str, str]],
        n_prompts: int,
        n_shots: int,
        seed: int = 0,
        ood: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """Like sample_prompts, but also returns the ground-truth answer for each prompt.

        Returns (prompts, answers) where answers[i] is the expected completion
        for prompts[i].  For OOD prompts the answer is g(f_k(query_x)).
        """
        if len(pool) < n_shots + 1:
            raise ValueError(
                f"pool size {len(pool)} < n_shots+1={n_shots + 1} for task {self.name!r}"
            )
        rng = random.Random(seed)
        prompts, answers = [], []
        for _ in range(n_prompts):
            sample = rng.sample(pool, n_shots + 1)
            demos, query = sample[:n_shots], sample[n_shots]
            query_x, query_y = query
            if ood:
                prompt = self.build_ood_prompt(demos, query_x)
                answer = self.ood_transform(query_y)
            else:
                prompt = self.build_prompt(demos, query_x)
                answer = query_y
            prompts.append(prompt)
            answers.append(answer)
        return prompts, answers


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

_ALL_ID_TASKS: Dict[str, List[Tuple[str, str]]] = {
    # Morphology (hand-curated, larger than Todd et al.)
    "present_to_past":      _PRESENT_TO_PAST,
    "singular_to_plural":   _SINGULAR_TO_PLURAL,
    # Translation (Todd et al.)
    "english_to_french":    _EXT_ENGLISH_TO_FRENCH,
    "english_to_spanish":   _EXT_ENGLISH_TO_SPANISH,
    "english_to_german":    _EXT_ENGLISH_TO_GERMAN,
    # Semantic (Todd et al.)
    "antonyms":             _EXT_ANTONYMS,
    "synonyms":             _EXT_SYNONYMS,
    # Classification (hand-curated, unique)
    "word_to_category":     _WORD_TO_CATEGORY,
    # Factual (Todd et al.)
    "country_to_capital":   _EXT_COUNTRY_TO_CAPITAL,
    "person_to_occupation": _EXT_PERSON_TO_OCCUPATION,
    "landmark_to_country":  _EXT_LANDMARK_TO_COUNTRY,
    "product_to_company":   _EXT_PRODUCT_TO_COMPANY,
}


def make_id_tasks(
    task_names: Optional[List[str]] = None,
) -> Dict[str, ICLTask]:
    """Return ID tasks, optionally filtered to a subset.

    Parameters
    ----------
    task_names : list of str, optional
        If provided, only return these tasks.  If ``None`` (default),
        return all 12 available tasks.
    """
    names = task_names if task_names is not None else list(_ALL_ID_TASKS.keys())
    out: Dict[str, ICLTask] = {}
    for name in names:
        if name not in _ALL_ID_TASKS:
            raise ValueError(
                f"Unknown task {name!r}. Available: {sorted(_ALL_ID_TASKS)}"
            )
        out[name] = ICLTask(name=name, pairs=_ALL_ID_TASKS[name])
    return out


def make_ood_tasks(
    task_names: Optional[List[str]] = None,
) -> Dict[str, ICLTask]:
    """Return OOD tasks (one per ID task).

    Parameters
    ----------
    task_names : list of str, optional
        ID task names to include.  Only OOD tasks whose parent ID task
        is in this list are returned.  If ``None``, all are returned.
    """
    id_tasks = make_id_tasks(task_names)
    return {
        ood_name: ICLTask(
            name=ood_name,
            pairs=id_tasks[parent_id_name].pairs,
            ood_transform=g_func,
            ood_name=g_desc,
        )
        for ood_name, (parent_id_name, g_func, g_desc) in OOD_SPEC.items()
        if parent_id_name in id_tasks
    }


# ---------------------------------------------------------------------------
# Experiment prompt builder
# ---------------------------------------------------------------------------

@dataclass
class ExperimentPrompts:
    """All prompts for one run of the subspace analysis experiment."""

    support_prompts:  Dict[str, List[str]]
    id_eval_prompts:  Dict[str, List[str]]
    ood_eval_prompts: Dict[str, List[str]]
    id_eval_answers:  Dict[str, List[str]]
    ood_eval_answers: Dict[str, List[str]]
    task_names:       List[str]
    ood_task_names:   List[str]


def build_experiment_prompts(
    n_support_prompts: int = 50,
    n_eval_prompts:    int = 60,
    n_shots:           int = 10,
    n_support_pairs:   int = 100,
    seed:              int = 0,
    task_names: Optional[List[str]] = None,
) -> ExperimentPrompts:
    """Build support + ID-eval + OOD-eval prompts for the subspace experiment.

    Parameters
    ----------
    n_support_prompts : int
        Prompts per ID task averaged to form task vector τ_k.
    n_eval_prompts : int
        Held-out prompts per task for R² / λ evaluation.
    n_shots : int
        In-context demonstrations per prompt.
    n_support_pairs : int
        Pairs in the support pool (rest go to eval pool).
        With 300+ pairs per task, n_support_pairs=100 leaves 200+ for eval.
    seed : int
    task_names : list of str, optional
        ID task names to include.  If ``None``, all 12 tasks are used.
    """
    id_tasks  = make_id_tasks(task_names)
    ood_tasks = make_ood_tasks(task_names)

    # Pre-compute all (support_pool, eval_pool) splits once, keyed by task name.
    # Splitting once and reusing guarantees the same partition is used everywhere.
    splits: Dict[str, Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]] = {
        name: task.support_eval_split(n_support=n_support_pairs, seed=seed)
        for name, task in id_tasks.items()
    }

    # ── Strict non-overlap assertion ─────────────────────────────────────────
    # Verify support_pool ∩ eval_pool = ∅ for every task.
    # This is guaranteed by the slice partition in support_eval_split, but we
    # assert explicitly so any future refactoring that breaks it is caught.
    for name, (sup, ev) in splits.items():
        sup_set = set(x for x, _ in sup)
        ev_set  = set(x for x, _ in ev)
        overlap = sup_set & ev_set
        if overlap:
            raise AssertionError(
                f"Task {name!r}: {len(overlap)} input(s) appear in BOTH support "
                f"and eval pools — strict non-overlap violated.\n"
                f"  Overlapping inputs: {sorted(overlap)[:5]} …"
            )
    # ─────────────────────────────────────────────────────────────────────────

    support_prompts:  Dict[str, List[str]] = {}
    id_eval_prompts:  Dict[str, List[str]] = {}
    id_eval_answers:  Dict[str, List[str]] = {}

    for name, task in id_tasks.items():
        support_pool, eval_pool = splits[name]
        support_prompts[name] = task.sample_prompts(support_pool, n_support_prompts, n_shots, seed=seed)
        prompts, answers = task.sample_prompts_with_answers(eval_pool, n_eval_prompts, n_shots, seed=seed + 1)
        id_eval_prompts[name] = prompts
        id_eval_answers[name] = answers

    ood_eval_prompts: Dict[str, List[str]] = {}
    ood_eval_answers: Dict[str, List[str]] = {}
    for ood_name, ood_task in ood_tasks.items():
        parent_id_name = OOD_SPEC[ood_name][0]
        _, eval_pool = splits[parent_id_name]   # reuse the same split — no re-shuffle
        prompts, answers = ood_task.sample_prompts_with_answers(
            eval_pool, n_eval_prompts, n_shots, seed=seed + 2, ood=True
        )
        ood_eval_prompts[ood_name] = prompts
        ood_eval_answers[ood_name] = answers

    return ExperimentPrompts(
        support_prompts=support_prompts,
        id_eval_prompts=id_eval_prompts,
        ood_eval_prompts=ood_eval_prompts,
        id_eval_answers=id_eval_answers,
        ood_eval_answers=ood_eval_answers,
        task_names=list(id_tasks.keys()),
        ood_task_names=list(ood_tasks.keys()),
    )
