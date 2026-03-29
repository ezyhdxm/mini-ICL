"""Unit tests for icl.real_llm.task_data — transforms, OOD spec, and data integrity."""

import pytest
from icl.real_llm.task_data import (
    _reverse,
    _shift_one,
    _leetspeak,
    _first_char,
    _capitalize,
    _capitalize_last,
    _double,
    ICLTask,
    make_id_tasks,
    make_ood_tasks,
    OOD_SPEC,
    INPUT_OOD_SPEC,
    _ALL_ID_TASKS,
)


# ---------------------------------------------------------------------------
# ICLTask.clean()
# ---------------------------------------------------------------------------

class TestICLTaskClean:
    def _make_task(self, pairs):
        # pad to 10 pairs minimum
        base = [("a", "b")] * 10
        return ICLTask(name="test", pairs=base + pairs)

    def test_drops_identity_pairs(self):
        task = self._make_task([("read", "read"), ("dog", "dog")])
        cleaned = task.clean(drop_identity=True, single_word_only=False)
        idents = [(x, y) for x, y in cleaned.pairs if x.lower() == y.lower()]
        assert len(idents) == 0

    def test_keeps_identity_when_disabled(self):
        task = self._make_task([("read", "read")])
        cleaned = task.clean(drop_identity=False, single_word_only=False)
        assert ("read", "read") in cleaned.pairs

    def test_drops_multiword_outputs(self):
        task = self._make_task([("hello", "good morning"), ("cat", "le chat")])
        cleaned = task.clean(drop_identity=False, single_word_only=True)
        multi = [(x, y) for x, y in cleaned.pairs if " " in y]
        assert len(multi) == 0

    def test_keeps_multiword_when_disabled(self):
        task = self._make_task([("hello", "good morning")])
        cleaned = task.clean(drop_identity=False, single_word_only=False)
        assert ("hello", "good morning") in cleaned.pairs

    def test_returns_new_task_instance(self):
        task = self._make_task([("read", "read")])
        cleaned = task.clean()
        assert cleaned is not task


# ---------------------------------------------------------------------------
# OOD transform: _reverse
# ---------------------------------------------------------------------------

class TestReverse:
    def test_basic(self):
        assert _reverse("actor") == "rotca"

    def test_single_char(self):
        assert _reverse("a") == "a"

    def test_palindrome(self):
        assert _reverse("racecar") == "racecar"

    def test_with_space(self):
        assert _reverse("software engineer") == "reenigne erawtfos"

    def test_empty(self):
        assert _reverse("") == ""


# ---------------------------------------------------------------------------
# OOD transform: _shift_one  (Caesar +1)
# ---------------------------------------------------------------------------

class TestShiftOne:
    def test_basic(self):
        assert _shift_one("actor") == "bdups"

    def test_z_wraps_to_a(self):
        assert _shift_one("z") == "a"

    def test_Z_wraps_to_A(self):
        assert _shift_one("Z") == "A"

    def test_preserves_case(self):
        result = _shift_one("Physicist")
        assert result[0].isupper()
        assert result[1:].islower()

    def test_non_alpha_unchanged(self):
        assert _shift_one("hello-world") == "ifmmp-xpsme"
        assert _shift_one("abc 123") == "bcd 123"

    def test_physicist(self):
        assert _shift_one("physicist") == "qiztjdjtu"

    def test_roundtrip_26_shifts(self):
        word = "abcxyz"
        shifted = word
        for _ in range(26):
            shifted = _shift_one(shifted)
        assert shifted == word


# ---------------------------------------------------------------------------
# OOD transform: _leetspeak
# ---------------------------------------------------------------------------

class TestLeetspeak:
    def test_actor(self):
        assert _leetspeak("actor") == "4ct0r"

    def test_physicist(self):
        assert _leetspeak("physicist") == "phys1c1st"

    def test_engineer(self):
        assert _leetspeak("engineer") == "3ng1n33r"

    def test_no_vowels(self):
        assert _leetspeak("rhythm") == "rhythm"

    def test_uppercase_vowels(self):
        assert _leetspeak("ACTOR") == "4CT0R"

    def test_non_vowel_unchanged(self):
        assert _leetspeak("bcdfg") == "bcdfg"

    def test_u_unchanged(self):
        # u/U are intentionally not substituted
        assert _leetspeak("ubuntu") == "ubuntu"

    def test_idempotent(self):
        # applying twice gives different result (digits are not vowels)
        first  = _leetspeak("actor")
        second = _leetspeak(first)
        assert first == second   # digits are not re-substituted


# ---------------------------------------------------------------------------
# OOD transforms: _first_char, _capitalize, _double
# ---------------------------------------------------------------------------

class TestFirstChar:
    def test_basic(self):
        assert _first_char("cat") == "c"

    def test_uppercase(self):
        assert _first_char("Einstein") == "E"

    def test_single_char(self):
        assert _first_char("x") == "x"

    def test_empty(self):
        assert _first_char("") == ""


class TestCapitalize:
    def test_lowercase(self):
        assert _capitalize("cat") == "Cat"

    def test_allcaps_becomes_titlecase(self):
        assert _capitalize("COLD") == "Cold"

    def test_mixed(self):
        assert _capitalize("hOT") == "Hot"

    def test_already_capitalized(self):
        assert _capitalize("Justice") == "Justice"


class TestCapitalizeLast:
    def test_basic(self):
        assert _capitalize_last("cat") == "caT"

    def test_last_already_upper(self):
        assert _capitalize_last("caT") == "caT"

    def test_longer_word(self):
        assert _capitalize_last("justice") == "justicE"

    def test_single_char(self):
        assert _capitalize_last("a") == "A"

    def test_empty(self):
        assert _capitalize_last("") == ""


class TestDouble:
    def test_basic(self):
        assert _double("cat") == "catcat"

    def test_with_space(self):
        assert _double("Tony Pua") == "Tony PuaTony Pua"

    def test_single_char(self):
        assert _double("a") == "aa"


# ---------------------------------------------------------------------------
# INPUT_OOD_SPEC: verify all entries reference known ID tasks
# ---------------------------------------------------------------------------

class TestInputOodSpec:
    def test_all_parents_exist(self):
        for ood_name, (parent, g_func, desc) in INPUT_OOD_SPEC.items():
            assert parent in _ALL_ID_TASKS, (
                f"INPUT_OOD_SPEC[{ood_name!r}] references unknown task {parent!r}"
            )

    def test_transforms_are_callable(self):
        for ood_name, (parent, g_func, desc) in INPUT_OOD_SPEC.items():
            assert callable(g_func), f"g_func for {ood_name!r} is not callable"

    def test_input_transform_pairs(self):
        tasks = make_ood_tasks(list({p for p, *_ in INPUT_OOD_SPEC.values()}))
        for ood_name, (parent, g_func, desc) in INPUT_OOD_SPEC.items():
            if ood_name not in tasks:
                continue
            task = tasks[ood_name]
            # Every pair must satisfy y == g(x)
            for x, y in task.pairs[:20]:
                assert y == g_func(x), (
                    f"{ood_name}: pair ({x!r}, {y!r}) but g(x)={g_func(x)!r}"
                )


# ---------------------------------------------------------------------------
# ICLTask: whitespace stripping in __post_init__
# ---------------------------------------------------------------------------

class TestICLTaskWhitespaceStrip:
    def test_strips_x_and_y(self):
        pairs = [(" cat ", " chat "), ("dog\t", "\tperro"), *[("a", "b")] * 9]
        task = ICLTask(name="test", pairs=pairs)
        assert task.pairs[0] == ("cat", "chat")
        assert task.pairs[1] == ("dog", "perro")

    def test_clean_pairs_unchanged(self):
        pairs = [("cat", "chat")] * 10
        task = ICLTask(name="test", pairs=pairs)
        assert task.pairs[0] == ("cat", "chat")

    def test_too_few_pairs_raises(self):
        with pytest.raises(ValueError, match="needs ≥ 10 pairs"):
            ICLTask(name="test", pairs=[("a", "b")] * 5)


# ---------------------------------------------------------------------------
# make_id_tasks / make_ood_tasks
# ---------------------------------------------------------------------------

class TestMakeIdTasks:
    def test_returns_all_12_by_default(self):
        tasks = make_id_tasks()
        assert len(tasks) == 12

    def test_subset(self):
        tasks = make_id_tasks(["english_to_french", "antonyms"])
        assert set(tasks.keys()) == {"english_to_french", "antonyms"}

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            make_id_tasks(["nonexistent_task"])

    def test_all_tasks_have_enough_pairs(self):
        tasks = make_id_tasks()
        for name, task in tasks.items():
            assert len(task.pairs) >= 10, f"{name} has too few pairs"

    def test_cleaning_applied_by_default(self):
        tasks = make_id_tasks(["english_to_french", "antonyms"])
        for name, task in tasks.items():
            identity = [(x, y) for x, y in task.pairs if x.lower() == y.lower()]
            multi    = [(x, y) for x, y in task.pairs if " " in y]
            assert len(identity) == 0, f"{name}: identity pairs remain after clean()"
            assert len(multi)    == 0, f"{name}: multi-word outputs remain after clean()"

    def test_cleaning_disabled(self):
        tasks = make_id_tasks(
            ["english_to_french"],
            drop_identity=False,
            single_word_only=False,
        )
        raw_count = len(_ALL_ID_TASKS["english_to_french"])
        assert len(tasks["english_to_french"].pairs) == raw_count

    def test_cleaned_pair_counts(self):
        tasks = make_id_tasks(["english_to_french", "antonyms", "product_to_company"])
        assert len(tasks["english_to_french"].pairs)  == 4585
        assert len(tasks["antonyms"].pairs)           == 2377
        # product_to_company is balanced: capped at 29 per label (Microsoft+Apple
        # were 45 % of raw data); rare labels kept as-is → 328 total.
        assert len(tasks["product_to_company"].pairs) == 328


class TestMakeOodTasks:
    ACTIVE = ["english_to_french", "antonyms", "product_to_company"]

    def test_active_ood_names(self):
        ood = make_ood_tasks(self.ACTIVE)
        expected = {"french_input_first", "antonym_input_cap_last", "product_input_double"}
        assert set(ood.keys()) == expected

    def test_input_transform_pairs_are_g_of_x(self):
        """For input-transform OOD tasks, every pair must satisfy y == g(x)."""
        ood = make_ood_tasks(self.ACTIVE)
        for ood_name, (parent, g_func, _) in INPUT_OOD_SPEC.items():
            if ood_name not in ood:
                continue
            task = ood[ood_name]
            for x, y in task.pairs[:10]:
                assert y == g_func(x), (
                    f"{ood_name}: pair ({x!r}, {y!r}) but g(x)={g_func(x)!r}"
                )

    def test_french_input_first_examples(self):
        ood = make_ood_tasks(self.ACTIVE)
        task = ood["french_input_first"]
        for x, y in task.pairs[:5]:
            assert y == x[0], f"expected first char of {x!r}, got {y!r}"

    def test_antonym_input_cap_last_examples(self):
        ood = make_ood_tasks(self.ACTIVE)
        task = ood["antonym_input_cap_last"]
        for x, y in task.pairs[:5]:
            assert y == x[:-1] + x[-1].upper(), f"expected cap_last({x!r}), got {y!r}"

    def test_product_input_double_examples(self):
        ood = make_ood_tasks(self.ACTIVE)
        task = ood["product_input_double"]
        for x, y in task.pairs[:5]:
            assert y == x + x, f"expected {x!r}+{x!r}, got {y!r}"


# ---------------------------------------------------------------------------
# Data integrity: no whitespace in pairs, minimum pair counts
# ---------------------------------------------------------------------------

class TestDataIntegrity:
    @pytest.mark.parametrize("name,pairs", list(_ALL_ID_TASKS.items()))
    def test_no_whitespace_in_pairs(self, name, pairs):
        for x, y in pairs:
            assert x == x.strip(), f"{name}: input {repr(x)} has leading/trailing whitespace"
            assert y == y.strip(), f"{name}: output {repr(y)} has leading/trailing whitespace"

    @pytest.mark.parametrize("name,pairs", list(_ALL_ID_TASKS.items()))
    def test_no_empty_pairs(self, name, pairs):
        for x, y in pairs:
            assert x, f"{name}: empty input found"
            assert y, f"{name}: empty output found"

    def test_pair_counts(self):
        expected_minimums = {
            "english_to_french":  4000,
            "antonyms":           2000,
            "person_to_occupation": 800,
        }
        for name, minimum in expected_minimums.items():
            assert len(_ALL_ID_TASKS[name]) >= minimum, \
                f"{name}: expected ≥{minimum} pairs, got {len(_ALL_ID_TASKS[name])}"
