"""The selection harness: the truncation guard, the boundary metric, and failure attribution.

The guard exists because an evaluation in this repo published 0.093 as a capability number while
silently discarding two thirds of the evidence it had been handed. These tests pin the behaviour
that would have caught it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.selection.eval_selection import contains  # noqa: E402
from eval.selection.selector import Truncation, truncation_of  # noqa: E402


class TestTruncationGuard:
    def test_a_clean_arm_reports_clean(self):
        t = Truncation()
        for n in (100, 200, 383):
            t.record(n, 384)
        assert t.truncated == 0
        assert t.report()["verdict"] == "clean"
        assert t.report()["share_truncated"] == 0.0

    def test_it_counts_every_item_that_overflowed(self):
        t = Truncation()
        for n in (100, 500, 900, 384):
            t.record(n, 384)
        got = t.report()
        assert got["truncated"] == 2
        assert got["share_truncated"] == 0.5
        assert got["worst_case_tokens_dropped"] == 900 - 384
        assert "not a capability measurement" in got["verdict"]

    def test_the_boundary_is_inclusive(self):
        """An input of exactly the window length fits; off-by-one here hides real truncation."""
        t = Truncation()
        t.record(384, 384)
        assert t.truncated == 0

    def test_tokens_seen_never_exceeds_the_window(self):
        t = Truncation()
        t.record(2000, 384)
        assert t.report()["mean_tokens_seen"] == 384.0
        assert t.report()["mean_tokens_given"] == 2000.0

    def test_it_measures_the_pair_the_model_is_given(self):
        class FakeTokenizer:
            def __call__(self, question, context, truncation=False):
                return {"input_ids": list(range(len(f"{question} {context}".split())))}

        got = truncation_of(FakeTokenizer(), [("q", "a b c"), ("q", " ".join("w" * 40))], limit=5)
        assert got.items == 2
        assert got.truncated == 1, "the 4-token pair fits a 5-token window; the 40-token one does not"


class TestContainment:
    def test_word_runs_not_characters(self):
        """The reason this metric is not `normalize(pred) in normalize(gold)`."""
        assert not contains("no", ["Strathy Township of Temagami, Northeastern Ontario"])

    def test_either_direction_counts(self):
        assert contains("Berkeley", ["University of California, Berkeley"])
        assert contains("Alachua County", ["Alachua"])

    def test_a_budget_excludes_a_run_on_that_merely_swallows_the_gold(self):
        pred = "Glenn Ford, Vince Edwards, Shirley Jones and Erin Gray, with Edward Albert Heimberger"
        gold = ["Edward Albert Heimberger"]
        assert contains(pred, gold), "unbounded containment accepts it"
        assert not contains(pred, gold, budget=3), "a bounded metric does not: it is not the same answer"

    def test_a_budget_keeps_a_genuine_extent_dispute(self):
        assert contains("2006", ["2006 season"], budget=3)
        assert contains("Rick Ducommun", ['Richard "Rick" Ducommun'], budget=3)

    def test_an_empty_prediction_matches_nothing(self):
        assert not contains("", ["anything"])
        assert not contains("   ", ["anything"])


class TestFailureAttribution:
    def test_categories_are_disjoint_and_ordered(self):
        from eval.open_domain.data import Item
        from eval.selection.eval_selection import classify

        item = Item(id="x", question="Who replaced the manager who began at Leeds United?",
                    gold=["Alex McLeish"])
        assert classify("", item, item.question) == "abstained"
        assert classify("Alex McLeish", item, item.question) == "span_boundary", \
            "an exact match is containment; callers filter exact matches before attributing"
        assert classify("Leeds United", item, item.question) == "bridge_entity_returned"
        assert classify("Trafalgar Square", item, item.question) == "wrong_span"


@pytest.mark.parametrize("path", ["eval/selection/data.py", "eval/selection/selector.py",
                                  "eval/selection/eval_selection.py", "eval/selection/label_audit.py"])
def test_the_harness_imports_without_a_model(path):
    """Nothing in the harness may load weights at import time; the audit must run on a laptop."""
    import importlib

    mod = importlib.import_module(path.replace("/", ".")[:-3])
    assert mod.__doc__, "each harness module states what it measures and why"
