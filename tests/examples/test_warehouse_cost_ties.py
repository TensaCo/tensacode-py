"""Explicit dispatch cost resolution; original counterexamples stay immutable."""

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.robotics.warehouse.coordination import Coordinator, order_cost_groups
from examples.robotics.warehouse.substrate_audit import (
    DEFAULT_OUTPUT,
    evaluate,
    fixtures,
    prepare,
)


def test_groups_are_anchored_not_chained_and_boundary_is_inclusive():
    rows = [(0.0, "z", None), (0.75, "a", None), (1.5, "b", None)]
    assert [r[1] for r in order_cost_groups(rows, 1.0, 0.0)] == ["a", "z", "b"]
    rows = [
        (0.0, "z", None),
        (1.0, "b", None),
        (math.nextafter(1.0, math.inf), "a", None),
    ]
    assert [r[1] for r in order_cost_groups(rows, 1.0, 0.0)] == ["b", "z", "a"]
    assert order_cost_groups(list(reversed(rows)), 1.0, 0.0) == order_cost_groups(
        rows, 1.0, 0.0
    )


def test_relative_cost_resolution_and_distinct_costs():
    rows = [(1e6, "z", None), (1e6 + 5e-7, "a", None), (1e6 + 2e-6, "b", None)]
    assert [r[1] for r in order_cost_groups(rows)] == ["a", "z", "b"]
    assert [r[1] for r in order_cost_groups(rows, 0.0, 0.0)] == ["z", "a", "b"]
    assert order_cost_groups([(1.0, "z", None), (1.000001, "a", None)])[0][1] == "z"


@pytest.mark.parametrize("tolerance", [-1, math.inf, math.nan])
def test_invalid_tolerance_rejected(tolerance):
    with pytest.raises(ValueError, match="tolerances"):
        Coordinator(cost_absolute_tolerance=tolerance)


@pytest.mark.parametrize("substrate", ["alg0", "alg1", "alg2"])
def test_original_permutation_counterexamples_now_choose_lexical_robot(substrate):
    receipt = json.loads((DEFAULT_OUTPUT / "tie-counterexamples.json").read_text())
    coordinator = Coordinator(substrate=substrate)
    for row in receipt["counterexamples"]:
        result = evaluate(row["case"], coordinator)
        assert result["assignments"][0]["robot_id"] == "a", row["case"]["id"]


def test_task_priority_is_lexical_and_input_order_independent():
    protocol = json.loads((DEFAULT_OUTPUT / "protocol.json").read_text())
    case = fixtures(protocol)[0]
    tasks, robots, scene = prepare(case)
    a, z = replace(tasks[0], id="a"), replace(tasks[0], id="z")
    for ordering in ([a, z], [z, a]):
        assignments = Coordinator().dispatch(
            ordering, list(reversed(robots)), scene, now=0.0
        )
        assert [(a.task_id, a.robot_id) for a in assignments] == [("a", "a")]


def test_versioned_replay_keeps_original_costs_and_resolves_recorded_decisions():
    original = json.loads((DEFAULT_OUTPUT / "tie-counterexamples.json").read_text())
    updated = json.loads((DEFAULT_OUTPUT / "v2/tie-counterexamples.json").read_text())
    assert original["actual_dispatch_mismatches"] == 140
    assert updated["actual_dispatch_mismatches"] == 0
    assert len(updated["counterexamples"]) == len(original["counterexamples"])
    for before, after in zip(original["counterexamples"], updated["counterexamples"]):
        assert before["case"] == after["case"]
        assert set(after["chosen"].values()) == {"a"}
        for substrate in ("alg0", "alg1", "alg2"):
            assert (
                before["results"][substrate]["gates"]
                == after["results"][substrate]["gates"]
            )
