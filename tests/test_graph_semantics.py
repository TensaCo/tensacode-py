import json

import pytest

from tensorcode.ops.graph import (
    ChoiceInput,
    Decide,
    Graph,
    Retrieve,
    Score,
    Transform,
)


def test_score_uses_identified_supplied_semantics_and_exposes_configuration():
    graph = Graph(nodes=("a", "b"), edges=(("a", "links", "b"),))
    operation = Score(
        lambda candidate, context: len(candidate.edges) * context["weight"],
        semantics="fixture.edge-count-v1",
    )

    assert operation(graph, context={"weight": 2.5}) == 2.5
    assert operation.configuration() == {
        "operation": "graph.score",
        "semantics": "fixture.edge-count-v1",
    }
    json.dumps(operation.configuration())


def test_supplied_callback_needs_identity_before_its_configuration_is_persistable():
    transform = Transform(lambda graph, context: graph)

    with pytest.raises(ValueError, match="identity"):
        transform.configuration()
    with pytest.raises(ValueError, match="semantics"):
        Score(lambda graph, context: 1.0, semantics="")


def test_retrieve_ranks_only_supplied_graphs_and_retains_sources():
    query = Graph(nodes=("query",), identity="q")
    low = Graph(nodes=("low",), sources=("source:low",), identity="low")
    high = Graph(nodes=("high",), sources=("source:high",), identity="high")
    operation = Retrieve(
        (low, high),
        lambda requested, candidate, context: context[candidate.identity],
        semantics="fixture.lookup-v1",
        limit=1,
    )

    matches = operation(query, context={"low": 0.25, "high": 0.75})

    assert tuple(match.value.identity for match in matches) == ("high",)
    assert matches[0].value.sources == ("source:high",)
    assert matches[0].score == 0.75
    assert operation.configuration()["items"][0]["identity"] == "low"
    json.dumps(operation.configuration())


def test_decide_scores_every_supplied_option_and_selects_first_best():
    objective = Graph(nodes=("objective",), identity="objective")
    first = Graph(nodes=("first",), identity="first")
    second = Graph(nodes=("second",), identity="second")
    operation = Decide(
        lambda requested, option, context: context[option.identity],
        semantics="fixture.utility-v1",
    )

    decision = operation(
        ChoiceInput(objective, (first, second)),
        context={"first": 4.0, "second": 4.0},
    )

    assert decision.value is first
    assert tuple(item.value for item in decision.scores) == (first, second)
    assert tuple(item.score for item in decision.scores) == (4.0, 4.0)
    assert decision.semantics == "fixture.utility-v1"


def test_supplied_semantics_must_return_finite_real_scores():
    graph = Graph(nodes=("a",))

    with pytest.raises(TypeError, match="real number"):
        Score(lambda candidate, context: "high", semantics="bad-v1")(graph)
    with pytest.raises(ValueError, match="finite"):
        Score(lambda candidate, context: float("nan"), semantics="bad-v2")(graph)
