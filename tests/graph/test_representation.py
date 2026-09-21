import json

import pytest

from tensorcode.ops.graph import Graph, JSONDecoder, JSONEncoder, SourceAnchor


def test_graph_freezes_extensible_attributes_away_from_caller_mutation():
    attributes = {"tags": ["review"], "owner": {"name": "Ada"}}
    graph = Graph(nodes=("case",), attributes=attributes)

    attributes["tags"].append("mutated")
    attributes["owner"]["name"] = "Grace"

    assert graph.attributes["tags"] == ("review",)
    assert graph.attributes["owner"]["name"] == "Ada"
    with pytest.raises(TypeError):
        graph.attributes["new"] = True


def test_equal_graph_attribute_maps_have_equal_hashes_regardless_of_input_order():
    first = Graph(nodes=("case",), attributes={"a": 1, "b": 2})
    second = Graph(nodes=("case",), attributes={"b": 2, "a": 1})

    assert first == second
    assert hash(first) == hash(second)


def test_source_anchor_contributes_its_explicit_source_reference():
    graph = Graph(
        nodes=("case",),
        source_anchors=(SourceAnchor("document:anchor", target="case"),),
    )

    assert graph.sources == ("document:anchor",)


def test_json_roundtrip_preserves_identity_relations_attributes_and_anchors():
    original = Graph(
        nodes=("case", "open", "closed"),
        edges=(("case", "status", "open"), ("case", "status", "closed")),
        sources=("document:a", "document:b"),
        identity="case-7",
        attributes={"schema": "fixture-v1"},
        node_attributes={"case": {"kind": "record"}},
        edge_attributes=({"asserted": True}, {"asserted": False}),
        source_anchors=(
            SourceAnchor("document:a", target="case", location={"page": 2}),
            SourceAnchor("document:b", target=1, location={"line": [8, 9]}),
        ),
    )

    payload = JSONDecoder(as_text=True)(original)
    restored = JSONEncoder()(payload)

    assert restored == original
    assert json.loads(payload)["schema"] == "tensorcode.graph/v1"
    assert JSONDecoder().configuration() == {
        "operation": "graph.json_decode",
        "as_text": False,
    }


def test_roundtrip_retains_competing_facts_in_order():
    graph = Graph(
        nodes=("ticket", "open", "closed"),
        edges=(("ticket", "state", "open"), ("ticket", "state", "closed")),
    )

    restored = JSONEncoder()(JSONDecoder()(graph))

    assert restored.edges == (
        ("ticket", "state", "open"),
        ("ticket", "state", "closed"),
    )


def test_json_encoder_rejects_unknown_referents_instead_of_inventing_nodes():
    payload = {
        "schema": "tensorcode.graph/v1",
        "nodes": [{"id": "known", "attributes": {}}],
        "edges": [
            {
                "source": "known",
                "relation": "mentions",
                "target": "missing",
                "attributes": {},
            }
        ],
        "sources": [],
        "source_anchors": [],
        "attributes": {},
        "identity": None,
    }

    with pytest.raises(ValueError, match="existing nodes"):
        JSONEncoder()(payload)


@pytest.mark.parametrize("index", [-1, True])
def test_edge_attribute_lookup_rejects_noncanonical_indices(index):
    graph = Graph(nodes=("a", "b"), edges=(("a", "link", "b"),))

    with pytest.raises(ValueError, match="edge index"):
        graph.attributes_for_edge(index)
