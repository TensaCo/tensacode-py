import pytest

from tensorcode.ops.graph import Graph, SourceAnchor


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


def test_representation_preserves_competing_edges_and_source_anchors():
    graph = Graph(
        nodes=("ticket", "open", "closed"),
        edges=(("ticket", "state", "open"), ("ticket", "state", "closed")),
        source_anchors=(SourceAnchor("document:a", target=0),),
    )
    assert graph.neighbors("ticket", relation="state") == ("open", "closed")
    assert graph.sources == ("document:a",)
    assert graph.source_anchors[0].target == 0


def test_representation_rejects_unknown_referents():
    with pytest.raises(ValueError, match="existing nodes"):
        Graph(nodes=("known",), edges=(("known", "mentions", "missing"),))


@pytest.mark.parametrize("index", [-1, True])
def test_edge_attribute_lookup_rejects_noncanonical_indices(index):
    graph = Graph(nodes=("a", "b"), edges=(("a", "link", "b"),))

    with pytest.raises(ValueError, match="edge index"):
        graph.attributes_for_edge(index)
