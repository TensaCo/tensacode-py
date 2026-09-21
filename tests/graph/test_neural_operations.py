import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from tensorcode.ops.graph import Graph, SourceAnchor
from tensorcode.ops.graph.neural import GraphClassifier, GraphEncoder
from tensorcode.ops.vec import Latent, Space
from tensorcode.tracing import trace


def test_core_graph_import_does_not_require_torch():
    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise ImportError('torch is unavailable')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from tensorcode.ops.graph import Graph
assert Graph(nodes=('node',)).nodes == ('node',)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )

    assert completed.returncode == 0, completed.stderr


def test_graph_encoder_preserves_node_correspondence_and_source_metadata():
    graph = Graph(
        nodes=("a", "b", "c"),
        edges=(("a", "link", "b"), ("b", "link", "c")),
        sources=("dataset:item-12",),
        identity="item-12",
        node_attributes={
            "a": {"features": [1.0, 0.0]},
            "b": {"features": [0.0, 1.0]},
            "c": {"features": [1.0, 1.0]},
        },
        source_anchors=(SourceAnchor("dataset:item-12", target="b"),),
    )
    space = Space("fixture.graph-nodes", 4, organization="sequence")
    encoder = GraphEncoder(2, 5, 4, space=space, steps=2)

    encoded = encoder(graph)

    assert isinstance(encoded, Latent)
    assert encoded.tensor.shape == (3, 4)
    assert encoded.space == space
    assert encoded.sources == ("dataset:item-12",)
    assert encoded.metadata["node_ids"] == ("a", "b", "c")
    assert encoded.metadata["graph_identity"] == "item-12"
    assert encoded.metadata["source_anchors"][0]["target"] == "b"
    json.dumps(encoder.configuration())


def test_graph_encoder_has_a_real_gradient_path_from_node_output_to_inputs():
    torch.manual_seed(7)
    graph = Graph(nodes=("a", "b"), edges=(("a", "link", "b"),))
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    encoder = GraphEncoder(
        2,
        5,
        3,
        space=Space("fixture.gradient", 3, organization="sequence"),
        steps=2,
    )

    loss = encoder(graph, node_features=features).tensor.square().sum()
    loss.backward()

    assert features.grad is not None
    assert torch.count_nonzero(features.grad).item() > 0
    assert all(parameter.grad is not None for parameter in encoder.parameters())
    assert any(
        torch.count_nonzero(parameter.grad).item() > 0
        for parameter in encoder.parameters()
    )


def test_graph_encoder_uses_operation_context_for_traceable_supplied_features():
    graph = Graph(nodes=("a", "b"), edges=(("a", "link", "b"),))
    features = torch.tensor([[1.0], [2.0]], requires_grad=True)
    encoder = GraphEncoder(
        1,
        3,
        2,
        space=Space("fixture.trace", 2, organization="sequence"),
        steps=1,
    )

    with trace() as session:
        encoded = encoder(graph, context={"node_features": features})

    encoded.tensor.sum().backward()
    replayed = session.replay(encoded)

    assert len(session.calls) == 1
    assert features.grad is not None
    assert torch.equal(replayed.tensor, encoded.tensor)


def test_graph_prediction_path_is_one_traceable_trainable_operation():
    graph = Graph(
        nodes=("a", "b"),
        edges=(("a", "link", "b"),),
        node_attributes={"a": {"features": [1.0]}, "b": {"features": [2.0]}},
    )
    encoder = GraphEncoder(
        1,
        3,
        2,
        space=Space("fixture.prediction", 2, organization="sequence"),
        steps=1,
    )
    model = GraphClassifier(encoder, labels=("no", "yes"))

    with trace() as session:
        prediction = model(graph)

    prediction.logits.sum().backward()
    replayed = session.replay(prediction)

    assert len(session.calls) == 1
    assert model.output.weight.grad is not None
    assert torch.equal(replayed.logits, prediction.logits)


def test_graph_classifier_head_matches_encoder_dtype():
    graph = Graph(
        nodes=("a",),
        node_attributes={"a": {"features": [1.0]}},
    )
    encoder = GraphEncoder(
        1,
        3,
        2,
        space=Space("fixture.float64", 2, organization="sequence"),
        steps=1,
    ).double()

    prediction = GraphClassifier(encoder, labels=("no", "yes"))(graph)

    assert prediction.logits.dtype == torch.float64


def test_graph_classifier_preserves_native_encoder_forward_hooks():
    graph = Graph(
        nodes=("a",),
        node_attributes={"a": {"features": [1.0]}},
    )
    encoder = GraphEncoder(
        1,
        3,
        2,
        space=Space("fixture.hooks", 2, organization="sequence"),
        steps=1,
    )
    observed = []
    handle = encoder.register_forward_hook(lambda module, inputs, output: observed.append(output))

    try:
        prediction = GraphClassifier(encoder, labels=("no", "yes"))(graph)
    finally:
        handle.remove()

    assert observed == [prediction.node_latent]


def _authored_topology_fixture(prefix, connected, count):
    nodes = tuple(f"{prefix}:{index}" for index in range(count))
    edges = (
        tuple((nodes[index], "fixture-link", nodes[index + 1]) for index in range(count - 1))
        if connected
        else ()
    )
    return Graph(
        nodes=nodes,
        edges=edges,
        identity=prefix,
        node_attributes={node: {"features": [1.0]} for node in nodes},
    )


def test_graph_classifier_trains_and_generalizes_to_disjoint_authored_graphs():
    torch.manual_seed(7)
    train = tuple(
        (_authored_topology_fixture(f"train-{label}-{count}", bool(label), count), label)
        for count in (3, 4, 5)
        for label in (0, 1)
    )
    held_out = tuple(
        (_authored_topology_fixture(f"test-{label}-{count}", bool(label), count), label)
        for count in (6, 7)
        for label in (0, 1)
    )
    encoder = GraphEncoder(
        1,
        8,
        6,
        space=Space("fixture.topology", 6, organization="sequence"),
        steps=2,
    )
    model = GraphClassifier(encoder, labels=("isolated", "connected"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    with torch.no_grad():
        before = sum(
            torch.nn.functional.cross_entropy(model(graph).logits[None], torch.tensor([label]))
            for graph, label in held_out
        ).item()
    initial_encoder = encoder.input_projection.weight.detach().clone()

    for _ in range(100):
        optimizer.zero_grad()
        loss = sum(
            torch.nn.functional.cross_entropy(model(graph).logits[None], torch.tensor([label]))
            for graph, label in train
        )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        after = sum(
            torch.nn.functional.cross_entropy(model(graph).logits[None], torch.tensor([label]))
            for graph, label in held_out
        ).item()
        predictions = tuple(model(graph).value for graph, _ in held_out)

    assert after < before * 0.25
    assert predictions == ("isolated", "connected", "isolated", "connected")
    assert not torch.equal(initial_encoder, encoder.input_projection.weight)
