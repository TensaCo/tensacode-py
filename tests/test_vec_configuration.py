import json

import pytest
import torch

from tensorcode.ops.vec import (
    Classify,
    Decide,
    Decode,
    ImageEncoder,
    Latent,
    Retrieve,
    Score,
    Space,
    TextEncoder,
    Transform,
)


class Similarity(torch.nn.Module):
    def forward(self, query, candidates):
        return (query.unsqueeze(-2) * candidates).sum(dim=-1)


class Scale(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, value):
        return value * self.factor


class WeightRepr(torch.nn.Linear):
    def extra_repr(self):
        return f"learned={self.weight.detach().flatten().tolist()}"


class PrivateScale(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self._factor = factor

    def forward(self, value):
        return value * self._factor


class ExplicitScale(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor
        self.opaque_runtime_helper = object()

    def forward(self, value):
        return value * self.factor

    def configuration(self):
        return {"factor": self.factor}


def combine_with_default(value, context, factor=2):
    return value + context["bias"] * factor


def test_operation_configurations_are_json_safe_and_include_constructor_semantics():
    source = Space("source", 2)
    target = Space("target", 3)
    operations = (
        Transform(torch.nn.Linear(2, 3), input_space=source, output_space=target),
        TextEncoder(vocabulary=("one", "two"), dimensions=2, space=source),
        Classify(torch.nn.Linear(2, 2), labels=("yes", "no"), input_space=source),
        ImageEncoder(
            in_channels=1,
            patch_size=2,
            dimensions=2,
            space=Space("patches", 2, organization="spatial"),
        ),
        Decode(torch.nn.Linear(2, 1), input_space=source, output="scalar regression"),
        Score(
            Similarity(),
            query_space=source,
            candidate_space=source,
            meaning="dot-product relevance",
        ),
        Decide(largest=False),
        Retrieve(k=2),
    )

    serialized = [json.loads(json.dumps(operation.configuration())) for operation in operations]

    assert serialized[0]["input_space"] == source.configuration()
    assert serialized[0]["output_space"] == target.configuration()
    assert serialized[1]["vocabulary"] == ["one", "two"]
    assert serialized[2]["labels"] == ["yes", "no"]
    assert serialized[4]["output"] == "scalar regression"
    assert serialized[5]["meaning"] == "dot-product relevance"
    assert serialized[6]["largest"] is False
    assert serialized[7]["k"] == 2


def test_module_configuration_tracks_architecture_but_not_learned_values():
    operation = Transform(torch.nn.Linear(2, 3))
    before = operation.configuration()

    with torch.no_grad():
        operation.module.weight.fill_(91.0)
        operation.module.bias.fill_(-37.0)

    assert operation.configuration() == before
    assert Transform(torch.nn.Linear(2, 4)).configuration() != before
    serialized = json.dumps(before)
    assert "91.0" not in serialized
    assert "-37.0" not in serialized


def test_module_configuration_does_not_trust_repr_that_contains_weights():
    operation = Transform(WeightRepr(2, 2, bias=False))
    before = operation.configuration()

    with torch.no_grad():
        operation.module.weight.add_(100)

    assert operation.configuration() == before


def test_module_configuration_includes_custom_json_safe_behavior_attributes():
    assert Transform(Scale(2)).configuration() != Transform(Scale(3)).configuration()


def test_module_configuration_includes_private_custom_behavior_attributes():
    assert Transform(PrivateScale(2)).configuration() != Transform(PrivateScale(3)).configuration()


def test_explicit_module_configuration_is_authoritative_over_opaque_runtime_attrs():
    first = Transform(ExplicitScale(2)).configuration()
    second = Transform(ExplicitScale(3)).configuration()

    assert first != second
    json.dumps(first)


def test_named_callback_defaults_are_part_of_configuration_identity():
    original = combine_with_default.__defaults__
    try:
        before = Transform(torch.nn.Identity(), combine=combine_with_default).configuration()
        combine_with_default.__defaults__ = (3,)
        after = Transform(torch.nn.Identity(), combine=combine_with_default).configuration()
    finally:
        combine_with_default.__defaults__ = original

    assert before != after
    assert before["combine"]["defaults"] == [2]


def test_configuration_rejects_closures_without_explicit_metadata():
    def make_combine(factor):
        return lambda value, context: value * factor

    operation = Transform(torch.nn.Identity(), combine=make_combine(2))

    with pytest.raises(ValueError, match="explicit configuration"):
        operation.configuration()


def test_classify_space_validation_preserves_existing_prediction_api():
    space = Space("classifier/features", 2)
    classify = Classify(
        torch.nn.Linear(2, 2, bias=False),
        labels=("left", "right"),
        input_space=space,
    )
    vector = torch.ones(2, requires_grad=True)

    prediction = classify(Latent(vector, space))

    assert prediction.logits.shape == (2,)
    assert prediction.value in ("left", "right")
    prediction.logits.sum().backward()
    assert vector.grad is not None
    with pytest.raises(ValueError, match="incompatible.*space"):
        classify(Latent(torch.ones(2), Space("other", 2)))


def test_classify_label_order_is_part_of_persistable_configuration():
    module_a = torch.nn.Linear(2, 2)
    module_b = torch.nn.Linear(2, 2)
    first = Classify(module_a, labels=("yes", "no")).configuration()
    swapped = Classify(module_b, labels=("no", "yes")).configuration()

    assert first != swapped


def test_operations_can_share_the_same_trainable_backbone_instance():
    backbone = torch.nn.Linear(2, 2, bias=False)
    transform = Transform(backbone)
    classify = Classify(backbone, labels=("a", "b"))
    value = torch.tensor([1.0, -1.0])
    before = classify(value).logits.detach().clone()

    transform(value).sum().backward()
    torch.optim.SGD(transform.parameters(), lr=0.2).step()

    assert transform.module is classify.module
    assert next(transform.parameters()) is next(classify.parameters())
    assert not torch.equal(before, classify(value).logits.detach())
