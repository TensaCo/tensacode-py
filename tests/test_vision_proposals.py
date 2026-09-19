"""Classifier proposals retain alternatives without pretending to understand scenes."""

from types import SimpleNamespace

import numpy as np
import pytest

from tensorcode.agent.vision_plugin import VisionPlugin
from tensorcode.records import Ref


def plugin(monkeypatch, tmp_path, probabilities=((0.55, 0.45),), labels=("cat", "dog")):
    monkeypatch.setenv("TENSORCODE_SCRATCH", str(tmp_path))
    vision = VisionPlugin("test-model")
    calls = []

    def predict(features):
        calls.append(features)
        return probabilities

    vision.model = {
        "size": 2,
        "labels": labels,
        "hierarchy": SimpleNamespace(describe=lambda values: values),
        "scaler": SimpleNamespace(transform=lambda values: values),
        "classifier": SimpleNamespace(predict_proba=predict),
    }
    return vision, calls


def test_all_classifier_alternatives_retained_with_stable_identity(monkeypatch, tmp_path):
    vision, calls = plugin(monkeypatch, tmp_path)
    ref = Ref("image:example")
    proposals = tuple(vision.interpret_image(np.zeros((2, 2, 3)), ref))
    assert len(calls) == 1
    assert len(proposals) == 2
    assert [p.score.value for p in proposals] == [0.55, 0.45]
    assert all(p.score.kind == "uncalibrated" for p in proposals)
    assert all("no calibration" in p.score.basis for p in proposals)
    assert proposals[0].graph.nodes == proposals[1].graph.nodes == (Ref("image:example/entity"),)
    assert [p.graph.propositions[1].roles["object"] for p in proposals] == ["cat", "dog"]
    for proposal in proposals:
        assert proposal.graph.image == ref
        assert proposal.graph.limitations == ("whole-image classification only; no scene structure inferred",)
        assert len(proposal.graph.propositions) == 2
        assert proposal.provenance == ("vision:test-model", "classifier.predict_proba")
    assert not hasattr(vision, "see")


@pytest.mark.parametrize("probabilities,labels", [
    ((0.5, 0.5), ("cat", "dog")),  # batch dimension missing
    (((0.5, 0.5), (0.5, 0.5)), ("cat", "dog")),
    (((1.0,),), ("cat", "dog")),
    (((float("nan"), 0.5),), ("cat", "dog")),
    (((float("inf"), 0.0),), ("cat", "dog")),
    (((-0.1, 1.1),), ("cat", "dog")),
    (((0.2, 0.2),), ("cat", "dog")),
    ((("bad", 0.5),), ("cat", "dog")),
    (((0.5, 0.5),), ("cat", "cat")),
    (((0.5, 0.5),), ("cat", "")),
    (((),), ()),
])
def test_malformed_distributions_produce_no_interpretation(monkeypatch, tmp_path, probabilities, labels):
    vision, _ = plugin(monkeypatch, tmp_path, probabilities, labels)
    image, ref = np.zeros((2, 2, 3)), Ref("image:example")
    assert tuple(vision.interpret_image(image, ref)) == ()


def test_unavailable_model_produces_no_interpretation(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORCODE_SCRATCH", str(tmp_path))
    vision = VisionPlugin("missing")
    assert tuple(vision.interpret_image(object(), Ref("image:example"))) == ()


def test_model_vocabulary_and_zero_score_alternatives_are_preserved(monkeypatch, tmp_path):
    vision, _ = plugin(monkeypatch, tmp_path, ((1.0, 0.0),), ("indoor workspace", "unfamiliar-layout"))
    proposals = tuple(vision.interpret_image(np.zeros((2, 2, 3)), Ref("image:renamed")))
    assert [p.graph.propositions[1].roles["object"] for p in proposals] == [
        "indoor workspace", "unfamiliar-layout"
    ]
    assert [p.score.value for p in proposals] == [1.0, 0.0]
    assert all(p.graph.nodes == (Ref("image:renamed/entity"),) for p in proposals)
