"""Grounding is an explicit, evidence-backed semantic alternative."""
from dataclasses import replace
import json

import pytest

from tensorcode.agent.grounding import MentionBinding, propose_grounding
from tensorcode.agent.interpretation import InterpretationWorkspace
from tensorcode.agent.understand import Act, SentenceAlternative
from tensorcode.language import Entity, Frame, Question, Request
from tensorcode.language.conventions import RequestInterpretation
from tensorcode.records import Ref


def setup(kind="request"):
    workspace = InterpretationWorkspace()
    source = workspace.add_source("compare the folder with the folder")
    image = workspace.add_source("scene", modality="image", payload=b"raw pixels")
    group = workspace.create_group(source.id)
    # Deliberately share an object: occurrence identity must win over Python identity.
    mention = Entity("name", "folder", {"unresolved": Entity("name", "owner")})
    frame = Frame("compare", {"left": mention, "right": mention})
    meaning = Request(frame) if kind == "request" else Question(frame, "polarity") if kind == "question" else frame
    act = Act(kind, meaning, frame)
    parent = workspace.propose(group.id, SentenceAlternative(None, (act,)))
    return workspace, group, parent, image


def binding(source, role="left", reference="world:first", path=None):
    return MentionBinding(path or ("acts", 0, "frame", "roles", role), Ref(reference), (source.id,), "inspected scene correspondence")


@pytest.mark.parametrize("kind", ["request", "question", "tell"])
def test_occurrences_are_independent_and_source_selection_is_unchanged(kind):
    workspace, group, parent, image = setup(kind)
    workspace.select(group.id, parent.id, reason="explicit caller selection")
    child = propose_grounding(workspace, group.id, parent.id, [binding(image), binding(image, "right", "world:second")])
    frame = child.payload.acts[0].frame
    assert frame.roles["left"].ref == Ref("world:first")
    assert frame.roles["right"].ref == Ref("world:second")
    assert frame.roles["left"].features["unresolved"].ref is None
    assert parent.payload.acts[0].frame.roles["left"].ref is None
    assert parent.payload.acts[0].frame.roles["right"].ref is None
    state = workspace.get(group.id)
    assert state.selected_id == parent.id
    assert state.history[-1].operation == "select"
    assert len(state.candidates) == 2
    meaning = child.payload.acts[0].meaning
    assert (meaning if kind == "tell" else meaning.frame) == frame
    assert f"grounding-parent:{parent.id}" in child.provenance
    audit = [json.loads(item[len("grounding:"):]) for item in child.provenance if item.startswith("grounding:")]
    assert audit[0]["evidence_ids"] == [image.id]
    assert audit[0]["path"] == ["acts", 0, "frame", "roles", "left"]
    # Caller mutation cannot modify retained proposals.
    frame.roles["left"].features.clear()
    assert "unresolved" in workspace.get(group.id).candidates[-1].payload.acts[0].frame.roles["left"].features


def test_nested_features_frames_and_tuple_occurrences():
    workspace, group, parent, image = setup()
    mention = Entity("name", "same")
    frame = Frame("outer", {"custom": Entity("name", "container", {"custom-feature": (Frame("inner", {"custom-role": mention}), mention)})})
    parent = workspace.propose(group.id, SentenceAlternative(None, (Act("request", Request(frame), frame),)))
    path = ("acts", 0, "frame", "roles", "custom", "features", "custom-feature", 0, "roles", "custom-role")
    child = propose_grounding(workspace, group.id, parent.id, [binding(image, path=path)])
    values = child.payload.acts[0].frame.roles["custom"].features["custom-feature"]
    assert values[0].roles["custom-role"].ref == Ref("world:first")
    assert values[1].ref is None


def test_existing_binding_conflict_requires_alternative_from_original():
    workspace, group, parent, image = setup()
    first = propose_grounding(workspace, group.id, parent.id, [binding(image)])
    with pytest.raises(ValueError, match="conflicts"):
        propose_grounding(workspace, group.id, first.id, [binding(image, reference="world:other")])
    second = propose_grounding(workspace, group.id, parent.id, [binding(image, reference="world:other")])
    compatible = propose_grounding(workspace, group.id, first.id, [binding(image)])
    assert second.payload.acts[0].frame.roles["left"].ref == Ref("world:other")
    assert compatible.payload.acts[0].frame.roles["left"].ref == Ref("world:first")


def test_request_convention_meaning_stays_consistent():
    workspace, group, parent, image = setup()
    act = parent.payload.acts[0]
    act = replace(act, interpretation=RequestInterpretation(act.meaning, "authored", "test"))
    parent = workspace.propose(group.id, replace(parent.payload, acts=(act,)))
    child = propose_grounding(workspace, group.id, parent.id, [binding(image)])
    act = child.payload.acts[0]
    assert act.frame == act.meaning.frame == act.interpretation.request.frame
    assert act.interpretation.convention_id == "authored"


def test_invalid_bindings_leave_workspace_unchanged():
    workspace, group, parent, image = setup()
    cases = [
        [binding(image), binding(image)],
        [binding(image, path=("acts", -1, "frame", "roles", "left"))],
        [binding(image, path=("acts", 0, "frame", "roles", "missing"))],
        [binding(image, path=("acts", 0, "frame", "roles", "left", "text"))],
        [binding(image, path=("acts", 0, "frame", "roles"))],
        [binding(image), MentionBinding(("acts", 0, "frame", "roles", "right"), Ref("world:x"), ("missing",), "test")],
        [],
    ]
    for bindings in cases:
        before = workspace.get(group.id)
        with pytest.raises((ValueError, KeyError)):
            propose_grounding(workspace, group.id, parent.id, bindings)
        assert workspace.get(group.id) == before
    other = workspace.create_group(image.id)
    with pytest.raises(ValueError, match="belong"):
        propose_grounding(workspace, other.id, parent.id, [binding(image)])


@pytest.mark.parametrize("kwargs", [{"basis": " "}, {"evidence_ids": ()}, {"reference": "world:x"}, {"path": ("acts", True, "frame", "roles", "left")}])
def test_binding_requires_explicit_identity_evidence_and_occurrence(kwargs):
    fields = dict(path=("acts", 0, "frame", "roles", "left"), reference=Ref("world:x"), evidence_ids=("source:x",), basis="inspection")
    with pytest.raises((TypeError, ValueError)):
        MentionBinding(**(fields | kwargs))
