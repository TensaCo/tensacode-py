"""Real text proposals survive an explicitly supplied grounding revision.

The cached parser infers the syntax. This test supplies the scene correspondence;
it does not claim that the agent learned identity or selected the right meaning.
"""
from pathlib import Path

import pytest

from tensorcode.agent import Agent
from tensorcode.agent.grounding import MentionBinding, propose_grounding
from tensorcode.agent.understand import LearnedReader
from tensorcode.language import Entity
from tensorcode.records import Ref


def test_learned_source_and_decoder_evidence_survive_grounding_without_selection():
    model = Path.home() / ".cache/tensorcode/models/ud_ewt_parser.pickle"
    if not model.exists():
        pytest.skip("cached learned parser is unavailable")
    agent = Agent([], reader=LearnedReader(model))
    interpreted = agent.interpret("Birds fly.")
    workspace = agent.interpretations
    group = workspace.get(interpreted.group_ids[0])
    candidates = [
        (candidate, index)
        for candidate in group.candidates
        for index, act in enumerate(candidate.payload.acts)
        if act.frame is not None and act.frame.predicate == "fly"
        and isinstance(act.frame.roles.get("subject"), Entity)
        and act.frame.roles["subject"].text == "Birds"
    ]
    assert candidates, "learned alternatives should include the declarative reading"
    parent, index = candidates[0]
    evidence = workspace.add_source(
        "Explicit test correspondence, not an inferred visual identity",
        modality="observation", provider="test:authored-correspondence",
        payload={"reference": "scene:birds", "mention": "Birds"},
    )
    child = propose_grounding(workspace, group.id, parent.id, [MentionBinding(
        ("acts", index, "frame", "roles", "subject"), Ref("scene:birds"),
        (evidence.id,), "test supplies this correspondence to isolate evidence preservation",
    )])
    assert child.payload.acts[index].frame.roles["subject"].ref == Ref("scene:birds")
    assert child.payload.metadata == parent.payload.metadata
    assert child.payload.metadata["model_artifact"]["sha256"]
    source = workspace.get_source(interpreted.source_id)
    assert source.text == "Birds fly."
    for anchor in child.payload.metadata["token_anchors"]:
        start, stop = anchor["char_span"]
        assert source.text[start:stop] == anchor["token"]
    child.payload.metadata["heads"].clear()
    retained = workspace.get(group.id)
    assert retained.candidates[-1].payload.metadata["heads"]
    assert retained.selected_id is None
    assert not retained.history
    assert parent.payload.acts[index].frame.roles["subject"].ref is None
