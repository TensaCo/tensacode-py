"""Alternative meanings survive procedural choices without becoming beliefs."""

from dataclasses import FrozenInstanceError

import pytest

from tensorcode.agent.interpretation import InterpretationWorkspace


def test_source_evidence_and_alternative_provenance_survive_decisions():
    workspace = InterpretationWorkspace()
    text = "  make a python hello world project\n"
    source = workspace.add_source(
        text, provider="grammar", metadata={"speaker": "user", "span": [0, len(text)]},
    )
    group = workspace.create_group(source.id, provenance=("sentence:0",))
    first = workspace.propose(group.id, ("make-python", "tell-project"), provenance=("reader:0",))
    second = workspace.propose(group.id, ("make-project",), provenance=("reader:1",))

    assert workspace.get(group.id).selected is None
    chosen = workspace.select(group.id, first.id, reason="reader preference")
    revised = workspace.select(group.id, second.id, reason="user clarified the intended object")

    assert chosen.selected.id == first.id
    assert revised.selected.id == second.id
    assert revised.candidates == (first, second)
    assert revised.provenance == ("sentence:0",)
    assert [entry.selected_id for entry in revised.history] == [first.id, second.id]
    assert [entry.reason for entry in revised.history] == [
        "reader preference", "user clarified the intended object",
    ]
    assert workspace.get_source(revised.source_id).text == text
    assert workspace.get_source(source.id).provider == "grammar"
    assert workspace.get_source(source.id).metadata["speaker"] == "user"


def test_mutable_inputs_and_every_read_path_are_detached_snapshots():
    workspace = InterpretationWorkspace()
    metadata = {"region": [1, 2, 3, 4]}
    source = workspace.add_source("image evidence reference", modality="image", metadata=metadata)
    metadata["region"][0] = 99
    source.metadata["region"][1] = 99
    assert workspace.get_source(source.id).metadata == {"region": [1, 2, 3, 4]}
    workspace.sources()[0].metadata["region"].clear()
    assert workspace.get_source(source.id).metadata == {"region": [1, 2, 3, 4]}

    group = workspace.create_group(source.id)
    payload = {"entities": [{"label": "folder"}]}
    candidate = workspace.propose(group.id, payload)
    payload["entities"][0]["label"] = "input mutation"
    candidate.payload["entities"].clear()
    chosen = workspace.select(group.id, candidate.id, reason="inspection policy")
    chosen.selected.payload["entities"][0]["label"] = "returned mutation"
    workspace.get(group.id).candidates[0].payload.clear()
    workspace.values()[0].candidates[0].payload.clear()
    assert workspace.get(group.id).selected.payload == {"entities": [{"label": "folder"}]}
    assert group.candidates == ()  # Earlier snapshots are not live handles.
    with pytest.raises(FrozenInstanceError):
        chosen.selected_id = None


def test_reject_unset_and_restore_retain_all_decisions():
    workspace = InterpretationWorkspace()
    source = workspace.add_source("ambiguous")
    group = workspace.create_group(source.id)
    a = workspace.propose(group.id, {"reading": "a"})
    b = workspace.propose(group.id, {"reading": "b"})
    workspace.select(group.id, a.id, reason="initial policy")
    rejected_other = workspace.reject(group.id, b.id, reason="contradicts observation")
    assert rejected_other.selected_id == a.id
    rejected_selected = workspace.reject(group.id, a.id, reason="new evidence")
    assert rejected_selected.selected is None
    assert all(c.rejected for c in rejected_selected.candidates)
    restored = workspace.select(group.id, b.id, reason="earlier observation was retracted")
    assert not restored.selected.rejected
    deferred = workspace.unset(group.id, reason="need clarification")
    assert deferred.selected is None
    assert len(deferred.candidates) == 2
    assert [r.operation for r in deferred.history] == ["select", "reject", "reject", "select", "unset"]
    assert [r.revision for r in deferred.history] == [1, 2, 3, 4, 5]
    assert deferred.revision == 5


def test_invalid_references_cannot_change_workspace_or_cross_groups():
    workspace = InterpretationWorkspace()
    with pytest.raises(KeyError):
        workspace.create_group("missing-source")
    assert workspace.values() == ()
    source = workspace.add_source("evidence")
    group = workspace.create_group(source.id)
    other = workspace.create_group(source.id)
    candidate = workspace.propose(other.id, "other reading")
    for operation in (workspace.select, workspace.reject):
        with pytest.raises(KeyError):
            operation(group.id, candidate.id, reason="wrong group")
        with pytest.raises(KeyError):
            operation(group.id, "missing-candidate", reason="bad reference")
    with pytest.raises(KeyError):
        workspace.propose("missing-group", "reading")
    with pytest.raises(KeyError):
        workspace.unset("missing-group", reason="missing")
    with pytest.raises(KeyError):
        workspace.get_source("missing-source")
    assert workspace.get(group.id) == group


@pytest.mark.parametrize("reason", ["", "  ", None, 4])
def test_decisions_require_reasons_and_fail_atomically(reason):
    workspace = InterpretationWorkspace()
    source = workspace.add_source("evidence")
    group = workspace.create_group(source.id)
    candidate = workspace.propose(group.id, "meaning")
    baseline = workspace.select(group.id, candidate.id, reason="initial preference")
    for operation, args in (
        (workspace.select, (group.id, candidate.id)),
        (workspace.reject, (group.id, candidate.id)),
        (workspace.unset, (group.id,)),
    ):
        with pytest.raises(ValueError):
            operation(*args, reason=reason)
        assert workspace.get(group.id) == baseline


def test_identical_inputs_have_distinct_stable_identities():
    workspace = InterpretationWorkspace()
    sources = [workspace.add_source("same") for _ in range(2)]
    groups = [workspace.create_group(sources[0].id) for _ in range(2)]
    candidates = [workspace.propose(groups[0].id, "same") for _ in range(2)]
    assert len({s.id for s in sources}) == 2
    assert len({g.id for g in groups}) == 2
    assert len({c.id for c in candidates}) == 2
    assert workspace.get(groups[0].id).candidates == tuple(candidates)


def test_provenance_is_a_sequence_of_strings_without_silent_string_splitting():
    workspace = InterpretationWorkspace()
    source = workspace.add_source("source")
    group = workspace.create_group(source.id)
    for invalid in ("reader", (42,)):
        with pytest.raises(TypeError):
            workspace.propose(group.id, "meaning", provenance=invalid)
    assert workspace.get(group.id).candidates == ()
