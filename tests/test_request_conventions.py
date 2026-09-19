"""Request interpretation is replaceable register knowledge with visible authority."""
from dataclasses import replace
import json

import pytest

from tensorcode.agent.understand import act_of, acts_of, indirect_request
from tensorcode.language.conventions import interpret_request, request_conventions
from tensorcode.language.semantics import Entity, Frame, Question


def question(modality="could", *, person=2, asked="polarity", polarity="positive"):
    return Question(Frame("open", {"subject": Entity("pronoun", "you", {"person": person}),
                                  "object": Entity("file", "report")},
                          {"modality": modality, "polarity": polarity, "tense": "present",
                           "mood": "interrogative", "aspect": "perfect"}), asked)


@pytest.mark.parametrize("modal", ["can", "could", "would", "will"])
def test_seed_preserves_request_and_explains_authority(modal):
    q = question(modal)
    act = act_of(q)
    assert act.kind == "request"
    assert act.interpretation.convention_id == "en.neutral.listener-modal-request"
    assert "defeasible" in act.interpretation.source
    assert act.frame.roles == {"object": q.frame.roles["object"]}
    assert act.frame.features == {"polarity": "positive", "aspect": "perfect", "mood": "imperative"}
    assert q.frame.features["modality"] == modal
    assert indirect_request(q) == act.meaning


@pytest.mark.parametrize("q", [question("must"), question(person=1), question(asked="object"),
                                  question(polarity="negative")])
def test_minimal_changes_preserve_literal_question(q):
    assert act_of(q).kind == "question"
    assert interpret_request(q) is None


def test_caller_can_replace_or_disable_convention_for_coordinated_meanings():
    seed = request_conventions()[0]
    local = replace(seed, id="local.might", source="caller register", frame={"modality": ["might"], "negated": [False]})
    acts = acts_of((question("might"), question("could")), [local])
    assert [act.kind for act in acts] == ["request", "question"]
    assert acts[0].interpretation.source == "caller register"
    assert act_of(question(), []).kind == "question"


def test_configured_data_replaces_seed_and_can_change_transformation(tmp_path, monkeypatch):
    path = tmp_path / "register.json"
    path.write_text(json.dumps([{
        "id": "local", "source": "test register", "question": {"asked": ["polarity"]},
        "frame": {"modality": ["might"]}, "subject": {"person": [2]},
        "remove_roles": [], "remove_features": ["mood"], "add_features": {"mood": "imperative", "register": "local"},
    }]))
    monkeypatch.setenv("TENSORCODE_REQUEST_CONVENTIONS", str(path))
    assert act_of(question()).kind == "question"
    act = act_of(question("might"))
    assert act.interpretation.convention_id == "local"
    assert "subject" in act.frame.roles
    assert act.frame.features["register"] == "local"


@pytest.mark.parametrize("data", [{}, [42], [{"id": "missing-fields"}],
    [{"id": "invalid", "source": "seed", "question": {}, "frame": {"modality": "could"}, "subject": {}}],
    [{"id": "invalid", "source": "", "question": {}, "frame": {}, "subject": {}}],
])
def test_malformed_conventions_fail_visibly(data):
    with pytest.raises(ValueError):
        request_conventions(data)


def test_missing_configured_file_does_not_restore_seed(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORCODE_REQUEST_CONVENTIONS", str(tmp_path / "missing.json"))
    with pytest.raises(FileNotFoundError):
        act_of(question())


def test_duplicate_authority_identifiers_are_rejected():
    seed = request_conventions()[0]
    with pytest.raises(ValueError, match="unique"):
        request_conventions([seed, seed])


def test_reader_propagates_override_and_keeps_quoted_requests_as_mentions():
    from tensorcode.agent.understand import read
    from tensorcode.language import ENGLISH

    text = "could you open the file?"
    assert read(ENGLISH, text, conventions=[])[0].acts[0].kind == "question"
    quoted = read(ENGLISH, f'"{text}"')[0].acts[0]
    assert quoted.kind == "mention"
    assert quoted.interpretation.convention_id == "en.neutral.listener-modal-request"
