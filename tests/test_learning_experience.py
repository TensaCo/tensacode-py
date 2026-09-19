"""Executed fixture trajectories teach rule conditions; projection is explicitly authored."""
from dataclasses import replace

import pytest

from tensorcode.agent.interpretation import InterpretationWorkspace
from tensorcode.agent.plugin import Call
from tensorcode.learning.experience import Projection, extract_transitions, fit_transitions
from tensorcode.outcomes import Receipt, Unknown


PROVIDER = "plugin:latch"
PROJECTION = Projection("latch-features/v1", lambda before, action: {**before, **dict(action.args)},
                        lambda after: after["open"], ("test:authored-observation-projection",))


class Latch:
    def __init__(self, powered, enabled):
        self.powered, self.enabled, self.open = powered, enabled, False

    def observe(self):
        return {"powered": self.powered, "enabled": self.enabled, "open": self.open}

    def step(self, press):
        self.open = self.powered and self.enabled and press


def trajectories():
    workspace = InterpretationWorkspace()
    train, held = [], []
    for trial in range(12):
        for powered in (False, True):
            for enabled in (False, True):
                for press in (False, True):
                    attempt = f"{trial}:{powered}:{enabled}:{press}"
                    (train if trial < 9 else held).append(attempt)
                    world = Latch(powered, enabled)
                    action = Call("latch", "press", (("press", press),))
                    workspace.add_source("", modality="observation", provider=PROVIDER,
                                         payload=world.observe(), metadata={"stage": "before_action", "attempt_id": attempt,
                                         "action": action, "receipt": None, "status": "observed"})
                    world.step(press)
                    workspace.add_source("", modality="observation", provider=PROVIDER,
                                         payload=world.observe(), metadata={"stage": "after_action", "attempt_id": attempt,
                                         "action": action, "receipt": Receipt(action, "applied"), "status": "observed"})
    return workspace, train, held


def model():
    workspace, train, held = trajectories()
    batch = extract_transitions(workspace.sources(), provider=PROVIDER)
    return fit_transitions(batch.transitions, projection=PROJECTION, train_attempt_ids=train,
                           evaluation_attempt_ids=held), batch, train, held


def test_rules_learn_interacting_conditions_from_executed_trajectories():
    learned, batch, train, held = model()
    assert learned.artifact.rules
    answer = learned.predict({"powered": True, "enabled": False, "open": False},
                             Call("latch", "press", (("press", True),)))
    assert not isinstance(answer, Unknown)
    assert answer.outcome is False
    assert set(answer.evidence.training_attempt_ids) <= set(train)
    assert set(answer.evidence.evaluation_attempt_ids) <= set(held)
    assert answer.evidence.source_ids
    assert answer.projection_provenance == PROJECTION.provenance
    assert learned.evaluation.accuracy == 1.0
    assert learned.evaluation.coverage > 0
    # A fresh execution establishes the expected outcome independently of the model.
    world = Latch(True, False)
    world.step(True)
    assert answer.outcome == world.observe()["open"]


def test_missing_observations_do_not_trigger_negated_rules_or_majority_default():
    learned, *_ = model()
    answer = learned.predict({}, Call("latch", "press", ()))
    assert isinstance(answer, Unknown)
    # Returned artifacts are detached; clients cannot bypass validation by editing one.
    artifact = learned.artifact
    artifact.rules.clear()
    assert learned.artifact.rules


def test_split_overlap_duplicates_and_omissions_are_rejected():
    _, batch, train, held = model()
    for training, evaluation in ((train + [held[0]], held), (train + [train[0]], held), (train[:-1], held)):
        with pytest.raises(ValueError):
            fit_transitions(batch.transitions, projection=PROJECTION, train_attempt_ids=training,
                            evaluation_attempt_ids=evaluation)


def test_heldout_contradiction_blocks_the_affected_rule():
    _, batch, train, held = model()
    altered = tuple(replace(t, after={**t.after, "open": True}) if t.attempt_id in held else t
                    for t in batch.transitions)
    learned = fit_transitions(altered, projection=PROJECTION, train_attempt_ids=train, evaluation_attempt_ids=held)
    answer = learned.predict({"powered": True, "enabled": False, "open": False}, Call("latch", "press", (("press", True),)))
    assert isinstance(answer, Unknown) and answer.reason == "unverified_transition"
    assert any("heldout contradictions" in reason for e in learned.evidence for reason in e.reasons)


@pytest.mark.parametrize("status", ["rejected", "indeterminate", "failed"])
def test_nonapplied_receipts_never_become_successful_training_pairs(status):
    workspace, *_ = trajectories()
    sources = list(workspace.sources()[:2])
    after = sources[1]
    sources[1] = replace(after, metadata={**after.metadata, "receipt": Receipt(after.metadata["action"], status)})
    batch = extract_transitions(sources, provider=PROVIDER)
    assert not batch.transitions and len(batch.exclusions) == 1


def test_missing_duplicate_mismatched_and_unavailable_sources_are_explicit_exclusions():
    workspace, *_ = trajectories()
    before, after = workspace.sources()[:2]
    cases = [(before,), (before, replace(before, id="different"), after),
             (before, replace(after, metadata={**after.metadata, "status": "unavailable"})),
             (before, replace(after, metadata={**after.metadata, "action": Call("other", "press", ())}))]
    for sources in cases:
        batch = extract_transitions(sources, provider=PROVIDER)
        assert not batch.transitions and batch.exclusions
    with pytest.raises(ValueError, match="duplicate source"):
        extract_transitions((before, before, after), provider=PROVIDER)
    assert not extract_transitions((before, after), provider="plugin:other").transitions


def test_training_source_aliasing_and_duplicate_attempts_are_rejected():
    _, batch, train, held = model()
    rows = list(batch.transitions)
    rows[-1] = replace(rows[-1], source_ids=rows[0].source_ids)
    with pytest.raises(ValueError, match="observation sources"):
        fit_transitions(rows, projection=PROJECTION, train_attempt_ids=train, evaluation_attempt_ids=held)
    with pytest.raises(ValueError, match="duplicate attempt"):
        fit_transitions(batch.transitions + batch.transitions[:1], projection=PROJECTION,
                        train_attempt_ids=train, evaluation_attempt_ids=held)


def test_array_action_equality_is_structural_and_mismatch_is_excluded():
    import numpy as np

    workspace, *_ = trajectories()
    before, after = workspace.sources()[:2]
    action = Call("latch", "press", (("force", np.array([0.25, 0.5])),))
    before = replace(before, metadata={**before.metadata, "action": action})
    action_copy = Call("latch", "press", (("force", np.array([0.25, 0.5])),))
    after = replace(after, metadata={**after.metadata, "action": action_copy, "receipt": Receipt(action_copy, "applied")})
    assert len(extract_transitions((before, after), provider=PROVIDER).transitions) == 1
    wrong = Call("latch", "press", (("force", np.array([0.25, 0.9])),))
    after = replace(after, metadata={**after.metadata, "receipt": Receipt(wrong, "applied")})
    batch = extract_transitions((before, after), provider=PROVIDER)
    assert not batch.transitions and "identity" in batch.exclusions[0].reason


def test_unsupported_default_and_unseen_feature_values_abstain():
    learned, *_ = model()
    # Inducer leaves this class as the majority of its final remainder.
    default = learned.predict({"powered": True, "enabled": True, "open": False},
                              Call("latch", "press", (("press", True),)))
    assert isinstance(default, Unknown) and default.reason == "unsupported_transition"
    novel = learned.predict({"powered": "unseen", "enabled": False, "open": False},
                            Call("latch", "press", (("press", True),)))
    assert isinstance(novel, Unknown) and novel.reason == "unseen_feature_value"


def test_training_fit_does_not_authorize_a_rule_without_heldout_support():
    _, batch, train, held = model()
    # Restrict heldout validation to the uncovered class; training stays unchanged.
    rows = tuple(t for t in batch.transitions if t.attempt_id in train or t.after["open"])
    selected_held = tuple(t.attempt_id for t in rows if t.attempt_id in held)
    learned = fit_transitions(rows, projection=PROJECTION, train_attempt_ids=train,
                              evaluation_attempt_ids=selected_held)
    prediction = learned.predict({"powered": True, "enabled": False, "open": False},
                                 Call("latch", "press", (("press", True),)))
    assert isinstance(prediction, Unknown) and prediction.reason == "unverified_transition"
    assert learned.evaluation.coverage == 0
    assert learned.evaluation.accuracy is None
