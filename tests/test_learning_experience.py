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


def test_action_family_contract_rejects_unseen_plugin_capability_and_argument_names():
    learned, *_ = model()
    before = {"powered": True, "enabled": False, "open": False}
    for action in (Call("other", "press", (("press", True),)),
                   Call("latch", "other", (("press", True),)),
                   Call("latch", "press", (("other", True),))):
        result = learned.predict(before, action)
        assert isinstance(result, Unknown) and result.reason == "unseen_action_family"
    duplicate = learned.predict(before, Call("latch", "press", (("press", True), ("press", False))))
    assert isinstance(duplicate, Unknown) and duplicate.reason == "invalid_action_family"
    assert learned.action_families[0].argument_names == ("press",)


def test_action_families_are_not_inferred_from_evaluation_data():
    _, batch, train, held = model()
    rows = []
    for transition in batch.transitions:
        if transition.attempt_id in held:
            action = replace(transition.action, capability="new-in-evaluation")
            transition = replace(transition, action=action, receipt=replace(transition.receipt, action=action))
        rows.append(transition)
    learned = fit_transitions(rows, projection=PROJECTION, train_attempt_ids=train, evaluation_attempt_ids=held)
    assert learned.evaluation.coverage == 0
    assert {family.capability for family in learned.action_families} == {"press"}


def test_rule_support_cannot_transfer_between_action_families_when_projection_omits_family():
    _, batch, train, held = model()
    rows = list(batch.transitions)
    # A new action family has a training outcome but no heldout support.
    original = next(t for t in rows if t.attempt_id in train and not t.before["enabled"])
    action = replace(original.action, capability="unvalidated")
    rows[rows.index(original)] = replace(original, action=action, receipt=replace(original.receipt, action=action))
    learned = fit_transitions(rows, projection=PROJECTION, train_attempt_ids=train, evaluation_attempt_ids=held)
    result = learned.predict(original.before, action)
    assert isinstance(result, Unknown) and result.reason == "unverified_transition"


def test_counterexample_suspends_rule_and_invalidates_retained_proposals():
    learned, batch, train, held = model()
    before = {"powered": True, "enabled": False, "open": False}
    action = Call("latch", "press", (("press", True),))
    prediction = learned.predict(before, action)
    snapshot = learned.snapshot()
    assert learned.is_current(prediction)
    assert prediction.model_id == snapshot.id == learned.id
    assert prediction.rule_id in snapshot.rule_ids and prediction.model_revision == 0
    event = learned.observe_outcome(prediction, True, source_ids=("source:actual-step",),
                                    reason="actual projected observation contradicted the rule")
    assert event.revision == 1 and event.rule_id == prediction.rule_id
    assert event.predicted is False and event.observed is True
    assert event.source_ids == ("source:actual-step",)
    assert learned.history == (event,)
    assert learned.revision == 1 and not learned.is_current(prediction)
    assert snapshot.revision == 0 and snapshot.history == ()
    result = learned.predict(before, action)
    assert isinstance(result, Unknown) and result.reason == "suspended_rule"
    refitted = fit_transitions(batch.transitions, projection=PROJECTION, train_attempt_ids=train, evaluation_attempt_ids=held)
    assert refitted.id != learned.id and refitted.revision == 0
    assert not isinstance(refitted.predict(before, action), Unknown)
    assert learned.snapshot().history == (event,)


def test_unknown_outcome_or_foreign_prediction_cannot_suspend_a_rule():
    learned, *_ = model()
    prediction = learned.predict({"powered": True, "enabled": False, "open": False},
                                 Call("latch", "press", (("press", True),)))
    with pytest.raises(ValueError, match="unknown outcome"):
        learned.observe_outcome(prediction, Unknown("unobserved"), source_ids=("source:missing",), reason="unseen")
    with pytest.raises(ValueError, match="unchanged rule"):
        learned.observe_outcome(replace(prediction, model_id="other"), True, source_ids=("source:new",), reason="wrong model")
    with pytest.raises(ValueError, match="source IDs"):
        learned.observe_outcome(prediction, True, source_ids=(), reason="unsupported")
    assert learned.observe_outcome(prediction, False, source_ids=("source:confirmed",), reason="same outcome") is None
    assert learned.revision == 0 and learned.history == ()


def test_induced_rule_generalizes_to_a_combination_absent_from_training():
    _, batch, train, held = model()
    withheld = {t.attempt_id for t in batch.transitions if t.attempt_id in train and
                not t.before["powered"] and not t.before["enabled"] and dict(t.action.args)["press"]}
    rows = tuple(t for t in batch.transitions if t.attempt_id not in withheld)
    training = [i for i in train if i not in withheld]
    learned = fit_transitions(rows, projection=PROJECTION, train_attempt_ids=training, evaluation_attempt_ids=held)
    before = {"powered": False, "enabled": False, "open": False}
    action = Call("latch", "press", (("press", True),))
    assert not any(t.before == before and t.action == action for t in rows if t.attempt_id in training)
    answer = learned.predict(before, action)
    assert not isinstance(answer, Unknown) and answer.outcome is False
    assert len(learned.artifact.rules[answer.rule_index].conditions) < len(PROJECTION.features(before, action))


def test_unseen_feature_names_abstain_and_model_identity_is_read_only():
    learned, *_ = model()
    before = {"powered": True, "enabled": False, "open": False, "novel": True}
    result = learned.predict(before, Call("latch", "press", (("press", True),)))
    assert isinstance(result, Unknown) and result.reason == "unseen_feature"
    for name in ("id", "model_id", "revision", "projection", "provider"):
        with pytest.raises(AttributeError):
            setattr(learned, name, None)


def test_projected_fit_examples_validate_content_not_just_source_identifiers():
    learned, batch, train, held = model()
    assert len(learned.examples) == len(batch.transitions)
    assert {e.attempt_id for e in learned.examples if e.split == "training"} == set(train)
    assert {e.attempt_id for e in learned.examples if e.split == "evaluation"} == set(held)
    original = batch.transitions[0]
    assert learned.validate_transition(original) is True
    altered_action = replace(original.action, args=(("press", not dict(original.action.args)["press"]),))
    mismatches = (
        replace(original, before={**original.before, "enabled": not original.before["enabled"]}),
        replace(original, after={**original.after, "open": not original.after["open"]}),
        replace(original, action=altered_action, receipt=replace(original.receipt, action=altered_action)),
        replace(original, source_ids=("source:unrelated", original.source_ids[1])),
        replace(original, provider="plugin:another"),
    )
    for mismatch in mismatches:
        result = learned.validate_transition(mismatch)
        assert isinstance(result, Unknown) and result.reason == "fit_evidence_mismatch"
    assert learned.validate_transition(original) is True


def test_fabricated_fitting_payloads_cannot_borrow_real_source_ids():
    _, batch, train, held = model()
    fabricated = tuple(replace(row, after={**row.after, "open": not row.after["open"]}) for row in batch.transitions)
    learned = fit_transitions(fabricated, projection=PROJECTION, train_attempt_ids=train, evaluation_attempt_ids=held)
    # The fit records what it actually consumed; validation against independently
    # retained execution rows detects fabricated labels with unchanged source IDs.
    for real in batch.transitions:
        result = learned.validate_transition(real)
        assert isinstance(result, Unknown) and result.reason == "fit_evidence_mismatch"


def test_projected_example_action_arrays_are_detached_and_compared_exactly():
    import numpy as np

    _, batch, train, held = model()
    rows = []
    for row in batch.transitions:
        action = replace(row.action, args=(("force", np.array([float(dict(row.action.args)["press"])])),))
        rows.append(replace(row, action=action, receipt=replace(row.receipt, action=action)))
    projection = Projection("ignore-force-for-this-fixture", lambda before, action: before,
                            lambda after: after["open"], ("authored:array-identity-test",))
    learned = fit_transitions(rows, projection=projection, train_attempt_ids=train, evaluation_attempt_ids=held)
    snapshot = learned.examples[0]
    snapshot.action.args[0][1][0] = 99
    assert learned.examples[0].action.args[0][1][0] != 99
    assert learned.validate_transition(rows[0]) is True
    mutated = replace(rows[0], action=snapshot.action, receipt=replace(rows[0].receipt, action=snapshot.action))
    assert isinstance(learned.validate_transition(mutated), Unknown)


def test_supplied_artifact_without_fit_examples_cannot_claim_evidence_backed_prediction():
    from tensorcode.learning.experience import LearnedTransitionModel

    learned, batch, *_ = model()
    supplied = LearnedTransitionModel(learned.artifact, learned.projection, learned.evidence,
        learned.evaluation, learned.policy, learned.provider, learned._feature_values,
        learned._family_evidence, learned.action_families, learned.snapshot().source_ids)
    assert supplied.examples == ()
    result = supplied.predict(batch.transitions[0].before, batch.transitions[0].action)
    assert isinstance(result, Unknown) and result.reason == "missing_fit_examples"
    checked = supplied.validate_transition(batch.transitions[0])
    assert isinstance(checked, Unknown) and checked.reason == "missing_fit_example"


def test_boolean_outcomes_cannot_validate_numeric_labels():
    _, batch, train, held = model()
    rows = tuple(replace(row, after={"open": int(row.after["open"])})
                 if row.attempt_id in train else row for row in batch.transitions)
    learned = fit_transitions(rows, projection=PROJECTION, train_attempt_ids=train, evaluation_attempt_ids=held)
    assert learned.artifact.rules
    assert learned.evaluation.predicted == 0 and learned.evaluation.correct == 0
    assert all(e.evaluation_correct == 0 for e in learned.evidence)
    result = learned.predict({"powered": True, "enabled": False, "open": False},
                             Call("latch", "press", (("press", True),)))
    assert isinstance(result, Unknown) and result.reason == "unverified_transition"


def test_boolean_features_cannot_borrow_numeric_domain_membership():
    _, batch, train, held = model()
    rows = tuple(replace(row, before={**row.before, "enabled": int(row.before["enabled"])})
                 if row.attempt_id in train else row for row in batch.transitions)
    learned = fit_transitions(rows, projection=PROJECTION, train_attempt_ids=train, evaluation_attempt_ids=held)
    assert learned.evaluation.coverage == 0
    result = learned.predict({"powered": True, "enabled": False, "open": False},
                             Call("latch", "press", (("press", True),)))
    assert isinstance(result, Unknown) and result.reason == "unseen_feature_value"


def test_typed_domains_retain_both_observed_boolean_and_numeric_values():
    _, batch, train, held = model()
    rows = tuple(replace(row, before={**row.before, "enabled": int(row.before["enabled"])})
                 if row.attempt_id in train[:8] else row for row in batch.transitions)
    learned = fit_transitions(rows, projection=PROJECTION, train_attempt_ids=train, evaluation_attempt_ids=held)
    values = learned._feature_values["enabled"]
    assert {(type(value), value) for value in values} == {(bool, False), (bool, True), (int, 0), (int, 1)}


def test_literal_validation_does_not_equate_boolean_and_numeric_values():
    from tensorcode.learning.experience import _matched
    from tensorcode.learning.induce import DecisionList, Rule
    from tensorcode.learning.literals import Literal

    for literal in (Literal("x", True), Literal("x", True, negated=True),
                    Literal("x", 1, kind="at_least")):
        artifact = DecisionList([Rule((literal,), "result")])
        value = True if literal.kind == "at_least" else 1
        assert _matched(artifact, frozenset({("x", value)})) is None
    assert _matched(DecisionList([Rule((Literal("x", 1),), "result")]), frozenset({("x", 1)})) == 0
