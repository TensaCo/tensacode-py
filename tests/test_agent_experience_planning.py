"""Induced one-step choices, exercised against real deterministic Gym dynamics.

The environment, exploration calls, projection, and desired state are authored.
Transition rules come from recorded executions, not authored capability effects.
"""
from copy import deepcopy
from dataclasses import replace

import pytest

gym = pytest.importorskip("gymnasium")
from examples.general_agent.gym_connection import GymPlugin
from tensorcode.agent import Agent
from tensorcode.agent.plugin import Call
from tensorcode.learning.experience import Projection, extract_transitions, fit_transitions
from tensorcode.outcomes import Unknown


PROJECTION = Projection(
    "gym-discrete-state-and-action",
    lambda observed, action: {"state": int(observed["transition"]["observation"]),
                              "action": dict(action.args)["action"]},
    lambda observed: int(observed["transition"]["observation"]),
    ("Authored projection of the environment's discrete observation and supplied action",))


def invoke(agent, plugin, capability, **args):
    events = []
    cap = next(c for c in plugin.capabilities() if c.name == capability)
    receipt = agent._invoke(plugin, cap, args, events)
    assert receipt.status == "applied", receipt
    return next(event["attempt_id"] for event in events if event["type"] == "receipt")


def latest(agent, plugin):
    return [s for s in agent.interpretations.sources()
            if s.provider == "plugin:" + plugin.name and s.modality == "observation"][-1]


@pytest.fixture
def trained():
    plugin = GymPlugin(gym.make("FrozenLake-v1", is_slippery=False), name="arbitrary-connection-name")
    agent = Agent([plugin])
    training, evaluation = [], []
    for trial in range(6):
        for action in (0, 1, 2, 3):
            invoke(agent, plugin, "reset", seed=0)
            attempt = invoke(agent, plugin, "step", action=action)
            (training if trial < 4 else evaluation).append(attempt)
    rows = extract_transitions(agent.interpretations.sources(), provider="plugin:" + plugin.name).transitions
    rows = tuple(row for row in rows if row.action.capability == "step")
    model = fit_transitions(rows, projection=PROJECTION, train_attempt_ids=training,
                            evaluation_attempt_ids=evaluation)
    invoke(agent, plugin, "reset", seed=0)
    try:
        yield agent, plugin, model
    finally:
        plugin.close()


def calls(plugin, actions=(0, 1)):
    return tuple(Call(plugin.name, "step", (("action", action),)) for action in actions)


def test_real_environment_learned_choice_executes_and_observes_goal(trained):
    agent, plugin, model = trained
    before = latest(agent, plugin)
    sequence = plugin.sequence
    plan = agent.propose_experience(model, before.id, calls(plugin), desired_outcome=4)
    assert plan.reason == "sole_supported_target_prediction"
    assert dict(plan.selected_call.args) == {"action": 1}
    assert plugin.sequence == sequence
    assert not tuple(agent.store.claims()) and not tuple(agent.store.propositions())
    assert not any(cap.effects for cap in plugin.capabilities())
    assert plan.record_source_id
    assert all(candidate.prediction.evidence.source_ids for candidate in plan.candidates)
    assert all(candidate.prediction.projection_provenance == PROJECTION.provenance for candidate in plan.candidates)
    result = agent.execute_experience(plan.id)
    assert result.receipt.status == "applied"
    assert result.verification is True
    assert result.reason == "prediction_confirmed"
    assert plugin.observe()["transition"]["observation"] == 4
    assert result.source_ids and result.record_source_id
    assert not tuple(agent.store.claims()) and not tuple(agent.store.propositions())
    sequence = plugin.sequence
    duplicate = agent.execute_experience(plan.id)
    assert duplicate.reason == "proposal_already_consumed"
    assert plugin.sequence == sequence
    # State 4 was never a training input; familiar action alone is not sufficient.
    unseen = agent.propose_experience(model, latest(agent, plugin).id, calls(plugin), 4)
    assert unseen.selected_call is None
    assert all(isinstance(candidate.prediction, Unknown) for candidate in unseen.candidates)


def test_unknown_alternatives_remain_unknown_without_vetoing_supported_goal(trained):
    agent, plugin, model = trained
    plan = agent.propose_experience(model, latest(agent, plugin).id, calls(plugin, (0, 1, 2)), 4)
    assert plan.selected_call is not None and plan.reason == "sole_supported_target_prediction"
    assert isinstance(plan.candidates[-1].prediction, Unknown)
    assert plan.candidates[-1].prediction.reason == "unsupported_transition"
    result = agent.execute_experience(plan.id)
    assert result.receipt.status == "applied" and result.verification is True
    # An untrained observed state supplies no supported action; defer without a guess.
    deferred = agent.propose_experience(model, latest(agent, plugin).id, calls(plugin), 4)
    sequence = plugin.sequence
    result = agent.execute_experience(deferred.id)
    assert result.receipt is None and isinstance(result.verification, Unknown)
    assert plugin.sequence == sequence


def test_stale_actual_world_rejects_before_dispatch(trained):
    agent, plugin, model = trained
    plan = agent.propose_experience(model, latest(agent, plugin).id, calls(plugin), 4)
    plugin.environment.unwrapped.s = 8  # independent external mutation
    # Gym's retained observation does not automatically reflect privileged state:
    # mutate through real step so observe_evidence exposes the changed transition.
    plugin.execute(Call(plugin.name, "step", (("action", 0),)))
    sequence = plugin.sequence
    result = agent.execute_experience(plan.id)
    assert result.receipt.status == "rejected"
    assert result.verification.reason == "action_not_applied"
    assert plugin.sequence == sequence


def test_detached_proposal_and_capability_drift_do_not_change_dispatch(trained, monkeypatch):
    agent, plugin, model = trained
    plan = agent.propose_experience(model, latest(agent, plugin).id, calls(plugin), 4)
    object.__setattr__(plan, "selected_call", calls(plugin)[0])
    result = agent.execute_experience(plan.id)
    assert result.verification is True
    assert plugin.observe()["transition"]["observation"] == 4
    invoke(agent, plugin, "reset", seed=0)
    plan = agent.propose_experience(model, latest(agent, plugin).id, calls(plugin), 4)
    original = plugin.capabilities()
    monkeypatch.setattr(plugin, "capabilities", lambda: tuple(replace(c, description="changed") for c in original))
    result = agent.execute_experience(plan.id)
    assert result.receipt.status == "rejected"
    assert "capability_model_changed" in result.receipt.error


def test_observed_counterexample_suspends_learned_rule(trained):
    agent, plugin, model = trained
    plan = agent.propose_experience(model, latest(agent, plugin).id, calls(plugin), 4)
    # Change the environment's actual transition table after training while
    # retaining the same observable current state. Verification must catch it.
    plugin.environment.unwrapped.P[0][1] = [(1.0, 8, 0.0, False)]
    result = agent.execute_experience(plan.id)
    assert result.receipt.status == "applied"
    assert result.verification is False and result.reason == "prediction_contradicted"
    assert model.revision == 1
    invoke(agent, plugin, "reset", seed=0)
    next_plan = agent.propose_experience(model, latest(agent, plugin).id, calls(plugin), 4)
    assert next_plan.selected_call is None
    assert any(isinstance(c.prediction, Unknown) and c.prediction.reason == "suspended_rule"
               for c in next_plan.candidates)


def test_foreign_evidence_and_wrong_provider_cannot_authorize_plan(trained):
    agent, plugin, model = trained
    source = latest(agent, plugin)
    with pytest.raises(ValueError, match="mounted provider"):
        agent.propose_experience(model, source.id, (Call("other", "step", (("action", 1),)),), 4)
    with pytest.raises(ValueError, match="exactly match"):
        agent.propose_experience(model, source.id, (Call(plugin.name, "step", ()),), 4)
    other = Agent([plugin])
    foreign = other.interpretations.add_source("copied observation", modality="observation",
                    provider=model.provider, payload=deepcopy(source.payload), metadata={"status": "observed"})
    with pytest.raises(ValueError, match="sample evidence"):
        other.propose_experience(model, foreign.id, calls(plugin), 4)


def test_boolean_goal_does_not_alias_numeric_state(trained):
    agent, plugin, model = trained
    plan = agent.propose_experience(model, latest(agent, plugin).id, calls(plugin), False)
    assert plan.selected_call is None  # predicted integer 0 is not boolean False
    assert plan.reason == "no_supported_target"


def test_unavailable_after_observation_never_verifies_applied_action(trained, monkeypatch):
    agent, plugin, model = trained
    plan = agent.propose_experience(model, latest(agent, plugin).id, calls(plugin), 4)
    sequence = plugin.sequence
    original = plugin.observe_evidence
    monkeypatch.setattr(plugin, "observe_evidence", lambda: original() if plugin.sequence == sequence
                        else Unknown("sensor_unavailable"))
    result = agent.execute_experience(plan.id)
    assert result.receipt.status == "applied"
    assert isinstance(result.verification, Unknown)
    assert result.reason == "after_observation_unavailable"
    assert model.revision == 0



def test_fabricated_training_outcomes_cannot_borrow_real_source_ids(trained):
    agent, plugin, original_model = trained
    rows = extract_transitions(agent.interpretations.sources(), provider=original_model.provider).transitions
    rows = tuple(row for row in rows if row.action.capability == "step")
    # Genuine attempt/source IDs and applied receipts do not authenticate altered
    # contents supplied to the fitting API. Preserve every ID while forging labels.
    altered = []
    for row in rows:
        after = deepcopy(row.after)
        after["transition"]["observation"] = int(after["transition"]["observation"]) + 100
        altered.append(replace(row, after=after))
    training = [row.attempt_id for row in rows[:16]]
    evaluation = [row.attempt_id for row in rows[16:]]
    forged = fit_transitions(altered, projection=PROJECTION, train_attempt_ids=training,
                             evaluation_attempt_ids=evaluation)
    sequence = plugin.sequence
    with pytest.raises(ValueError, match="sample content disagrees"):
        agent.propose_experience(forged, latest(agent, plugin).id, calls(plugin), 104)
    assert plugin.sequence == sequence
    assert not tuple(agent.store.claims()) and not tuple(agent.store.propositions())
