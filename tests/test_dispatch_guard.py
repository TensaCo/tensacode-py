"""Hypothetical choices require a fresh dispatch check without bypassing conditions."""
import pytest

from tensorcode.agent import Agent
from tensorcode.agent.plugin import Capability, Plugin, Precondition
from tensorcode.outcomes import Receipt, Unknown


class Counter(Plugin):
    def __init__(self):
        super().__init__("counter")
        self.value = 0

    def observe_evidence(self):
        return {"value": self.value}

    def execute(self, action, *, key):
        self.value += 1
        return Receipt(action, "applied", idempotency_key=key)

    def precondition_holds(self, condition, args):
        return False


@pytest.mark.parametrize("decision", [False, Unknown("stale_prediction"), None, 1, "yes"])
def test_only_boolean_true_from_guard_allows_dispatch(decision):
    actor = Counter()
    agent = Agent([actor])
    events = []
    seen = []

    def guard(source_ids):
        seen.extend(agent.interpretations.get_source(s) for s in source_ids)
        return decision

    receipt = agent._invoke(actor, Capability("increment", ()), {}, events, before_dispatch=guard)
    assert receipt.status == "rejected" and actor.value == 0
    assert len(seen) == 1 and seen[0].payload == {"value": 0}
    assert seen[0].metadata["stage"] == "before_action"
    assert not any(event["type"] == "act" for event in events)
    after = agent.interpretations.sources()[-1]
    assert after.metadata["receipt"] == receipt
    assert after.payload == {"value": 0}
    assert not agent.store.claims() and not agent.store.propositions()


def test_guard_error_rejects_and_retains_the_failed_check():
    actor = Counter()
    agent = Agent([actor])
    events = []

    def guard(source_ids):
        raise RuntimeError("projection unavailable")

    receipt = agent._invoke(actor, Capability("increment", ()), {}, events, before_dispatch=guard)
    assert receipt.status == "rejected" and actor.value == 0
    check = next(event for event in events if event["type"] == "dispatch_guard")
    assert check["reason"] == "dispatch_guard_error"
    assert "projection unavailable" in check["detail"]


def test_valid_guard_reuses_linked_before_evidence_and_observes_actual_result():
    actor = Counter()
    agent = Agent([actor])
    events = []

    def guard(source_ids):
        source = agent.interpretations.get_source(source_ids[0])
        assert source.payload == {"value": 0}
        assert source.metadata["action"].capability == "increment"
        return True

    receipt = agent._invoke(actor, Capability("increment", ()), {}, events, before_dispatch=guard)
    assert receipt.status == "applied" and actor.value == 1
    before, after = agent.interpretations.sources()
    assert before.metadata["attempt_id"] == after.metadata["attempt_id"]
    assert after.payload == {"value": 1}
    assert [event["type"] for event in events] == [
        "observation", "dispatch_guard", "act", "observation", "receipt"]


def test_guard_cannot_override_failed_capability_preconditions():
    actor = Counter()
    agent = Agent([actor])
    calls = []
    cap = Capability("increment", (), preconditions=(Precondition("ready", {}),))
    receipt = agent._invoke(actor, cap, {}, [], before_dispatch=lambda ids: calls.append(ids) or True)
    assert receipt.status == "rejected" and actor.value == 0 and calls == []
