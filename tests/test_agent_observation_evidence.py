"""Raw observations remain source evidence, including around uncertain actions."""
import json

import pytest

from tensorcode.agent.core import Agent
from tensorcode.agent.plugin import Capability, Plugin, Precondition
from tensorcode.outcomes import Receipt, Unknown


class Sensor(Plugin):
    def __init__(self, name="sensor", payload=None, error=None):
        super().__init__(name)
        self.payload = payload
        self.error = error
        self.observed = 0

    def observe_evidence(self):
        self.observed += 1
        if self.error:
            raise self.error
        return self.payload


class ActingSensor(Sensor):
    def __init__(self, *, status="applied", exception=False, payload=None, observation_error=None):
        super().__init__("actor", payload if payload is not None else {"state": [0], "pixels": b"png"}, observation_error)
        self.status = status
        self.exception = exception
        self.calls = 0
        self.returned = None

    def execute(self, action, *, key):
        self.calls += 1
        self.payload["state"][0] += 1
        if self.exception:
            raise TimeoutError("reply lost after possible effect")
        self.returned = Receipt(action, self.status, idempotency_key=key)
        return self.returned


def sources(agent):
    return tuple(source for source in agent.interpretations.sources() if source.modality == "observation")


def test_perception_snapshots_raw_payload_without_assertions_or_raw_events():
    payload = {"pixels": b"png bytes", "scene": [{"uninterpreted": "relation"}]}
    sensor = Sensor(payload=payload)
    agent = Agent([sensor])
    events = []
    agent.perceive(events)
    [source] = sources(agent)
    assert source.provider == "plugin:sensor"
    assert source.metadata["stage"] == "perception"
    assert source.metadata["status"] == "observed"
    assert source.metadata["observed_at"]
    assert source.metadata["action"] is None
    assert source.payload == payload
    payload["scene"][0]["uninterpreted"] = "changed live sensor"
    source.payload["scene"].clear()
    assert sources(agent)[0].payload["scene"] == [{"uninterpreted": "relation"}]
    assert agent.store.propositions() == []
    assert agent.store.claims() == []
    assert not agent.interpretations.values()
    serialized = json.dumps(events)
    assert "png bytes" not in serialized and "uninterpreted" not in serialized
    assert source.id in serialized


def test_observation_errors_do_not_block_other_providers_or_become_world_facts():
    broken = Sensor("broken", error=ValueError("camera offline"))
    working = Sensor("working", payload={"pixels": b"valid"})
    agent = Agent([broken, working, Plugin("silent")])
    events = []
    agent.perceive(events)
    retained = sources(agent)
    assert [source.provider for source in retained] == ["plugin:broken", "plugin:working"]
    assert retained[0].metadata["status"] == "error"
    assert retained[0].metadata["error"] == {"type": "ValueError", "message": "camera offline"}
    assert retained[0].payload is None
    assert retained[1].metadata["status"] == "observed"
    assert working.observed == 1
    assert not agent.store.claims() and not agent.store.propositions()
    assert json.dumps(events)


def test_uncopyable_observation_is_retained_as_error_without_suppressing_other_sources():
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("snapshot unavailable")
    agent = Agent([Sensor("uncopyable", Uncopyable()), Sensor("valid", 0)])
    agent.perceive([])
    assert [source.metadata["status"] for source in sources(agent)] == ["error", "observed"]
    assert sources(agent)[1].payload == 0


@pytest.mark.parametrize("status", ["applied", "rejected", "failed", "indeterminate"])
def test_action_sources_link_exact_call_receipt_and_before_after_evidence(status):
    actor = ActingSensor(status=status)
    agent = Agent([actor])
    events = []
    returned = agent._invoke(actor, Capability("change", ()), {"input": "explicit"}, events)
    assert returned is actor.returned
    before, after = sources(agent)
    assert before.metadata["stage"] == "before_action"
    assert after.metadata["stage"] == "after_action"
    assert before.metadata["attempt_id"] == after.metadata["attempt_id"]
    assert before.metadata["attempt_id"]
    assert before.metadata["action"] == after.metadata["action"] == returned.action
    assert before.metadata["action"].arg("input") == "explicit"
    assert before.metadata["receipt"] is None
    assert after.metadata["receipt"] == returned
    assert after.metadata["receipt"].status == status
    assert before.payload["state"] == [0]
    assert after.payload["state"] == [1]
    assert actor.observed == 2 and actor.calls == 1
    assert events[-1]["attempt_id"] == before.metadata["attempt_id"]
    assert not agent.store.propositions()
    json.dumps(events)


def test_missing_and_failed_after_observations_preserve_indeterminate_receipt_without_retry():
    class LosingSensor(ActingSensor):
        def observe_evidence(self):
            if self.calls:
                raise OSError("camera lost")
            return super().observe_evidence()
    actor = LosingSensor(exception=True)
    agent = Agent([actor, Plugin("unavailable")])
    events = []
    receipt = agent._invoke(actor, Capability("change", ()), {}, events)
    assert receipt.status == "indeterminate" and receipt.retryable is False
    assert "TimeoutError" in receipt.error
    assert actor.calls == 1
    retained = sources(agent)
    assert [(s.provider, s.metadata["stage"], s.metadata["status"]) for s in retained] == [
        ("plugin:actor", "before_action", "observed"),
        ("plugin:unavailable", "before_action", "unavailable"),
        ("plugin:actor", "after_action", "error"),
        ("plugin:unavailable", "after_action", "unavailable"),
    ]
    assert all(s.metadata["receipt"] == receipt for s in retained if s.metadata["stage"] == "after_action")


def test_precondition_rejection_has_observation_pair_without_executor_dispatch():
    actor = ActingSensor()
    agent = Agent([actor])
    cap = Capability("change", (), preconditions=(Precondition("authorized", {}),))
    receipt = agent._invoke(actor, cap, {}, [])
    assert receipt.status == "rejected" and actor.calls == 0
    before, after = sources(agent)
    assert before.payload == after.payload
    assert after.metadata["receipt"] == receipt


def test_invoking_provider_and_explicit_observers_included_once_even_when_unmounted():
    actor = ActingSensor()
    mounted = Sensor("mounted", payload="ambient")
    observer = Sensor("explicit", payload="other viewpoint")
    agent = Agent([mounted, mounted])
    agent._invoke(actor, Capability("change", ()), {}, [], observers=(observer, actor, mounted))
    assert actor.observed == observer.observed == mounted.observed == 2
    retained = sources(agent)
    assert len(retained) == 6
    assert {s.provider for s in retained} == {"plugin:actor", "plugin:mounted", "plugin:explicit"}
    assert len({s.metadata["attempt_id"] for s in retained}) == 1
    agent._invoke(actor, Capability("change", ()), {}, [])
    assert len({s.metadata["attempt_id"] for s in sources(agent)}) == 2


def test_unknown_observation_is_explicit_missing_coverage_not_a_world_value():
    agent = Agent([Sensor(payload=Unknown("no_signal", "occluded"))])
    agent.perceive([])
    [source] = sources(agent)
    assert source.metadata["status"] == "unavailable"
    assert source.metadata["reason"] == "no_signal"
    assert source.payload is None
    assert not agent.store.propositions()


def test_uncopyable_action_linkage_cannot_suppress_execution_or_replace_actual_receipt():
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("executor argument cannot be snapshotted")
    argument = Uncopyable()
    actor = ActingSensor()
    agent = Agent([actor])
    events = []
    receipt = agent._invoke(actor, Capability("change", ()), {"opaque": argument}, events)
    assert receipt is actor.returned
    assert receipt.status == "applied"
    assert receipt.action.arg("opaque") is argument
    assert actor.calls == 1
    before, after = sources(agent)
    assert before.metadata["attempt_id"] == after.metadata["attempt_id"]
    for source in (before, after):
        assert source.metadata["status"] == "error"
        assert source.metadata["linkage_snapshot_error"]["type"] == "RuntimeError"
        assert source.metadata["action"] is None and source.metadata["receipt"] is None
    assert after.metadata["receipt_status"] == "applied"
    assert events[-1]["status"] == "applied"
