"""Whole task behavior across real effects, revisions, and dishonest action models."""

from tensorcode.agent.core import Agent
from tensorcode.agent.filesystem import FileSystemPlugin
from tensorcode.agent.plugin import Call, Capability, Effect, Param, Plugin, Precondition
from tensorcode.goals import Condition, GoalSpec
from tensorcode.outcomes import Receipt, Unknown


def file_goal(path="demo/src/main.py", text="payload", *, invariants=()):
    return GoalSpec((Condition("content", {"path": path, "text": text}),), invariants=invariants)


class RecordedFilesystem(FileSystemPlugin):
    def __init__(self, root):
        super().__init__(root, refinements=False)
        self.calls = []

    def execute(self, act, *, key):
        self.calls.append(act)
        return super().execute(act, key=key)


def test_nested_creation_records_each_observed_step(tmp_path):
    plugin = RecordedFilesystem(tmp_path)
    agent = Agent([plugin])
    result = agent.pursue(file_goal())
    assert result.status == "done", result
    assert result.verified is True
    assert (tmp_path / "demo/src/main.py").read_text() == "payload"
    assert [call.capability for call in plugin.calls] == ["mkdir", "mkdir", "write_file"]
    task = agent.tasks.get(result.task_id)
    assert task.status == "done"
    assert len(task.attempts[0].steps) == 3
    assert all(step.verified is True and step.receipt.status == "applied" for step in task.attempts[0].steps)


def test_suspend_and_resume_replans_without_repeating_created_directory(tmp_path):
    plugin = RecordedFilesystem(tmp_path)
    agent = Agent([plugin])
    first = agent.pursue(file_goal(), max_steps=1)
    assert first.status == "suspended"
    assert (tmp_path / "demo").is_dir()
    assert not (tmp_path / "demo/src").exists()
    second = agent.pursue(task_id=first.task_id)
    assert second.status == "done"
    assert [(c.capability, c.arg("path")) for c in plugin.calls] == [
        ("mkdir", "demo"), ("mkdir", "demo/src"), ("write_file", "demo/src/main.py")]
    task = agent.tasks.get(first.task_id)
    assert [attempt.status for attempt in task.attempts] == ["suspended", "done"]
    assert [len(attempt.steps) for attempt in task.attempts] == [1, 2]
    assert second.task_id == first.task_id


def test_revision_after_partial_execution_keeps_old_history_and_new_destination(tmp_path):
    plugin = RecordedFilesystem(tmp_path)
    agent = Agent([plugin])
    original = file_goal("old/main.py", "first")
    revised = file_goal("new/main.py", "second")
    first = agent.pursue(original, max_steps=1)
    assert first.status == "suspended"
    agent.tasks.revise(first.task_id, revised, reason="destination changed")
    second = agent.pursue(task_id=first.task_id)
    assert second.status == "done"
    assert (tmp_path / "old").is_dir()
    assert not (tmp_path / "old/main.py").exists()
    assert (tmp_path / "new/main.py").read_text() == "second"
    task = agent.tasks.get(first.task_id)
    assert [revision.goal for revision in task.history] == [original, revised]
    assert [attempt.revision for attempt in task.attempts] == [1, 2]
    assert task.attempts[0].steps[0].call.arg("path") == "old"
    assert task.attempts[0].plan.steps[-1].action.arg("path") == "old/main.py"


def test_existing_readme_is_held_through_creation(tmp_path):
    (tmp_path / "README.md").write_text("keep")
    plugin = RecordedFilesystem(tmp_path)
    agent = Agent([plugin])
    held = Condition("content", {"path": "README.md", "text": "keep"})
    events = []
    result = agent.pursue(file_goal(invariants=(held,)), events=events)
    assert result.status == "done"
    assert (tmp_path / "README.md").read_text() == "keep"
    assert len([e for e in events if e["type"] == "condition" and e["stage"] == "before_step"]) == 3
    assert len([e for e in events if e["type"] == "condition" and e["stage"] == "after_step"]) == 3


def c(pred):
    return Condition(pred, {"item": "x"})


class ModeledWorld(Plugin):
    def __init__(self):
        super().__init__("model", planning_enabled=True)
        self.state = {"seed": True, "a": False, "b": False, "held": True}
        self.calls = []
        self.lie = False
        self.clobber = False
        self.change_precondition = False
        self.change_held = False
        self.held_observations = 0

    def capabilities(self):
        return (
            Capability("first", (Param("x", "item"),), (Effect("a", {"item": "x"}),),
                       preconditions=(Precondition("seed", {"item": "x"}),)),
            Capability("second", (Param("x", "item"),), (Effect("b", {"item": "x"}),),
                       preconditions=(Precondition("a", {"item": "x"}),)),
        )

    def enumerate_actions(self, goal):
        return [Call(self.name, cap.name, (("x", "x"),)) for cap in self.capabilities()]

    def observe_condition(self, condition):
        if condition.args != {"item": "x"} or condition.pred not in self.state:
            return Unknown("unobserved")
        if condition.pred == "held":
            self.held_observations += 1
            # Planning, before first, after first, before second: an external
            # change appears only at the next action boundary.
            if self.change_held and self.held_observations == 4:
                self.state["held"] = False
        value = self.state[condition.pred]
        return not value if condition.negated else value

    def precondition_holds(self, condition, args):
        if self.change_precondition and condition.pred == "seed":
            self.state["seed"] = False
        return self.observe_condition(Condition(condition.pred, {r: args[p] for r, p in condition.roles.items()}, condition.negated))

    def execute(self, act, *, key):
        self.calls.append(act.capability)
        if not self.lie:
            self.state["a" if act.capability == "first" else "b"] = True
        if self.clobber and act.capability == "second":
            self.state["a"] = False
        return Receipt(act, "applied", idempotency_key=key)


def test_applied_receipt_without_effect_stops_dependents():
    world = ModeledWorld()
    world.lie = True
    result = Agent([world]).pursue(GoalSpec((c("b"),)))
    assert result.status == "failed"
    assert world.calls == ["first"]
    assert len(result.steps) == 1
    assert result.steps[0].receipt.status == "applied"
    assert result.steps[0].verified is False


def test_external_invariant_change_stops_before_next_dispatch():
    world = ModeledWorld()
    world.change_held = True
    result = Agent([world]).pursue(GoalSpec((c("b"),), invariants=(c("held"),)))
    assert result.status == "failed"
    assert world.calls == ["first"]
    assert result.steps[0].verified is True
    assert result.verified is False
    assert "held condition" in result.reason


def test_final_whole_goal_check_detects_undeclared_clobber():
    world = ModeledWorld()
    world.clobber = True
    events = []
    result = Agent([world]).pursue(GoalSpec((c("a"), c("b"))), events=events)
    assert world.calls == ["first", "second"]
    assert all(step.verified is True for step in result.steps)
    assert result.status == "failed"
    assert result.verified is False
    assert any(e["type"] == "condition" and e["stage"] == "task_complete" and e["status"] == "fails" for e in events)


def test_precondition_is_observed_again_after_planning():
    world = ModeledWorld()
    world.change_precondition = True
    result = Agent([world]).pursue(GoalSpec((c("b"),)))
    assert result.status == "failed"
    assert world.calls == []
    assert result.steps[0].receipt.status == "rejected"
    assert "precondition seed" in result.reason


class SeedObserver(Plugin):
    def __init__(self, name="sensor"):
        super().__init__(name, planning_enabled=True)
        self.available = True

    def observe_condition(self, condition):
        if condition.pred != "seed" or condition.args != {"item": "x"}:
            return Unknown("unobserved")
        return not self.available if condition.negated else self.available


class SharedEvidenceWorld(ModeledWorld):
    def __init__(self, *, owner_result=None, change_sensor=None):
        super().__init__()
        self.owner_result = owner_result
        self.change_sensor = change_sensor

    def observe_condition(self, condition):
        if condition.pred == "seed":
            return Unknown("sensor_owned")
        return super().observe_condition(condition)

    def precondition_holds(self, condition, args):
        if condition.pred == "seed":
            if self.change_sensor is not None:
                self.change_sensor.available = False
            return Unknown("sensor_owned") if self.owner_result is None else self.owner_result
        return super().precondition_holds(condition, args)


def test_cross_plugin_evidence_enables_planned_execution():
    world = SharedEvidenceWorld()
    result = Agent([world, SeedObserver()]).pursue(GoalSpec((c("b"),)))
    assert result.status == "done"
    assert world.calls == ["first", "second"]
    assert result.verified is True


def test_owner_and_shared_evidence_conflict_blocks_execution():
    world = SharedEvidenceWorld(owner_result=False)
    result = Agent([world, SeedObserver()]).pursue(GoalSpec((c("b"),)))
    assert result.status == "failed"
    assert world.calls == []
    assert result.receipt.status == "rejected"
    assert "precondition seed: unknown" in result.reason


def test_new_shared_conflict_overrides_positive_owner_check():
    first = SeedObserver("first_sensor")
    second = SeedObserver("second_sensor")
    world = SharedEvidenceWorld(owner_result=True, change_sensor=second)
    result = Agent([world, first, second]).pursue(GoalSpec((c("b"),)))
    assert result.status == "failed"
    assert world.calls == []
    assert result.receipt.status == "rejected"
    assert "precondition seed: unknown" in result.reason


def test_invalid_shared_observation_cannot_be_hidden_by_owner_true():
    class MalformedSensor(SeedObserver):
        def observe_condition(self, condition):
            if condition.pred == "seed" and not self.available:
                return "true"
            return super().observe_condition(condition)

    sensor = MalformedSensor()
    world = SharedEvidenceWorld(owner_result=True, change_sensor=sensor)
    result = Agent([world, sensor]).pursue(GoalSpec((c("b"),)))
    assert result.status == "failed"
    assert world.calls == []
    assert result.receipt.status == "rejected"
