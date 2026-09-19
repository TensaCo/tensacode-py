"""Task history survives revisions and mutation of external payloads."""

from types import SimpleNamespace

import pytest

from tensorcode.agent.tasks import TaskLedger
from tensorcode.outcomes import Unknown


def outcome(status="failed", **kwargs):
    return SimpleNamespace(status=status, plan=kwargs.get("plan"),
                           receipt=kwargs.get("receipt"), verified=kwargs.get("verified"),
                           reason=kwargs.get("reason", ""))


def test_revision_preserves_old_goal_and_attempt_after_correction():
    ledger = TaskLedger()
    initial_goal = {"destination": ["scratch"]}
    task = ledger.create("make a project", initial_goal)
    plan = {"destination": ["scratch"]}
    receipt = {"written": ["scratch/main.py"]}
    ledger.record(task.id, outcome(plan=plan, receipt=receipt, verified=False,
                                   reason="missing README"))
    initial_goal["destination"].append("tampered")
    plan["destination"].clear()
    receipt["written"].clear()

    revised = ledger.revise(task.id, {"destination": ["Documents"]},
                            reason="user corrected destination")
    assert revised.id == task.id
    assert revised.source == "make a project"
    assert revised.revision == 2
    assert revised.status == "ready"
    assert revised.revisions[0].goal == {"destination": ["scratch"]}
    assert revised.revisions[1].reason == "user corrected destination"
    assert revised.attempts[0].revision == 1
    assert revised.attempts[0].plan == {"destination": ["scratch"]}
    assert revised.attempts[0].receipt == {"written": ["scratch/main.py"]}
    completed = ledger.record(task.id, outcome("done", verified=True))
    assert [attempt.revision for attempt in completed.attempts] == [1, 2]
    assert completed.status == "done"


def test_returned_snapshots_cannot_rewrite_ledger():
    ledger = TaskLedger()
    task = ledger.create("request", {"items": []})
    task.goal["items"].append("changed")
    task.history[0].goal["items"].append("changed")
    ledger.record(task.id, outcome(plan={"args": []}))
    for snapshot in (ledger.get(task.id), ledger.values()[0], next(iter(ledger))):
        snapshot.goal["items"].append("changed")
        snapshot.attempts[0].plan["args"].append("changed")
    assert ledger.get(task.id).goal == {"items": []}
    assert ledger.get(task.id).history[0].goal == {"items": []}
    assert ledger.get(task.id).attempts[0].plan == {"args": []}


@pytest.mark.parametrize("status, expected", [
    ("done", "done"), ("failed", "failed"), ("unverified", "unverified"),
    ("unknown", "blocked"), ("not_understood", "blocked"), ("declined", "blocked"),
])
def test_task_status_retains_distinct_attempt_status(status, expected):
    ledger = TaskLedger()
    task = ledger.create("request")
    result = ledger.record(task.id, outcome(status, verified=Unknown("no observation")))
    assert result.status == expected
    assert result.attempts[0].status == status
    assert isinstance(result.attempts[0].verified, Unknown)


def test_completed_task_requires_revision_before_another_attempt():
    ledger = TaskLedger()
    task = ledger.create("request")
    ledger.record(task.id, outcome("done"))
    with pytest.raises(ValueError, match="revised"):
        ledger.record(task.id, outcome("done"))
    assert len(ledger.get(task.id).attempts) == 1
    ledger.revise(task.id, "new goal", reason="changed request")
    assert ledger.record(task.id, outcome("done")).revision == 2


def test_invalid_updates_leave_history_unchanged():
    ledger = TaskLedger()
    task = ledger.create("request")
    for reason in ("", " \n "):
        with pytest.raises(ValueError, match="nonempty reason"):
            ledger.revise(task.id, "new goal", reason=reason)
    with pytest.raises(ValueError, match="unsupported"):
        ledger.record(task.id, outcome("answered"))
    assert ledger.get(task.id) == task
    with pytest.raises(KeyError):
        ledger.get("missing")
    assert ledger.create("request").id != task.id
    assert len(ledger) == 2
