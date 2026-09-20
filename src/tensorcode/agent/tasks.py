"""Task identity and revision history for one agent's lifetime.

The ledger is in memory, not disk persistence. Goals are data supplied by callers;
they need not come from a lexical resource. Plans and receipts belong to attempts,
so revising a goal never rewrites what an earlier attempt tried or observed.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from functools import wraps
from threading import RLock
from typing import Any, Callable, Iterator
from uuid import uuid4

from .task_dependencies import InterpretationDependency


def _dependencies(values):
    if not isinstance(values, tuple) or any(not isinstance(value, InterpretationDependency) for value in values):
        raise TypeError("task dependencies must be a tuple of InterpretationDependency values")
    for value in values:
        value.__post_init__()
    return deepcopy(values)


def _goal_interpretation_id(value):
    if value is not None and (type(value) is not str or not value.strip()):
        raise ValueError("goal_interpretation_id must be a nonempty string or None")
    return value


@dataclass(frozen=True)
class TaskRevision:
    revision: int
    goal: Any
    reason: str = ""
    dependencies: tuple[InterpretationDependency, ...] = ()
    goal_interpretation_id: str | None = None


@dataclass(frozen=True)
class StepAttempt:
    step_id: str
    call: Any
    receipt: Any
    verified: Any


@dataclass(frozen=True)
class TaskAttempt:
    revision: int
    status: str
    plan: Any = None
    receipt: Any = None
    verified: Any = None
    reason: str = ""
    steps: tuple[StepAttempt, ...] = ()


@dataclass(frozen=True)
class Task:
    id: str
    source: str
    goal: Any
    revision: int = 1
    status: str = "ready"
    revisions: tuple[TaskRevision, ...] = ()
    attempts: tuple[TaskAttempt, ...] = ()
    dependencies: tuple[InterpretationDependency, ...] = ()
    goal_interpretation_id: str | None = None

    @property
    def history(self) -> tuple[TaskRevision, ...]:
        """Every goal revision, including the current one."""
        return self.revisions


_TASK_STATUS = {
    "done": "done",
    "failed": "failed",
    "unverified": "unverified",
    "unknown": "blocked",
    "not_understood": "blocked",
    "declined": "blocked",
    "suspended": "suspended",
}


def _locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


class TaskLedger:
    """Store detached snapshots of tasks, revisions, and execution attempts.

    Inputs and returned records are deep-copied: mutable goal arguments or receipt
    payloads cannot retroactively change the ledger. Payloads must consequently
    support ``copy.deepcopy``. Returned records are snapshots, not live handles;
    use ``get`` to read the current state after recording or revising a task.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._lock = RLock()

    @_locked
    def create(self, source: str, goal: Any = None, *, dependencies=(),
               goal_interpretation_id: str | None = None) -> Task:
        goal_interpretation_id = _goal_interpretation_id(goal_interpretation_id)
        dependencies = _dependencies(dependencies)
        task = Task(
            id=f"task:{uuid4().hex}",
            source=source,
            goal=deepcopy(goal),
            revisions=(TaskRevision(1, deepcopy(goal), dependencies=deepcopy(dependencies),
                                    goal_interpretation_id=goal_interpretation_id),),
            dependencies=dependencies,
            goal_interpretation_id=goal_interpretation_id,
        )
        self._tasks[task.id] = task
        return deepcopy(task)

    @_locked
    def get(self, task_id: str) -> Task:
        return deepcopy(self._tasks[task_id])

    @_locked
    def current_revision(self, task_id: str) -> int:
        """Read authorization version without invoking payload-copy callbacks."""
        return self._tasks[task_id].revision

    @_locked
    def revise(self, task_id: str, goal: Any, *, reason: str, dependencies=None,
               goal_interpretation_id: str | None = None,
               expected_revision: int | None = None,
               before_commit: Callable[[], bool] | None = None) -> Task:
        """Revise a goal without replacing intervening revisions or attempts.

        Dependencies retain their existing default; the current goal-projection
        link clears unless explicitly supplied for this revision. Historical
        links remain unchanged. ``expected_revision`` supplies compare-and-swap
        admission. Even without it, reentrant payload copying cannot overwrite
        a newer revision. Returned snapshots may precede a same-revision receipt
        delivered during snapshot copying; that receipt remains in the ledger.
        An optional ``before_commit`` guard runs after all copying and must return
        exactly True. Its result is followed by fresh ledger revision validation;
        the guard is not authorization to replace a concurrently revised task.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("a task revision requires a nonempty reason")
        goal_interpretation_id = _goal_interpretation_id(goal_interpretation_id)
        if expected_revision is not None and (type(expected_revision) is not int or expected_revision < 1):
            raise ValueError("expected_revision must be a positive integer or None")
        if before_commit is not None and not callable(before_commit):
            raise TypeError("before_commit must be callable or None")
        task = self._tasks[task_id]
        admitted_revision = task.revision

        def current():
            latest = self._tasks[task_id]
            if expected_revision is not None and latest.revision != expected_revision:
                raise ValueError("stale expected task revision")
            if latest.revision != admitted_revision:
                raise ValueError("task revision changed while copying revision payloads")
            return latest

        current()  # Reject stale admission before invoking any payload copy.
        dependencies = _dependencies(task.dependencies if dependencies is None else dependencies)
        current()
        copied_goal = deepcopy(goal)
        current()
        historical_goal = deepcopy(goal)
        current()
        historical_dependencies = deepcopy(dependencies)
        current()
        revision = admitted_revision + 1
        entry = TaskRevision(revision, historical_goal, reason, historical_dependencies,
                             goal_interpretation_id)

        def revised(latest):
            return replace(latest, goal=copied_goal, revision=revision, status="ready",
                           revisions=latest.revisions + (entry,), dependencies=dependencies,
                           goal_interpretation_id=goal_interpretation_id)

        # Returning a detached snapshot can itself invoke payload callbacks.
        # Prepare it before committing, then revalidate and reread the stored
        # task after every such copy has finished. No copies follow the write.
        result = deepcopy(revised(current()))
        current()
        if before_commit is not None and before_commit() is not True:
            raise ValueError("task revision before_commit guard rejected")
        self._tasks[task_id] = revised(current())
        return result

    @_locked
    def record(self, task_id: str, outcome: Any, *, revision: int | None = None) -> Task:
        """Record an outcome against the revision actually attempted.

        Attempt status preserves the outcome's distinction (e.g. declined versus
        unknown); task status groups these as blocked. An explicit older revision
        retains delayed receipts without changing the current task's status.
        A completed revision cannot acquire another attempt. Omitting revision
        preserves the API's current-revision behavior.
        """
        task = self._tasks[task_id]
        revision = task.revision if revision is None else revision
        if type(revision) is not int or revision < 1 or not any(item.revision == revision for item in task.revisions):
            raise ValueError("attempt revision must identify a known positive task revision")
        if any(attempt.revision == revision and attempt.status == "done" for attempt in task.attempts):
            raise ValueError("a completed task must be revised before another attempt")
        status = outcome.status
        if status not in _TASK_STATUS:
            raise ValueError(f"unsupported task outcome status: {status!r}")
        attempt = deepcopy(TaskAttempt(
            revision=revision,
            status=status,
            plan=outcome.plan,
            receipt=outcome.receipt,
            verified=outcome.verified,
            reason=outcome.reason,
            steps=getattr(outcome, "steps", ()),
        ))
        # Payload copying may invoke user code. Preserve a revision made by such
        # a reentrant callback instead of writing our earlier snapshot over it.
        task = self._tasks[task_id]
        if any(previous.revision == revision and previous.status == "done" for previous in task.attempts):
            raise ValueError("a completed task must be revised before another attempt")
        updated = replace(task, status=_TASK_STATUS[status] if revision == task.revision else task.status,
                          attempts=task.attempts + (attempt,))
        self._tasks[task_id] = updated
        return deepcopy(updated)

    @_locked
    def values(self) -> tuple[Task, ...]:
        return tuple(deepcopy(task) for task in self._tasks.values())

    def __iter__(self) -> Iterator[Task]:
        return iter(self.values())

    @_locked
    def __len__(self) -> int:
        return len(self._tasks)
