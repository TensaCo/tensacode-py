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
from typing import Any, Iterator
from uuid import uuid4

from .task_dependencies import InterpretationDependency


def _dependencies(values):
    if not isinstance(values, tuple) or any(not isinstance(value, InterpretationDependency) for value in values):
        raise TypeError("task dependencies must be a tuple of InterpretationDependency values")
    for value in values:
        value.__post_init__()
    return deepcopy(values)


@dataclass(frozen=True)
class TaskRevision:
    revision: int
    goal: Any
    reason: str = ""
    dependencies: tuple[InterpretationDependency, ...] = ()


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
    def create(self, source: str, goal: Any = None, *, dependencies=()) -> Task:
        dependencies = _dependencies(dependencies)
        task = Task(
            id=f"task:{uuid4().hex}",
            source=source,
            goal=deepcopy(goal),
            revisions=(TaskRevision(1, deepcopy(goal), dependencies=deepcopy(dependencies)),),
            dependencies=dependencies,
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
    def revise(self, task_id: str, goal: Any, *, reason: str, dependencies=None) -> Task:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("a task revision requires a nonempty reason")
        task = self._tasks[task_id]
        dependencies = _dependencies(task.dependencies if dependencies is None else dependencies)
        revision = task.revision + 1
        updated = replace(
            task,
            goal=deepcopy(goal),
            revision=revision,
            status="ready",
            revisions=task.revisions + (TaskRevision(revision, deepcopy(goal), reason, deepcopy(dependencies)),),
            dependencies=dependencies,
        )
        self._tasks[task_id] = updated
        return deepcopy(updated)

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
