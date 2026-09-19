"""Task identity and revision history for one agent's lifetime.

The ledger is in memory, not disk persistence. Goals are data supplied by callers;
they need not come from a lexical resource. Plans and receipts belong to attempts,
so revising a goal never rewrites what an earlier attempt tried or observed.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Iterator
from uuid import uuid4


@dataclass(frozen=True)
class TaskRevision:
    revision: int
    goal: Any
    reason: str = ""


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


class TaskLedger:
    """Store detached snapshots of tasks, revisions, and execution attempts.

    Inputs and returned records are deep-copied: mutable goal arguments or receipt
    payloads cannot retroactively change the ledger. Payloads must consequently
    support ``copy.deepcopy``. Returned records are snapshots, not live handles;
    use ``get`` to read the current state after recording or revising a task.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def create(self, source: str, goal: Any = None) -> Task:
        task = Task(
            id=f"task:{uuid4().hex}",
            source=source,
            goal=deepcopy(goal),
            revisions=(TaskRevision(1, deepcopy(goal)),),
        )
        self._tasks[task.id] = task
        return deepcopy(task)

    def get(self, task_id: str) -> Task:
        return deepcopy(self._tasks[task_id])

    def revise(self, task_id: str, goal: Any, *, reason: str) -> Task:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("a task revision requires a nonempty reason")
        task = self._tasks[task_id]
        revision = task.revision + 1
        updated = replace(
            task,
            goal=deepcopy(goal),
            revision=revision,
            status="ready",
            revisions=task.revisions + (TaskRevision(revision, deepcopy(goal), reason),),
        )
        self._tasks[task_id] = updated
        return deepcopy(updated)

    def record(self, task_id: str, outcome: Any) -> Task:
        """Record a request outcome against its current goal revision.

        Attempt status preserves the outcome's distinction (e.g. declined versus
        unknown); task status groups these as blocked. A completed task must be
        explicitly revised before another attempt can be recorded.
        """
        task = self._tasks[task_id]
        if task.status == "done":
            raise ValueError("a completed task must be revised before another attempt")
        status = outcome.status
        if status not in _TASK_STATUS:
            raise ValueError(f"unsupported task outcome status: {status!r}")
        attempt = deepcopy(TaskAttempt(
            revision=task.revision,
            status=status,
            plan=outcome.plan,
            receipt=outcome.receipt,
            verified=outcome.verified,
            reason=outcome.reason,
            steps=getattr(outcome, "steps", ()),
        ))
        updated = replace(task, status=_TASK_STATUS[status], attempts=task.attempts + (attempt,))
        self._tasks[task_id] = updated
        return deepcopy(updated)

    def values(self) -> tuple[Task, ...]:
        return tuple(deepcopy(task) for task in self._tasks.values())

    def __iter__(self) -> Iterator[Task]:
        return iter(self.values())

    def __len__(self) -> int:
        return len(self._tasks)
