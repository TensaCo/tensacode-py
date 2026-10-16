"""Read-only scheduler receipts for observed-state mission question answering.

The adapter deliberately has no dispatch/finish interface. Loading an archived
receipt cannot resume a robot mission or certify that its source is authentic.
"""

from copy import deepcopy
from types import SimpleNamespace

from .coordination import Assignment
from .jobs import Capability, JobScheduler, validate_capabilities
from .language import Task


class RecordedScheduler:
    def __init__(self, snapshot: dict, source: str = "supplied scheduler snapshot"):
        if (
            not isinstance(snapshot, dict)
            or set(snapshot)
            != {"schema", "tasks", "capabilities", "coordinator", "events"}
            or snapshot["schema"] != "tensorcode.warehouse.scheduler.v1"
        ):
            raise ValueError("Unsupported scheduler receipt")
        snapshot = deepcopy(snapshot)
        self.tasks = [
            Task(
                **{
                    **row,
                    "dependencies": tuple(row["dependencies"]),
                    "evidence_sources": tuple(row["evidence_sources"]),
                }
            )
            for row in snapshot["tasks"]
        ]
        self.capabilities = tuple(
            Capability(
                **{
                    **row,
                    "object_ids": tuple(row["object_ids"]),
                    "destination_ids": tuple(row["destination_ids"]),
                    "operations": tuple(row["operations"]),
                    "allowed_pairs": None
                    if row.get("allowed_pairs") is None
                    else tuple(tuple(pair) for pair in row["allowed_pairs"]),
                }
            )
            for row in snapshot["capabilities"]
        )
        validate_capabilities(self.tasks, self.capabilities)
        state = snapshot["coordinator"]
        if set(state) != {
            "completed",
            "failed",
            "active",
            "outcomes",
            "last_gates",
            "substrate",
        }:
            raise ValueError("Malformed coordinator receipt")
        known = {task.id for task in self.tasks}
        completed, failed = set(state["completed"]), set(state["failed"])
        active = {key: Assignment(**value) for key, value in state["active"].items()}
        if (
            (completed | failed | set(active)) - known
            or completed & failed
            or (completed | failed) & set(active)
        ):
            raise ValueError("Contradictory task lifecycle receipt")
        if not isinstance(snapshot["events"], list) or any(
            not isinstance(e, dict) or "type" not in e for e in snapshot["events"]
        ):
            raise ValueError("Malformed lifecycle event history")
        self.coordinator = SimpleNamespace(
            completed=completed, failed=failed, active=active
        )
        self.events = snapshot["events"]
        self.source = source

    def explain(self, question: str):
        return {
            **JobScheduler.explain(self, question),
            "scope": "recorded_scheduler_receipt",
            "receipt_source": self.source,
        }
