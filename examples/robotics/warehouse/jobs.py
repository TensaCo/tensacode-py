"""Capability-checked task lifecycle and explanations over observed scene state.

This layer does not add robot skills. A mission is admitted only when every
grounded job is within at least one supplied robot's explicit execution scope.
Scope is an engineering capability contract, not an observed object location.
"""

import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace

from .coordination import Coordinator, RobotState
from .language import Instruction, Task, ground_instruction
from .representation import SceneGraph


@dataclass(frozen=True)
class Capability:
    robot_id: str
    object_ids: tuple[str, ...]
    destination_ids: tuple[str, ...]
    operations: tuple[str, ...] = ("pick_place",)
    allowed_pairs: tuple[tuple[str, str], ...] | None = None

    def supports(self, task: Task) -> bool:
        return (
            task.operation in self.operations
            and task.object_id in self.object_ids
            and task.destination_id in self.destination_ids
            and (
                self.allowed_pairs is None
                or (task.object_id, task.destination_id) in self.allowed_pairs
            )
        )


class UnsupportedMission(ValueError):
    def __init__(self, tasks):
        self.tasks = tuple(tasks)
        super().__init__(
            "No robot supports the requested object/destination pair: "
            + ", ".join(self.tasks)
        )


def validate_capabilities(tasks: list[Task], capabilities: tuple[Capability, ...]):
    if len({c.robot_id for c in capabilities}) != len(capabilities):
        raise ValueError("Duplicate robot capability contract")
    unsupported = [
        task.id for task in tasks if not any(c.supports(task) for c in capabilities)
    ]
    if unsupported:
        raise UnsupportedMission(unsupported)


@dataclass
class JobScheduler:
    tasks: list[Task]
    capabilities: tuple[Capability, ...]
    coordinator: Coordinator = field(default_factory=Coordinator)
    events: list[dict] = field(default_factory=list)

    def __post_init__(self):
        validate_capabilities(self.tasks, self.capabilities)

    @classmethod
    def from_instruction(
        cls,
        instruction: Instruction,
        scene: SceneGraph,
        now: float,
        capabilities: tuple[Capability, ...],
        substrate="alg0",
        max_age=2.0,
    ):
        return cls(
            ground_instruction(instruction, scene, now, max_age),
            capabilities,
            Coordinator(substrate=substrate),
        )

    def dispatch(
        self,
        robots: list[RobotState],
        scene: SceneGraph,
        now: float,
        routes=None,
        max_age=2.0,
    ):
        contracts = {c.robot_id: c for c in self.capabilities}
        scoped = []
        for robot in robots:
            if robot.id not in contracts:
                raise ValueError(f"No explicit capability contract for {robot.id}")
            contract = contracts[robot.id]
            scoped.append(
                replace(
                    robot,
                    capabilities=tuple(
                        set(robot.capabilities) & set(contract.operations)
                    ),
                    allowed_object_ids=contract.object_ids,
                    allowed_destination_ids=contract.destination_ids,
                    allowed_pairs=contract.allowed_pairs,
                )
            )
        assignments = self.coordinator.dispatch(
            self.tasks, scoped, scene, now, routes, max_age
        )
        self.events.append(
            {
                "time": now,
                "scene_revision": scene.revision,
                "type": "dispatch_checked",
                "gates": self.coordinator.last_gates,
                "assigned": [asdict(a) for a in assignments],
            }
        )
        return assignments

    def finish(self, task_id: str, success: bool, evidence: str, now: float):
        self.coordinator.finish(task_id, success, evidence)
        self.events.append(
            {
                "time": now,
                "type": "outcome",
                "task_id": task_id,
                "success": success,
                "evidence": evidence,
            }
        )

    def snapshot(self):
        """Data-only receipt for QA/review; observed gates are preserved exactly."""
        return deepcopy(
            {
                "schema": "tensorcode.warehouse.scheduler.v1",
                "tasks": [asdict(task) for task in self.tasks],
                "capabilities": [
                    asdict(capability) for capability in self.capabilities
                ],
                "coordinator": {
                    "completed": sorted(self.coordinator.completed),
                    "failed": sorted(self.coordinator.failed),
                    "active": {
                        key: asdict(value)
                        for key, value in self.coordinator.active.items()
                    },
                    "outcomes": self.coordinator.outcomes,
                    "last_gates": self.coordinator.last_gates,
                    "substrate": self.coordinator.substrate,
                },
                "events": self.events,
            }
        )

    def explain(self, question: str):
        match = re.fullmatch(
            r"why is ([a-z0-9_-]+) waiting\??", question.lower().strip()
        )
        if not match:
            return {
                "status": "unsupported",
                "text": "Supported lifecycle question: Why is ROBOT_ID waiting?",
            }
        robot_id = match[1]
        if robot_id not in {c.robot_id for c in self.capabilities}:
            return {
                "status": "unknown",
                "text": "No capability contract establishes that robot.",
            }
        active = [a for a in self.coordinator.active.values() if a.robot_id == robot_id]
        if active:
            return {
                "status": "supported",
                "text": f"{robot_id} is assigned, not waiting.",
                "active_tasks": [a.task_id for a in active],
                "source": "coordinator.active",
            }
        checks = [
            (index, e)
            for index, e in enumerate(self.events)
            if e["type"] == "dispatch_checked"
        ]
        if not checks:
            return {"status": "unknown", "text": "No dispatch check has occurred."}
        latest_index, latest = checks[-1]
        blocked = []
        contract = next(c for c in self.capabilities if c.robot_id == robot_id)
        for task in self.tasks:
            if not contract.supports(task):
                continue
            if task.id in self.coordinator.completed:
                continue
            if task.id in self.coordinator.failed:
                blocked.append(
                    {
                        "task_id": task.id,
                        "reasons": ["execution_failed_no_automatic_retry"],
                    }
                )
                continue
            gates = latest["gates"].get(task.id, {}).get(robot_id)
            if gates is not None:
                blocked.append(
                    {
                        "task_id": task.id,
                        "reasons": [
                            k
                            for k, v in gates.items()
                            if k not in {"cost", "eligible"} and v == 0.0
                        ],
                        "gates": gates,
                    }
                )
        return {
            "status": "supported",
            "robot_id": robot_id,
            "checked_at": latest["time"],
            "scene_revision": latest["scene_revision"],
            "tasks": blocked,
            "text": "All jobs supported by this robot completed."
            if not blocked
            else "Waiting reasons are the recorded dispatch gates.",
            "source": f"events/{latest_index}",
        }
