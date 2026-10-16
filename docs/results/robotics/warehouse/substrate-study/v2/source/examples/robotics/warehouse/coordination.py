"""Deterministic assignment with task dependencies and exclusive reservations.

This is dispatch, not collision avoidance: zone names are supplied route
resources. A robot's own motion controller must still enforce physical safety.
"""

from dataclasses import asdict, dataclass, field
from math import dist, isfinite

from .language import Task
from .representation import SceneGraph


def build_dispatch_kernel():
    """Compile eligibility and sensor-position assignment cost through ALG.0/1/2."""
    from tensorcode.algorithm import AlgorithmBuilder

    b = AlgorithmBuilder(name="warehouse_dispatch", dtype="float64")
    robot = b.input("robot_position", (3,))
    obj = b.input("object_position", (3,))
    dependencies = b.input("dependencies_ready")
    resources = b.input("resources_free")
    available = b.input("robot_available")
    fresh = b.input("evidence_fresh")
    capable = b.input("robot_capable")
    eligible = dependencies * resources * available * fresh * capable
    delta = robot - obj
    score = (delta * delta).sum()
    return b.build({"eligible": eligible, "cost": score})


@dataclass(frozen=True)
class RobotState:
    id: str
    position: tuple[float, float, float]
    capabilities: tuple[str, ...] = ("pick_place",)
    available: bool = True
    allowed_object_ids: tuple[str, ...] | None = None
    allowed_destination_ids: tuple[str, ...] | None = None
    allowed_pairs: tuple[tuple[str, str], ...] | None = None

    def __post_init__(self):
        if (
            not self.id
            or len(self.position) != 3
            or not all(isfinite(v) for v in self.position)
        ):
            raise ValueError("Robot needs an identity and finite estimated 3D position")


@dataclass(frozen=True)
class Assignment:
    task_id: str
    robot_id: str
    object_id: str
    destination_id: str
    resources: tuple[str, ...]
    travel_distance_m: float


def order_cost_groups(candidates, absolute_tolerance=1e-12, relative_tolerance=1e-12):
    """Sort finite (cost, identity, payload) rows with anchored tolerance groups.

    The absolute tolerance has squared-metre units for dispatch. Each group is
    anchored at its minimum *raw* cost; membership never chains through a later
    member. Within a group, identity wins. This is deterministic for fixed costs,
    but rounding near a group boundary can still change group membership.
    """
    if any(not isfinite(t) or t < 0 for t in (absolute_tolerance, relative_tolerance)):
        raise ValueError("Cost tolerances must be finite and nonnegative")
    if any(not isfinite(row[0]) for row in candidates):
        raise ValueError("Assignment costs must be finite")
    rows = sorted(candidates, key=lambda row: row[:2])
    ordered = []
    start = 0
    while start < len(rows):
        anchor = rows[start][0]
        tolerance = absolute_tolerance + relative_tolerance * abs(anchor)
        stop = start + 1
        while stop < len(rows) and rows[stop][0] - anchor <= tolerance:
            stop += 1
        ordered.extend(sorted(rows[start:stop], key=lambda row: row[1]))
        start = stop
    return ordered


@dataclass
class Coordinator:
    completed: set[str] = field(default_factory=set)
    active: dict[str, Assignment] = field(default_factory=dict)
    outcomes: list[dict] = field(default_factory=list)
    substrate: str = "alg0"
    failed: set[str] = field(default_factory=set)
    last_gates: dict[str, dict] = field(default_factory=dict)
    cost_absolute_tolerance: float = 1e-12
    cost_relative_tolerance: float = 1e-12
    _machine: object = field(default=None, repr=False, init=False)

    def __post_init__(self):
        order_cost_groups(
            [], self.cost_absolute_tolerance, self.cost_relative_tolerance
        )
        self._machine = build_dispatch_kernel().compile(self.substrate)

    def dispatch(
        self,
        tasks: list[Task],
        robots: list[RobotState],
        scene: SceneGraph,
        now: float,
        routes: dict[str, tuple[str, ...]] | None = None,
        max_age: float = 2.0,
    ) -> list[Assignment]:
        if len({t.id for t in tasks}) != len(tasks) or len(
            {r.id for r in robots}
        ) != len(robots):
            raise ValueError("Task and robot IDs must be unique")
        task_ids = {t.id for t in tasks}
        if any(set(t.dependencies) - task_ids - self.completed for t in tasks):
            raise ValueError("Unknown task dependency")
        reachable = set(self.completed)
        remaining = list(tasks)
        while remaining:
            ready = [t for t in remaining if set(t.dependencies) <= reachable]
            if not ready:
                raise ValueError("Cyclic task dependencies")
            reachable.update(t.id for t in ready)
            remaining = [t for t in remaining if t not in ready]
        busy = {a.robot_id for a in self.active.values()}
        reserved = {r for a in self.active.values() for r in a.resources}
        assignments = []
        self.last_gates = {}
        # Ready tasks compete for resources in stable lexical-ID order.
        for task in sorted(tasks, key=lambda task: task.id):
            if (
                task.id in self.completed
                or task.id in self.active
                or task.id in self.failed
            ):
                continue
            obj, dest = (
                scene.entities[task.object_id],
                scene.entities[task.destination_id],
            )
            resources = (
                f"object:{obj.id}",
                f"destination:{dest.id}",
                *((routes or {}).get(task.id, ())),
            )
            eligible = []
            self.last_gates[task.id] = {}
            for robot in robots:
                inputs = {
                    "robot_position": robot.position,
                    "object_position": obj.position,
                    "dependencies_ready": float(
                        set(task.dependencies) <= self.completed
                    ),
                    "resources_free": float(not set(resources) & reserved),
                    "robot_available": float(robot.available and robot.id not in busy),
                    "evidence_fresh": float(
                        scene.fresh(obj, now, max_age)
                        and scene.fresh(dest, now, max_age)
                    ),
                    "robot_capable": float(
                        task.operation in robot.capabilities
                        and (
                            robot.allowed_object_ids is None
                            or obj.id in robot.allowed_object_ids
                        )
                        and (
                            robot.allowed_destination_ids is None
                            or dest.id in robot.allowed_destination_ids
                        )
                        and (
                            robot.allowed_pairs is None
                            or (obj.id, dest.id) in robot.allowed_pairs
                        )
                    ),
                }
                output = self._machine(inputs)
                self.last_gates[task.id][robot.id] = {
                    **{
                        key: value
                        for key, value in inputs.items()
                        if key not in {"robot_position", "object_position"}
                    },
                    "eligible": float(output["eligible"]),
                    "cost": float(output["cost"]),
                }
                if float(output["eligible"]) > 0.5:
                    eligible.append((float(output["cost"]), robot.id, robot))
            if not eligible:
                continue
            robot = order_cost_groups(
                eligible, self.cost_absolute_tolerance, self.cost_relative_tolerance
            )[0][2]
            assignment = Assignment(
                task.id,
                robot.id,
                obj.id,
                dest.id,
                tuple(resources),
                dist(robot.position, obj.position) + dist(obj.position, dest.position),
            )
            self.active[task.id] = assignment
            assignments.append(assignment)
            busy.add(robot.id)
            reserved.update(resources)
        return assignments

    def finish(self, task_id: str, success: bool, evidence: str):
        if not evidence:
            raise ValueError("Completion requires an execution-evidence reference")
        assignment = self.active.pop(task_id)
        if success:
            self.completed.add(task_id)
        else:
            self.failed.add(task_id)
        self.outcomes.append(
            {**asdict(assignment), "success": success, "evidence": evidence}
        )


def independent_assignment_baseline(tasks, robots, scene):
    """Explicit weak ablation: nearest robot per task, no joint reservations."""
    return {
        t.id: min(
            robots,
            key=lambda r: (
                dist(r.position, scene.entities[t.object_id].position),
                r.id,
            ),
        ).id
        for t in tasks
    }
