"""Explicit restricted English -> instruction tree -> evidence-grounded task DAG.

Grammar: move/bring/carry PLACEABLE to DESTINATION; compose clauses with
`while` (parallel) and `then`/`;` (sequence). References are a kind or ID,
optionally preceded by a color, optionally `in`/`on` a containing entity or
`left of`/`right of` another entity (warehouse x axis). No learned parser.
Unsupported language is rejected, never silently reduced to keywords.
"""

import re
from dataclasses import asdict, dataclass, field

from .representation import Entity, SceneGraph


class GroundingError(ValueError):
    def __init__(self, status: str, message: str, candidates=()):
        super().__init__(message)
        self.status = status
        self.candidates = tuple(candidates)


@dataclass(frozen=True)
class Reference:
    name: str
    color: str | None = None
    relation: str | None = None
    anchor: "Reference | None" = None


@dataclass(frozen=True)
class Instruction:
    operation: str
    object: Reference | None = None
    destination: Reference | None = None
    children: tuple["Instruction", ...] = ()

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Task:
    id: str
    object_id: str
    destination_id: str
    dependencies: tuple[str, ...]
    scene_revision: int
    evidence_sources: tuple[str, ...]
    operation: str = "pick_place"


COLORS = {"red", "green", "blue", "yellow", "magenta", "orange", "white", "black"}


def reference_from_dict(value, depth=0):
    if (
        depth > 8
        or not isinstance(value, dict)
        or set(value) != {"name", "color", "relation", "anchor"}
    ):
        raise ValueError("Invalid reference object")
    if not isinstance(value["name"], str) or not re.fullmatch(
        r"[a-z0-9_-]+", value["name"]
    ):
        raise ValueError("Invalid reference name")
    if value["color"] is not None and (
        not isinstance(value["color"], str) or value["color"] not in COLORS
    ):
        raise ValueError("Unsupported color")
    relation = value["relation"]
    if relation is not None and (
        not isinstance(relation, str)
        or relation not in {"in", "on", "left of", "right of"}
    ):
        raise ValueError("Unsupported spatial relation")
    if (relation is None) != (value["anchor"] is None):
        raise ValueError("Relation requires exactly one anchor")
    return Reference(
        value["name"],
        value["color"],
        relation,
        reference_from_dict(value["anchor"], depth + 1) if relation else None,
    )


def instruction_from_dict(data: dict, _depth: int = 0) -> Instruction:
    """Validate untrusted parser output; never execute model-generated code."""
    if (
        _depth > 16
        or not isinstance(data, dict)
        or set(data) != {"operation", "object", "destination", "children"}
    ):
        raise ValueError("Invalid instruction object or excessive tree depth")

    operation = data["operation"]
    if not isinstance(operation, str):
        raise ValueError("Invalid instruction operation")  # noqa: TRY004 - malformed JSON contract
    if not isinstance(data["children"], list) or len(data["children"]) > 64:
        raise ValueError("Invalid instruction children")
    if operation == "pick_place" and not data["children"]:
        return Instruction(
            operation,
            reference_from_dict(data["object"]),
            reference_from_dict(data["destination"]),
        )
    if (
        operation in {"sequence", "parallel"}
        and data["object"] is None
        and data["destination"] is None
        and len(data["children"]) >= 2
    ):
        return Instruction(
            operation,
            children=tuple(
                instruction_from_dict(c, _depth + 1) for c in data["children"]
            ),
        )
    raise ValueError("Invalid instruction operation or structure")


def parse_reference(text: str) -> Reference:
    text = re.sub(r"^(?:the|a|an)\s+", "", text.strip().lower())
    parts = re.split(r"\s+(left of|right of|in|on)\s+", text, maxsplit=1)
    words = parts[0].split()
    color = words.pop(0) if words and words[0] in COLORS else None
    if len(words) != 1 or not re.fullmatch(r"[a-z0-9_-]+", words[0]):
        raise ValueError(f"Unsupported reference: {text!r}")
    return Reference(
        words[0],
        color,
        parts[1] if len(parts) > 1 else None,
        parse_reference(parts[2]) if len(parts) > 1 else None,
    )


def parse_instruction(text: str) -> Instruction:
    if not isinstance(text, str) or not 0 < len(text) <= 4096:
        raise ValueError("Instruction must contain 1..4096 characters")
    text = text.strip().lower().rstrip(".")
    sequential = re.split(r"\s*;\s*(?:then\s+)?|\s+(?:and\s+)?then\s+", text)
    if len(sequential) > 1:
        return Instruction(
            "sequence", children=tuple(parse_instruction(x) for x in sequential)
        )
    parallel = re.split(r"\s+while\s+", text)
    if len(parallel) > 1:
        return Instruction(
            "parallel", children=tuple(parse_instruction(x) for x in parallel)
        )
    match = re.fullmatch(r"(?:move|bring|carry)\s+(.+?)\s+to\s+(.+)", text)
    if not match:
        raise ValueError(
            "Supported commands: move/bring/carry X to Y, composed with while/then"
        )
    return Instruction(
        "pick_place", parse_reference(match[1]), parse_reference(match[2])
    )


def resolve(
    reference: Reference, scene: SceneGraph, now: float, max_age: float = 2.0
) -> Entity:
    candidates = [
        e
        for e in scene.entities.values()
        if reference.name in {e.id.lower(), e.kind.lower()}
        and (reference.color is None or e.attributes.get("color") == reference.color)
    ]
    candidates = [e for e in candidates if scene.fresh(e, now, max_age)]
    if reference.anchor:
        anchor = resolve(reference.anchor, scene, now, max_age)
        if reference.relation == "in":
            candidates = [e for e in candidates if e.parent_id == anchor.id]
        elif reference.relation == "on":
            candidates = [
                e
                for e in candidates
                if any(
                    r.subject == e.id
                    and r.object == anchor.id
                    and r.predicate == "on"
                    and 0 <= now - r.observed_at <= max_age
                    and r.confidence >= 0.7
                    for r in scene.relations
                )
            ]
        else:
            direction = -1 if reference.relation == "left of" else 1
            candidates = [
                e
                for e in candidates
                if direction * (e.position[0] - anchor.position[0])
                > e.uncertainty_m + anchor.uncertainty_m
            ]
    if not candidates:
        raise GroundingError(
            "unknown", "No fresh, confident observation supports the reference"
        )
    if len(candidates) != 1:
        raise GroundingError(
            "ambiguous",
            "Reference matches multiple observed entities",
            [e.id for e in candidates],
        )
    return candidates[0]


def ground_instruction(
    tree: Instruction, scene: SceneGraph, now: float, max_age: float = 2.0
) -> list[Task]:
    tasks = []

    def visit(node, dependencies):
        if node.operation == "sequence":
            terminal = dependencies
            for child in node.children:
                terminal = visit(child, terminal)
            return terminal
        if node.operation == "parallel":
            return tuple(
                t for child in node.children for t in visit(child, dependencies)
            )
        if (
            node.operation != "pick_place"
            or node.object is None
            or node.destination is None
        ):
            raise ValueError("Invalid instruction tree")
        obj = resolve(node.object, scene, now, max_age)
        dest = resolve(node.destination, scene, now, max_age)
        if obj.id == dest.id:
            raise GroundingError("invalid", "An object cannot be its own destination")
        if obj.kind not in {"parcel", "box", "cube", "item"} or dest.kind not in {
            "station",
            "tray",
            "bin",
            "shelf",
        }:
            raise GroundingError(
                "invalid", "Pick/place requires a movable object and receiving surface"
            )
        sources = {obj.source, dest.source}
        for reference in (node.object, node.destination):
            while reference.anchor:
                anchor = resolve(reference.anchor, scene, now, max_age)
                sources.add(anchor.source)
                if reference.relation == "on":
                    sources.update(
                        r.source
                        for r in scene.relations
                        if r.object == anchor.id
                        and r.predicate == "on"
                        and 0 <= now - r.observed_at <= max_age
                        and r.confidence >= 0.7
                    )
                reference = reference.anchor
        task = Task(
            f"task-{len(tasks) + 1}",
            obj.id,
            dest.id,
            tuple(dependencies),
            scene.revision,
            tuple(sorted(sources)),
        )
        tasks.append(task)
        return (task.id,)

    visit(tree, ())
    return tasks


@dataclass(frozen=True)
class Answer:
    status: str
    text: str
    entity_ids: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    scene_revision: int = 0
    details: dict = field(default_factory=dict)


def answer_question(
    question: str, scene: SceneGraph, now: float, max_age: float = 2.0
) -> Answer:
    question = question.strip().lower().rstrip("?.")
    count = re.fullmatch(r"how many ([a-z0-9_-]+)(?: are there)?", question)
    if count:
        kind = count[1].removesuffix("s")
        known = [e for e in scene.entities.values() if e.kind == kind]
        fresh = [e for e in known if scene.fresh(e, now, max_age)]
        if not fresh:
            return Answer(
                "unknown",
                "No fresh observations establish that count.",
                scene_revision=scene.revision,
            )
        return Answer(
            "supported",
            f"I have fresh observations of {len(fresh)} {count[1]}; unseen objects are not counted.",
            tuple(e.id for e in fresh),
            tuple(sorted({e.source for e in fresh})),
            scene.revision,
            {
                "observed_count": len(fresh),
                "stale_or_uncertain_count": len(known) - len(fresh),
            },
        )
    contents = re.fullmatch(r"what is (in|on) (.+)", question)
    spatial = re.fullmatch(r"is (.+?) (left of|right of) (.+)", question)
    if contents or spatial:
        try:
            if contents:
                relation, reference = contents.groups()
                anchor = resolve(parse_reference(reference), scene, now, max_age)
                candidates = [
                    e for e in scene.entities.values() if scene.fresh(e, now, max_age)
                ]
                sources = {anchor.source}
                if relation == "in":
                    candidates = [e for e in candidates if e.parent_id == anchor.id]
                else:
                    relations = [
                        r
                        for r in scene.relations
                        if r.object == anchor.id
                        and r.predicate == "on"
                        and 0 <= now - r.observed_at <= max_age
                        and r.confidence >= 0.7
                    ]
                    candidates = [
                        e for e in candidates if e.id in {r.subject for r in relations}
                    ]
                    sources.update(r.source for r in relations)
                if not candidates:
                    return Answer(
                        "unknown",
                        "No fresh evidence establishes the contents; this does not establish emptiness.",
                        (anchor.id,),
                        (anchor.source,),
                        scene.revision,
                    )
                sources.update(e.source for e in candidates)
                return Answer(
                    "supported",
                    f"Observed {relation} {anchor.id}: "
                    + ", ".join(e.id for e in candidates),
                    tuple(e.id for e in candidates),
                    tuple(sorted(sources)),
                    scene.revision,
                )
            left, relation, right = spatial.groups()
            first = resolve(parse_reference(left), scene, now, max_age)
            second = resolve(parse_reference(right), scene, now, max_age)
            separation = first.position[0] - second.position[0]
            uncertainty = first.uncertainty_m + second.uncertainty_m
            if abs(separation) <= uncertainty:
                return Answer(
                    "unknown",
                    "Position uncertainty prevents that spatial comparison.",
                    (first.id, second.id),
                    tuple(sorted({first.source, second.source})),
                    scene.revision,
                )
            yes = separation < 0 if relation == "left of" else separation > 0
            return Answer(
                "supported",
                ("Yes" if yes else "No") + "; compared along the warehouse x axis.",
                (first.id, second.id),
                tuple(sorted({first.source, second.source})),
                scene.revision,
                {"answer": yes},
            )
        except GroundingError as exc:
            return Answer(
                exc.status, str(exc), exc.candidates, scene_revision=scene.revision
            )
        except ValueError as exc:
            return Answer("unsupported", str(exc), scene_revision=scene.revision)
    match = re.fullmatch(r"where is (.+)", question)
    if not match:
        return Answer(
            "unsupported",
            "Supported questions: where is X; how many KIND; what is in/on X; is X left/right of Y?",
            scene_revision=scene.revision,
        )
    try:
        entity = resolve(parse_reference(match[1]), scene, now, max_age)
    except GroundingError as exc:
        return Answer(
            exc.status, str(exc), exc.candidates, scene_revision=scene.revision
        )
    except ValueError as exc:
        return Answer("unsupported", str(exc), scene_revision=scene.revision)
    position = tuple(round(v, 3) for v in entity.position)
    return Answer(
        "supported",
        f"{entity.id} was observed at {position} m in the warehouse frame.",
        (entity.id,),
        (entity.source,),
        scene.revision,
        {
            "position_m": position,
            "age_s": now - entity.observed_at,
            "uncertainty_m": entity.uncertainty_m,
            "confidence": entity.confidence,
        },
    )
