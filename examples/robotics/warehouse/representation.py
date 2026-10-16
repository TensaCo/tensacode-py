"""Data-only, revisioned sensor scene for the bounded warehouse example.

Nothing in this module queries simulator state. Producers must attach their
observation provenance; accepting a record does not certify its provenance.
"""

from dataclasses import asdict, dataclass, field
from math import isfinite


@dataclass(frozen=True)
class Entity:
    id: str
    kind: str
    attributes: dict[str, str]
    position: tuple[float, float, float]
    observed_at: float
    source: str
    confidence: float = 1.0
    uncertainty_m: float = 0.0
    parent_id: str | None = None

    def __post_init__(self):
        if not self.id or not self.source or not self.kind:
            raise ValueError(
                "Entity identity, kind and observation source are required"
            )
        if len(self.position) != 3 or not all(isfinite(v) for v in self.position):
            raise ValueError("Position must contain three finite values")
        if not isfinite(self.observed_at) or not 0 <= self.confidence <= 1:
            raise ValueError("Invalid observation timestamp or confidence")
        if not isfinite(self.uncertainty_m) or self.uncertainty_m < 0:
            raise ValueError("Invalid positional uncertainty")


@dataclass(frozen=True)
class Relation:
    subject: str
    predicate: str
    object: str
    observed_at: float
    source: str
    confidence: float = 1.0


@dataclass
class SceneGraph:
    entities: dict[str, Entity] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    revision: int = 0

    def upsert(self, entity: Entity):
        previous = self.entities.get(entity.id)
        if previous and entity.observed_at < previous.observed_at:
            raise ValueError(
                "Out-of-order evidence cannot overwrite a newer observation"
            )
        if entity.parent_id == entity.id:
            raise ValueError("An entity cannot contain itself")
        cursor = entity.parent_id
        seen = {entity.id}
        while cursor is not None:
            if cursor in seen:
                raise ValueError("Scene composition must be acyclic")
            seen.add(cursor)
            parent = self.entities.get(cursor)
            if parent is None:
                raise ValueError(f"Unknown composition parent: {cursor}")
            cursor = parent.parent_id
        self.revision += 1
        self.history.append(
            {
                "revision": self.revision,
                "previous": asdict(previous) if previous else None,
                "observation": asdict(entity),
            }
        )
        self.entities[entity.id] = entity

    def add_relation(self, relation: Relation):
        if (
            relation.subject not in self.entities
            or relation.object not in self.entities
        ):
            raise ValueError("Relation endpoints must be observed entities")
        if (
            not relation.source
            or not isfinite(relation.observed_at)
            or not 0 <= relation.confidence <= 1
        ):
            raise ValueError("Invalid relation evidence")
        self.relations.append(relation)
        self.revision += 1

    def fresh(
        self,
        entity: Entity,
        now: float,
        max_age: float = 2.0,
        min_confidence: float = 0.7,
    ):
        return (
            0 <= now - entity.observed_at <= max_age
            and entity.confidence >= min_confidence
        )

    def to_dict(self):
        return {
            "revision": self.revision,
            "entities": [asdict(e) for e in self.entities.values()],
            "relations": [asdict(r) for r in self.relations],
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data):
        """Load a saved observation scene, checking structure and parent cycles."""
        if not isinstance(data, dict) or set(data) != {
            "revision",
            "entities",
            "relations",
            "history",
        }:
            raise ValueError("Invalid scene document")
        if (
            not isinstance(data["revision"], int)
            or data["revision"] < 0
            or not isinstance(data["history"], list)
        ):
            raise ValueError("Invalid scene revision history")
        scene = cls()
        pending = []
        for row in data["entities"]:
            entity = Entity(**{**row, "position": tuple(row["position"])})
            if entity.id in {e.id for e in pending}:
                raise ValueError("Duplicate entity ID")
            pending.append(entity)
        while pending:
            ready = [
                e
                for e in pending
                if e.parent_id is None or e.parent_id in scene.entities
            ]
            if not ready:
                raise ValueError("Unknown or cyclic composition parent")
            for entity in ready:
                scene.upsert(entity)
                pending.remove(entity)
        for row in data["relations"]:
            scene.add_relation(Relation(**row))
        if data["revision"] < scene.revision:
            raise ValueError("Revision predates current scene records")
        scene.revision = data["revision"]
        scene.history = data["history"]
        return scene
