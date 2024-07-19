from typing import Protocol, TypedDict
from pydantic import UUID4


class OpOverrideDiscriminator(TypedDict):
    op: str
    run: UUID4
    object_type: str
    latent_format: str


class TaggedObject(Protocol):
    _tc_engine_meta: dict[Engine.ID, Engine.Meta]
    _tc_op_overrides: dict[tuple[OpOverrideDiscriminator], Operation]
    _tc_logs: dict[Run.ID, list[str]]
