from typing import Protocol
from tensacode.internal.types.obj_op_latent_type_triple import OBJ_OP_LATENT_TYPE_TRIPLE
from tensacode.internal.utils.pydantic import HasID


class TaggedObject(Protocol):
    _tc_engine_meta: dict[Engine.ID, Engine.Meta]
    _tc_op_overrides: dict[tuple[HasGUID.GUID], Operation]
    _tc_logs: dict[Run.ID, list[str]]
