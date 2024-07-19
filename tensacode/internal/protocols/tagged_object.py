from typing import Optional, Annotated
from pydantic import UUID4, BaseModel
from typing import Protocol, Any
from tensacode.core.base.base_engine import BaseEngine
from tensacode.core.base.ops.base_op import BaseOp
from tensacode.internal.protocols.latent_types import LatentType, LATENT_TYPE_STR
from tensacode.internal.utils.misc import advanced_equality_check, WILDCARD
from tensacode.internal.utils.pydantic import SubclassField


class TCSelector(BaseModel):
    engine_type: SubclassField(BaseEngine) | None
    engine_id: str | None
    run_id: UUID4 | None
    operator_type: SubclassField(BaseOp) | None
    operator_id: str | None
    object_type: SubclassField(Any) | None
    object_id: str | None
    latent_type: SubclassField(LatentType) | None

    def __eq__(self, other: Any) -> bool:
        if super().__eq__(other):
            return True
        if not isinstance(other, TCSelector):
            return False
        return advanced_equality_check(self, other)


class TaggedObject(Protocol):
    _tc_config_overrides: dict[TCSelector, dict[str, Any]]
    _tc_enc_cache: dict[TCSelector, LatentType]
    _tc_op_overrides: dict[TCSelector, BaseOp]
