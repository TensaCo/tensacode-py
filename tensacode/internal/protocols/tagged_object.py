from typing import Optional
from pydantic import UUID4, BaseModel
from typing import Protocol, Any
from tensacode.core.base.base_engine import BaseEngine
from tensacode.core.base.ops.base_op import BaseOp
from tensacode.internal.protocols.latent_types import LatentType


class TCSelector(BaseModel):
    engine_type: Optional[str]
    engine_id: Optional[str]
    run_id: Optional[UUID4]
    object_type: Optional[str]
    object_id: Optional[str]
    latent_type: Optional[str]


class TaggedObject(Protocol):
    _tc_config_overrides: dict[TCSelector, dict[str, Any]]
    _tc_enc_cache: dict[TCSelector, LatentType]
    _tc_op_overrides: dict[TCSelector, BaseOp]
