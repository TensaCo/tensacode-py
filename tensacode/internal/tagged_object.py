from typing import Optional, Annotated
from pydantic import UUID4, BaseModel
from typing import Protocol, Any
from tensacode.core.base.base_engine import BaseEngine
from tensacode.core.base.ops.base_op import BaseOp
from tensacode.internal.protocols.latent_types import LatentType
from tensacode.internal.utils.misc import advanced_equality_check
from tensacode.internal.utils.pydantic import SubclassField


class TaggedObject(Protocol):
    _tc_config_overrides: dict[TCSelector, dict[str, Any]]
    _tc_enc_cache: dict[TCSelector, LatentType]
    _tc_op_overrides: dict[TCSelector, BaseOp]
    # TODO: this is not implemented yet
