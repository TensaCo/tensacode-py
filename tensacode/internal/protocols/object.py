from typing import Protocol
from tensacode.internal.types.obj_op_latent_type_triple import OBJ_OP_LATENT_TYPE_TRIPLE


class SupportsTCOpOverride(Protocol):
    __tc_op_overrides__: dict[OBJ_OP_LATENT_TYPE_TRIPLE, Operation]
    __tc_logs__: list[str]
