from typing import Protocol
from uuid import UUID4
from abc import ABC

from tensacode.internal import consts


class HasID(Protocol):
    ID: type[UUID4] = UUID4
    id: UUID4


class HasVersion(Protocol):
    version: str = consts.VERSION


class Serializable(HasID, HasVersion, ABC):
    # TODO: implement in the big BaseModels
    pass
