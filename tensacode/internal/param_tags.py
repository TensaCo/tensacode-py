from dataclasses import dataclass, field
from typing import Any, Annotated, TypeVar, Union

from tensacode.internal.latent import LatentType


@dataclass
class EncodeTag:
    encode_args: tuple = field(default_factory=tuple)
    encode_kwargs: dict = field(default_factory=dict)


T = TypeVar("T")

Encoded = Annotated[Union[T, Union[LatentType, Any]], EncodeTag()]


@dataclass
class AutofillTag:
    autofill_args: tuple = field(default_factory=tuple)
    autofill_kwargs: dict = field(default_factory=dict)


T = TypeVar("T")

Autofilled = Annotated[Union[T, Union[LatentType, Any]], AutofillTag()]
