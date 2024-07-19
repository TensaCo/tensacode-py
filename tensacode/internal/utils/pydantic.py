import uuid
from typing_extensions import Annotated
from typing import List, Union
from numbers import Number
from pydantic import PlainValidator, PlainSerializer, UUID4
from typing_extensions import Protocol
from tensacode.internal.utils import consts


class HasID(Protocol):
    ID: type[UUID4] = UUID4
    id: UUID4


class HasVersion(Protocol):
    version: str = consts.VERSION


from typing import Type, Any, Optional
from pydantic import Field, field_validator, field_serializer


def find_subclass_by_name(base_class: Type, name: str) -> Optional[Type]:
    def recursive_search(cls: Type) -> Optional[Type]:
        if cls.__name__ == name:
            return cls
        for subclass in cls.__subclasses__():
            result = recursive_search(subclass)
            if result:
                return result
        return None

    return recursive_search(base_class)


def SubclassField(
    base_class: Type,
    field_extras: dict[str, Any] = {},
    extra_annotations: dict[str, Any] = {},
):
    def wrapper(cls: Type) -> Any:
        return Field(
            ...,
            serialization_alias=f"{cls.__name__}_name",
            validation_alias=f"{cls.__name__}_name",
            **field_extras,
        )

    @field_validator(cls.__name__, mode="before")
    @classmethod
    def validate_subclass(cls, value):
        if isinstance(value, str):
            subclass = find_subclass_by_name(base_class, value)
            if subclass is None:
                raise ValueError(
                    f"No subclass of {base_class.__name__} found with name: {value}"
                )
            return subclass
        return value

    @field_serializer(cls.__name__)
    def serialize_subclass(self, value: Type) -> str:
        return value.__name__

    annotations = [wrapper, validate_subclass, serialize_subclass]
    annotations.extend(extra_annotations.values())

    return Annotated[Type[base_class], *annotations]


Complex = Annotated[
    complex,
    PlainValidator(
        lambda x: x if isinstance(x, complex) else complex(x.get("real", x.get("imag")))
    ),
    PlainSerializer(lambda x: {"real": x.real, "imag": x.imag}, return_type=dict),
]


Tensor = Union[Number, list["Tensor"]]


try:
    import numpy as np

    Numpy = Annotated[
        np.ndarray,
        PlainValidator(lambda x: x if isinstance(x, np.ndarray) else np.array(x)),
        PlainSerializer(lambda x: x.tolist(), return_type=list),
    ]
    Tensor = Tensor | Numpy
except ImportError:
    pass


try:
    import torch

    PTTensor = Annotated[
        torch.Tensor,
        PlainValidator(lambda x: x if isinstance(x, torch.Tensor) else torch.tensor(x)),
        PlainSerializer(lambda x: x.tolist(), return_type=list),
    ]
except ImportError:
    pass

try:
    import tensorflow as tf

    TFTensor = Annotated[
        tf.Tensor,
        PlainValidator(lambda x: x if isinstance(x, tf.Tensor) else tf.tensor(x)),
        PlainSerializer(lambda x: x.numpy().tolist(), return_type=list),
    ]
except ImportError:
    pass

try:
    import jax

    JAXArray = Annotated[
        jax.Array,
        PlainValidator(lambda x: x if isinstance(x, jax.Array) else jax.array(x)),
        PlainSerializer(lambda x: x.tolist(), return_type=list),
    ]
except ImportError:
    pass
