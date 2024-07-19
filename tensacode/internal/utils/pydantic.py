import uuid
from typing_extensions import Annotated
from typing import List, Union
from numbers import Number
from pydantic import PlainValidator, PlainSerializer, UUID4
from typing_extensions import Protocol
from tensacode.internal.utils.string import to_capital_camel_case

from typing import Generic, TypeVar

T = TypeVar("T")


def get_all_subclasses_recursive(cls: type[T]) -> set[type[T]]:
    def get_subclasses(cls: type[T]) -> set[type[T]]:
        return set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in get_subclasses(c)]
        )

    return get_subclasses(cls)


def _assert_valid_guid(value: str) -> str:
    try:
        classname, uuid_str = value.split(":", 1)
        if not classname:
            raise ValueError("Classname must be non-empty")
        uuid_obj = uuid.UUID(uuid_str)
        if uuid_obj.version != 4:
            raise ValueError("Only UUID version 4 is supported")

        # Check if the classname matches any HasID subclass
        valid_classnames = {
            cls.classname() for cls in get_all_subclasses_recursive(HasGUID)
        }
        if classname not in valid_classnames:
            raise ValueError(
                f"Classname '{classname}' does not match any HasGUID subclass"
            )

        return value
    except ValueError as e:
        raise ValueError(f"Invalid GUID format: {e}")


class HasID(Protocol):
    ID: type[UUID4] = UUID4
    id: UUID4


class HasGUID(HasID, Protocol):
    @classmethod
    def classname(cls) -> str:
        return to_capital_camel_case(cls.__name__)

    T_GUID: type[str] = Annotated[str, PlainValidator(_assert_valid_guid)]

    @property
    def guid(self) -> T_GUID:
        return f"{self.classname}:{self.id}"


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
