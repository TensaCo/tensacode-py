import uuid
from typing_extensions import Annotated
from typing import List, Union
from numbers import Number
from pydantic import PlainValidator, PlainSerializer, UUID4
from typing_extensions import Protocol


class HasID(Protocol):
    ID: type[UUID4] = UUID4
    id: UUID4


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
