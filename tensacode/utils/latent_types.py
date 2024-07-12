from pydantic import BaseModel
from typing import Literal, TypeVar


LATENT_TYPES = Literal[
    "object",
    "text",
    "image",
    "audio",
    "video",
    "vector",
    "grid",
    "graph-linked-list",
    "graph-adjacency-list",
]
