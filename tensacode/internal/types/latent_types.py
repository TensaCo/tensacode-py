from typing import Literal, TypedDict


LATENT_TYPES = Literal[
    "object",
    "bytes",
    "text",
    "image",
    "audio",
    "video",
    "vector",
    "grid",
    "graph-adjacency-list",
    "graph-adjacency-matrix",
]

LATENT_TYPE_OBJECT = object
LATENT_TYPE_BYTES = bytes
LATENT_TYPE_TEXT = str
LATENT_TYPE_IMAGE = Tensor["h", "w", "c"]
LATENT_TYPE_AUDIO = Tensor["s", "c"]
LATENT_TYPE_VIDEO = Tensor["t", "h", "w", "c"]
LATENT_TYPE_VECTOR = Tensor["d"]
LATENT_TYPE_GRID = Tensor[..., "d"]


class LATENT_TYPE_GRAPH_ADJACENCY_LIST(TypedDict):
    nodes: Tensor["n", "d"]
    adj_list: list[tuple[int, int]]


class LATENT_TYPE_GRAPH_ADJACENCY_MATRIX(TypedDict):
    nodes: Tensor["n", "d"]
    adj_mat: Tensor["n", "n", "d"]


class AnthropomorphicKeys(TypedDict):
    vision: LATENT_TYPE_IMAGE
    audio: LATENT_TYPE_AUDIO
    text: LATENT_TYPE_TEXT
    graph: LATENT_TYPE_GRAPH_ADJACENCY_LIST
