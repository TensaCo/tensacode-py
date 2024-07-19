from typing import Literal, TypedDict, Generic, TypeVar


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


class ANTHROPOMORPHIC(TypedDict):
    vision: LATENT_TYPE_IMAGE
    audio: LATENT_TYPE_AUDIO
    text: LATENT_TYPE_TEXT
    graph: LATENT_TYPE_GRAPH_ADJACENCY_LIST


LATENT_TYPE = Literal[
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
    "anthropomorphic",
]

latent_types: dict[LATENT_TYPE, type] = {
    "object": LATENT_TYPE_OBJECT,
    "bytes": LATENT_TYPE_BYTES,
    "text": LATENT_TYPE_TEXT,
    "image": LATENT_TYPE_IMAGE,
    "audio": LATENT_TYPE_AUDIO,
    "video": LATENT_TYPE_VIDEO,
    "vector": LATENT_TYPE_VECTOR,
    "grid": LATENT_TYPE_GRID,
    "graph-adjacency-list": LATENT_TYPE_GRAPH_ADJACENCY_LIST,
    "graph-adjacency-matrix": LATENT_TYPE_GRAPH_ADJACENCY_MATRIX,
    "anthropomorphic": ANTHROPOMORPHIC,
}
