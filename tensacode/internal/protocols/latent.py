from typing import Literal, TypedDict, Generic, TypeVar


LatentTypeObject = object
LatentTypeBytes = bytes
LatentTypeText = str
LatentTypeImage = Tensor["h", "w", "c"]
LatentTypeAudio = Tensor["s", "c"]
LatentTypeVideo = Tensor["t", "h", "w", "c"]
LatentTypeVector = Tensor["d"]
LatentTypeGrid = Tensor[..., "d"]


class LatentTypeGraphAdjacencyList(TypedDict):
    nodes: Tensor["n", "d"]
    adj_list: list[tuple[int, int]]


class LatentTypeGraphAdjacencyMatrix(TypedDict):
    nodes: Tensor["n", "d"]
    adj_mat: Tensor["n", "n", "d"]


class Anthropomorphic(TypedDict):
    vision: LatentTypeImage
    audio: LatentTypeAudio
    text: LatentTypeText
    graph: LatentTypeGraphAdjacencyList


LatentType = Literal[
    LatentTypeObject,
    LatentTypeBytes,
    LatentTypeText,
    LatentTypeImage,
    LatentTypeAudio,
    LatentTypeVideo,
    LatentTypeVector,
    LatentTypeGrid,
    LatentTypeGraphAdjacencyList,
    LatentTypeGraphAdjacencyMatrix,
    Anthropomorphic,
]

LATENT_TYPE_STR = Literal[
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

latent_types_mapping: dict[LATENT_TYPE_STR, LatentType] = {
    "object": LatentTypeObject,
    "bytes": LatentTypeBytes,
    "text": LatentTypeText,
    "image": LatentTypeImage,
    "audio": LatentTypeAudio,
    "video": LatentTypeVideo,
    "vector": LatentTypeVector,
    "grid": LatentTypeGrid,
    "graph-adjacency-list": LatentTypeGraphAdjacencyList,
    "graph-adjacency-matrix": LatentTypeGraphAdjacencyMatrix,
    "anthropomorphic": Anthropomorphic,
}
