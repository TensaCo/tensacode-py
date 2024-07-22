from typing import Literal, TypedDict, Generic, TypeVar
import networkx as nx


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

# [(parent, child)]
LATENT_TYPE_DISTS_ADJ_LIST: list[tuple[LatentType, LatentType]] = [
    (LatentType, LatentTypeObject),
    (LatentType, LatentTypeBytes),
    (LatentType, LatentTypeText),
    (LatentType, LatentTypeImage),
    (LatentType, LatentTypeAudio),
    (LatentType, LatentTypeVideo),
    (LatentType, LatentTypeVector),
    (LatentType, LatentTypeGrid),
    (LatentType, LatentTypeGraphAdjacencyList),
    (LatentType, LatentTypeGraphAdjacencyMatrix),
    (LatentType, Anthropomorphic),
    (LatentTypeGrid, LatentTypeImage),
    (LatentTypeGrid, LatentTypeAudio),
    (LatentTypeGrid, LatentTypeVideo),
]

# Create a directed graph
LATENT_TYPE_GRAPH = nx.DiGraph(LATENT_TYPE_DISTS_ADJ_LIST)


def latent_type_subtype_distance(sub: LatentType, parent: LatentType) -> int | None:
    """
    Compute the minimum distance from parent to sub in the latent type hierarchy.

    Args:
        sub (LatentType): The subtype to measure distance from.
        parent (LatentType): The parent type to measure distance to.

    Returns:
        int: The minimum distance from sub to parent. Returns -1 if there's no path.
    """
    if sub == parent:
        return 0
    try:
        return nx.shortest_path_length(LATENT_TYPE_GRAPH, sub, parent)
    except nx.NetworkXNoPath:
        return None


def are_latent_subtypes(sub: LatentType, parent: LatentType) -> bool:
    """
    Check if sub is a subtype of parent in the latent type hierarchy.
    """
    return latent_type_subtype_distance(sub, parent) is not None


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
