from dataclasses import dataclass


@dataclass(frozen=True)
class Graph:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str, str], ...] = ()
    sources: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'nodes', tuple(self.nodes))
        object.__setattr__(self, 'edges', tuple(tuple(e) for e in self.edges))
        object.__setattr__(self, 'sources', tuple(self.sources))
        if not all(isinstance(v, str) for v in (*self.nodes, *self.sources)):
            raise TypeError('Node identities and source references must be strings')
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError('Duplicate node identity')
        for edge in self.edges:
            if len(edge) != 3 or not all(isinstance(v, str) for v in edge) or edge[0] not in self.nodes or edge[2] not in self.nodes:
                raise ValueError('Edges must connect existing nodes')

    def neighbors(self, node, *, relation=None):
        return tuple(b for a, r, b in self.edges if a == node and (relation is None or relation == r))
