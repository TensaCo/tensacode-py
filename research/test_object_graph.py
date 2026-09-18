from dataclasses import dataclass, field

import tensacode as tc
from research.object_graph import from_records, to_records


@dataclass
class Node:
    name: str
    children: list = field(default_factory=list)
    parent: object = None


def test_object_graph_identity_and_cycles_round_trip():
    reg = tc.TypeRegistry()
    reg.register(Node, identity="name")
    root = Node("root")
    child = Node("child", parent=root)
    root.children.append(child)
    conv = to_records([root], reg)
    assert len(conv.records) == 2 and conv.report.lossless
    (rebuilt,), report = from_records(conv, reg)
    assert rebuilt.children[0].parent is rebuilt and report.lossless


def test_identity_collisions_are_reported_not_merged():
    reg = tc.TypeRegistry()
    reg.register(Node, identity="name")
    conv = to_records([Node("same"), Node("same")], reg)
    assert len(conv.records) == 2 and any("identity collision" in loss for loss in conv.report.losses)
