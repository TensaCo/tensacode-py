from tensacode.internal.tcir.nodes import Node


def merge_identical(node: Node):
    """Merge identical nodes to convert a node tree to a (potentially cyclic) graph"""
    node.merge_with_identical(node)
