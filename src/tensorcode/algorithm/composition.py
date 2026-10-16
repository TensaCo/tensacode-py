"""Compile bounded prerequisite graphs into portable scheduling computations."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
import numpy as np
from .ir import Algorithm, AlgorithmBuilder, _name


def compile_schedule(dependencies: Mapping[str, Sequence[str]]) -> Algorithm:
    """Compile task readiness; indices follow mapping insertion order.

    Inputs ``completed`` and ``enabled`` are vectors in task order. Values >= .5
    mean true. Output ``ready`` is one exactly when a task is enabled, unfinished,
    and all its prerequisites are complete. The artifact does not execute tasks,
    choose among ready tasks, infer prerequisites, or certify physical completion.
    Callers supply observations and dispatch handlers explicitly.
    """
    if not isinstance(dependencies, Mapping) or not 1 <= len(dependencies) <= 256:
        raise ValueError("schedules require 1 to 256 named tasks")
    names = list(dependencies)
    for name in names:
        _name(name)
    indices = {name: i for i, name in enumerate(names)}
    adjacency = np.zeros((len(names), len(names)))
    for name, parents in dependencies.items():
        if isinstance(parents, (str, bytes)) or not isinstance(parents, Sequence):
            raise ValueError("prerequisites must be a sequence of task names")
        if any(not isinstance(p, str) or p not in indices for p in parents):
            raise ValueError("unknown prerequisite")
        if len(set(parents)) != len(parents):
            raise ValueError("duplicate prerequisite")
        for parent in parents:
            adjacency[indices[name], indices[parent]] = 1
    visiting, visited = set(), set()

    def visit(name):
        if name in visiting:
            raise ValueError("cyclic prerequisites")
        if name in visited:
            return
        visiting.add(name)
        for parent in dependencies[name]:
            visit(parent)
        visiting.remove(name)
        visited.add(name)

    for name in names:
        visit(name)
    b = AlgorithmBuilder(name="prerequisite_schedule")
    completed = b.where(b.input("completed", (len(names),)) >= 0.5, 1.0, 0.0)
    enabled = b.where(b.input("enabled", (len(names),)) >= 0.5, 1.0, 0.0)
    missing = (b.constant(adjacency) @ (1 - completed).reshape(len(names), 1)).reshape(
        len(names)
    )
    return b.build({"ready": b.where(missing < 0.5, enabled * (1 - completed), 0.0)})
