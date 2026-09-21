"""Find Python modules potentially affected by a change, without importing code.

    python examples/dependency_impact.py src/tensorcode --changed ops/vec/latent.py

Uses an authored static-import algorithm through graph operations. This is useful
code analysis, not learned reasoning: dynamic imports and runtime conditions are
not resolved, so the report is a partial analysis rather than a build guarantee.
"""
from __future__ import annotations

import argparse
import ast
from importlib.util import resolve_name
import json
from pathlib import Path

from tensorcode import trace
from tensorcode.ops.graph import Graph, JSONDecoder, Score, SourceAnchor, Transform


def module_name(root, path):
    parts = list(path.relative_to(root).with_suffix('').parts)
    if parts[-1] == '__init__':
        parts.pop()
    return '.'.join([root.name, *parts])


def import_graph(root):
    root = Path(root).resolve()
    if not root.is_dir() or not (root / '__init__.py').is_file():
        raise ValueError('Supply a Python package directory containing __init__.py')
    files = {}
    for path in sorted(root.rglob('*.py')):
        relative = path.relative_to(root)
        if any(part.startswith('.') or part == '__pycache__' for part in relative.parts):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            continue
        files[module_name(root, path)] = path
    edges, attributes, anchors = [], [], []
    for name, path in files.items():
        source = path.relative_to(root).as_posix()
        anchors.append(SourceAnchor(source, target=name))
        package = name if path.name == '__init__.py' else name.rpartition('.')[0]
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'), filename=source)):
            targets = []
            if isinstance(node, ast.Import):
                targets = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ''
                if node.level:
                    try:
                        base = resolve_name('.' * node.level + base, package)
                    except ImportError:
                        continue
                for item in node.names:
                    child = base + '.' + item.name
                    targets.append(child if child in files else base)
            for target in sorted(set(targets)):
                if target in files:
                    edges.append((name, 'imports', target))
                    attributes.append({'file': source, 'line': node.lineno})
    return Graph(tuple(files), tuple(edges), identity=root.name,
                 edge_attributes=tuple(attributes), source_anchors=tuple(anchors))


def impact(graph, context):
    changed = context['changed']
    if changed not in graph.nodes:
        raise ValueError(f'Changed module is not in the package: {changed}')
    affected = {changed}
    while True:
        incoming = {source for source, _, target in graph.edges if target in affected}
        if incoming <= affected:
            break
        affected.update(incoming)
    selected = [i for i, (source, _, target) in enumerate(graph.edges)
                if source in affected and target in affected]
    return Graph(tuple(node for node in graph.nodes if node in affected),
                 tuple(graph.edges[i] for i in selected), identity=graph.identity,
                 attributes={'changed': changed, 'analysis': 'static reverse imports'},
                 edge_attributes=tuple(graph.edge_attributes[i] for i in selected),
                 source_anchors=tuple(anchor for anchor in graph.source_anchors
                                      if anchor.target in affected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('--changed', type=Path, required=True, help='Python file relative to package')
    args = parser.parse_args()
    root = args.package.resolve()
    changed_path = (root / args.changed).resolve()
    if not changed_path.is_relative_to(root):
        parser.error('--changed must stay inside the package')
    graph = import_graph(root)
    analyze = Transform(impact, identity='example.static-import-impact.v1', replayable=True)
    count = Score(lambda value, context: len(value.nodes), semantics='affected-module-count')
    with trace() as session:
        affected = analyze(graph, context={'changed': module_name(root, changed_path)})
        total = count(affected)
    print(json.dumps({'affected_count': int(total), 'trace_calls': len(session.calls),
                      'graph': JSONDecoder()(affected)}, indent=2))


if __name__ == '__main__':
    main()
