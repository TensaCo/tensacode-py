import importlib.util
from pathlib import Path


def example():
    spec = importlib.util.spec_from_file_location('dependency_example', Path(__file__).parents[2] / 'examples/dependency_impact.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_relative_imports_and_transitive_impact_retain_source_files(tmp_path):
    root = tmp_path / 'app'
    root.mkdir()
    (root / '__init__.py').write_text('')
    (root / 'base.py').write_text('VALUE = 1\n')
    (root / 'service.py').write_text('from .base import VALUE\n')
    (root / 'api.py').write_text('from . import service\n')
    (root / 'unrelated.py').write_text('import json\n')
    mod = example()
    graph = mod.import_graph(root)
    affected = mod.impact(graph, {'changed': 'app.base'})
    assert set(affected.nodes) == {'app.base', 'app.service', 'app.api'}
    assert {anchor.source for anchor in affected.source_anchors} == {'base.py', 'service.py', 'api.py'}
    assert ('app.service', 'imports', 'app.base') in graph.edges
    assert ('app.api', 'imports', 'app.service') in graph.edges


def test_symlinked_python_file_outside_package_is_not_read(tmp_path):
    root = tmp_path / 'app'
    root.mkdir()
    (root / '__init__.py').write_text('')
    outside = tmp_path / 'secret.py'
    outside.write_text('this is deliberately not Python')
    (root / 'leak.py').symlink_to(outside)
    assert example().import_graph(root).nodes == ('app',)
