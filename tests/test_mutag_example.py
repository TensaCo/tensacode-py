"""Check dataset isolation; fixtures here only test archive parsing."""
import importlib.util
from pathlib import Path
import zipfile

import pytest


def example():
    spec = importlib.util.spec_from_file_location('mutag_example', Path(__file__).parents[1] / 'examples/mutag.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def archive(tmp_path, edges='1, 2\n2, 1\n3, 4\n4, 3\n'):
    path = tmp_path / 'graphs.zip'
    files = {'graph_labels': '-1\n1\n', 'graph_indicator': '1\n1\n2\n2\n',
             'node_labels': '0\n1\n2\n0\n', 'A': edges, 'edge_labels': '1\n1\n2\n2\n'}
    with zipfile.ZipFile(path, 'w') as z:
        for name, content in files.items():
            z.writestr(f'MUTAG/MUTAG_{name}.txt', content)
    return path


def test_archive_preserves_graph_boundaries_and_local_edges(tmp_path):
    data = example().read(archive(tmp_path))
    assert data == [
        {'id': 1, 'label': -1, 'node_labels': [0, 1], 'edges': [(0, 1, 1), (1, 0, 1)]},
        {'id': 2, 'label': 1, 'node_labels': [2, 0], 'edges': [(0, 1, 2), (1, 0, 2)]},
    ]


def test_cross_graph_edge_rejected(tmp_path):
    with pytest.raises(ValueError, match='crosses graph'):
        example().read(archive(tmp_path, edges='1, 3\n2, 1\n3, 4\n4, 3\n'))


def test_split_is_disjoint_reproducible_and_complete():
    mod = example()
    train, test = mod.split_ids(188, seed=7)
    assert len(train) == 150 and len(test) == 38
    assert not set(train) & set(test)
    assert sorted(train + test) == list(range(188))
    assert (train, test) == mod.split_ids(188, seed=7)
