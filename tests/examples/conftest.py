"""Skip experiment tests when the repository's ``.development`` tree is absent.

Some tests exercise internal experiment scripts under ``.development/``. That
tree is tracked in git but deliberately not shipped in the sdist, so these
tests skip instead of failing when run from an unpacked source distribution.
"""
from pathlib import Path

import pytest

_DEVELOPMENT = Path(__file__).parents[2] / '.development'
_SKIP = pytest.mark.skip(
    reason='needs the repository .development tree, which the sdist does not ship')


def pytest_collection_modifyitems(config, items):
    if _DEVELOPMENT.is_dir():
        return
    needs = {}
    for item in items:
        path = Path(str(item.fspath))
        if path not in needs:
            needs[path] = path.suffix == '.py' and "'.development/" in path.read_text()
        if needs[path]:
            item.add_marker(_SKIP)
