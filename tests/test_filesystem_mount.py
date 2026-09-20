"""Explicit app mounting of the real filesystem adapter."""
import subprocess
import sys

import pytest

from examples.general_agent.plugins import mount, mount_all
from agent_test_support import selected_agent as Agent
from tensorcode.agent.filesystem import FileSystemPlugin
from tensorcode.language import verbnet, wordnet


@pytest.mark.parametrize("spec", ["filesystem", "filesystem:", "filesystem:   "])
def test_root_must_be_explicit(spec):
    with pytest.raises(ValueError, match="explicit existing root"):
        mount(spec)


def test_root_must_already_exist_and_be_directory(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="choose an existing directory"):
        mount(f"filesystem:{missing}")
    assert not missing.exists()
    file = tmp_path / "file"
    file.write_text("unchanged")
    with pytest.raises(ValueError, match="choose an existing directory"):
        mount(f"filesystem:{file}")


def test_mount_has_no_view_and_uses_explicit_root(tmp_path):
    mounted = mount(f"filesystem:{tmp_path}")
    assert isinstance(mounted.plugin, FileSystemPlugin)
    assert mounted.plugin.root == tmp_path
    assert mounted.name == mounted.plugin.name
    assert not mounted.has_view
    assert mounted.view() is None


def test_different_filesystem_mounts_have_distinct_provider_names(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    mounted = mount_all([f"filesystem:{left}", f"filesystem:{right}"])
    assert mounted[0].plugin.name != mounted[1].plugin.name


def test_filesystem_mount_does_not_import_computerworld(tmp_path):
    script = """
import sys
from examples.general_agent.plugins import mount
mount('filesystem:' + sys.argv[1])
assert not any('computerworld' in name or name == 'examples.general_agent.desktop' for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True)


def test_mounted_agent_does_not_supply_a_project_recipe(tmp_path):
    from agent_test_support import fixture_goal_selector
    if wordnet.find_wordnet() is None or verbnet.find_verbnet() is None:
        pytest.skip("requires WordNet and VerbNet data")
    mounted = mount(f"filesystem:{tmp_path}")
    turn = Agent([mounted.plugin], goal_selector=fixture_goal_selector('build-26.1-1', frame_index=0)).turn("make a python project")
    assert turn.outcomes and turn.outcomes[0].status == "declined", turn
    assert list(tmp_path.iterdir()) == []
