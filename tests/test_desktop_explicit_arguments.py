"""Real computerworld execution receives explicit arguments, never descriptions."""
import pytest

pytest.importorskip("computerworld")

from examples.browser_agents.worlds import desktop_world
from examples.browser_agents.worlds.runtime import CwWorld
from examples.general_agent.desktop import DesktopPlugin, path_ref
from tensorcode.agent.plugin import Call, Capability, Effect, Param
from tensorcode.language import Entity
from tensorcode.records import Ref


@pytest.fixture
def desktop():
    world = CwWorld(desktop_world(), 0)
    # Authored command models isolate transport. This is not a learned capability test.
    learned = [
        (Capability("make", (Param("path", "directory"),)), "mkdir {0}"),
        (Capability("write", (Param("path", "file"), Param("text", "text"))), "echo {1} > {0}"),
        (Capability("move", (Param("source", "path"), Param("destination", "path")),
                    effects=(Effect("has_location", {"undergoer": "source", "goal": "destination"}),)),
         "mv {0} {1}"),
    ]
    return world, DesktopPlugin(world, learn=False, learned=learned)


def call(plugin, name, **args):
    return plugin.execute(Call(plugin.name, name, tuple(args.items())), key=None)


def test_valid_explicit_paths_and_exact_rename_destination(desktop):
    world, plugin = desktop
    assert call(plugin, "write", path=path_ref("/home/agent/source.txt"), text="evidence").status == "applied"
    destination = "/home/agent/completely-different.txt"
    result = call(plugin, "move", source=path_ref("/home/agent/source.txt"), destination=path_ref(destination))
    assert result.status == "applied"
    assert world.read(destination).strip() == "evidence"
    assert not world.exists("/home/agent/source.txt")
    cap, _ = plugin.command_for("move")
    assert plugin.holds(cap, {"source": path_ref("/home/agent/source.txt"),
                              "destination": path_ref(destination)}) is True
    assert plugin.log[-1][0] == f"test -e {destination} && echo yes || echo no"


def test_invalid_arguments_reject_before_terminal_or_world_mutation(desktop):
    world, plugin = desktop
    baseline = world.state_hash()
    log_count = len(plugin.log)
    invalid = [
        Call("another-provider", "make", (("path", path_ref("/home/agent/no")),)),
        Call(plugin.name, "make", ()),
        Call(plugin.name, "make", (("path", path_ref("/home/agent/no")), ("extra", "x"))),
        Call(plugin.name, "make", (("path", Ref("entity:/home/agent/no")),)),
        Call(plugin.name, "make", (("path", Ref("path:relative")),)),
        Call(plugin.name, "make", (("path", Entity("path", "/home/agent/no")),)),
        Call(plugin.name, "make", (("path", "/home/agent/no"),)),
        Call(plugin.name, "make", (("path", path_ref("/home/agent/no")), ("path", path_ref("/home/agent/other")))),
        Call(plugin.name, "list_directory", (("directory", Ref("app:terminal")),)),
        Call(plugin.name, "open_application", (("app", Ref("path:terminal")),)),
    ]
    for act in invalid:
        assert plugin.execute(act, key=None).status == "rejected"
    assert len(plugin.log) == log_count
    assert world.state_hash() == baseline


def test_listing_missing_directory_is_not_home_or_empty_success(desktop):
    _, plugin = desktop
    assert call(plugin, "list_directory", directory=path_ref("/home/agent/no-such-directory")).status == "failed"
    assert call(plugin, "list_directory", directory=path_ref("/home/agent")).status == "applied"
    assert "no-such-directory" in plugin.log[-2][0]


def test_no_description_resolution_override_or_hidden_helpers():
    for name in ("refer", "denote", "kind_fits", "_resolve", "_find", "app_for", "_is_directory"):
        assert name not in DesktopPlugin.__dict__
