"""Learned action models must not certify observations they cannot express."""
from dataclasses import replace

import pytest

from examples.general_agent.desktop import DesktopPlugin, path_ref
from examples.general_agent import discover
from tensorcode.agent.plugin import Capability, Effect, Param, Precondition
from tensorcode.outcomes import Unknown


def desktop(output="yes"):
    plugin = object.__new__(DesktopPlugin)
    plugin.run = lambda command: (True, [output])
    return plugin


def test_supported_effect_does_not_hide_unsupported_content():
    cap = Capability("write", (Param("path", "file"),), effects=(
        Effect("be", {"undergoer": "path"}),
        Effect("contain", {"undergoer": "path"}),
    ))
    result = desktop().holds(cap, {"path": path_ref("/file")})
    assert isinstance(result, Unknown)
    assert result.reason == "unsupported_effect"


@pytest.mark.parametrize("effect,args", [
    (Effect("be", {"undergoer": "missing"}), {}),
    (Effect("be", {"undergoer": "path", "unhandled": "other"}), {"path": path_ref("/file")}),
    (Effect("has_location", {"undergoer": "path", "goal": "missing"}), {"path": path_ref("/file")}),
])
def test_incomplete_bindings_or_roles_are_unknown(effect, args):
    assert isinstance(desktop().holds(Capability("check", (), (effect,)), args), Unknown)


@pytest.mark.parametrize("pred,negated,output,expected", [
    ("be", False, "yes", True), ("be", True, "no", True),
    ("destroyed", False, "no", True), ("destroyed", True, "yes", True),
    ("be", False, "no", False),
])
def test_existence_polarity(pred, negated, output, expected):
    cap = Capability("check", (), (Effect(pred, {"undergoer": "path"}, negated),))
    assert desktop(output).holds(cap, {"path": path_ref("/file")}) is expected


def test_negated_destination_requires_absence():
    cap = Capability("check", (), (Effect("has_location", {"undergoer": "path", "goal": "dest"}, True),))
    args = {"path": path_ref("/file"), "dest": path_ref("/folder")}
    assert desktop("no").holds(cap, args) is True
    assert desktop("yes").holds(cap, args) is False


def test_unrecognized_check_output_is_not_false_evidence():
    cap = Capability("check", (), (Effect("be", {"undergoer": "path"}),))
    assert isinstance(desktop("permission denied").holds(cap, {"path": path_ref("/file")}), Unknown)


def test_new_and_modified_contents_do_not_induce_unbound_containment():
    before = discover.Scene({"/old": "file"}, {"/old": "before"})
    after = discover.Scene({"/old": "file", "/new": "file"}, {"/old": "after", "/new": "hello"})
    effects, _ = discover.effects_from(before.diff(after), ["/old", "/new"], ["old", "new"])
    assert effects == [Effect("be", {"undergoer": "new"})]


def test_failed_read_is_missing_observation_not_empty_text():
    class Actor:
        def listing(self, directory):
            return [("secret", False)]
        def run(self, command):
            return False, []
    scene = discover.observe(Actor(), root="/home")
    assert scene.entries == {"/home/secret": "file"}
    assert scene.text == {}


def test_kind_widening_preserves_preconditions(monkeypatch):
    class World:
        def snapshot(self):
            return None
        def restore(self, snapshot):
            pass
    class Actor:
        def run(self, command):
            return True, []
    scenes = iter([discover.Scene({}, {}), discover.Scene({"/new": "dir"}, {})])
    monkeypatch.setattr(discover, "setup_scene", lambda *args, **kwargs: None)
    monkeypatch.setattr(discover, "observe", lambda plugin: next(scenes))
    cap = Capability("test", (Param("path", "file"),), preconditions=(Precondition("be", {"undergoer": "path"}),))
    widened = discover.widen_kinds(Actor(), World(), cap, discover.Usage("test", "test {0}", ("file",), "test"))
    assert widened == replace(cap, params=(Param("path", "path"),))


def test_unrepresentable_mutation_with_output_is_not_learned_as_read(monkeypatch):
    class World:
        def snapshot(self):
            return None
        def restore(self, snapshot):
            pass
    class Actor:
        def run(self, command):
            return True, ["changed contents"]
    path = f"{discover.PROBE}/file0.txt"
    before = discover.Scene({path: "file"}, {path: "before"})
    after = discover.Scene({path: "file"}, {path: "after"})
    scenes = iter([before, after, before, after])
    monkeypatch.setattr(discover, "usages", lambda command: [discover.Usage(command, command + " {0}", ("file",), "modify")])
    monkeypatch.setattr(discover, "setup_scene", lambda *args, **kwargs: None)
    monkeypatch.setattr(discover, "observe", lambda plugin: next(scenes))
    assert discover.discover(Actor(), World(), ["modify"]) == []
