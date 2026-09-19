"""Connection routing is keyed by identity and actual adapters, never desktop type."""
import json

import pytest

from examples.general_agent.connections import ConnectionRegistry, connection_id
from examples.general_agent import plugins
from tensorcode.agent.plugin import Capability, Plugin


def test_ids_and_descriptors_are_stable_without_engines(tmp_path):
    specs = [f"filesystem:{tmp_path}", f"filesystem:{tmp_path}", "self"]
    configured = plugins.specs_descriptors(specs)
    mounted = plugins.mount_all(specs)
    assert [d["id"] for d in configured] == [m.id for m in mounted]
    assert len({m.id for m in mounted}) == 3
    assert len({m.plugin.name for m in mounted}) == 3
    assert all(d["status"] == "configured" for d in configured)
    assert json.loads(json.dumps(mounted[0].descriptor()))["kind"] == "filesystem"
    assert connection_id(" filesystem: /x ") == connection_id("filesystem:/x")


def test_novel_kind_registered_without_chatbot_dispatch(monkeypatch):
    monkeypatch.setattr(plugins, "FACTORIES", dict(plugins.FACTORIES))
    monkeypatch.setattr(plugins, "FACTORY_METADATA", dict(plugins.FACTORY_METADATA))
    class Measurement(Plugin):
        def capabilities(self):
            return [Capability("sample", (), effect_kind="read")]
    plugins.register_factory("novel", lambda arg, callback: plugins.Mounted(arg, Measurement(arg)),
                             connection_kind="spectrometer", about="Supplied instrument")
    mounted = plugins.mount_all(["novel:alpha", "novel:beta"])
    registry = ConnectionRegistry(mounted)
    assert [d["kind"] for d in registry.descriptors()] == ["spectrometer"] * 2
    assert registry.select([mounted[1].id]) == [mounted[1]]
    assert registry.select([mounted[0].id]) == [mounted[0]]
    assert registry.select([]) == []
    assert registry.descriptors()[0]["capabilities"][0]["name"] == "sample"
    assert plugins.specs_descriptors(["novel:alpha"])[0]["kind"] == "spectrometer"


def test_unknown_kinds_do_not_claim_working_adapters():
    for descriptor in plugins.specs_descriptors(["not-registered:local", "unregistered:example"]):
        assert descriptor["status"] == "unavailable"
        assert descriptor["capabilities"] == []
        assert not descriptor["selectable"]
    with pytest.raises(ValueError, match="no plugin"):
        plugins.mount("not-registered:local")


def test_selection_validation_and_resource_evidence():
    mounted = plugins.Mounted("renamed", Plugin("provider"), id="a", kind="arbitrary")
    unavailable = plugins.Mounted("offline", Plugin("offline"), id="b", status="unavailable")
    registry = ConnectionRegistry([mounted, unavailable])
    resource = registry.register_resource("upload-1", "scene.mp4", "video/mp4", "/files/upload-1", 10)
    assert resource.descriptor()["kind"] == "attachment"
    assert resource.descriptor()["capabilities"] == []
    assert json.loads(json.dumps(registry.descriptors()))[-1]["resource"]["size"] == 10
    for ids, message in [(["a", "a"], "duplicate"), (["missing"], "unknown"),
                         (["b"], "unavailable"), (["upload-1"], "evidence")]:
        with pytest.raises(ValueError, match=message):
            registry.select(ids)
    with pytest.raises(ValueError, match="unique"):
        registry.register(mounted)


def test_descriptor_does_not_invoke_preview():
    def view():
        raise AssertionError("inventory must not trigger screenshots")
    adapter = plugins.Mounted("device", Plugin("device"), view=view,
                              id="device", preview={"media_type": "image/png"})
    assert adapter.descriptor()["preview"]["media_type"] == "image/png"
