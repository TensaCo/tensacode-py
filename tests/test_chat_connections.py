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


def test_resource_inventory_is_chat_scoped_metadata_only_and_survives_reload(tmp_path):
    import base64
    import queue
    import sqlite3
    from examples.general_agent.chat_api import ChatApplication
    from examples.general_agent.chat_store import ChatStore
    from examples.general_agent.server import Hub
    from test_chat_application import Handler

    store = ChatStore(tmp_path)
    left, right = store.create_chat(), store.create_chat()
    own = store.upload("clip.mp4", "video/mp4", base64.b64encode(b"video evidence").decode())
    other = store.upload("private.txt", "text/plain", base64.b64encode(b"other chat").decode())
    store.upload("unsent.png", "image/png", base64.b64encode(b"not in any chat").decode())
    store.add_message(left["id"], "user", "left", attachment_ids=[own["id"], own["id"]])
    store.add_message(right["id"], "user", "right", attachment_ids=[other["id"]])
    configured = [{"id": "adapter", "name": "Tool", "kind": "novel", "status": "configured"}]
    app = ChatApplication(store, queue.Queue(), Hub(), configured)
    app.receive({"type": "connections", "chat_id": left["id"], "connections": [
        {**configured[0], "status": "connected"},
        {"id": "removed-adapter", "name": "Old", "status": "connected"},
        {"id": "injected-resource", "resource": {"url": "/wrong"}, "selectable": False}]})
    assert app.connections_for_chat() == configured
    # Reading connection cards must never fetch attachment blobs.
    def deny_blobs(operation, table, column, *unused):
        if operation == sqlite3.SQLITE_READ and table == "attachments" and column == "data":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    store.db.set_authorizer(deny_blobs)
    handler = Handler("/api/connections?chat_id=" + left["id"])
    app.routes(handler)
    assert handler.status == 200
    descriptors = handler.json()["connections"]
    assert [d["id"] for d in descriptors] == ["adapter", "attachment:" + own["id"]]
    resource = descriptors[-1]
    assert resource["resource"]["url"] == own["content_url"]
    assert resource["preview"]["media_type"] == "video/mp4"
    assert not resource["selectable"] and resource["capabilities"] == []
    assert '"data":' not in json.dumps(resource)
    chat = Handler("/api/chats/" + left["id"])
    app.routes(chat)
    assert chat.json()["connections"] == descriptors
    store.close()
    reopened = ChatStore(tmp_path)
    try:
        restored = ChatApplication(reopened, queue.Queue(), Hub(), configured)
        assert restored.connections_for_chat(left["id"])[-1] == resource
        assert restored.connections_for_chat(right["id"])[-1]["id"] == "attachment:" + other["id"]
    finally:
        reopened.close()


def test_submit_publishes_scoped_resource_connection_and_forbids_selecting_it(tmp_path):
    import base64
    import queue
    from examples.general_agent.chat_api import ChatApplication
    from examples.general_agent.chat_store import ChatStore
    from examples.general_agent.server import Hub

    store = ChatStore(tmp_path)
    try:
        chat = store.create_chat()
        upload = store.upload("notes.txt", "text/plain", base64.b64encode(b"source").decode())
        hub = Hub()
        app = ChatApplication(store, queue.Queue(), hub)
        app.submit(chat["id"], {"text": "read", "attachment_ids": [upload["id"]]})
        event = next(event for event in hub.history if event["type"] == "connections")
        assert event["chat_id"] == chat["id"]
        assert event["connections"][0]["id"] == "attachment:" + upload["id"]
        with pytest.raises(ValueError, match="connection id"):
            app.submit(chat["id"], {"text": "execute", "connection_ids": ["attachment:" + upload["id"]]})
    finally:
        store.close()


def test_worker_resource_shutdown_and_failed_mount_inventory(monkeypatch):
    import queue
    from types import SimpleNamespace
    import tensorcode.agent
    from tensorcode.agent.interpretation import InterpretationWorkspace
    from examples.general_agent.server import worker

    class AgentFixture:
        def __init__(self, *args, **kwargs):
            self.grammar = object()
            self.interpretations = InterpretationWorkspace()
        def turn(self, text, images=()):
            return SimpleNamespace(events=[], reply="fixture reply", seconds=0)

    monkeypatch.setattr(tensorcode.agent, "Agent", AgentFixture)
    def failed_mount(argument, callback):
        raise RuntimeError("connection unavailable")
    monkeypatch.setitem(plugins.FACTORIES, "broken-fixture", failed_mount)
    descriptor = plugins.specs_descriptors(["broken-fixture"])[0]
    inbox, events = queue.Queue(), queue.Queue()
    attachment = {"id": "file1", "name": "evidence.txt", "media_type": "text/plain",
                  "size": 3, "data": b"raw", "content_url": "/files/file1"}
    inbox.put({"chat_id": "chat1", "message_id": "message1", "text": "one", "connection_ids": [],
               "attachments": [attachment]})
    inbox.put({"chat_id": "chat1", "message_id": "message2", "text": "two",
               "connection_ids": [descriptor["id"]], "attachments": []})
    inbox.put(None)
    worker(inbox, events, 10, specs=("broken-fixture",))
    emitted = []
    while not events.empty():
        emitted.append(events.get_nowait())
    inventories = [event["connections"] for event in emitted if event["type"] == "connections"]
    assert len(inventories) == 2
    assert all(inventory[-1]["id"] == "attachment:file1" for inventory in inventories)
    assert inventories[-1][0]["status"] == "unavailable"
    assert inventories[-1][0]["selectable"] is False
    assert "RuntimeError" in inventories[-1][0]["error"]
    assert all("data" not in connection for inventory in inventories for connection in inventory)
