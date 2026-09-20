"""Transport-neutral connection descriptors and explicit per-chat selection.

Kinds are adapter-supplied labels, never a dispatch vocabulary for the chatbot.
A resource connection retains an uploaded artifact; it does not imply a model can
understand it or that an executable plugin exists for it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol
from uuid import NAMESPACE_URL, uuid5


def connection_id(spec: str, occurrence: int = 1) -> str:
    """Stable across process restarts; duplicate specs remain distinct mounts."""
    kind, separator, argument = spec.partition(":")
    normalized = kind.strip() + (separator + argument.strip() if separator else "")
    return "connection-" + uuid5(NAMESPACE_URL, f"tensorcode:{normalized}:{occurrence}").hex


class ConnectionAdapter(Protocol):
    """Adapters own their plugin, availability, capabilities, and optional preview."""
    id: str
    plugin: Any

    def descriptor(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ResourceConnection:
    id: str
    name: str
    media_type: str
    url: str
    size: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def descriptor(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "kind": "attachment", "status": "available",
                "about": "Uploaded source evidence; understanding depends on available interpreters.",
                "capabilities": [], "preview": {"url": self.url, "media_type": self.media_type},
                "resource": {"url": self.url, "media_type": self.media_type, "size": self.size,
                             "metadata": dict(self.metadata)}, "selectable": False}


class ConnectionRegistry:
    """A registry has no active-selection state; each chat supplies its own IDs."""

    def __init__(self, adapters: Iterable[ConnectionAdapter] = ()) -> None:
        self._adapters: dict[str, ConnectionAdapter] = {}
        self._resources: dict[str, ResourceConnection] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: ConnectionAdapter) -> None:
        if not adapter.id or adapter.id in self._adapters or adapter.id in self._resources:
            raise ValueError(f"connection ID must be nonempty and unique: {adapter.id!r}")
        self._adapters[adapter.id] = adapter

    def register_resource(self, id: str, name: str, media_type: str, url: str,
                          size: int, **metadata: Any) -> ResourceConnection:
        if not id or id in self._adapters or id in self._resources:
            raise ValueError(f"connection ID must be nonempty and unique: {id!r}")
        if size < 0:
            raise ValueError("resource size must be nonnegative")
        resource = ResourceConnection(id, name, media_type, url, size, metadata)
        self._resources[id] = resource
        return resource

    def get(self, id: str) -> ConnectionAdapter | ResourceConnection:
        if id in self._adapters:
            return self._adapters[id]
        if id in self._resources:
            return self._resources[id]
        raise ValueError(f"unknown connection: {id}")

    def descriptors(self) -> list[dict[str, Any]]:
        return [adapter.descriptor() for adapter in (*self._adapters.values(), *self._resources.values())]

    def select(self, ids: Iterable[str]) -> list[ConnectionAdapter]:
        selected = list(ids)
        if len(set(selected)) != len(selected):
            raise ValueError("duplicate connection selection")
        result = []
        for id in selected:
            adapter = self.get(id)
            if id not in self._adapters:
                raise ValueError(f"resource connection is evidence, not an executable adapter: {id}")
            if adapter.descriptor()["status"] != "connected":
                raise ValueError(f"connection is unavailable: {id}")
            result.append(adapter)
        return result


def attachment_connection(attachment: dict[str, Any]) -> ResourceConnection:
    """Describe one explicitly scoped attachment using metadata only.

    Scope is selected by the caller's chat transcript, never by listing the global
    upload store. Raw bytes and unrelated attachment metadata cannot enter cards.
    """
    return ResourceConnection(
        "attachment:" + attachment["id"], attachment["name"], attachment["media_type"],
        attachment["content_url"], attachment["size"], {"attachment_id": attachment["id"]})
