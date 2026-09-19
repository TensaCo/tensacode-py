"""What the agent is plugged into, chosen at run time.

The chat app used to build one desktop and one pair of eyes, in code. The agent itself never
required that — it takes a list of plugins and knows nothing about what is in it — so the app
was the part that hardcoded the world. Here a plugin is *mounted by name*, any number of
times, and the page adapts to whatever is mounted: no computer, one, or several.

A mount says three things: what it is called, the plugin the agent gets, and how to take its
picture if it has one. Everything else — capabilities, vocabulary, what it can refer to — the
plugin already declares to the agent through the normal protocol.

    mount("desktop")            a computerworld machine with the standard desktop
    mount("desktop:note")       the same engine with a note-taking task on the desktop
    mount("vision")             optional trained category-proposal adapter
    mount("filesystem:/path")   explicit existing local root

Register additional constructors with ``register_factory``. Descriptor kinds are open
labels and require no server or UI type dispatch. Browser endpoints and additional adapters must be supplied
explicitly; configuration alone never means a connection is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .connections import connection_id


@dataclass
class Mounted:
    """One plugin the agent is using, with the means to show it if it can be shown."""

    name: str                                   # human display name; routing uses id
    plugin: Any
    view: Callable[[], bytes | None] = lambda: None
    about: str = ""

    id: str = ""
    kind: str = "plugin"
    status: str = "connected"
    preview: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    close: Callable[[], None] = lambda: None

    def descriptor(self) -> dict[str, Any]:
        """Describe declared affordances without taking a screenshot."""
        capabilities = [{"name": c.name, "description": c.description,
                         "effect_kind": c.effect_kind,
                         "params": [{"name": p.name, "kind": p.kind, "role": p.role} for p in c.params]}
                        for c in self.plugin.capabilities()]
        return {"id": self.id, "name": self.name, "kind": self.kind,
                "status": self.status, "about": self.about, "capabilities": capabilities,
                "preview": self.preview, "metadata": dict(self.metadata),
                "selectable": self.status == "connected"}

    @property
    def has_view(self) -> bool:
        """Whether this plugin can be looked at, which is what puts a monitor on the page."""
        try:
            return self.view() is not None
        except Exception:  # noqa: BLE001
            return False


def _desktop(argument: str, on_step) -> Mounted:
    from examples.browser_agents.worlds import desktop_world, note_world
    from examples.browser_agents.worlds.runtime import CwWorld
    from examples.general_agent.desktop import DesktopPlugin

    if argument == "note":
        definition, about = note_world("Set up a project called demo under ~/Projects.", 1), \
            "a machine with a task written on its desktop"
    else:
        definition, about = desktop_world(), "a desktop machine"
    world = CwWorld(definition, 0)
    plugin = DesktopPlugin(world, on_step=on_step)
    name = f"desktop:{argument}" if argument else "desktop"
    return Mounted(name=name, plugin=plugin, view=lambda: plugin.surface.png(), about=about,
                   kind="virtual_os", preview={"media_type": "image/png", "transport": "frames"})


def _vision(argument: str, on_step) -> Mounted:
    from tensorcode.agent.vision_plugin import VisionPlugin

    plugin = VisionPlugin()
    return Mounted(name="vision", plugin=plugin, kind="vision",
                   status="connected" if plugin.model is not None else "unavailable",
                   about="Whole-image category proposals; no holistic scene understanding.",
                   metadata={"model_loaded": plugin.model is not None})


def _self(argument: str, on_step) -> Mounted:
    from tensorcode.agent.discourse import DiscoursePlugin

    # it reports on the agent that owns it, which does not exist until every plugin is
    # mounted — hence `attach`, called once the agent is built
    return Mounted(name="self", plugin=DiscoursePlugin(), kind="introspection",
                   about="explains what it did and what it can do")


def _filesystem(argument: str, on_step) -> Mounted:
    from pathlib import Path

    from tensorcode.agent.filesystem import FileSystemPlugin

    if not argument:
        raise ValueError("filesystem requires an explicit existing root: --plugin filesystem:/path/to/directory")
    root = Path(argument).expanduser().absolute()
    name = f"filesystem:{root}"
    try:
        plugin = FileSystemPlugin(root, name=name)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot mount filesystem root {root}: {exc}; choose an existing directory") from exc
    return Mounted(name=name, plugin=plugin, kind="filesystem",
                   about=f"creates files under {root}")


def _browser(argument: str, on_step) -> Mounted:
    from .browser_connection import BrowserPlugin

    plugin = BrowserPlugin(argument)
    return Mounted(name=plugin.name, plugin=plugin, kind="browser", view=plugin.screenshot,
                   preview={"media_type": "image/png", "transport": "frames"},
                   about="Explicitly connected real Chromium tab; raw DOM and pixels, typed actions.",
                   close=plugin.close)


def _gym(argument: str, on_step) -> Mounted:
    from .gym_connection import GymPlugin

    plugin = GymPlugin.from_id(argument)
    return Mounted(name=plugin.name, plugin=plugin, kind="gym", view=plugin.screenshot,
                   about="Real Gymnasium environment; explicit reset/step and raw observations.",
                   metadata={"action_space": repr(plugin.environment.action_space),
                             "observation_space": repr(plugin.environment.observation_space),
                             "render_mode": plugin.environment.render_mode}, close=plugin.close)


#: name -> how to mount it. The key before the colon in a spec.
FACTORIES: dict[str, Callable[[str, Any], Mounted]] = {"desktop": _desktop, "vision": _vision, "self": _self,
                                                          "filesystem": _filesystem, "browser": _browser, "gym": _gym}


def mount(spec: str, on_step=None) -> Mounted:
    """Mount one plugin from a spec like ``desktop`` or ``desktop:note``."""
    kind, _, argument = spec.partition(":")
    factory = FACTORIES.get(kind.strip())
    if factory is None:
        raise ValueError(f"no plugin called {kind!r}; have {', '.join(sorted(FACTORIES))}")
    got = factory(argument.strip(), on_step)
    got.id = connection_id(spec)
    if got.kind == "plugin":
        got.kind = FACTORY_METADATA.get(kind.strip(), {}).get("kind", kind.strip())
    return got


def attach_agent(mounted: list[Mounted], agent: Any) -> None:
    """Hand the agent to any plugin that reports on it."""
    for m in mounted:
        if hasattr(m.plugin, "attach"):
            m.plugin.attach(agent)


def mount_all(specs: list[str], on_step=None) -> list[Mounted]:
    """Mount independently identified connections, including duplicate specs."""
    out: list[Mounted] = []
    occurrences: dict[str, int] = {}
    names: set[str] = set()
    for spec in specs:
        try:
            got = mount(spec, on_step)
        except Exception:
            for previous in out:
                previous.close()
            raise
        base_id = connection_id(spec)
        occurrences[base_id] = occurrences.get(base_id, 0) + 1
        got.id = connection_id(spec, occurrences[base_id])
        base_name = got.name
        suffix = 2
        while got.name in names:
            got.name = f"{base_name}#{suffix}"
            suffix += 1
        names.add(got.name)
        got.plugin.name = got.name
        out.append(got)
    return out


FACTORY_METADATA = {
    "gym": {"kind": "gym", "about": "Explicit Gymnasium environment; requires optional installed dependencies"},
    "browser": {"kind": "browser", "about": "Explicit CDP connection to an existing Chromium tab"},
    "desktop": {"kind": "virtual_os", "about": "Computerworld virtual desktop"},
    "filesystem": {"kind": "filesystem", "about": "Explicitly mounted local filesystem root"},
    "vision": {"kind": "vision", "about": "Category proposals if a trained model is installed"},
    "self": {"kind": "introspection", "about": "Agent capability and trace inspection"},
}


def register_factory(kind: str, factory: Callable[[str, Any], Mounted], *,
                     connection_kind: str | None = None, about: str = "") -> None:
    """Register a real adapter constructor; arbitrary kinds require no UI changes."""
    if not kind or kind.strip() != kind or ":" in kind or kind in FACTORIES:
        raise ValueError(f"invalid or already registered connection kind: {kind!r}")
    FACTORIES[kind] = factory
    FACTORY_METADATA[kind] = {"kind": connection_kind or kind, "about": about}


def specs_descriptors(specs: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
    """Inventory configuration without constructing engines or promising availability.

    Worker descriptors replace these after initialization. Unknown kinds are unavailable;
    the inventory never advertises unimplemented browser/gym adapters.
    """
    result = []
    occurrences: dict[str, int] = {}
    for spec in specs:
        kind = spec.partition(":")[0].strip()
        base_id = connection_id(spec)
        occurrences[base_id] = occurrences.get(base_id, 0) + 1
        metadata = FACTORY_METADATA.get(kind, {"kind": kind, "about": ""})
        result.append({"id": connection_id(spec, occurrences[base_id]), "name": spec,
                       "kind": metadata["kind"], "about": metadata["about"],
                       "status": "configured" if kind in FACTORIES else "unavailable",
                       "capabilities": [], "preview": None, "selectable": kind in FACTORIES})
    return result
