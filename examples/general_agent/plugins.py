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
    mount("vision")             the image pipeline, for pictures the person pastes in

Adding a kind of plugin is adding an entry to ``FACTORIES``; nothing in the server or the page
needs to know it exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Mounted:
    """One plugin the agent is using, with the means to show it if it can be shown."""

    name: str                                   # unique on the page: "desktop", "desktop:note"
    plugin: Any
    view: Callable[[], bytes | None] = lambda: None
    about: str = ""

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
    return Mounted(name=name, plugin=plugin, view=lambda: plugin.surface.png(), about=about)


def _vision(argument: str, on_step) -> Mounted:
    from tensorcode.agent.vision_plugin import VisionPlugin

    return Mounted(name="vision", plugin=VisionPlugin(), about="looks at pictures you paste in")


def _self(argument: str, on_step) -> Mounted:
    from tensorcode.agent.discourse import DiscoursePlugin

    # it reports on the agent that owns it, which does not exist until every plugin is
    # mounted — hence `attach`, called once the agent is built
    return Mounted(name="self", plugin=DiscoursePlugin(),
                   about="explains what it did and what it can do")


#: name -> how to mount it. The key before the colon in a spec.
FACTORIES: dict[str, Callable[[str, Any], Mounted]] = {"desktop": _desktop, "vision": _vision, "self": _self}


def mount(spec: str, on_step=None) -> Mounted:
    """Mount one plugin from a spec like ``desktop`` or ``desktop:note``."""
    kind, _, argument = spec.partition(":")
    factory = FACTORIES.get(kind.strip())
    if factory is None:
        raise ValueError(f"no plugin called {kind!r}; have {', '.join(sorted(FACTORIES))}")
    return factory(argument.strip(), on_step)


def attach_agent(mounted: list[Mounted], agent: Any) -> None:
    """Hand the agent to any plugin that reports on it."""
    for m in mounted:
        if hasattr(m.plugin, "attach"):
            m.plugin.attach(agent)


def mount_all(specs: list[str], on_step=None) -> list[Mounted]:
    """Mount each spec, making names unique when the same kind is mounted twice."""
    out: list[Mounted] = []
    for spec in specs:
        got = mount(spec, on_step)
        if any(m.name == got.name for m in out):
            got.name = f"{got.name}#{sum(1 for m in out if m.name.split('#')[0] == got.name) + 1}"
        out.append(got)
    return out
