"""Linux desktop accessibility provider (AT-SPI 2), for real native windows.

This reads the same tree screen readers use, over D-Bus, through ``gi``'s ``Atspi``
bindings. It needs no screenshot and no DOM, so it is the primary route on a real Linux
desktop; where an app's tree is thin (GTK4 apps expose deep anonymous panels, canvas-drawn
content exposes nothing), fuse it with the vision provider.

    provider = AtspiProvider()
    scene = provider.perceive(Target(window="Text Editor"))   # one window
    scene = provider.perceive(Target())                       # every showing window

Read-only: this module only queries. It never invokes an accessible action.

Notes from this machine (GNOME 4x on X11): ``Atspi.init()`` works without setting
``toolkit-accessibility``; application nodes have no Component interface (no extents), so
geometry comes from their frames down; GTK4 apps put several unnamed ``panel`` levels
between the frame and the widgets, so the walk goes deep and filters by state.
"""

from __future__ import annotations

import time
from typing import Any, Iterator

from .protocol import Element, PerceivedScene, Region, Target, TextBlock, center, own

ROLE_MAP = {
    "push button": "button", "toggle button": "button", "button": "button", "link": "link",
    "entry": "textbox", "text": "textbox", "password text": "textbox", "spin button": "textbox", "search box": "textbox",
    "check box": "checkbox", "check menu item": "checkbox", "radio button": "option", "radio menu item": "option",
    "combo box": "combobox", "page tab": "tab", "list item": "row", "table row": "row", "table cell": "cell",
    "menu item": "menuitem", "menu": "menuitem", "slider": "slider", "list box": "list", "list": "list",
    "label": "text", "static": "text", "heading": "text", "paragraph": "text", "caption": "text",
    "image": "image", "icon": "image", "canvas": "image",
    "frame": "window", "window": "window", "dialog": "dialog", "alert": "dialog",
}
TEXTY = {"text", "image"}
CONTAINERS = {"panel", "filler", "scroll pane", "viewport", "tool bar", "menu bar", "layered pane", "split pane", "table", "tree", "tree table", "document frame", "document web", "section", "group", "form", "list box", "list", "page tab list", "notebook", "status bar", "header", "footer", "internal frame", "application", "frame", "window", "dialog", "alert", "canvas", "drawing area"}
STATE_FLAGS = (("CHECKED", "checked"), ("EDITABLE", "editable"), ("FOCUSED", "focused"), ("EXPANDED", "expanded"), ("SELECTED", "selected"), ("BUSY", "busy"))


class AtspiProvider:
    name = "atspi"
    reliability = 0.93  # a real accessibility tree, but names can be missing or stale

    def __init__(self, max_nodes: int = 4000, max_depth: int = 18, include_offscreen: bool = False) -> None:
        self.max_nodes, self.max_depth, self.include_offscreen = max_nodes, max_depth, include_offscreen
        self._atspi = None

    # -- availability

    def available(self) -> bool:
        try:
            self._load()
        except Exception:
            return False
        try:
            return self._desktop() is not None
        except Exception:
            return False

    def _load(self) -> Any:
        if self._atspi is None:
            import gi

            gi.require_version("Atspi", "2.0")
            from gi.repository import Atspi

            Atspi.init()
            self._atspi = Atspi
        return self._atspi

    def _desktop(self) -> Any:
        return self._load().get_desktop(0)

    # -- inventory

    def applications(self) -> list[tuple[int, str]]:
        desktop = self._desktop()
        out = []
        for i in range(desktop.get_child_count()):
            try:
                app = desktop.get_child_at_index(i)
                out.append((i, app.get_name() or ""))
            except Exception:
                continue
        return out

    def windows(self) -> list[tuple[str, str, tuple[int, int, int, int]]]:
        """(application, window title, box) for every showing top-level frame."""
        Atspi = self._load()
        found = []
        desktop = self._desktop()
        for i in range(desktop.get_child_count()):
            try:
                app = desktop.get_child_at_index(i)
                app_name = app.get_name() or ""
                for j in range(app.get_child_count()):
                    frame = app.get_child_at_index(j)
                    if frame is None:
                        continue
                    box = self._extents(frame)
                    if box and self._showing(frame):
                        found.append((app_name, frame.get_name() or "", box))
            except Exception:
                continue
        return found

    # -- perception

    def perceive(self, target: Target) -> PerceivedScene:
        Atspi = self._load()
        t0 = time.perf_counter()
        elements: list[Element] = []
        texts: list[TextBlock] = []
        regions: list[Region] = []
        seen = [0]
        for app_name, frame in self._frames(target):
            box = self._extents(frame) or (0, 0, 0, 0)
            regions.append(Region("window", frame.get_name() or app_name, box, 0.95, own("atspi", "frame extents", app_name)))
            for node, depth, path in self._walk(frame, seen):
                self._absorb(node, app_name, frame.get_name() or app_name, path, elements, texts, regions)
        scene = PerceivedScene(tuple(elements), tuple(texts), tuple(regions), (), "", ", ".join(r.label for r in regions if r.kind == "window"),
                               next((r.label for r in regions if r.kind == "dialog"), ""), False, "atspi",
                               {"atspi": (time.perf_counter() - t0) * 1e3, "nodes": float(seen[0])})
        return scene  # AT-SPI extents are already screen coordinates

    # -- internals

    def _frames(self, target: Target) -> Iterator[tuple[str, Any]]:
        desktop = self._desktop()
        for i in range(desktop.get_child_count()):
            try:
                app = desktop.get_child_at_index(i)
                app_name = app.get_name() or ""
            except Exception:
                continue
            if target.app and target.app.lower() not in app_name.lower():
                continue
            try:
                n = app.get_child_count()
            except Exception:
                continue
            for j in range(n):
                try:
                    frame = app.get_child_at_index(j)
                except Exception:
                    continue
                if frame is None or not self._showing(frame):
                    continue
                title = ""
                try:
                    title = frame.get_name() or ""
                except Exception:
                    pass
                if target.window and target.window.lower() not in title.lower():
                    continue
                yield app_name, frame

    def _walk(self, node: Any, seen: list[int], depth: int = 0, path: str = "") -> Iterator[tuple[Any, int, str]]:
        if depth > self.max_depth or seen[0] >= self.max_nodes:
            return
        try:
            n = node.get_child_count()
        except Exception:
            n = 0
        for i in range(n):
            if seen[0] >= self.max_nodes:
                return
            try:
                child = node.get_child_at_index(i)
            except Exception:
                continue
            if child is None:
                continue
            seen[0] += 1
            child_path = f"{path}/{i}"
            if not self.include_offscreen and not self._showing(child):
                continue
            yield child, depth + 1, child_path
            yield from self._walk(child, seen, depth + 1, child_path)

    def _absorb(self, node: Any, app: str, window: str, path: str, elements: list, texts: list, regions: list) -> None:
        try:
            role_name = node.get_role_name()
        except Exception:
            return
        role = ROLE_MAP.get(role_name)
        box = self._extents(node)
        if box is None or box[2] <= 0 or box[3] <= 0:
            return
        name = ""
        try:
            name = (node.get_name() or "").strip()
        except Exception:
            pass
        value = self._value(node)
        state = self._states(node)
        provenance = own("atspi", f"role={role_name}", app, locator=path)
        if role in (None, "window", "dialog"):
            if role in ("window", "dialog"):
                regions.append(Region(role, name, box, 0.95, provenance))
            elif role_name in CONTAINERS:
                regions.append(Region("panel", name, box, 0.8, provenance))
            return
        if role in TEXTY and not self._interactive(node):
            body = value or name
            if body:
                texts.append(TextBlock(body, box, window, 0, 0.95, provenance))
            return
        elements.append(Element(role, name, value, "", window, box, center(box), state, 0.95, provenance))

    def _value(self, node: Any) -> str:
        """A field's own content. The Text/Value interfaces are module functions, not node methods."""
        Atspi = self._load()
        try:
            node.clear_cache()  # AT-SPI caches per node; a field just typed into needs a fresh read
        except Exception:
            pass
        try:
            count = Atspi.Text.get_character_count(node)
            if count:
                return Atspi.Text.get_text(node, 0, min(count, 400))
        except Exception:
            pass
        try:
            return str(Atspi.Value.get_current_value(node))
        except Exception:
            return ""

    def _states(self, node: Any) -> frozenset[str]:
        Atspi = self._load()
        try:
            ss = node.get_state_set()
        except Exception:
            return frozenset()
        out = {flag for state, flag in STATE_FLAGS if self._has(ss, state)}
        if not self._has(ss, "ENABLED") or not self._has(ss, "SENSITIVE"):
            out.add("disabled")
        return frozenset(out)

    def _has(self, state_set: Any, name: str) -> bool:
        Atspi = self._load()
        try:
            return bool(state_set.contains(getattr(Atspi.StateType, name)))
        except Exception:
            return False

    def _interactive(self, node: Any) -> bool:
        try:
            ss = node.get_state_set()
        except Exception:
            return False
        return self._has(ss, "FOCUSABLE") or self._has(ss, "EDITABLE")

    def _showing(self, node: Any) -> bool:
        try:
            ss = node.get_state_set()
        except Exception:
            return False
        return self._has(ss, "SHOWING") and self._has(ss, "VISIBLE")

    def _extents(self, node: Any) -> tuple[int, int, int, int] | None:
        Atspi = self._load()
        try:
            e = node.get_extents(Atspi.CoordType.SCREEN)
        except Exception:
            return None
        if e is None or e.width <= 0 or e.height <= 0 or e.width > 20000:
            return None
        return (int(e.x), int(e.y), int(e.width), int(e.height))
