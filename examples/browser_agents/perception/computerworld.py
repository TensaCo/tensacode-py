"""Perception from the computerworld engine's declared structured scene.

``CwProvider`` turns ``env.scene(w, h)`` into the same ``PerceivedScene`` the DOM and
accessibility providers produce, so a mind does not change with the environment.

Two mappings are deliberate, and are how one mind runs on three different bodies:

* **Roles.** Engine semantic roles (``heading``, ``button``, ``textbox``, ``text``) map onto
  the protocol's vocabulary, as every provider maps its own.
* **Sections.** A node inside window *n* is reported in the section named by that window's
  title, matching how the DOM provider reports a window/dialog label. Shell chrome (dock,
  top bar) has no section.
The engine draws one control as several nodes (a hit region and its icon or text). They are
merged per interaction id, keeping the largest box.
"""

from __future__ import annotations

import time

from .protocol import Element, PerceivedScene, Region, Target, TextBlock, own

ROLE_MAP = {
    "heading": "text", "button": "button", "textbox": "textbox", "text": "text", "link": "link",
    "checkbox": "checkbox", "image": "image", "list": "list", "row": "row", "cell": "cell", "tab": "tab",
}
#: engine interaction ids that are window chrome rather than content
CHROME = ("drag", "resize", "minimize", "maximize", "close", "focus")


def transform_of(node: dict) -> dict:
    return node.get("transform") or {"a": 1024, "b": 0, "c": 0, "d": 1024, "tx": 0, "ty": 0}


def scene_point(node: dict) -> tuple[int, int]:
    """A node's centre in scene pixels, through its transform (the engine's documented formula)."""
    t, b = transform_of(node), node["bounds"]
    x, y = b["x"] + b["width"] // 2, b["y"] + b["height"] // 2
    return (int((t["a"] * x + t["c"] * y) / 1024) + t["tx"], int((t["b"] * x + t["d"] * y) / 1024) + t["ty"])


def scene_box(node: dict) -> tuple[int, int, int, int]:
    t, b = transform_of(node), node["bounds"]
    x = int((t["a"] * b["x"] + t["c"] * b["y"]) / 1024) + t["tx"]
    y = int((t["b"] * b["x"] + t["d"] * b["y"]) / 1024) + t["ty"]
    return (x, y, max(1, int(b["width"] * t["a"] / 1024)), max(1, int(b["height"] * t["d"] / 1024)))


def window_id(interaction: str) -> str | None:
    parts = interaction.split(":")
    return parts[1] if len(parts) > 2 and parts[0] == "window" else None


def windows_in(raw: dict) -> dict[str, tuple[str, tuple[int, int, int, int]]]:
    """Window id -> literal focus-region label and box supplied by the engine."""
    found: dict[str, tuple[str, tuple[int, int, int, int]]] = {}
    for node in raw.get("nodes", ()):
        interaction = node.get("interaction") or ""
        wid = window_id(interaction)
        if wid is None:
            continue
        label = (node.get("semantic") or {}).get("label") or ""
        box = scene_box(node)
        if not interaction.endswith(":focus"):
            continue
        title = label
        old = found.get(wid)
        if old is None or box[2] * box[3] > old[1][2] * old[1][3]:
            found[wid] = (title or (old[0] if old else ""), box)
        elif not old[0] and title:
            found[wid] = (title, old[1])
    return found


class CwProvider:
    """Transport the engine's declared semantics; no pixel or language inference."""

    name = "computerworld"
    reliability = 0.98

    def available(self) -> bool:
        try:
            import computerworld  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return True

    def perceive(self, target: Target) -> PerceivedScene:
        t0 = time.perf_counter()
        surface = target.detail.get("surface") or target.page
        raw = surface.scene()
        windows = windows_in(raw)

        def section_for(node: dict, interaction: str) -> str:
            wid = window_id(interaction)
            if wid is not None:
                return windows.get(wid, ("", ()))[0]
            point = scene_point(node)
            for title, box in windows.values():
                if box[0] <= point[0] <= box[0] + box[2] and box[1] <= point[1] <= box[1] + box[3]:
                    return title
            return ""

        by_interaction: dict[str, Element] = {}
        loose: list[Element] = []
        texts: list[TextBlock] = []
        for node in raw.get("nodes", ()):
            semantic = node.get("semantic") or {}
            primitive = node.get("primitive") or {}
            interaction = node.get("interaction") or ""
            role = ROLE_MAP.get(semantic.get("role") or "", "unknown")
            engine_label = semantic.get("label") or ""
            box, section = scene_box(node), section_for(node, interaction)
            text = (primitive.get("text") or "").strip()
            if interaction or role in ("button", "textbox", "checkbox", "link"):
                name = engine_label.strip()
                state = {"editable"} if role == "textbox" else set()
                if semantic.get("disabled"):
                    state.add("disabled")
                if semantic.get("focused"):
                    state.add("focused")
                element = Element(
                    role=role, name=name, value=str(semantic.get("value") or ""),
                    section=section, box=box, point=scene_point(node), state=frozenset(state),
                    provenance=own("computerworld", "scene+semantics", f"role={semantic.get('role')} label={engine_label!r}", interaction or str(node.get("id"))),
                )
                if not interaction:
                    loose.append(element)
                    continue
                kept = by_interaction.get(interaction)
                if kept is None or box[2] * box[3] > kept.box[2] * kept.box[3]:
                    by_interaction[interaction] = element
            elif text and primitive.get("kind") == "text":
                texts.append(TextBlock(text=text, box=box, section=section,
                                       provenance=own("computerworld", "scene+text", f"size={primitive.get('size')}", str(node.get("id")))))
        elements = [e for e in (*by_interaction.values(), *loose)]
        elements.sort(key=lambda e: (e.box[1], e.box[0]))
        texts.sort(key=lambda t: (t.box[1], t.box[0]))
        regions = [Region(kind="window", label=title, box=box, provenance=own("computerworld", "scene+window")) for title, box in windows.values()]
        return PerceivedScene(
            elements=tuple(elements), texts=tuple(texts), regions=tuple(regions),
            title=(raw.get("title") or ""), source=self.name,
            timings_ms={"scene": (time.perf_counter() - t0) * 1e3},
        )
