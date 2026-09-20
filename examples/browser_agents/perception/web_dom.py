"""Web DOM / ARIA provider: the page's own accessibility-relevant structure.

This is what the browser agents have always used (``browser.PERCEIVE_JS``), expressed as a
provider so it can be swapped or fused with others. Confidence is high but not 1: a DOM name
can still be wrong about what a user sees (an aria-label that lies, a stale label).
"""

from __future__ import annotations

import time

from ..browser import PERCEIVE_JS
from .protocol import Element, PerceivedScene, Provenance, Region, Table, Target, TextBlock, own

STATE_KEYS = (("checked", "checked"), ("disabled", "disabled"), ("current", "current"))


class WebDomProvider:
    name = "web-dom"
    reliability = 0.95

    def available(self) -> bool:
        return True

    def perceive(self, target: Target) -> PerceivedScene:
        page = target.page
        t0 = time.perf_counter()
        raw = page.evaluate(PERCEIVE_JS)
        elements = []
        for c in raw["controls"]:
            state = {name for key, name in STATE_KEYS if c.get(key)}
            if c["role"] == "textbox":
                state.add("editable")
            elements.append(Element(
                role=c["role"], name=c["name"], value=c["value"] or (c["shown"] or ""), hint=c["hint"], section=c["section"],
                box=tuple(c["box"]), point=tuple(c["point"]) if c["point"] else None, state=frozenset(state),
                confidence=0.97, provenance=own("web-dom", "aria+hit-test", locator=f"box{tuple(c['box'])}"),
            ))
        texts = tuple(TextBlock(t["text"], tuple(t["box"]), t["section"], t["seq"], 0.97, own("web-dom", "text-node")) for t in raw["texts"])
        regions = tuple(Region("graphic", g["label"], tuple(g["box"]), 0.97, own("web-dom", "canvas/img")) for g in raw["graphics"])
        if raw["dialog"]:
            regions += (Region("dialog", raw["dialog"][:80], (0, 0, 0, 0), 0.97, own("web-dom", "role=dialog")),)
        tables = tuple(Table(t["label"], tuple(t["header"]), tuple(tuple(r) for r in t["rows"]), own("web-dom", "table")) for t in raw["tables"])
        return PerceivedScene(tuple(elements), texts, regions, tables, raw["url"], raw["title"], raw["dialog"], raw["busy"], "web-dom",
                              {"dom": (time.perf_counter() - t0) * 1e3})


class RecordedDomProvider(WebDomProvider):
    """Transport an explicitly supplied DOM capture, including its supplied labels.

    Labels are DOM evidence; this provider does not infer pixel visibility.
    """

    name = "web-dom"

    def __init__(self, screen: dict) -> None:
        self.screen = screen

    def perceive(self, target: Target) -> PerceivedScene:
        elements = []
        for c in self.screen["controls"]:
            name = c["name"]
            state = {n for k, n in STATE_KEYS if c.get(k)} | ({"editable"} if c["role"] == "textbox" else set())
            elements.append(Element(c["role"], name, c["value"] or c.get("shown", ""), c["hint"], c["section"], tuple(c["box"]),
                                    tuple(c["point"]) if c["point"] else None, frozenset(state), 0.97,
                                    own("web-dom", "aria+hit-test", "recorded")))
        texts = tuple(TextBlock(t["text"], tuple(t["box"]), t["section"], t["seq"], 0.97, own("web-dom", "text-node", "recorded")) for t in self.screen["texts"])
        regions = tuple(Region("graphic", g["label"], tuple(g["box"]), 0.97, own("web-dom", "canvas/img", "recorded")) for g in self.screen["graphics"])
        return PerceivedScene(tuple(elements), texts, regions, (), self.screen["url"], self.screen["title"], self.screen["dialog"], False, "web-dom")
