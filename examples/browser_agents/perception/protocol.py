"""One perception protocol, several interchangeable backends.

A *provider* answers one question: what is on this target right now? The target is a web
page, a desktop window, or a screen image. The answer is a ``PerceivedScene``: elements
(role, name, value, box, hit point, state), text blocks, and regions (windows, dialogs,
graphics). Every item carries **where it came from** (``Provenance``), **how sure** the
provider is (``confidence``), and **what disagreed** (``conflicts``), so fusing two
providers never silently collapses two different readings into one confident answer.

    provider.perceive(target) -> PerceivedScene -> .to_screen()  (what agents already use)

Backends live next to this file: ``web_dom`` (page DOM/ARIA), ``atspi`` (Linux desktop
accessibility), ``visual`` (OCR + UI detector + CV rules), ``fixture`` (recorded scenes,
for tests), ``uia``/``ax`` (Windows/macOS stubs with the interface spelled out), and
``fusion`` (several providers merged by agreement).

All providers report coordinates in one frame: screen space when the target is a desktop
window (accessibility already does; a pixel provider adds ``Target.image_origin``, the
position of the captured image), and page space for a web page.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Protocol, Sequence, runtime_checkable

Box = tuple[int, int, int, int]

# roles the protocol speaks; providers map their own vocabulary onto these
ROLES = frozenset({"button", "textbox", "checkbox", "combobox", "tab", "option", "link", "menuitem", "slider", "list", "row", "cell", "text", "image", "window", "dialog", "unknown"})
STATES = frozenset({"checked", "current", "disabled", "editable", "focused", "expanded", "selected", "busy"})


@dataclass(frozen=True)
class Provenance:
    """Where one piece of perception came from."""

    source: str  # "web-dom" | "atspi" | "vision" | "fixture" | "uia" | "ax"
    locator: str = ""  # a path, selector, accessible id, or box
    method: str = ""  # how it was obtained, e.g. "aria+hit-test", "ocr+rules"
    detail: str = ""  # the specific rule or interface that produced it

    def __str__(self) -> str:
        return f"{self.source}:{self.method}" + (f" ({self.detail})" if self.detail else "")


@dataclass(frozen=True)
class Conflict:
    """Two providers read the same thing differently. Kept, not averaged."""

    field: str
    mine: Any
    theirs: Any
    source: str

    def __str__(self) -> str:
        return f"{self.field}={self.mine!r} vs {self.theirs!r} from {self.source}"


@dataclass(frozen=True)
class Element:
    role: str
    name: str = ""
    value: str = ""
    hint: str = ""
    section: str = ""
    box: Box = (0, 0, 0, 0)
    point: tuple[int, int] | None = None  # where to click; None when nothing is hittable
    state: frozenset[str] = frozenset()
    confidence: float = 1.0
    provenance: tuple[Provenance, ...] = ()
    conflicts: tuple[Conflict, ...] = ()

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(p.source for p in self.provenance))


@dataclass(frozen=True)
class TextBlock:
    text: str
    box: Box = (0, 0, 0, 0)
    section: str = ""
    seq: int = 0  # live-region announcement order, 0 if not an announcement
    confidence: float = 1.0
    provenance: tuple[Provenance, ...] = ()
    conflicts: tuple[Conflict, ...] = ()

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(p.source for p in self.provenance))


@dataclass(frozen=True)
class Region:
    kind: str  # "window" | "dialog" | "graphic" | "panel"
    label: str = ""
    box: Box = (0, 0, 0, 0)
    confidence: float = 1.0
    provenance: tuple[Provenance, ...] = ()


@dataclass(frozen=True)
class Table:
    label: str
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    provenance: tuple[Provenance, ...] = ()


@dataclass
class PerceivedScene:
    elements: tuple[Element, ...] = ()
    texts: tuple[TextBlock, ...] = ()
    regions: tuple[Region, ...] = ()
    tables: tuple[Table, ...] = ()
    url: str = ""
    title: str = ""
    dialog: str = ""
    busy: bool = False
    source: str = ""
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_screen(self):  # -> browser.Screen
        """The value existing agents consume. State and confidence collapse to the old fields."""
        from ..browser import Control, Graphic, Screen, Table as ScreenTable, Text

        controls = tuple(
            Control(
                role=e.role, name=e.name, value=e.value, checked=(True if "checked" in e.state else False if e.role == "checkbox" else None),
                disabled="disabled" in e.state, hint=e.hint, group="", section=e.section, input_type="",
                box=e.box, point=e.point, current="current" in e.state or "selected" in e.state, shown=e.value if e.role == "combobox" else "",
            )
            for e in self.elements
        )
        texts = tuple(Text(role="alert" if t.seq else "text", text=t.text, section=t.section, box=t.box, seq=t.seq) for t in self.texts)
        graphics = tuple(Graphic(label=r.label, box=r.box) for r in self.regions if r.kind == "graphic")
        tables = tuple(ScreenTable(t.label, t.header, t.rows) for t in self.tables)
        return Screen(self.url, self.title, controls, texts, tables, self.dialog, self.busy, graphics)

    def named(self, name: str) -> list[Element]:
        return [e for e in self.elements if e.name == name]

    @property
    def uncertain(self) -> list[Element | TextBlock]:
        """Everything a provider (or fusion) is not sure about: low confidence or a live disagreement."""
        return [x for x in (*self.elements, *self.texts) if x.conflicts or x.confidence < 0.5]


@dataclass
class Target:
    """What to perceive. Providers use the fields they understand and ignore the rest."""

    page: Any = None  # Playwright page (web DOM)
    image: Any = None  # np.ndarray screenshot (vision)
    grab: Callable[[], Any] | None = None  # produce a screenshot when ``image`` is absent
    app: str | None = None  # accessibility: application name
    window: str | None = None  # accessibility: window/frame title (substring match)
    image_origin: tuple[int, int] = (0, 0)  # where ``image`` sits in screen space; providers that read pixels add it, so every provider reports screen coordinates
    detail: dict[str, Any] = field(default_factory=dict)

    def picture(self):
        if self.image is None and self.grab is not None:
            self.image = self.grab()
        return self.image


@runtime_checkable
class Provider(Protocol):
    """Anything that can answer ``perceive``. Backends are swappable per body or per task."""

    name: str
    reliability: float  # 0..1 prior on this source's structure, used to resolve conflicts in fusion

    def available(self) -> bool: ...

    def perceive(self, target: Target) -> PerceivedScene: ...


# ------------------------------------------------------------------ helpers


def own(source: str, method: str, detail: str = "", locator: str = "") -> tuple[Provenance, ...]:
    return (Provenance(source, locator, method, detail),)


def center(box: Box) -> tuple[int, int]:
    x, y, w, h = box
    return (int(x + w / 2), int(y + h / 2))


def iou(a: Box, b: Box) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def inside(pt: tuple[int, int] | None, box: Box, pad: int = 2) -> bool:
    return pt is not None and box[0] - pad <= pt[0] <= box[0] + box[2] + pad and box[1] - pad <= pt[1] <= box[1] + box[3] + pad


def shift(scene: PerceivedScene, dx: int, dy: int) -> PerceivedScene:
    """Move a scene's coordinates (screen -> image, or window -> screen)."""
    if (dx, dy) == (0, 0):
        return scene
    mv = lambda b: (b[0] + dx, b[1] + dy, b[2], b[3])  # noqa: E731
    mvp = lambda p: None if p is None else (p[0] + dx, p[1] + dy)  # noqa: E731
    scene.elements = tuple(replace(e, box=mv(e.box), point=mvp(e.point)) for e in scene.elements)
    scene.texts = tuple(replace(t, box=mv(t.box)) for t in scene.texts)
    scene.regions = tuple(replace(r, box=mv(r.box)) for r in scene.regions)
    return scene


def elements_by_name(scene: PerceivedScene) -> dict[str, list[Element]]:
    out: dict[str, list[Element]] = {}
    for e in scene.elements:
        out.setdefault(e.name, []).append(e)
    return out


def describe(scene: PerceivedScene, limit: int = 12) -> list[str]:
    """Short lines for logs: role, name, source(s), confidence, and any disagreement."""
    lines = []
    for e in scene.elements[:limit]:
        marks = "+".join(e.sources)
        lines.append(f"{e.role} {e.name!r} [{marks} {e.confidence:.2f}]" + (f" ⚠ {'; '.join(str(c) for c in e.conflicts)}" if e.conflicts else ""))
    return lines


def merge_scenes(scenes: Iterable[PerceivedScene]) -> PerceivedScene:
    """Concatenate scenes without fusing (used for providers covering disjoint areas)."""
    scenes = list(scenes)
    return PerceivedScene(
        elements=tuple(e for s in scenes for e in s.elements),
        texts=tuple(t for s in scenes for t in s.texts),
        regions=tuple(r for s in scenes for r in s.regions),
        tables=tuple(t for s in scenes for t in s.tables),
        url=next((s.url for s in scenes if s.url), ""),
        title=next((s.title for s in scenes if s.title), ""),
        dialog=next((s.dialog for s in scenes if s.dialog), ""),
        busy=any(s.busy for s in scenes),
        source="+".join(dict.fromkeys(s.source for s in scenes if s.source)),
    )


def match_pairs(mine: Sequence[Any], theirs: Sequence[Any], *, min_iou: float = 0.4) -> list[tuple[int, int]]:
    """Greedy one-to-one geometric matching between two lists of boxed items."""
    scored: list[tuple[float, int, int]] = []
    for i, a in enumerate(mine):
        for j, b in enumerate(theirs):
            overlap = iou(a.box, b.box)
            point_hit = inside(getattr(a, "point", None), b.box) or inside(getattr(b, "point", None), a.box)
            if overlap >= min_iou or (point_hit and overlap > 0.05):
                scored.append((overlap + (0.5 if point_hit else 0.0), i, j))
    scored.sort(reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs = []
    for _, i, j in scored:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j))
    return pairs
