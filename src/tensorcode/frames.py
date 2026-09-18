"""Three orientations over the same claims: what it is, where it is, how it was known.

The store indexes claims by subject, predicate and scope — all *semantic*. That is not
enough for a mind that has a body: "what is next to the pointer" and "what did I see with
my eyes rather than read in a tree" are ordinary questions, and answering them by scanning
everything is what makes them feel impossible.

    semantic   (subject, predicate)                what it is about
    spatial    (window, region, box)               where it is, relative to the screen and the pointer
    modality   (pixels, structure, text, hearsay)  how it came to be known

The same claim sits in all three. That is the point: awareness spreading in one frame pulls
in neighbours from the others (``Frames.links`` feeds ``awareness.Awareness``), which is
what makes nucleation cross modalities. Where two modalities disagree about the same thing,
``disagreements`` reports it rather than letting the stronger index win silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .records import ClaimRecord, Ref, Store

#: how a claim came to be known, from its evidence's method and source
MODALITIES = ("pixels", "structure", "text", "hearsay", "inference", "other")

_MODALITY_PATTERNS = (
    ("pixels", re.compile(r"ocr|pixel|vision|screenshot|segment", re.I)),
    ("structure", re.compile(r"dom|scene|atspi|accessib|ax\b|uia|tree", re.I)),
    ("hearsay", re.compile(r"hearsay|heard|told|said|rumou?r", re.I)),
    ("inference", re.compile(r"derive|rule|infer", re.I)),
    ("text", re.compile(r"parse|text|read|grammar|utterance|note", re.I)),
)


def modality_of(rec: ClaimRecord) -> tuple[str, ...]:
    """Which modalities this claim rests on, in a fixed order (a claim may have several)."""
    found: set[str] = set()
    for e in rec.evidence:
        blob = f"{e.method or ''} {e.source.id}"
        for name, pattern in _MODALITY_PATTERNS:
            if pattern.search(blob):
                found.add(name)
                break
        else:
            found.add("other")
    return tuple(m for m in MODALITIES if m in found)


@dataclass(frozen=True)
class Place:
    """Where something is, in screen terms. ``None`` fields are simply unknown."""

    window: str | None = None
    region: str | None = None
    box: tuple[int, int, int, int] | None = None  # x, y, w, h

    @property
    def center(self) -> tuple[float, float] | None:
        if self.box is None:
            return None
        x, y, w, h = self.box
        return (x + w / 2, y + h / 2)

    def distance(self, other: Place | tuple[float, float]) -> float | None:
        """Pixels between centres, or ``None`` when either side has no box."""
        here = self.center
        there = other if isinstance(other, tuple) else other.center
        if here is None or there is None:
            return None
        return ((here[0] - there[0]) ** 2 + (here[1] - there[1]) ** 2) ** 0.5

    def near(self, other: Place | tuple[float, float], slack: float = 120.0) -> bool:
        d = self.distance(other)
        return d is not None and d <= slack

    @property
    def known(self) -> bool:
        return self.window is not None or self.region is not None or self.box is not None


def place_of(rec: ClaimRecord, mind: Store) -> Place:
    """Read a place off the claim's subject: its entity payload's geometry, or ``in``/``at`` claims."""
    window = region = box = None
    payload = mind.entities.get(rec.claim.subject)
    if payload is not None:
        raw = getattr(payload, "box", None)
        if isinstance(raw, (list, tuple)) and len(raw) == 4 and all(isinstance(v, (int, float)) for v in raw):
            box = tuple(int(v) for v in raw)  # type: ignore[assignment]
        section = getattr(payload, "section", None)
        if isinstance(section, str) and section:
            window = section
    if rec.claim.predicate in ("in", "in_table") and isinstance(rec.claim.object, str):
        region = rec.claim.object
    if rec.claim.predicate == "at" and isinstance(rec.claim.object, (list, tuple)) and len(rec.claim.object) == 4:
        box = tuple(int(v) for v in rec.claim.object)  # type: ignore[assignment]
    return Place(window, region, box)


@dataclass(frozen=True)
class Binding:
    """One claim, seen from all three frames at once."""

    claim_id: str
    subject: Ref
    predicate: str
    place: Place
    modality: tuple[str, ...]


@dataclass(frozen=True)
class Disagreement:
    """Two live claims about the same thing, from different modalities, with different objects."""

    subject: Ref
    predicate: str
    readings: tuple[tuple[str, Any, tuple[str, ...]], ...]  # (claim id, object, modalities)

    def describe(self) -> str:
        parts = ", ".join(f"{obj!r} via {'+'.join(mods) or 'unknown'}" for _, obj, mods in self.readings)
        return f"{self.subject} {self.predicate}: {parts}"


class Frames:
    """Three indexes over one mind, rebuilt when the store's revision moves."""

    def __init__(self, mind: Store, *, modality: Callable[[ClaimRecord], tuple[str, ...]] = modality_of,
                 place: Callable[[ClaimRecord, Store], Place] = place_of) -> None:
        self.mind = mind
        self._modality_of, self._place_of = modality, place
        self._bindings: dict[str, Binding] = {}
        self._by_modality: dict[str, set[str]] = {}
        self._by_window: dict[str, set[str]] = {}
        self._by_region: dict[str, set[str]] = {}
        self._placed: list[tuple[str, Place]] = []
        self._grid: dict[tuple[int, int], list[str]] = {}  # coarse cells, so adjacency is a lookup
        self._cell = 128
        self._at = -1

    def index(self) -> None:
        if self._at == self.mind.revision:
            return
        self._bindings, self._by_modality, self._by_window, self._by_region, self._placed = {}, {}, {}, {}, []
        self._grid = {}
        for cid, rec in sorted(self.mind._claims.items()):
            if rec.retracted:
                continue
            mods = self._modality_of(rec)
            where = self._place_of(rec, self.mind)
            self._bindings[cid] = Binding(cid, rec.claim.subject, rec.claim.predicate, where, mods)
            for m in mods:
                self._by_modality.setdefault(m, set()).add(cid)
            if where.window:
                self._by_window.setdefault(where.window, set()).add(cid)
            if where.region:
                self._by_region.setdefault(where.region, set()).add(cid)
            if where.box is not None:
                self._placed.append((cid, where))
                cx, cy = where.center  # type: ignore[misc]
                self._grid.setdefault((int(cx // self._cell), int(cy // self._cell)), []).append(cid)
        self._at = self.mind.revision

    # -- the three frames

    def binding(self, claim: Any) -> Binding | None:
        self.index()
        return self._bindings.get(claim if isinstance(claim, str) else claim.id)

    def semantic(self, subject: Ref | None = None, predicate: str | None = None) -> list[ClaimRecord]:
        return self.mind.claims(subject=subject, predicate=predicate)

    def spatial(self, *, window: str | None = None, region: str | None = None,
                near: Place | tuple[float, float] | None = None, slack: float = 120.0) -> list[ClaimRecord]:
        """Claims located in a window or region, or within ``slack`` pixels of a point."""
        self.index()
        ids: set[str] | None = None
        if window is not None:
            ids = set(self._by_window.get(window, ()))
        if region is not None:
            ids = set(self._by_region.get(region, ())) if ids is None else ids & self._by_region.get(region, set())
        if near is not None:
            close = {cid for cid, place in self._placed if place.near(near, slack)}
            ids = close if ids is None else ids & close
        chosen = sorted(ids if ids is not None else self._bindings)
        return [self.mind._claims[cid] for cid in chosen if cid in self.mind._claims and not self.mind._claims[cid].retracted]

    def modality(self, kind: str) -> list[ClaimRecord]:
        self.index()
        return [self.mind._claims[cid] for cid in sorted(self._by_modality.get(kind, ()))
                if not self.mind._claims[cid].retracted]

    # -- cross-frame

    def disagreements(self) -> list[Disagreement]:
        """Where modalities conflict about one (subject, predicate). Nothing is resolved here."""
        self.index()
        groups: dict[tuple[Ref, str], list[tuple[str, Any, tuple[str, ...]]]] = {}
        for cid, b in sorted(self._bindings.items()):
            rec = self.mind._claims[cid]
            groups.setdefault((b.subject, b.predicate), []).append((cid, rec.claim.object, b.modality))
        out = []
        for (subject, predicate), readings in sorted(groups.items(), key=lambda kv: (kv[0][0].id, kv[0][1])):
            objects = {repr(obj) for _, obj, _ in readings}
            modalities = {mods for _, _, mods in readings}
            if len(objects) > 1 and len(modalities) > 1:
                out.append(Disagreement(subject, predicate, tuple(readings)))
        return out

    def links(self, claim_id: str, mind: Store, *, slack: float = 80.0, spatial_weight: float = 0.6,
              modality_weight: float = 0.25) -> Iterable[tuple[str, str, float]]:
        """Extra adjacency for awareness: what is beside this on screen, and what was seen with it."""
        self.index()
        b = self._bindings.get(claim_id)
        if b is None:
            return ()
        out: list[tuple[str, str, float]] = []
        if b.place.box is not None:
            cx, cy = b.place.center  # type: ignore[misc]
            reach = int(slack // self._cell) + 1
            cell_x, cell_y = int(cx // self._cell), int(cy // self._cell)
            for dx in range(-reach, reach + 1):
                for dy in range(-reach, reach + 1):
                    for cid in self._grid.get((cell_x + dx, cell_y + dy), ()):
                        if cid != claim_id and self._bindings[cid].place.near(b.place, slack):
                            out.append((cid, "spatial", spatial_weight))
        if b.place.window:
            for cid in sorted(self._by_window.get(b.place.window, ()))[:64]:
                if cid != claim_id:
                    out.append((cid, "window", modality_weight))
        return list(dict.fromkeys(out))
