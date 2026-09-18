"""Fuse several providers into one scene, keeping disagreement visible.

The rules, in one place:

* **Structure follows the most reliable provider that saw the element.** Accessibility (DOM,
  AT-SPI) beats pixels, so a fused element keeps the accessible role, box and hit point.
* **A missing field is filled from a less reliable provider**, with that provider's
  confidence (times a small penalty) and its provenance recorded. This is how vision names
  icon-only controls and reads canvas content that carries no accessible name.
* **Two different readings never collapse into one confident value.** If both providers have
  a value and they differ: the more reliable one is kept, its confidence drops (to at most
  the ratio of the two reliabilities), and the other reading is recorded as a ``Conflict``.
  If the providers are equally reliable, the field becomes empty with both readings recorded,
  which is the scene-level way of saying Unknown.
* **Elements only one provider saw are kept**, with their own source and confidence, so
  fusion is additive (vision fills gaps) rather than a filter. But a weaker provider's element
  that sits *inside* an interactive element the stronger one already has is the same control
  seen partially (a checkbox's label, a button's text): it is merged in, not added as a rival
  the agent might click instead.
* Agreement raises confidence: two independent sources reading the same text is worth more
  than either alone (``1 - (1-a)(1-b)``, capped).

Every fused item lists every source in ``provenance``, so a belief built on it can be
explained down to "AT-SPI said the role, OCR said the name".
"""

from __future__ import annotations

import re
import time
from dataclasses import replace
from typing import Sequence

from .protocol import Conflict, Element, PerceivedScene, Provider, Region, Target, TextBlock, center as center_of, match_pairs

FILL_PENALTY = 0.9  # a field borrowed from a less reliable provider is worth slightly less


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().casefold()


def agree(a: float, b: float) -> float:
    return round(min(0.99, 1 - (1 - a) * (1 - b)), 3)


def _combine_field(field: str, mine: str, theirs: str, my_conf: float, their_conf: float, my_rel: float, their_rel: float, their_source: str):
    """(value, confidence, conflicts) for one text field of two matched items."""
    if _norm(mine) == _norm(theirs):
        return mine, agree(my_conf, their_conf), ()
    if mine and not theirs:
        return mine, my_conf, ()
    if theirs and not mine:
        return theirs, round(their_conf * FILL_PENALTY, 3), ()
    conflict = (Conflict(field, mine, theirs, their_source),)
    if my_rel > their_rel:
        return mine, round(min(my_conf, their_rel / my_rel), 3), conflict
    if their_rel > my_rel:
        return theirs, round(min(their_conf, my_rel / their_rel), 3), conflict
    return "", 0.0, conflict  # equally reliable and different: refuse to pick


class FusedProvider:
    """Several providers merged. ``reliability`` is the best of its parts."""

    name = "fused"

    def __init__(self, providers: Sequence[Provider], *, min_iou: float = 0.4) -> None:
        if not providers:
            raise ValueError("fusion needs at least one provider")
        self.providers = sorted(providers, key=lambda p: -p.reliability)
        self.min_iou = min_iou
        self.name = "+".join(p.name for p in self.providers)
        self.reliability = self.providers[0].reliability

    def available(self) -> bool:
        return any(p.available() for p in self.providers)

    def perceive(self, target: Target) -> PerceivedScene:
        t0 = time.perf_counter()
        scenes = []
        for p in self.providers:
            if not p.available():
                continue
            scene = p.perceive(target)
            scenes.append((p, scene))
        if not scenes:
            raise RuntimeError("no provider in the fusion is available")
        (base_provider, fused), *rest = scenes
        reliability = {base_provider.name: base_provider.reliability}
        for provider, scene in rest:
            reliability[provider.name] = provider.reliability
            fused = fuse_scenes(fused, scene, reliability[base_provider.name], provider.reliability, min_iou=self.min_iou)
        fused.source = self.name
        fused.timings_ms = {f"{p.name}:{k}": v for p, s in scenes for k, v in s.timings_ms.items()} | {"fusion": (time.perf_counter() - t0) * 1e3}
        return fused


def fuse_scenes(primary: PerceivedScene, secondary: PerceivedScene, primary_reliability: float, secondary_reliability: float, *, min_iou: float = 0.4) -> PerceivedScene:
    """Merge ``secondary`` into ``primary`` (the more reliable one) by geometric agreement."""
    elements = list(primary.elements)
    pairs = dict(match_pairs(primary.elements, secondary.elements, min_iou=min_iou))
    for i, j in pairs.items():
        mine, theirs = primary.elements[i], secondary.elements[j]
        conflicts = list(mine.conflicts)
        name, name_conf, name_conflict = _combine_field("name", mine.name, theirs.name, mine.confidence, theirs.confidence, primary_reliability, secondary_reliability, theirs.sources[0] if theirs.sources else "?")
        value, value_conf, value_conflict = _combine_field("value", mine.value, theirs.value, mine.confidence, theirs.confidence, primary_reliability, secondary_reliability, theirs.sources[0] if theirs.sources else "?")
        conflicts += [*name_conflict, *value_conflict]
        if mine.role != theirs.role and mine.role and theirs.role:
            conflicts.append(Conflict("role", mine.role, theirs.role, theirs.sources[0] if theirs.sources else "?"))
        elements[i] = replace(
            mine, name=name, value=value, hint=mine.hint or theirs.hint,
            state=mine.state | {s for s in theirs.state if s not in ("editable",)},
            point=mine.point or theirs.point,
            confidence=round(min(max(name_conf, 0.0) if name else mine.confidence, agree(mine.confidence, theirs.confidence)), 3),
            provenance=mine.provenance + theirs.provenance, conflicts=tuple(conflicts),
        )
    matched_secondary = set(pairs.values())
    # a weaker provider often sees only part of a control (the label of a checkbox, the text of a
    # button). If its element sits inside an interactive element the stronger provider already has,
    # it is the same thing: merge its name in rather than adding a rival the agent might click.
    from .protocol import inside as _inside

    for j, theirs in enumerate(secondary.elements):
        if j in matched_secondary:
            continue
        host = next((i for i, mine in enumerate(elements)
                     if mine.role in ("button", "checkbox", "textbox", "combobox", "tab", "option", "link", "menuitem")
                     and _inside(theirs.point or center_of(theirs.box), mine.box, pad=2) and theirs.box[2] * theirs.box[3] <= 1.2 * mine.box[2] * mine.box[3]), None)
        if host is None:
            elements.append(theirs)
            continue
        mine = elements[host]
        name, name_conf, name_conflict = _combine_field("name", mine.name, theirs.name, mine.confidence, theirs.confidence, primary_reliability, secondary_reliability, theirs.sources[0] if theirs.sources else "?")
        elements[host] = replace(mine, name=name or mine.name, confidence=mine.confidence if mine.name else round(min(mine.confidence, name_conf or mine.confidence), 3),
                                 provenance=mine.provenance + theirs.provenance, conflicts=mine.conflicts + name_conflict)

    texts = list(primary.texts)
    text_pairs = dict(match_pairs(primary.texts, secondary.texts, min_iou=0.45))
    for i, j in text_pairs.items():
        mine, theirs = primary.texts[i], secondary.texts[j]
        text, conf, conflict = _combine_field("text", mine.text, theirs.text, mine.confidence, theirs.confidence, primary_reliability, secondary_reliability, theirs.sources[0] if theirs.sources else "?")
        texts[i] = replace(mine, text=text, confidence=conf, provenance=mine.provenance + theirs.provenance, conflicts=mine.conflicts + conflict)
    texts += [t for j, t in enumerate(secondary.texts) if j not in set(text_pairs.values())]

    regions = list(primary.regions)
    region_pairs = dict(match_pairs(primary.regions, secondary.regions, min_iou=0.5))
    for i, j in region_pairs.items():
        mine, theirs = primary.regions[i], secondary.regions[j]
        label, conf, _ = _combine_field("label", mine.label, theirs.label, mine.confidence, theirs.confidence, primary_reliability, secondary_reliability, theirs.provenance[0].source if theirs.provenance else "?")
        regions[i] = replace(mine, label=label or mine.label, confidence=conf, provenance=mine.provenance + theirs.provenance)
    regions += [r for j, r in enumerate(secondary.regions) if j not in set(region_pairs.values())]

    return PerceivedScene(
        tuple(elements), tuple(texts), tuple(regions), primary.tables or secondary.tables,
        primary.url or secondary.url, primary.title or secondary.title, primary.dialog or secondary.dialog,
        primary.busy or secondary.busy, f"{primary.source}+{secondary.source}",
    )
