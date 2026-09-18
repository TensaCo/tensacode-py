"""Noticing change: snapshots of what was perceived, and the difference between two of them.

Perception as it stands answers "what is there now". It cannot answer "what changed", which is
the most basic temporal perception there is: a snapshot scope retracts what left view and keeps
no record of the leaving. This module keeps the record.

    snap  = snapshot(items, tag="turn:3")           # what was perceived, reduced to identity + attributes
    diff(before, snap)                              # typed changes, each with provenance and a time
    Volatility().watch(...).is_volatile(key)        # what changes every frame anyway, learned not declared

A live screen changes constantly — clocks tick, carets blink, terminals scroll — so a diff that
reports everything is useless. Volatility is *measured*: a key that changed in most recent
comparisons is volatile, and a change to it is flagged rather than silently dropped, so both the
signal and the noise can be counted.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from .outcomes import Score

APPEARED, DISAPPEARED, MOVED, VALUE, WINDOW_OPENED, WINDOW_CLOSED, FOCUS, OCCLUDED = (
    "appeared", "disappeared", "moved", "value_changed", "window_opened", "window_closed", "focus_changed", "occluded")
REPLACED = "replaced"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Item:
    """One perceived thing, reduced to what a difference needs."""

    key: str
    kind: str = "control"  # control | text | window
    label: str = ""
    value: str = ""
    window: str = ""
    where: tuple[int, int, int, int] | None = None
    focused: bool = False

    @property
    def centre(self) -> tuple[int, int] | None:
        if self.where is None:
            return None
        x, y, w, h = self.where
        return (x + w // 2, y + h // 2)


@dataclass(frozen=True)
class Snapshot:
    """What was perceived at one moment, addressable by identity."""

    at: datetime
    items: Mapping[str, Item]
    tag: str = ""

    @property
    def windows(self) -> tuple[str, ...]:
        return tuple(sorted({i.window for i in self.items.values() if i.window}))

    def __len__(self) -> int:
        return len(self.items)


@dataclass(frozen=True)
class Change:
    """One difference, with what it was, what it is, and when it was noticed."""

    kind: str
    key: str
    at: datetime
    label: str = ""
    window: str = ""
    was: str | None = None
    now: str | None = None
    moved_by: tuple[int, int] | None = None
    volatile: bool = False  # this key changes on its own; reported so it can be filtered, not hidden
    brought: int = 0  # parts that came or went with this whole, reported as one change rather than many

    def describe(self) -> str:
        what = self.label or self.key
        where = f" in {self.window}" if self.window else ""
        if self.kind == APPEARED:
            return f"{what} appeared{where}"
        if self.kind == DISAPPEARED:
            return f"{what} went away{where}"
        if self.kind == OCCLUDED:
            return f"{what} is covered by {self.now}, not gone"
        if self.kind == REPLACED:
            return f"{self.was} became {self.now}{where}"
        if self.kind == WINDOW_OPENED:
            return f"the {what} window opened" + (f" (with {self.brought} things in it)" if self.brought else "")
        if self.kind == WINDOW_CLOSED:
            return f"the {what} window closed" + (f" (taking {self.brought} things with it)" if self.brought else "")
        if self.kind == MOVED:
            dx, dy = self.moved_by or (0, 0)
            return f"{what} moved{where} by ({dx}, {dy})"
        if self.kind == FOCUS:
            return f"focus moved to {what}{where}" if self.now == "focused" else f"{what} lost focus{where}"
        return f"{what}{where} changed from {self.was!r} to {self.now!r}"


@dataclass(frozen=True)
class Changes:
    """The difference between two snapshots, separated into signal and self-moving noise."""

    since: datetime
    at: datetime
    all: tuple[Change, ...] = ()

    @property
    def steady(self) -> tuple[Change, ...]:
        """Changes to things that do not change on their own."""
        return tuple(c for c in self.all if not c.volatile)

    @property
    def volatile(self) -> tuple[Change, ...]:
        return tuple(c for c in self.all if c.volatile)

    def of(self, *kinds: str) -> tuple[Change, ...]:
        return tuple(c for c in self.steady if c.kind in kinds)

    def summary(self, limit: int = 8) -> str:
        if not self.steady:
            return "nothing changed" if not self.volatile else f"nothing changed (besides {len(self.volatile)} things that change on their own)"
        lines = [c.describe() for c in self.steady[:limit]]
        if len(self.steady) > limit:
            lines.append(f"and {len(self.steady) - limit} more")
        return "; ".join(lines)

    def __len__(self) -> int:
        return len(self.steady)


@dataclass(frozen=True)
class ChangePolicy:
    """What counts as a change, and what is beneath notice."""

    move_threshold: int = 8  # pixels; below this a box wobble is not a move
    report_moves: bool = True
    volatile_after: int = 3  # comparisons needed before a key can be judged volatile
    volatile_fraction: float = 0.5  # changed in at least this share of them
    window_of: int = 12  # how many recent comparisons the judgement rests on


class Volatility:
    """Which keys change on their own, learned by watching rather than declared.

    A clock's text changes every comparison; a file listing does not. Nothing here knows what a
    clock is: it knows that key has changed in most of the comparisons it has seen.
    """

    def __init__(self, policy: ChangePolicy | None = None) -> None:
        self.policy = policy or ChangePolicy()
        self._seen: dict[str, deque[bool]] = defaultdict(lambda: deque(maxlen=self.policy.window_of))

    def watch(self, before: Snapshot, after: Snapshot) -> "Volatility":
        """Record, for every key present in both, whether its value moved."""
        for key, now in after.items.items():
            was = before.items.get(key)
            if was is None:
                continue
            self._seen[key].append(was.value != now.value or _shifted(was, now, self.policy.move_threshold))
        return self

    def is_volatile(self, key: str) -> bool:
        seen = self._seen.get(key)
        if not seen or len(seen) < self.policy.volatile_after:
            return False
        return (sum(seen) / len(seen)) >= self.policy.volatile_fraction

    def rate(self, key: str) -> Score | None:
        seen = self._seen.get(key)
        if not seen:
            return None
        return Score(round(sum(seen) / len(seen), 3), "frequency", f"changed in {sum(seen)} of {len(seen)} comparisons")

    def volatile_keys(self) -> tuple[str, ...]:
        return tuple(sorted(k for k in self._seen if self.is_volatile(k)))


def snapshot(items: Iterable[Item | Mapping[str, Any]], *, at: datetime | None = None, tag: str = "") -> Snapshot:
    """A snapshot from perceived items; mappings are accepted so a caller can stay loose."""
    built: dict[str, Item] = {}
    for raw in items:
        item = raw if isinstance(raw, Item) else Item(**{k: v for k, v in raw.items() if k in Item.__dataclass_fields__})
        built[item.key] = item
    return Snapshot(at or _now(), built, tag)


def diff(before: Snapshot | None, after: Snapshot, *, policy: ChangePolicy | None = None,
         volatility: Volatility | None = None) -> Changes:
    """Typed differences between two snapshots.

    Unmatched keys are paired by label and position before being called appeared/disappeared, so a
    control whose identity shifted reads as a move rather than a death and a birth.
    """
    policy = policy or ChangePolicy()
    if before is None:
        return Changes(after.at, after.at, ())
    volatile = (lambda key: volatility.is_volatile(key)) if volatility is not None else (lambda key: False)
    changes: list[Change] = []
    gone = {k: v for k, v in before.items.items() if k not in after.items}
    fresh = {k: v for k, v in after.items.items() if k not in before.items}
    renamed = _pair_up(gone, fresh, policy)
    for key, now in after.items.items():
        was = before.items.get(key) or renamed.get(key)
        if was is None:
            changes.append(Change(WINDOW_OPENED if now.kind == "window" else APPEARED, key, after.at,
                                  now.label or now.value, now.window, None, now.value or now.label, volatile=volatile(key)))
            continue
        if was.value != now.value:
            changes.append(Change(VALUE, key, after.at, now.label or now.key, now.window, was.value, now.value, volatile=volatile(key)))
        if was.focused != now.focused:
            changes.append(Change(FOCUS, key, after.at, now.label or now.key, now.window,
                                  "focused" if was.focused else "not focused", "focused" if now.focused else "not focused",
                                  volatile=volatile(key)))
        if policy.report_moves and _shifted(was, now, policy.move_threshold):
            b, a = was.centre, now.centre
            changes.append(Change(MOVED, key, after.at, now.label or now.key, now.window, str(was.where), str(now.where),
                                  (a[0] - b[0], a[1] - b[1]) if a and b else None, volatile=volatile(key)))
    matched_back = set(renamed.values())
    for key, was in gone.items():
        if was in matched_back:
            continue
        changes.append(Change(WINDOW_CLOSED if was.kind == "window" else DISAPPEARED, key, after.at,
                              was.label or was.value, was.window, was.value or was.label, None, volatile=volatile(key)))
    changes = _mark_occlusions(_pair_replacements(_subsume_parts(_drop_consequent_moves(changes), before, after),
                                                  before, after),
                               before, _window_bounds(before, after))
    changes.sort(key=lambda c: (_ORDER.get(c.kind, 9), c.window, c.key))
    return Changes(before.at, after.at, tuple(changes))


def _drop_consequent_moves(changes: list[Change]) -> list[Change]:
    """A thing whose text changed and therefore shifted has not also moved.

    Text is laid out, so a longer word pushes its own box sideways. Reporting that as a move is how
    a clock ticking over produces two pieces of news instead of one — and the move, unlike the tick,
    is not recognisably volatile, so it survives into the answer as pure noise.
    """
    said = {c.key for c in changes if c.kind in (VALUE, REPLACED)}
    return [c for c in changes if not (c.kind == MOVED and c.key in said)]


def _subsume_parts(changes: list[Change], before: Snapshot, after: Snapshot) -> list[Change]:
    """A window opening is one change, not one per control inside it.

    Perception reports parts; a mind should notice the whole. If every item of a window is new (the
    region did not exist before), the parts are folded into a single window change carrying how
    many came with it. A window that was already there keeps reporting its parts individually,
    because then the parts are the news.
    """
    had = _by_window(before)
    has = _by_window(after)
    wholes: dict[str, str] = {}
    for window, now_keys in has.items():
        if not window:
            continue
        if not had.get(window) and now_keys:
            wholes[window] = APPEARED
    for window, was_keys in had.items():
        if not window:
            continue
        if not has.get(window) and was_keys:
            wholes[window] = DISAPPEARED
    if not wholes:
        return changes
    # where each new or departing whole is, so parts the structure did not attribute can be placed
    # inside it by geometry — the spatial frame doing part-whole binding the DOM did not
    bounds = {w: _bounds(after if v == APPEARED else before, w) for w, v in wholes.items()}
    kept: list[Change] = []
    folded: dict[str, int] = {}
    for change in changes:
        verdict = wholes.get(change.window)
        if verdict == APPEARED and change.kind in (APPEARED, WINDOW_OPENED):
            folded[change.window] = folded.get(change.window, 0) + 1
            continue
        if verdict == DISAPPEARED and change.kind in (DISAPPEARED, WINDOW_CLOSED):
            folded[change.window] = folded.get(change.window, 0) + 1
            continue
        if not change.window and change.kind in (APPEARED, DISAPPEARED):
            side = after if change.kind == APPEARED else before
            item = side.items.get(change.key)
            inside = _containing(bounds, item, wholes, APPEARED if change.kind == APPEARED else DISAPPEARED)
            if inside is not None:
                folded[inside] = folded.get(inside, 0) + 1
                continue
        kept.append(change)
    for window, count in sorted(folded.items()):
        kind = WINDOW_OPENED if wholes[window] == APPEARED else WINDOW_CLOSED
        at = after.at
        kept.append(Change(kind, f"window:{window}", at, window, window,
                           None if kind == WINDOW_OPENED else window, window if kind == WINDOW_OPENED else None,
                           brought=count - 1 if count > 1 else 0))
    return kept


def _pair_replacements(changes: list[Change], before: Snapshot, after: Snapshot, slack: int = 6) -> list[Change]:
    """One thing standing where another stood is a substitution, not a death and a birth.

    A top bar that read "Terminal" and now reads "Text Editor" has not lost one label and gained
    another: the same slot now says something else. Reporting it as two changes is how a diff turns
    one fact into noise.
    """
    gone = [c for c in changes if c.kind == DISAPPEARED]
    fresh = [c for c in changes if c.kind == APPEARED]
    if not gone or not fresh:
        return changes
    used: set[str] = set()
    pairs: dict[str, Change] = {}
    for new_change in fresh:
        now_item = after.items.get(new_change.key)
        if now_item is None or now_item.where is None:
            continue
        for old_change in gone:
            if old_change.key in used:
                continue
            was_item = before.items.get(old_change.key)
            if was_item is None or was_item.where is None or was_item.kind != now_item.kind:
                continue
            # a text slot holding a longer word is wider: position and height identify the slot, width does not
            wx, wy, _, wh = was_item.where
            nx, ny, _, nh = now_item.where
            if abs(wx - nx) <= slack and abs(wy - ny) <= slack and abs(wh - nh) <= slack:
                used.add(old_change.key)
                pairs[new_change.key] = Change(REPLACED, new_change.key, new_change.at, now_item.label or new_change.label,
                                               now_item.window, old_change.label or old_change.was,
                                               new_change.label or new_change.now, volatile=new_change.volatile)
                break
    if not pairs:
        return changes
    out = []
    for change in changes:
        if change.kind == DISAPPEARED and change.key in used:
            continue
        out.append(pairs.get(change.key, change) if change.kind == APPEARED else change)
    return out


def _window_bounds(before: Snapshot, after: Snapshot) -> dict[str, tuple[int, int, int, int]]:
    """Where each window that has just appeared now sits."""
    had, has = _by_window(before), _by_window(after)
    opened = [w for w, keys in has.items() if w and keys and not had.get(w)]
    return {w: b for w in opened if (b := _bounds(after, w)) is not None}


def _mark_occlusions(changes: list[Change], before: Snapshot, opened: Mapping[str, tuple[int, int, int, int]]) -> list[Change]:
    """A thing under a window that just opened is covered, not destroyed.

    Perception cannot tell "closed" from "hidden behind": both are simply absent. Geometry can,
    and getting it wrong is how an agent comes to believe a window was closed when the user merely
    opened another one on top of it.
    """
    if not opened:
        return changes
    out: list[Change] = []
    for change in changes:
        if change.kind in (DISAPPEARED, WINDOW_CLOSED):
            item = before.items.get(change.key) or next((i for i in before.items.values() if i.window == change.window), None)
            box = _bounds(before, change.window) if change.kind == WINDOW_CLOSED else (item.where if item else None)
            coverer = _covered_by(box, opened)
            if coverer is not None:
                out.append(Change(OCCLUDED, change.key, change.at, change.label, change.window,
                                  was="in view", now=coverer, volatile=change.volatile, brought=change.brought))
                continue
        out.append(change)
    return out


def _covered_by(box: tuple[int, int, int, int] | None, opened: Mapping[str, tuple[int, int, int, int]]) -> str | None:
    """Which newly opened window covers most of a box, if one covers most of it."""
    if box is None:
        return None
    x, y, w, h = box
    area = max(1, w * h)
    for window, (ox, oy, ow, oh) in sorted(opened.items()):
        overlap = max(0, min(x + w, ox + ow) - max(x, ox)) * max(0, min(y + h, oy + oh) - max(y, oy))
        if overlap / area >= 0.6:
            return window
    return None


def _bounds(snap: Snapshot, window: str) -> tuple[int, int, int, int] | None:
    """The rectangle a window's known parts occupy."""
    boxes = [i.where for i in snap.items.values() if i.window == window and i.where is not None]
    if not boxes:
        return None
    left = min(b[0] for b in boxes)
    top = min(b[1] for b in boxes)
    right = max(b[0] + b[2] for b in boxes)
    bottom = max(b[1] + b[3] for b in boxes)
    return (left, top, right - left, bottom - top)


def _containing(bounds: Mapping[str, tuple[int, int, int, int] | None], item: Item | None,
                wholes: Mapping[str, str], want: str, margin: int = 24) -> str | None:
    """Which appearing/departing whole a stray part sits inside, if any."""
    if item is None or item.where is None:
        return None
    centre = item.centre
    if centre is None:
        return None
    for window, box in bounds.items():
        if box is None or wholes.get(window) != want:
            continue
        x, y, w, h = box
        if (x - margin) <= centre[0] <= (x + w + margin) and (y - margin) <= centre[1] <= (y + h + margin):
            return window
    return None


def _by_window(snap: Snapshot) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for key, item in snap.items.items():
        out.setdefault(item.window, set()).add(key)
    return out


_ORDER = {WINDOW_OPENED: 0, WINDOW_CLOSED: 1, REPLACED: 2, APPEARED: 3, DISAPPEARED: 4, OCCLUDED: 5, VALUE: 6, FOCUS: 7, MOVED: 8}


def _shifted(was: Item, now: Item, threshold: int) -> bool:
    a, b = was.centre, now.centre
    if a is None or b is None:
        return False
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) >= threshold


def _pair_up(gone: Mapping[str, Item], fresh: Mapping[str, Item], policy: ChangePolicy) -> dict[str, Item]:
    """Match disappeared items to appeared ones that are plainly the same thing under a new key."""
    pairs: dict[str, Item] = {}
    taken: set[str] = set()
    for new_key, now in sorted(fresh.items()):
        best, best_cost = None, None
        for old_key, was in sorted(gone.items()):
            if old_key in taken or was.kind != now.kind or was.window != now.window:
                continue
            if was.label != now.label or not was.label:
                continue
            a, b = was.centre, now.centre
            cost = (abs(a[0] - b[0]) + abs(a[1] - b[1])) if a and b else 0
            if best_cost is None or cost < best_cost:
                best, best_cost = old_key, cost
        if best is not None:
            taken.add(best)
            pairs[new_key] = gone[best]
    return pairs


@dataclass
class Watcher:
    """Snapshots over time, so "what changed since X" can be asked of any earlier moment."""

    policy: ChangePolicy = field(default_factory=ChangePolicy)
    keep: int = 40
    volatility: Volatility = field(init=False)
    history: list[Snapshot] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.volatility = Volatility(self.policy)

    def see(self, items: Iterable[Item | Mapping[str, Any]], *, tag: str = "", at: datetime | None = None) -> Changes:
        """Take a snapshot, learn what is volatile, and return what changed since the last one."""
        snap = snapshot(items, at=at, tag=tag)
        previous = self.history[-1] if self.history else None
        if previous is not None:
            self.volatility.watch(previous, snap)
        changes = diff(previous, snap, policy=self.policy, volatility=self.volatility)
        self.history.append(snap)
        del self.history[: max(0, len(self.history) - self.keep)]
        return changes

    def since(self, tag: str) -> Changes | None:
        """What changed between the snapshot with that tag and the latest one."""
        marked = next((s for s in reversed(self.history) if s.tag == tag), None)
        if marked is None or not self.history:
            return None
        return diff(marked, self.history[-1], policy=self.policy, volatility=self.volatility)

    def since_first(self, tag: str) -> Changes | None:
        """What changed since the FIRST snapshot carrying that tag — a turn boundary, not the latest frame."""
        marked = next((s for s in self.history if s.tag == tag), None)
        if marked is None or not self.history:
            return None
        return diff(marked, self.history[-1], policy=self.policy, volatility=self.volatility)

    def latest(self) -> Snapshot | None:
        return self.history[-1] if self.history else None

    def tagged(self, tag: str) -> Snapshot | None:
        return next((s for s in reversed(self.history) if s.tag == tag), None)


# ------------------------------------------------------------------ from claims


PERCEPT_PREDICATES = ("label", "reads", "value", "shows", "checked", "current", "announces")


def attribute_windows(items: Sequence[Item], *, margin: int = 8) -> list[Item]:
    """Give window-less items the window whose area they sit in.

    The scene graph attributes an item to a window only when the DOM says so, and it does not say
    so for window chrome: a title bar, a Close button and a scrollbar all come through with no
    window at all. Geometry knows better. Without this, anything that asks a question *about a
    window* — what closed, what was covered — can only see the minority of items the DOM labelled,
    which is how a closed window leaves most of its contents behind as still-existing objects.
    """
    bounds: dict[str, tuple[int, int, int, int]] = {}
    for window in {i.window for i in items if i.window}:
        boxes = [i.where for i in items if i.window == window and i.where is not None]
        if not boxes:
            continue
        left, top = min(b[0] for b in boxes), min(b[1] for b in boxes)
        right, bottom = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
        bounds[window] = (left, top, right - left, bottom - top)
    if not bounds:
        return list(items)
    out: list[Item] = []
    for item in items:
        if item.window or item.where is None or item.centre is None:
            out.append(item)
            continue
        cx, cy = item.centre
        inside = [(w * h, name) for name, (x, y, w, h) in bounds.items()
                  if (x - margin) <= cx <= (x + w + margin) and (y - margin) <= cy <= (y + h + margin)]
        # the smallest containing window: a dialog inside a window belongs to the dialog
        out.append(replace(item, window=min(inside)[1]) if inside else item)
    return out


def items_from_claims(records: Sequence[Any], entity_of: Any) -> list[Item]:
    """Build snapshot items from perceptual claims (the vocabulary ``parse``-to-scene-graph emits).

    One item per perceived subject: its label, whatever value it carries, where it is, and which
    window it belongs to. Callers with a different vocabulary can build ``Item``s themselves.
    """
    by_subject: dict[str, dict[str, Any]] = {}
    for rec in records:
        claim = rec.claim
        if claim.predicate not in PERCEPT_PREDICATES:
            continue
        row = by_subject.setdefault(claim.subject.id, {"key": claim.subject.id})
        entity = entity_of(claim.subject)
        box = getattr(entity, "box", None)
        if box is not None and "where" not in row:
            row["where"] = tuple(int(v) for v in box)
        section = getattr(entity, "section", None)
        if section and "window" not in row:
            row["window"] = str(section)
        role = getattr(entity, "role", None)
        row["kind"] = "text" if claim.predicate in ("reads", "announces") else ("window" if role == "window" else "control")
        if claim.predicate == "label":
            row["label"] = str(claim.object)
        elif claim.predicate == "current":
            row["focused"] = bool(claim.object)
        else:
            row["value"] = str(getattr(claim.object, "value", claim.object))
    return [Item(**{k: v for k, v in row.items() if k in Item.__dataclass_fields__}) for row in by_subject.values()]
