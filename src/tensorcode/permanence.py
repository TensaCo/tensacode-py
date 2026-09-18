"""Object permanence: a thing that leaves view is occluded, not destroyed.

Snapshot scopes retract what is no longer perceived, which is right for *beliefs about now* and
wrong for *objects*. A window scrolled behind another still exists; a file still exists when the
terminal is cleared. Without permanence an agent cannot say "the Files window is still open, I
just can't see it", and identity does not survive occlusion.

The trap is that naive permanence re-introduces the stale-belief bug that retraction was there to
prevent. So an object file keeps two different things apart:

    exists          the object is known to exist (survives leaving view)
    present         it was perceived in the latest observation (does not)

Those two are independent, which is the point: three states, not two. In view; out of view but
believed to exist (occluded, scrolled away, behind another window); and *gone* — positively known
not to exist any more, because the window it lived in was seen to close. Only evidence of
destruction moves an object to the third state; simply not seeing it never does.

and every attribute is dated. Asking for an attribute of an absent object gets the value, when it
was last seen, and ``stale=True`` — never a bare assertion about the present. ``assertable``
exists so a caller can refuse to act on a stale attribute at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

from .change import Item, Snapshot
from .records import Claim, Evidence, Ref, Store

OBJECTS = Ref("scope:objects")

# what an object file writes to the store that can change from one look to the next
_CHANGING = frozenset({"last_seen", "in_view", "state", "exists"})


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Attribute:
    """A value, when it was seen, and whether that is old news."""

    name: str
    value: Any
    seen_at: datetime
    stale: bool

    def describe(self) -> str:
        when = self.seen_at.isoformat(timespec="seconds")
        return f"{self.name}={self.value!r}" + (f" (as of {when}, not visible now)" if self.stale else "")


@dataclass
class ObjectFile:
    """What is known about one thing, across every time it has been seen."""

    ref: Ref
    key: str
    kind: str = "control"
    label: str = ""
    window: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    seen_at: dict[str, datetime] = field(default_factory=dict)
    first_seen: datetime = field(default_factory=_now)
    last_seen: datetime = field(default_factory=_now)
    times_seen: int = 1
    present: bool = True
    exists: bool = True  # positively known to be gone only when its container was seen to close
    gone_at: datetime | None = None
    gone_because: str = ""
    returns: int = 0  # how many times it came back after going away

    @property
    def state(self) -> str:
        """in_view | out_of_view | gone — the distinction a snapshot scope cannot make."""
        if not self.exists:
            return "gone"
        return "in_view" if self.present else "out_of_view"

    def attribute(self, name: str) -> Attribute | None:
        if name not in self.attributes:
            return None
        return Attribute(name, self.attributes[name], self.seen_at.get(name, self.last_seen), not self.present)

    def assertable(self, name: str, *, within: timedelta | None = None, now: datetime | None = None) -> bool:
        """May a caller state this attribute as current? Only if seen now, or fresh enough to risk."""
        if name not in self.attributes:
            return False
        if not self.exists:
            return False  # an attribute of a destroyed object is never current
        if self.present:
            return True
        if within is None:
            return False
        return ((now or _now()) - self.seen_at.get(name, self.last_seen)) <= within

    def describe(self) -> str:
        if not self.exists:
            state = f"gone since {(self.gone_at or self.last_seen).isoformat(timespec='seconds')}"
            state += f" ({self.gone_because})" if self.gone_because else ""
        elif self.present:
            state = "in view"
        else:
            state = f"not in view since {self.last_seen.isoformat(timespec='seconds')}"
        return f"{self.label or self.key} ({self.kind}{', ' + self.window if self.window else ''}) — {state}"


@dataclass(frozen=True)
class ObservationReport:
    """What one observation did to the registry, so permanence can be measured."""

    seen: int = 0
    new: int = 0
    rematched: int = 0  # same object under a different key
    returned: int = 0  # was absent, is present again
    absent: int = 0
    identities_kept: int = 0  # objects that kept their file across this observation
    wrong_about_gone: int = 0  # we had written something off, and there it was


class Objects:
    """A registry of object files, updated by observation.

    Identity is by key when the key is stable, and by (kind, window, name) when it is not — the
    scene-graph keys carry an occurrence index, so a list losing an earlier row renames every row
    after it. Matching on the name first is what keeps a file attached to its object.

    What counts as the name depends on what kind of thing it is, and getting this wrong is how a
    measurement of permanence comes back empty. A control is named by its label: its value changes
    while it stays the same control. A *line of read-only text* has no label and no life apart from
    its content — the line "report.pdf" in a terminal **is** that content, and its key is only its
    position in a scrolling region. Identify such an item by its value, or a file follows the slot
    instead of the object, and a line that scrolls away looks like a line whose text changed.
    """

    def __init__(self, *, forget_absent_after: timedelta | None = None) -> None:
        # keyed by the file's own id, never by a screen key: screen keys are recycled, and a
        # registry keyed by them loses one object every time another takes over its slot
        self.files: dict[str, ObjectFile] = {}
        self._by_key: dict[str, str] = {}
        self._by_identity: dict[tuple[str, str, str], str] = {}
        self.forget_absent_after = forget_absent_after
        self._counter = 0

    # ------------------------------------------------------------- observing

    def observe(self, snap: Snapshot) -> ObservationReport:
        """Update from one snapshot: what is here, what is merely not visible, what is new."""
        seen: set[str] = set()
        new = rematched = returned = wrong_about_gone = 0
        for key, item in snap.items.items():
            file = self._match(item, key)
            if file is None:
                file = self._create(item, snap.at)
                new += 1
            else:
                if file.key != key:
                    rematched += 1
                    self._rekey(file, key)
                if not file.present:
                    returned += 1
                    file.returns += 1
                if not file.exists:  # seeing it is stronger evidence than our belief that it was gone
                    file.exists, file.gone_at, file.gone_because = True, None, ""
                    wrong_about_gone += 1
                file.times_seen += 1
            file.present = True
            file.last_seen = snap.at
            file.label = item.label or file.label
            file.window = item.window or file.window
            self._by_key[item.key] = file.ref.id
            self._by_identity[(file.kind, file.window, _name_of(item))] = file.ref.id
            for name, value in _attributes(item).items():
                if file.attributes.get(name) != value or name not in file.seen_at:
                    file.attributes[name] = value
                file.seen_at[name] = snap.at
            seen.add(file.ref.id)
        absent = 0
        for fid, file in list(self.files.items()):
            if fid in seen:
                continue
            if file.present:
                file.present = False
            if file.exists:
                absent += 1  # gone objects are not "absent": we are not waiting for them to come back
            if self.forget_absent_after is not None and (snap.at - file.last_seen) > self.forget_absent_after:
                self._drop(file)
        return ObservationReport(len(snap.items), new, rematched, returned, absent,
                                 identities_kept=len(snap.items) - new, wrong_about_gone=wrong_about_gone)

    def closed(self, window: str, *, at: datetime | None = None, why: str = "its window closed") -> int:
        """A window was *seen* to close: what lived in it is gone, not merely out of sight.

        This is the only way an object stops existing. Absence never implies it, because absence is
        what occlusion looks like; a close event is evidence of destruction and nothing else is.
        """
        when = at or _now()
        marked = 0
        for file in self.files.values():
            if file.window != window or not file.exists:
                continue
            file.exists, file.present = False, False
            file.gone_at, file.gone_because = when, why
            marked += 1
        return marked

    def gone(self) -> list[ObjectFile]:
        return sorted((f for f in self.files.values() if not f.exists), key=lambda f: f.key)

    # ------------------------------------------------------------- reading

    def get(self, key: str) -> ObjectFile | None:
        """The file for a screen key, or for a file id."""
        fid = self._by_key.get(key)
        return self.files.get(fid) if fid else self.files.get(key)

    def find(self, label: str, *, kind: str | None = None, window: str | None = None) -> list[ObjectFile]:
        rows = [f for f in self.files.values() if f.label == label
                and (kind is None or f.kind == kind) and (window is None or f.window == window)]
        return sorted(rows, key=lambda f: (not f.present, f.key))

    def present(self) -> list[ObjectFile]:
        return sorted((f for f in self.files.values() if f.present), key=lambda f: f.key)

    def absent(self) -> list[ObjectFile]:
        """Out of view but believed to exist — not the same set as ``gone``."""
        return sorted((f for f in self.files.values() if not f.present and f.exists), key=lambda f: f.key)

    def known(self) -> list[ObjectFile]:
        return sorted(self.files.values(), key=lambda f: f.key)

    def windows(self, *, including_absent: bool = True) -> list[str]:
        rows = self.files.values() if including_absent else self.present()
        return sorted({f.window for f in rows if f.window})

    # ------------------------------------------------------------- claims

    def remember(self, mind: Store, *, source: str = "obs:objects", at: datetime | None = None) -> int:
        """Write existence and identity into a non-snapshot scope, so they survive retraction.

        Only existence and identity go here. Attributes stay on the file, dated, because writing
        them as plain claims is exactly how a stale belief would get asserted as current.

        The handful that do change — whether it is in view, when it was last seen — are *replaced*,
        not appended. Appending looks harmless and is not: a claim per object per turn is a leak
        with a scope name, and on a desktop of a few hundred objects it outgrows perception itself
        within a few dozen turns.
        """
        when = at or _now()
        written = 0
        for file in self.known():
            for predicate, value in (("exists", file.exists), ("is_a", file.kind), ("label", file.label),
                                     ("last_seen", file.last_seen), ("in_view", file.present),
                                     ("state", file.state)):
                if value == "" or value is None:
                    continue
                live = [r for r in mind.claims(file.ref, predicate, scope=OBJECTS) if not r.retracted]
                if any(r.claim.object == value for r in live):
                    continue
                if predicate in _CHANGING and live:  # supersede: one current value, no history kept
                    mind.forget([r.id for r in live])
                mind.tell(Claim(file.ref, predicate, value, scope=OBJECTS),
                          Evidence(Ref(source), when, method="object-permanence"))
                written += 1
        return written

    # ------------------------------------------------------------- internals

    def _create(self, item: Item, at: datetime) -> ObjectFile:
        self._counter += 1
        file = ObjectFile(Ref(f"object:{self._counter}"), item.key, item.kind, item.label, item.window,
                          first_seen=at, last_seen=at)
        self.files[file.ref.id] = file
        self._by_key[item.key] = file.ref.id
        self._by_identity[(item.kind, item.window, _name_of(item))] = file.ref.id
        return file

    def _match(self, item: Item, key: str) -> ObjectFile | None:
        """The file this item belongs to, if any.

        Name before key, and never a key whose file names something else: in a scrolling region the
        keys are recycled, so trusting the key first hands one object's file to the next occupant of
        its slot and quietly overwrites what was known about it.
        """
        named = self._by_name(item)
        if named is not None:
            return named
        held = self.files.get(self._by_key.get(key, ""))
        if held is None:
            return None
        name, held_name = _name_of(item), _identity_name(held)
        if name and held_name and name != held_name:
            return None
        return held

    def _by_name(self, item: Item) -> ObjectFile | None:
        name = _name_of(item)
        if not name:
            return None
        fid = self._by_identity.get((item.kind, item.window, name))
        return self.files.get(fid) if fid else None

    def _rekey(self, file: ObjectFile, key: str) -> None:
        """The same object under a new screen key. Only the index moves; the file stays put."""
        if self._by_key.get(file.key) == file.ref.id:
            self._by_key.pop(file.key, None)
        file.key = key
        self._by_key[key] = file.ref.id

    def _drop(self, file: ObjectFile) -> None:
        self.files.pop(file.ref.id, None)
        if self._by_key.get(file.key) == file.ref.id:
            self._by_key.pop(file.key, None)
        if self._by_identity.get((file.kind, file.window, _identity_name(file))) == file.ref.id:
            self._by_identity.pop((file.kind, file.window, _identity_name(file)), None)


def _name_of(item: Item) -> str:
    """What identifies this item across observations (see ``Objects``)."""
    if item.label:
        return item.label
    return item.value if item.kind == "text" else ""


def _identity_name(file: ObjectFile) -> str:
    if file.label:
        return file.label
    return str(file.attributes.get("value", "")) if file.kind == "text" else ""


def _attributes(item: Item) -> Mapping[str, Any]:
    out: dict[str, Any] = {}
    if item.value != "":
        out["value"] = item.value
    if item.where is not None:
        out["where"] = item.where
    out["focused"] = item.focused
    return out


def still_there(objects: Objects, label: str) -> str:
    """A sentence an agent can say honestly about something not in view."""
    files = objects.find(label)
    if not files:
        return f"I have no record of {label}."
    file = files[0]
    if file.present:
        return f"{label} is on screen now."
    if not file.exists:
        return (f"{label} is gone — {file.gone_because} at "
                f"{(file.gone_at or file.last_seen).isoformat(timespec='seconds')}.")
    return (f"{label} was there when I last saw it "
            f"({file.last_seen.isoformat(timespec='seconds')}); I can't see it now, so I can't say it still is.")


def describe_attribute(objects: Objects, label: str, name: str) -> str:
    """Report an attribute without pretending an old reading is a current one."""
    files = objects.find(label)
    if not files:
        return f"I have no record of {label}."
    attribute = files[0].attribute(name)
    if attribute is None:
        return f"I never noted the {name} of {label}."
    return f"{label}: {attribute.describe()}"


def known_objects(snapshots: Iterable[Snapshot]) -> Objects:
    """Replay snapshots into a fresh registry (used by measurements and tests)."""
    objects = Objects()
    for snap in snapshots:
        objects.observe(snap)
    return objects
