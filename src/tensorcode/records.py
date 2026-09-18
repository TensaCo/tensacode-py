"""Typed native values by default; explicit graph records where identity matters.

The world/state substrate has exactly two record kinds:

* an **entity**: a stable ``Ref`` bound to an ordinary typed Python value
  (dataclass, Pydantic model, enum, primitive). Identity is *assigned*.
* a **claim**: a proposition ``(subject, predicate, object)`` qualified by the
  interval in which it is asserted to hold and an optional scope. Identity is
  *content-derived*: the same proposition from two sources is one claim with
  two pieces of evidence.

Observations, documents, sources, hypotheses, and contexts are entities whose
values have domain types. Evidence, intervals, and scores are plain values.
Executable structure (plans) lives in ``actions.py``, not here: a relationship
between facts is not a scheduling dependency.

Serialization only reconstructs types that were explicitly registered. Unknown
type names come back as ``Opaque`` values; nothing is ever imported by name.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import cached_property
from typing import Any, Callable, Iterable, Mapping, Sequence

from .outcomes import Score

# --------------------------------------------------------------------------- refs


@dataclass(frozen=True, order=True)
class Ref:
    """A stable reference: ``"<kind>:<name>"``. Equality is identity."""

    id: str

    def __post_init__(self) -> None:
        kind, sep, name = self.id.partition(":")
        if not (kind and sep and name):
            raise ValueError(f"Ref must look like 'kind:name', got {self.id!r}")

    @property
    def kind(self) -> str:
        return self.id.partition(":")[0]

    def __str__(self) -> str:
        return self.id


@dataclass(frozen=True)
class Interval:
    """Closed interval; ``None`` means unbounded on that side."""

    start: datetime | None = None
    end: datetime | None = None

    def __post_init__(self) -> None:
        if self.start and self.end and self.end < self.start:
            raise ValueError("interval end precedes start")

    @classmethod
    def at(cls, t: datetime) -> Interval:
        return cls(t, t)

    def contains(self, t: datetime) -> bool:
        return (self.start is None or self.start <= t) and (self.end is None or t <= self.end)

    def overlap(self, other: Interval) -> Interval | None:
        starts = [s for s in (self.start, other.start) if s is not None]
        ends = [e for e in (self.end, other.end) if e is not None]
        start, end = (max(starts) if starts else None), (min(ends) if ends else None)
        if start and end and end < start:
            return None
        return Interval(start, end)


@dataclass(frozen=True)
class Evidence:
    """Why a claim is on record: which source said it, when, and how it was extracted."""

    source: Ref
    observed_at: datetime  # when the source made the observation (not when the fact holds)
    locator: str | None = None  # where inside the source, e.g. "text[18:33]"
    method: str | None = None  # e.g. "snmp-poll", "parse:note-rules@1"
    confidence: Score | None = None  # the extractor's or source's own score, with its kind
    derived_from: tuple[str, ...] = ()  # premise claim ids, when a rule derived this; retracting a premise withdraws it


@dataclass(frozen=True)
class Claim:
    subject: Ref
    predicate: str
    object: Any  # a Ref or an encodable typed value
    valid: Interval = Interval()
    scope: Ref | None = None  # None: the shared world; otherwise a hypothesis/context entity

    @cached_property
    def id(self) -> str:
        """Content identity: equal propositions share an id regardless of source."""
        canonical = json.dumps(_canonical(self), sort_keys=True, separators=(",", ":"))
        return "claim:" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _canonical(v: Any) -> Any:
    """Deterministic, registry-free form used only for hashing (never for reconstruction)."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, Ref):
        return {"$ref": v.id}
    if isinstance(v, datetime):
        return {"$datetime": v.isoformat()}
    if isinstance(v, date):
        return {"$date": v.isoformat()}
    if isinstance(v, enum.Enum):
        return {"$enum": f"{type(v).__module__}.{type(v).__qualname__}", "value": _canonical(v.value)}
    if isinstance(v, (list, tuple)):
        return [_canonical(x) for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted((_canonical(x) for x in v), key=lambda x: json.dumps(x, sort_keys=True))
    if isinstance(v, Mapping):
        return {"$map": sorted([[_canonical(k), _canonical(x)] for k, x in v.items()], key=json.dumps)}
    if dataclasses.is_dataclass(v) or _is_model(v):
        return {"$type": f"{type(v).__module__}.{type(v).__qualname__}", "fields": {k: _canonical(x) for k, x in _fields_of(v).items()}}
    raise EncodeError(f"claim objects must be values, got {type(v).__qualname__}")


@dataclass(frozen=True)
class Retraction:
    reason: str
    evidence: tuple[Evidence, ...] = ()


@dataclass
class ClaimRecord:
    claim: Claim
    evidence: list[Evidence] = field(default_factory=list)
    retracted: Retraction | None = None

    @property
    def id(self) -> str:
        return self.claim.id


@dataclass(frozen=True)
class Conflict:
    """Two live claims about a functional predicate that cannot both hold."""

    a: ClaimRecord
    b: ClaimRecord
    during: Interval


@dataclass(frozen=True)
class Var:
    name: str


_ANY = object()

# ---------------------------------------------------------------------- codec


class EncodeError(TypeError):
    pass


@dataclass(frozen=True)
class Opaque:
    """A value whose type name is not registered here. Preserved, never imported."""

    type_name: str
    data: Any


@dataclass
class ConversionReport:
    losses: list[str] = field(default_factory=list)
    opaque: list[str] = field(default_factory=list)

    @property
    def lossless(self) -> bool:
        return not self.losses and not self.opaque


@dataclass(frozen=True)
class Encoded:
    data: Any
    report: ConversionReport


@dataclass(frozen=True)
class _Registration:
    cls: type
    name: str
    identity: Callable[[Any], Any] | None  # entity types only


class TypeRegistry:
    """Explicit allow-list of types that may be encoded and reconstructed."""

    def __init__(self, *, parent: TypeRegistry | None = None) -> None:
        self._by_name: dict[str, _Registration] = {}
        self._by_cls: dict[type, _Registration] = {}
        self._parent = parent

    def register(
        self, cls: type, *, name: str | None = None, identity: str | Callable[[Any], Any] | None = None
    ) -> type:
        """Register a value type, or an entity type when ``identity`` is given.

        ``identity`` is a field name or a function returning a stable key.
        """
        if isinstance(identity, str):
            attr = identity
            identity = lambda obj: getattr(obj, attr)  # noqa: E731
        reg = _Registration(cls, name or cls.__name__, identity)
        if (existing := self.lookup_name(reg.name)) and existing.cls is not cls:
            raise ValueError(f"type name {reg.name!r} already registered for {existing.cls!r}")
        self._by_name[reg.name] = reg
        self._by_cls[cls] = reg
        return cls

    def lookup_name(self, name: str) -> _Registration | None:
        return self._by_name.get(name) or (self._parent.lookup_name(name) if self._parent else None)

    def lookup_cls(self, cls: type) -> _Registration | None:
        return self._by_cls.get(cls) or (self._parent.lookup_cls(cls) if self._parent else None)

    def is_entity(self, obj: Any) -> bool:
        reg = self.lookup_cls(type(obj))
        return bool(reg and reg.identity)


_BUILTINS = TypeRegistry()


def _is_model(obj: Any) -> bool:
    return hasattr(type(obj), "model_fields") and hasattr(obj, "model_dump")


def _fields_of(obj: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    if _is_model(obj):
        return {name: getattr(obj, name) for name in type(obj).model_fields}
    raise EncodeError(f"cannot enumerate fields of {type(obj).__qualname__}")


def encode(
    value: Any,
    registry: TypeRegistry,
    *,
    entity_ref: Callable[[Any], Ref] | None = None,
    root_path: str = "$",
    seen: dict[int, str] | None = None,
) -> Encoded:
    """Encode a value into JSON-compatible data.

    Cycles through *values* raise ``EncodeError`` (declare one of the types as an
    entity to break the cycle). Shared mutable sub-values are duplicated and the
    aliasing is reported as a loss.
    """
    report = ConversionReport()
    first_seen: dict[int, str] = {} if seen is None else seen  # shared across records by to_records
    stack: set[int] = set()

    def enc(v: Any, path: str) -> Any:
        if v is None or isinstance(v, (bool, int, float, str)):
            return v
        if isinstance(v, Ref):
            return {"$ref": v.id}
        if entity_ref is not None and path != root_path and registry.is_entity(v):
            return {"$ref": entity_ref(v).id}
        if isinstance(v, Opaque):
            report.opaque.append(f"{path}: {v.type_name}")
            return {"$type": v.type_name, "fields": v.data}
        if isinstance(v, datetime):
            return {"$datetime": v.isoformat()}
        if isinstance(v, date):
            return {"$date": v.isoformat()}
        mutable = isinstance(v, (list, dict, set)) or dataclasses.is_dataclass(v) or _is_model(v)
        if id(v) in stack:
            raise EncodeError(f"cycle through value at {path}; register one of its types as an entity")
        if mutable and id(v) in first_seen:
            report.losses.append(f"aliasing: {path} is the same object as {first_seen[id(v)]}")
        elif mutable:
            first_seen[id(v)] = path
        stack.add(id(v))
        try:
            if isinstance(v, enum.Enum):
                reg = registry.lookup_cls(type(v)) or _BUILTINS.lookup_cls(type(v))
                if not reg:
                    raise EncodeError(f"unregistered enum {type(v).__qualname__} at {path}")
                return {"$enum": reg.name, "value": v.value}
            if isinstance(v, (list, tuple)):
                items = [enc(x, f"{path}[{i}]") for i, x in enumerate(v)]
                return items if isinstance(v, list) else {"$tuple": items}
            if isinstance(v, (set, frozenset)):
                items = sorted((enc(x, f"{path}{{}}") for x in v), key=lambda x: json.dumps(x, sort_keys=True))
                return {"$set": items}
            if isinstance(v, Mapping):
                if all(isinstance(k, str) and not k.startswith("$") for k in v):
                    return {k: enc(x, f"{path}.{k}") for k, x in v.items()}
                return {"$map": [[enc(k, f"{path}<key>"), enc(x, f"{path}[{k!r}]")] for k, x in v.items()]}
            if dataclasses.is_dataclass(v) or _is_model(v):
                reg = registry.lookup_cls(type(v)) or _BUILTINS.lookup_cls(type(v))
                if not reg:
                    raise EncodeError(f"unregistered type {type(v).__qualname__} at {path}")
                return {"$type": reg.name, "fields": {k: enc(x, f"{path}.{k}") for k, x in _fields_of(v).items()}}
            raise EncodeError(f"no encoding for {type(v).__qualname__} at {path}")
        finally:
            stack.discard(id(v))

    return Encoded(enc(value, root_path), report)


def decode(data: Any, registry: TypeRegistry, *, resolve: Callable[[Ref], Any] | None = None) -> tuple[Any, ConversionReport]:
    """Reconstruct a value. Only registered types are instantiated; others become ``Opaque``."""
    report = ConversionReport()

    def dec(d: Any, path: str) -> Any:
        if d is None or isinstance(d, (bool, int, float, str)):
            return d
        if isinstance(d, list):
            return [dec(x, f"{path}[{i}]") for i, x in enumerate(d)]
        if not isinstance(d, dict):
            raise EncodeError(f"malformed data at {path}")
        if "$ref" in d:
            ref = Ref(d["$ref"])
            return resolve(ref) if resolve else ref
        if "$datetime" in d:
            return datetime.fromisoformat(d["$datetime"])
        if "$date" in d:
            return date.fromisoformat(d["$date"])
        if "$tuple" in d:
            return tuple(dec(x, path) for x in d["$tuple"])
        if "$set" in d:
            return frozenset(dec(x, path) for x in d["$set"])
        if "$map" in d:
            return {dec(k, path): dec(v, path) for k, v in d["$map"]}
        if "$enum" in d:
            reg = registry.lookup_name(d["$enum"]) or _BUILTINS.lookup_name(d["$enum"])
            if not reg:
                report.opaque.append(f"{path}: {d['$enum']}")
                return Opaque(d["$enum"], d["value"])
            return reg.cls(d["value"])
        if "$type" in d:
            reg = registry.lookup_name(d["$type"]) or _BUILTINS.lookup_name(d["$type"])
            if not reg:
                report.opaque.append(f"{path}: {d['$type']}")
                return Opaque(d["$type"], d["fields"])
            fields = {k: dec(v, f"{path}.{k}") for k, v in d["fields"].items()}
            if hasattr(reg.cls, "model_validate"):
                return reg.cls.model_validate(fields)
            init = {f.name for f in dataclasses.fields(reg.cls) if f.init}
            obj = reg.cls(**{k: v for k, v in fields.items() if k in init})
            for k, v in fields.items():
                if k not in init:
                    object.__setattr__(obj, k, v)
            return obj
        return {k: dec(v, f"{path}.{k}") for k, v in d.items()}

    return dec(data, "$"), report


for _cls in (Interval, Evidence, Claim, Score, Retraction):
    _BUILTINS.register(_cls)

# --------------------------------------------------------------------- rewrites


@dataclass(frozen=True)
class Put:
    ref: Ref
    value: Any


@dataclass(frozen=True)
class SetField:
    ref: Ref
    path: tuple[str | int, ...]
    value: Any


@dataclass(frozen=True)
class Tell:
    claim: Claim
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class Retract:
    claim_id: str
    reason: str
    evidence: tuple[Evidence, ...] = ()


Edit = Put | SetField | Tell | Retract


@dataclass(frozen=True)
class Patch:
    """A *proposed* change. Inert until a store applies it."""

    edits: tuple[Edit, ...]
    base_revision: int
    rationale: str = ""


@dataclass(frozen=True)
class Commit:
    revision: int
    patch: Patch
    added: tuple[str, ...] = ()  # claim ids that became live (new or revived)
    retracted: tuple[str, ...] = ()  # claim ids that stopped being live, including withdrawn derivations


class StaleRevision(RuntimeError):
    pass


def _replace_path(value: Any, path: Sequence[str | int], new: Any) -> Any:
    """Copy-on-write update of a nested field."""
    if not path:
        return new
    head, rest = path[0], path[1:]
    if isinstance(value, list):
        copy = list(value)
        copy[head] = _replace_path(value[head], rest, new)  # type: ignore[index]
        return copy
    if isinstance(value, dict):
        return {**value, head: _replace_path(value[head], rest, new)}
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(value, **{head: _replace_path(getattr(value, head), rest, new)})  # type: ignore[arg-type]
    if _is_model(value):
        return value.model_copy(update={head: _replace_path(getattr(value, head), rest, new)})
    raise TypeError(f"cannot update field {head!r} of {type(value).__qualname__}")


# ------------------------------------------------------------------------ store


@dataclass(frozen=True)
class Subgraph:
    entities: dict[Ref, Any]
    claims: list[ClaimRecord]


class Store:
    """An in-memory world/state graph. A reference implementation, not a database."""

    def __init__(self, registry: TypeRegistry | None = None) -> None:
        self.registry = registry or TypeRegistry()
        self.revision = 0
        self.entities: dict[Ref, Any] = {}
        self.functional: set[str] = set()
        self._claims: dict[str, ClaimRecord] = {}
        self._by_subject: dict[Ref, set[str]] = defaultdict(set)
        self._by_object: dict[Ref, set[str]] = defaultdict(set)
        self._by_predicate: dict[str, set[str]] = defaultdict(set)
        self._by_scope: dict[Ref | None, set[str]] = defaultdict(set)
        self._dependents: dict[str, set[str]] = defaultdict(set)  # premise id -> ids of claims derived from it

    def _index(self, rec: ClaimRecord) -> None:
        c = rec.claim
        self._by_subject[c.subject].add(rec.id)
        self._by_predicate[c.predicate].add(rec.id)
        self._by_scope[c.scope].add(rec.id)
        if isinstance(c.object, Ref):
            self._by_object[c.object].add(rec.id)

    def _unindex(self, rec: ClaimRecord) -> None:
        c = rec.claim
        for index, key in ((self._by_subject, c.subject), (self._by_predicate, c.predicate), (self._by_scope, c.scope), (self._by_object, c.object)):
            if isinstance(key, Ref) or index is not self._by_object:
                index.get(key, set()).discard(rec.id)

    # -- schema
    def declare(self, predicate: str, *, functional: bool) -> None:
        """A functional predicate has at most one object per subject, scope, and instant."""
        (self.functional.add if functional else self.functional.discard)(predicate)

    # -- committed single edits
    def put(self, ref: Ref, value: Any) -> Ref:
        return self._commit(Patch((Put(ref, value),), self.revision)).patch.edits[0].ref  # type: ignore[union-attr]

    def tell(self, claim: Claim, *evidence: Evidence) -> ClaimRecord:
        if not evidence:
            raise ValueError("a claim needs at least one piece of evidence")
        self._commit(Patch((Tell(claim, evidence),), self.revision))
        return self._claims[claim.id]

    def apply(self, patch: Patch) -> Commit:
        return self._commit(patch)

    def _commit(self, patch: Patch) -> Commit:
        if patch.base_revision != self.revision:
            raise StaleRevision(f"patch based on revision {patch.base_revision}, store is at {self.revision}")
        staged: dict[Ref, Any] = {}  # only the entities this patch touches
        for edit in patch.edits:  # validate everything before mutating anything
            if isinstance(edit, SetField):
                current = staged[edit.ref] if edit.ref in staged else self.entities.get(edit.ref, _ANY)
                if current is _ANY:
                    raise KeyError(edit.ref)
                staged[edit.ref] = _replace_path(current, edit.path, edit.value)
            elif isinstance(edit, Put):
                staged[edit.ref] = edit.value
            elif isinstance(edit, Retract) and edit.claim_id not in self._claims:
                raise KeyError(edit.claim_id)
            elif isinstance(edit, Tell) and not edit.evidence:
                raise ValueError("a claim needs at least one piece of evidence")
        self.entities.update(staged)
        added: list[str] = []
        retracted: list[str] = []
        for edit in patch.edits:
            if isinstance(edit, Tell):
                rec = self._claims.get(edit.claim.id)
                if rec is None:
                    rec = self._claims[edit.claim.id] = ClaimRecord(edit.claim)
                    self._index(rec)
                    added.append(rec.id)
                elif rec.retracted:  # new evidence revives a withdrawn claim
                    rec.retracted = None
                    rec.evidence.clear()
                    added.append(rec.id)
                for e in edit.evidence:
                    if e not in rec.evidence:
                        rec.evidence.append(e)
                        for premise in e.derived_from:
                            self._dependents[premise].add(rec.id)
            elif isinstance(edit, Retract):
                self._retract(edit.claim_id, Retraction(edit.reason, edit.evidence), retracted)
        self.revision += 1
        return Commit(self.revision, patch, tuple(dict.fromkeys(added)), tuple(dict.fromkeys(retracted)))

    def _retract(self, claim_id: str, why: Retraction, out: list[str]) -> None:
        """Retract a claim, then withdraw derivations whose every line of support depended on something retracted."""
        stack = [(claim_id, why)]
        while stack:
            cid, reason = stack.pop()
            rec = self._claims.get(cid)
            if rec is None or rec.retracted:
                continue
            rec.retracted = reason
            out.append(cid)
            for dep in sorted(self._dependents.get(cid, ())):
                d = self._claims.get(dep)
                if d is None or d.retracted:
                    continue
                supported = any(
                    not e.derived_from or all((p := self._claims.get(x)) is not None and not p.retracted for x in e.derived_from) for e in d.evidence
                )
                if not supported:
                    stack.append((dep, Retraction(f"premise {cid} withdrawn")))

    def forget(self, claim_ids: Iterable[str]) -> None:
        """Evict claims from working memory entirely (not a retraction: no history is kept)."""
        for cid in claim_ids:
            rec = self._claims.pop(cid, None)
            if rec is not None:
                self._unindex(rec)
                self._dependents.pop(cid, None)

    # -- queries
    def get(self, ref: Ref) -> Any:
        return self.entities[ref]

    def claim(self, claim_id: str) -> ClaimRecord:
        return self._claims[claim_id]

    def claims(
        self,
        subject: Ref | None = None,
        predicate: str | None = None,
        object: Any = _ANY,
        *,
        at: datetime | None = None,
        scope: Ref | None | object = _ANY,
        include_retracted: bool = False,
    ) -> list[ClaimRecord]:
        candidates: list[Iterable[str]] = []
        if subject is not None:
            candidates.append(self._by_subject.get(subject, ()))
        if isinstance(object, Ref):
            candidates.append(self._by_object.get(object, ()))
        if predicate is not None:
            candidates.append(self._by_predicate.get(predicate, ()))
        if scope is not _ANY:
            candidates.append(self._by_scope.get(scope, ()))  # type: ignore[arg-type]
        ids: Iterable[str] = min(candidates, key=len) if candidates else self._claims
        out = []
        for cid in ids:
            rec = self._claims.get(cid)
            c = rec.claim if rec else None
            if rec is None or (rec.retracted and not include_retracted):
                continue
            if subject is not None and c.subject != subject:
                continue
            if predicate is not None and c.predicate != predicate:
                continue
            if object is not _ANY and c.object != object:
                continue
            if scope is not _ANY and c.scope != scope:
                continue
            if at is not None and not c.valid.contains(at):
                continue
            out.append(rec)
        return sorted(out, key=lambda r: r.id)

    def match(self, *patterns: tuple[Any, str, Any], at: datetime | None = None, with_support: bool = False) -> list:
        """Conjunctive pattern query. ``Var`` terms bind; everything else must be equal.

        With ``with_support=True`` each result is ``(bindings, claim ids that matched)``.
        """
        results: list[tuple[dict[str, Any], tuple[str, ...]]] = [({}, ())]
        for s, p, o in patterns:
            nxt = []
            for binding, support in results:
                s_b = binding.get(s.name, s) if isinstance(s, Var) else s
                o_b = binding.get(o.name, o) if isinstance(o, Var) else o
                for rec in self.claims(
                    subject=None if isinstance(s_b, Var) else s_b,
                    predicate=p,
                    object=_ANY if isinstance(o_b, Var) else o_b,
                    at=at,
                ):
                    b = dict(binding)
                    if isinstance(s_b, Var):
                        b[s_b.name] = rec.claim.subject
                    if isinstance(o_b, Var):
                        if o_b.name in b and b[o_b.name] != rec.claim.object:
                            continue
                        b[o_b.name] = rec.claim.object
                    nxt.append((b, support + (rec.id,)))
            results = nxt
        return results if with_support else [b for b, _ in results]

    def conflicts(self, subject: Ref | None = None) -> list[Conflict]:
        groups: dict[tuple, list[ClaimRecord]] = defaultdict(list)
        for rec in self.claims(subject=subject):
            if rec.claim.predicate in self.functional:
                groups[(rec.claim.subject, rec.claim.predicate, rec.claim.scope)].append(rec)
        found = []
        for recs in groups.values():
            for i, a in enumerate(recs):
                for b in recs[i + 1 :]:
                    if a.claim.object != b.claim.object and (during := a.claim.valid.overlap(b.claim.valid)):
                        found.append(Conflict(a, b, during))
        return found

    def neighborhood(self, ref: Ref, depth: int = 1) -> Subgraph:
        seen, frontier, claims = {ref}, [ref], {}
        for _ in range(depth):
            nxt = []
            for r in frontier:
                linked = self.claims(subject=r) + self.claims(object=r)
                if r in self.entities:
                    linked_refs = _refs_in(self.entities[r])
                else:
                    linked_refs = []
                for rec in linked:
                    claims[rec.id] = rec
                    linked_refs += [rec.claim.subject] + ([rec.claim.object] if isinstance(rec.claim.object, Ref) else [])
                for other in linked_refs:
                    if other not in seen:
                        seen.add(other)
                        nxt.append(other)
            frontier = nxt
        return Subgraph({r: self.entities[r] for r in sorted(seen) if r in self.entities}, sorted(claims.values(), key=lambda c: c.id))

    # -- persistence
    def to_json(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "functional": sorted(self.functional),
            "entities": {r.id: encode(v, self.registry).data for r, v in sorted(self.entities.items())},
            "claims": [
                {
                    "claim": encode(rec.claim, self.registry).data,
                    "evidence": [encode(e, self.registry).data for e in rec.evidence],
                    "retracted": encode(rec.retracted, self.registry).data if rec.retracted else None,
                }
                for rec in sorted(self._claims.values(), key=lambda r: r.id)
            ],
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any], registry: TypeRegistry) -> tuple[Store, ConversionReport]:
        store, report = cls(registry), ConversionReport()

        def dec(d: Any) -> Any:
            value, rep = decode(d, registry)
            report.opaque.extend(rep.opaque)
            return value

        store.functional = set(data["functional"])
        store.entities = {Ref(r): dec(v) for r, v in data["entities"].items()}
        for row in data["claims"]:
            claim = dec(row["claim"])
            rec = store._claims[claim.id] = ClaimRecord(claim, [dec(e) for e in row["evidence"]])
            rec.retracted = dec(row["retracted"]) if row["retracted"] else None
            store._index(rec)
            for e in rec.evidence:
                for premise in e.derived_from:
                    store._dependents[premise].add(rec.id)
            if isinstance(claim.object, Ref):
                store._by_object[claim.object].add(claim.id)
        store.revision = data["revision"]
        return store, report


def _refs_in(value: Any) -> list[Ref]:
    if isinstance(value, Ref):
        return [value]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [r for v in value for r in _refs_in(v)]
    if isinstance(value, Mapping):
        return [r for v in value.values() for r in _refs_in(v)]
    if dataclasses.is_dataclass(value) or _is_model(value):
        return [r for v in _fields_of(value).values() for r in _refs_in(v)]
    return []
