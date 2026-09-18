"""Awareness: a bounded, salience-weighted active set nucleated from the claim graph.

Percepts and the current utterance *seed* awareness; activation then spreads along links
that already exist in the store — a shared subject, a claim about the object of another,
a premise or a derivation, a shared source, and whatever extra adjacency a caller supplies
(spatial neighbours from ``frames.py``, for instance). Spreading is bounded by a budget and
a floor, so a mind with ten thousand claims still thinks over dozens.

That bound is also the practical point: ``think`` over ``Awareness.view()`` only sees the
aware set, so a rule whose patterns would otherwise scan all of memory costs what the
aware set costs, not what memory costs.

Nothing here is heuristic in the sense of being unexplainable: activation is the best
multiplicative path from a seed, ``why`` returns that path, and iteration order is sorted,
so two runs of the same mind give the same aware set.
"""

from __future__ import annotations

import heapq
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from .records import ClaimRecord, Ref, Store

#: claims by source, per store and revision — shared, because building it is O(memory)
_SOURCE_INDEX: "weakref.WeakKeyDictionary[Any, tuple[int, dict[Ref, set[str]]]]" = weakref.WeakKeyDictionary()

#: how much activation survives each kind of hop, before ``decay``
LINK_WEIGHTS: Mapping[str, float] = {
    "subject": 1.0,  # another claim about the same subject
    "object": 0.8,  # a claim about the thing this one points at
    "mention": 0.7,  # a claim that points at this one's subject
    "premise": 0.9,  # what this claim was derived from
    "dependent": 0.9,  # what was derived from this claim
    "source": 0.4,  # read in the same observation
}


@dataclass(frozen=True)
class AwarenessPolicy:
    """What a caller chooses: how far awareness spreads and how much of it it can hold."""

    budget: int = 64  # at most this many claims are aware at once
    decay: float = 0.6  # activation multiplier per hop
    floor: float = 0.05  # below this a claim is not aware at all
    max_hops: int = 3
    weights: Mapping[str, float] = field(default_factory=lambda: dict(LINK_WEIGHTS))
    fade: float = 0.5  # what carries over when a cycle ends
    fan_out: int = 32  # neighbours considered per link kind, so a huge observation cannot dominate
    max_group: int = 256  # a link shared by more claims than this says nothing specific; ignore it

    def weight(self, kind: str) -> float:
        return self.weights.get(kind, 0.5)


@dataclass(frozen=True)
class Activation:
    """Why a claim is aware: how strongly, and by which path from which seed."""

    claim_id: str
    strength: float
    seed: str
    path: tuple[tuple[str, str], ...] = ()  # (link kind, claim id) hops from the seed, in order

    @property
    def hops(self) -> int:
        return len(self.path)


class Awareness:
    """The active set over one mind. Seed it, spread, then think over ``view()``."""

    def __init__(self, mind: Store, policy: AwarenessPolicy | None = None,
                 extra_links: Callable[[str, Store], Iterable[tuple[str, str, float]]] | None = None) -> None:
        self.mind = mind
        self.policy = policy or AwarenessPolicy()
        self.extra_links = extra_links  # claim id -> (neighbour id, link kind, weight)
        self._act: dict[str, Activation] = {}
        self._pending: dict[str, Activation] = {}  # seeds not yet spread
        self._source_index: dict[Ref, set[str]] = {}
        self._indexed_at = -1
        self._generation = 0  # bumped whenever activation changes, so ``aware`` can cache
        self._aware_cache: list[ClaimRecord] | None = None
        self._aware_ids_cache: set[str] = set()
        self._aware_key: tuple[int, int] | None = None

    # -- seeding

    def seed(self, what: Iterable[Any] | Any, strength: float = 1.0, *, label: str | None = None) -> list[str]:
        """Seed from claim ids, ``ClaimRecord``s, ``Claim``s, or a ``Ref`` (all its claims)."""
        ids: list[str] = []
        for item in what if isinstance(what, (list, tuple, set, frozenset)) else [what]:
            if isinstance(item, Ref):
                ids += [r.id for r in self.mind.claims(subject=item)] + [r.id for r in self.mind.claims(object=item)]
            elif isinstance(item, ClaimRecord):
                ids.append(item.id)
            elif isinstance(item, str):
                ids.append(item)
            elif hasattr(item, "id"):
                ids.append(item.id)
            else:
                raise TypeError(f"cannot seed awareness from {type(item).__qualname__}")
        seeded = []
        for cid in sorted(dict.fromkeys(ids)):
            if cid not in self.mind._claims or self.mind._claims[cid].retracted:
                continue
            a = Activation(cid, strength, label or cid)
            if self._act.get(cid, Activation(cid, -1.0, "")).strength < strength:
                self._act[cid] = a
                self._pending[cid] = a
                self._generation += 1
            seeded.append(cid)
        return seeded

    # -- spreading

    def spread(self) -> list[Activation]:
        """Grow the active set from its seeds. Returns what became newly aware, strongest first."""
        p = self.policy
        heap: list[tuple[float, str, Activation]] = [(-a.strength, cid, a) for cid, a in sorted(self._pending.items())]
        heapq.heapify(heap)
        self._pending.clear()
        newly: list[Activation] = []
        while heap:
            neg, cid, act = heapq.heappop(heap)
            if self._act.get(cid) is not act and -neg < self._act.get(cid, act).strength:
                continue  # a stronger path to this claim was already taken
            if act.hops >= p.max_hops or len(self._act) >= p.budget * 4:
                continue  # stop growing the frontier; the budget trims the result anyway
            for nid, kind, weight in self._links(cid):
                strength = act.strength * p.decay * weight
                if strength < p.floor:
                    continue
                known = self._act.get(nid)
                if known is not None and known.strength >= strength:
                    continue
                nxt = Activation(nid, strength, act.seed, act.path + ((kind, nid),))
                self._act[nid] = nxt
                self._generation += 1
                newly.append(nxt)
                heapq.heappush(heap, (-strength, nid, nxt))
        return sorted(newly, key=lambda a: (-a.strength, a.claim_id))

    def _links(self, claim_id: str) -> list[tuple[str, str, float]]:
        rec = self.mind._claims.get(claim_id)
        if rec is None or rec.retracted:
            return []
        c, w, fan = rec.claim, self.policy.weight, self.policy.fan_out
        out: list[tuple[str, str, float]] = []

        def take(ids: Any, kind: str) -> None:
            """Up to ``fan_out`` neighbours of one kind, chosen by id so the choice is stable.

            A group bigger than ``max_group`` is skipped rather than sampled: "read in the
            same observation as three thousand other things" is not evidence of relatedness,
            and scanning it would make nucleation cost grow with memory.
            """
            if not ids or len(ids) > self.policy.max_group:
                return
            for other in heapq.nsmallest(fan, ids):
                if other != claim_id:
                    out.append((other, kind, w(kind)))

        take(self.mind._by_subject.get(c.subject, ()), "subject")
        if isinstance(c.object, Ref):
            take(self.mind._by_subject.get(c.object, ()), "object")
        take(self.mind._by_object.get(c.subject, ()), "mention")
        for e in rec.evidence:
            for premise in e.derived_from:
                out.append((premise, "premise", w("premise")))
        take(self.mind._dependents.get(claim_id, ()), "dependent")
        for source in sorted({e.source for e in rec.evidence}):
            take(self._sources().get(source, ()), "source")
        if self.extra_links is not None:
            out += [(nid, kind, weight) for nid, kind, weight in self.extra_links(claim_id, self.mind)]
        live = [(nid, kind, weight) for nid, kind, weight in out
                if nid in self.mind._claims and not self.mind._claims[nid].retracted]
        return list(dict.fromkeys(live))

    def _sources(self) -> dict[Ref, set[str]]:
        """Claims by the observation they were read in.

        Cached per store and revision: building it is O(memory), so a fresh awareness each
        cycle must not pay for it again. Without this, nucleation is O(memory) rather than
        O(aware set), which is the whole thing awareness is for.
        """
        cached = _SOURCE_INDEX.get(self.mind)
        if cached is not None and cached[0] == self.mind.revision:
            return cached[1]
        index: dict[Ref, set[str]] = {}
        for cid, rec in self.mind._claims.items():
            if rec.retracted:
                continue
            for e in rec.evidence:
                index.setdefault(e.source, set()).add(cid)
        _SOURCE_INDEX[self.mind] = (self.mind.revision, index)
        return index

    # -- reading

    def aware(self) -> list[ClaimRecord]:
        """The active set, strongest first, trimmed to the budget. Retracted claims fall out."""
        key = (self.mind.revision, self._generation)
        if self._aware_cache is not None and self._aware_key == key:
            return self._aware_cache
        live = [(a, self.mind._claims[a.claim_id]) for a in self._act.values()
                if a.claim_id in self.mind._claims and not self.mind._claims[a.claim_id].retracted]
        live.sort(key=lambda pair: (-pair[0].strength, pair[0].claim_id))
        self._aware_cache = [rec for _, rec in live[: self.policy.budget]]
        self._aware_ids_cache = {rec.id for rec in self._aware_cache}
        self._aware_key = key
        return self._aware_cache

    def aware_ids(self) -> set[str]:
        self.aware()
        return self._aware_ids_cache

    def salience(self, claim: Any) -> float:
        cid = claim if isinstance(claim, str) else claim.id
        a = self._act.get(cid)
        return a.strength if a is not None and cid in self.aware_ids() else 0.0

    def why(self, claim: Any) -> list[str]:
        """The path that made this claim aware, as lines, or a note that it is not."""
        cid = claim if isinstance(claim, str) else claim.id
        a = self._act.get(cid)
        if a is None or cid not in self.aware_ids():
            return [f"{cid} is not aware"]
        lines = [f"{self._describe(a.seed)} (seed, {a.strength:.3f} at the end of the path)" if not a.path
                 else f"{self._describe(a.seed)} (seed)"]
        for kind, nid in a.path:
            lines.append(f"  --{kind}--> {self._describe(nid)}")
        if a.path:
            lines.append(f"  = {a.strength:.3f}")
        return lines

    def _describe(self, cid: str) -> str:
        rec = self.mind._claims.get(cid)
        if rec is None:
            return cid
        c = rec.claim
        obj = getattr(c.object, "value", c.object)
        return f"{c.subject} {c.predicate} {obj!r}"

    # -- between cycles

    def fade(self, factor: float | None = None) -> list[str]:
        """End of a cycle: activation decays, and what falls under the floor stops being aware."""
        f = self.policy.fade if factor is None else factor
        dropped = []
        for cid, a in sorted(self._act.items()):
            strength = a.strength * f
            if strength < self.policy.floor or cid not in self.mind._claims or self.mind._claims[cid].retracted:
                dropped.append(cid)
            else:
                self._act[cid] = Activation(cid, strength, a.seed, a.path)
                self._generation += 1
        for cid in dropped:
            self._act.pop(cid, None)
            self._pending.pop(cid, None)
        self._generation += 1
        return dropped

    def clear(self) -> None:
        self._act.clear()
        self._pending.clear()
        self._generation += 1

    def view(self) -> AwareView:
        """A store-like view restricted to the aware set, for ``think`` and intention code."""
        return AwareView(self.mind, self.aware_ids())


_ANY = object()


class AwareView:
    """A read-only look at a mind through awareness.

    Queries walk the aware set itself rather than filtering the store's answer, so a rule
    whose pattern would otherwise scan all of memory costs what awareness costs. That is
    the point of bounding it; delegating to ``Store.claims`` would keep the scan.
    """

    def __init__(self, mind: Store, ids: set[str]) -> None:
        self._mind, self._ids = mind, ids

    def __getattr__(self, name: str) -> Any:
        return getattr(self._mind, name)

    def _records(self) -> list[ClaimRecord]:
        live = (self._mind._claims.get(cid) for cid in self._ids)
        return [r for r in live if r is not None and not r.retracted]

    def claims(self, subject: Ref | None = None, predicate: str | None = None, object: Any = _ANY, *,
               at: Any = None, scope: Any = _ANY, include_retracted: bool = False) -> list[ClaimRecord]:
        out = []
        for rec in self._records():
            c = rec.claim
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

    def match(self, *patterns: Any, at: Any = None, with_support: bool = False) -> list:
        from .records import Var

        results: list[tuple[dict[str, Any], tuple[str, ...]]] = [({}, ())]
        for s, p, o in patterns:
            nxt = []
            for binding, support in results:
                s_b = binding.get(s.name, s) if isinstance(s, Var) else s
                o_b = binding.get(o.name, o) if isinstance(o, Var) else o
                for rec in self.claims(subject=None if isinstance(s_b, Var) else s_b, predicate=p,
                                       object=_ANY if isinstance(o_b, Var) else o_b, at=at):
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


def nucleate(mind: Store, seeds: Sequence[Any], policy: AwarenessPolicy | None = None,
             extra_links: Callable[[str, Store], Iterable[tuple[str, str, float]]] | None = None) -> Awareness:
    """Seed and spread in one call: the common shape at the top of a cycle."""
    a = Awareness(mind, policy, extra_links)
    a.seed(list(seeds))
    a.spread()
    return a
