"""A small cognitive substrate over the claim store.

    perceive / parse   ->  Fragment          (claims extracted from one input, with locators)
    integrate(mind, *fragments) -> Thought   (merge into working memory; report what changed)
    think(mind, rules, since=thought) -> Thought   (bottom-up rules fire on what changed)
    explain(mind, claim)                     (why a claim is believed: observations and rules)

A ``Fragment`` can be a *snapshot* of a scope (for example, everything currently on
screen). Claims in that scope that the new snapshot no longer contains are retracted,
and derivations that depended only on them are withdrawn with them.

Perception can be wrong for a frame. An optional ``Corroboration`` policy keeps chosen
perceived claims *tentative* until they are seen the same way in ``k`` frames (or confirmed
by an action's outcome); derivations inherit tentativeness from their premises, and a
tentative claim a later frame contradicts is retracted. Rules and intentions that must not
act on a single glance ask for the established view.

Rules are ordinary Python: patterns select premises, a function derives claims. Every
derived claim cites its premises, so every thought can be explained. Rules may call
other operations (``tc.classify``, ``tc.parse``); those calls appear in the trace.
Nothing here decides what to *do*; that is ``choose`` over intentions.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

from .outcomes import Score
from .records import Claim, ClaimRecord, Evidence, Patch, Put, Ref, Retract, Store, Tell


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Fragment:
    """Claims extracted from one input."""

    source: Ref  # the input these claims came from (a frame, a message, a document)
    claims: tuple[tuple[Claim, str | None], ...]  # (claim, locator within the source)
    entities: tuple[tuple[Ref, Any], ...] = ()  # typed payloads, e.g. geometry for a control
    snapshot_of: Ref | None = None  # if set, this fragment is everything currently true in that scope
    method: str = "parse"
    observed_at: datetime = field(default_factory=_now)
    confidence: Mapping[str, Score] = field(default_factory=dict)  # claim id -> extractor's score


@dataclass(frozen=True)
class Thought:
    """What changed in the mind. Empty thoughts are normal."""

    added: tuple[ClaimRecord, ...] = ()
    retracted: tuple[ClaimRecord, ...] = ()
    established: tuple[ClaimRecord, ...] = ()  # tentative claims that became established (see ``Corroboration``)

    def __add__(self, other: Thought) -> Thought:
        return Thought(self.added + other.added, self.retracted + other.retracted, self.established + other.established)

    @property
    def empty(self) -> bool:
        return not self.added and not self.retracted and not self.established

    def about(self, predicate: str) -> list[Claim]:
        return [r.claim for r in self.added if r.claim.predicate == predicate]


@dataclass
class Corroboration:
    """When a perceived claim may be acted on: after ``k`` consistent frames, or confirmation.

    * ``predicates``: which perceived predicates this governs (None = all). Others are
      established on sight, which keeps, say, button geometry instant while text readings
      wait for a second look.
    * A frame counts toward ``k`` only if the extractor's confidence (when it gave one) is at
      least ``min_confidence``; below that a claim stays tentative until ``confirm``-ed.
    * Derived claims are established only when every premise of some line of support is.
    * A tentative claim that a later frame contradicts is retracted: in a snapshot scope by
      no longer being perceived; elsewhere by a different object for a functional predicate.

    State is per mind and per run: create a fresh policy for each episode.
    """

    k: int = 2
    min_confidence: float = 0.0
    predicates: frozenset[str] | None = None
    frames: dict[str, int] = field(default_factory=dict)  # claim id -> frames that corroborated it
    confirmed: set[str] = field(default_factory=set)

    def governs(self, predicate: str) -> bool:
        return self.predicates is None or predicate in self.predicates

    def established(self, mind: Store, claim_id: str, _seen: frozenset[str] = frozenset()) -> bool:
        rec = mind._claims.get(claim_id)
        if rec is None or rec.retracted or claim_id in _seen:
            return False
        if claim_id in self.confirmed:
            return True
        for e in rec.evidence:
            if e.derived_from:
                if all(self.established(mind, p, _seen | {claim_id}) for p in e.derived_from):
                    return True
            elif not self.governs(rec.claim.predicate) or self.frames.get(claim_id, 0) >= self.k:
                return True
        return False

    def tentative(self, mind: Store, claim_id: str) -> bool:
        return claim_id in mind._claims and not self.established(mind, claim_id)

    def confirm(self, mind: Store, claim_id: str) -> Thought:
        """An action's outcome bore the claim out: establish it (and what now rests on it)."""
        before = self.established(mind, claim_id)
        self.confirmed.add(claim_id)
        return Thought(established=tuple(mind.claim(i) for i in self._closure(mind, [claim_id]))) if not before else Thought()

    def view(self, mind: Store) -> EstablishedView:
        return EstablishedView(mind, self)

    def _count(self, claim_id: str, score: Score | None) -> None:
        if score is None or score.value >= self.min_confidence:
            self.frames[claim_id] = self.frames.get(claim_id, 0) + 1

    def _closure(self, mind: Store, ids: Iterable[str]) -> list[str]:
        """Claims now established among ``ids`` and the derivations resting on them."""
        out, stack, seen = [], [i for i in ids], set()
        while stack:
            cid = stack.pop()
            if cid in seen:
                continue
            seen.add(cid)
            if self.established(mind, cid):
                out.append(cid)
                stack.extend(sorted(mind._dependents.get(cid, ())))
        return list(dict.fromkeys(out))


class EstablishedView:
    """A read-only look at a mind that hides tentative claims (everything else delegates)."""

    def __init__(self, mind: Store, policy: Corroboration) -> None:
        self._mind, self._policy = mind, policy

    def __getattr__(self, name: str) -> Any:
        return getattr(self._mind, name)

    def claims(self, *args: Any, **kwargs: Any) -> list[ClaimRecord]:
        return [r for r in self._mind.claims(*args, **kwargs) if self._policy.established(self._mind, r.id)]

    def match(self, *patterns: Any, with_support: bool = False) -> list:
        found = [(b, s) for b, s in self._mind.match(*patterns, with_support=True) if all(self._policy.established(self._mind, c) for c in s)]
        return found if with_support else [b for b, _ in found]


def integrate(mind: Store, *fragments: Fragment, remember_retracted: bool = False, corroboration: Corroboration | None = None) -> Thought:
    """Merge fragments into working memory.

    Snapshot scopes are ephemeral: claims that fall out of a snapshot are retracted and,
    unless ``remember_retracted``, forgotten along with derivations that depended on them.
    With a ``Corroboration`` policy, each fragment is one frame of evidence for its claims
    (see there); ``Thought.established`` lists claims that crossed the threshold.
    """
    policy = corroboration
    was_tentative = {cid for f in fragments for c, _ in f.claims if (cid := c.id) in mind._claims and policy.tentative(mind, cid)} if policy else set()
    edits: list[Any] = []
    for f in fragments:
        edits += [Put(ref, value) for ref, value in f.entities]
        present = set()
        for claim, locator in f.claims:
            if f.snapshot_of is not None and claim.scope != f.snapshot_of:
                raise ValueError(f"snapshot of {f.snapshot_of} contains a claim in scope {claim.scope}")
            present.add(claim.id)
            live = claim.id in mind._claims and not mind._claims[claim.id].retracted
            if policy is not None:
                policy._count(claim.id, f.confidence.get(claim.id))
                if f.snapshot_of is None and claim.predicate in mind.functional:
                    edits += [Retract(r.id, "contradicted by a later observation", (Evidence(f.source, f.observed_at, locator, f.method),))
                              for r in mind.claims(claim.subject, claim.predicate, scope=claim.scope) if r.claim.object != claim.object and policy.tentative(mind, r.id)]
            if f.snapshot_of is not None and live:
                continue  # still perceived; no need to pile up evidence every frame
            edits.append(Tell(claim, (Evidence(f.source, f.observed_at, locator, f.method, f.confidence.get(claim.id)),)))
        if f.snapshot_of is not None:
            edits += [Retract(rec.id, "no longer perceived", (Evidence(f.source, f.observed_at, method=f.method),)) for rec in mind.claims(scope=f.snapshot_of) if rec.id not in present]
    if not edits:
        return _newly_established(mind, policy, was_tentative, ())
    commit = mind.apply(Patch(tuple(edits), mind.revision))
    thought = Thought(tuple(mind.claim(i) for i in commit.added), tuple(mind.claim(i) for i in commit.retracted))
    thought += _newly_established(mind, policy, was_tentative, commit.added)
    if policy is not None:
        for cid in commit.retracted:  # contradicted or no longer perceived: the count starts over
            policy.frames.pop(cid, None)
            policy.confirmed.discard(cid)
    if not remember_retracted:
        mind.forget(commit.retracted)
    return thought


def _newly_established(mind: Store, policy: Corroboration | None, was_tentative: set[str], added: Sequence[str]) -> Thought:
    if policy is None or not was_tentative:
        return Thought()
    flipped = [cid for cid in was_tentative if cid not in added and policy.established(mind, cid)]
    return Thought(established=tuple(mind.claim(i) for i in policy._closure(mind, flipped) if i not in added))


@dataclass(frozen=True)
class Rule:
    """When the patterns match (and at least one matched claim is new), derive claims."""

    name: str
    when: tuple[tuple[Any, str, Any], ...]
    then: Callable[[Mapping[str, Any], Store], Iterable[Claim | tuple[Claim, Score] | Fragment]]
    version: str = "1"  # a Fragment output is an act of reading: integrated as knowledge, not as a derivation
    reacts_to: str = "new_premises"  # or "any_change": also fire when something was retracted (e.g. a spinner vanished)
    established_only: bool = False  # with a Corroboration policy: premises must be established, and ``then`` sees the established view


@dataclass
class ThinkStats:
    rounds: int = 0
    firings: int = 0
    derived: int = 0
    ms: float = 0.0


def think(mind: Store, rules: Sequence[Rule], *, since: Thought, max_rounds: int = 8, stats: ThinkStats | None = None, forget_withdrawn: bool = True,
          corroboration: Corroboration | None = None) -> Thought:
    """Bottom-up inference: fire rules on matches that involve something new, until quiet.

    Semi-naive evaluation: a match whose premises are all old already fired earlier. A claim
    that just became established counts as new for ``established_only`` rules.
    """
    t0 = _time.perf_counter()
    stats = stats if stats is not None else ThinkStats()
    total = Thought()
    frontier = {r.id for r in since.added} | {r.id for r in since.established}
    changed = bool(since.added or since.retracted or since.established)
    view = corroboration.view(mind) if corroboration is not None else None
    for round_no in range(max_rounds):
        if not frontier and not (round_no == 0 and changed):
            break
        stats.rounds += 1
        edits: list[Tell] = []
        readings: list[Fragment] = []
        for rule in rules:
            strict = rule.established_only and view is not None
            for bindings, support in (view if strict else mind).match(*rule.when, with_support=True):
                if frontier.isdisjoint(support) and not (rule.reacts_to == "any_change" and round_no == 0 and changed):
                    continue
                stats.firings += 1
                for out in rule.then(bindings, view if strict else mind):
                    if isinstance(out, Fragment):
                        readings.append(out)
                        continue
                    claim, score = out if isinstance(out, tuple) else (out, None)
                    edits.append(Tell(claim, (Evidence(Ref(f"rule:{rule.name}"), _now(), method=f"derive@{rule.version}", confidence=score, derived_from=tuple(sorted(set(support)))),)))
        if not edits and not readings:
            break
        added_ids: list[str] = []
        if edits:
            added_ids += mind.apply(Patch(tuple(edits), mind.revision)).added
        if readings:
            added_ids += [r.id for r in integrate(mind, *readings).added]
        added = tuple(mind.claim(i) for i in dict.fromkeys(added_ids))
        stats.derived += len(added)
        total += Thought(added)
        frontier = {r.id for r in added}
    stats.ms += (_time.perf_counter() - t0) * 1e3
    return total


def _shown(value: Any) -> Any:
    """How a claim's object reads in an explanation.

    ``Score`` and friends are worth unwrapping to their number, but a value that carries a
    unit alongside it (a quantity) loses its point when reduced to a bare float: "60.0"
    instead of "60 coin" is exactly the confusion the unit exists to prevent.
    """
    if hasattr(value, "value") and hasattr(value, "unit"):
        return str(value)
    return getattr(value, "value", value)


def explain(mind: Store, claim_id: str, *, depth: int = 6) -> list[str]:
    """Indented lines: the claim, then each line of support down to observations."""
    lines: list[str] = []

    def show(cid: str, level: int, seen: frozenset[str]) -> None:
        rec = mind._claims.get(cid)
        pad = "  " * level
        if rec is None:
            lines.append(f"{pad}(forgotten {cid})")
            return
        c = rec.claim
        obj = _shown(c.object)
        lines.append(f"{pad}{c.subject} {c.predicate} {obj!r}" + (" [retracted]" if rec.retracted else ""))
        if level >= depth or cid in seen:
            return
        for e in rec.evidence:
            conf = f" ({e.confidence.kind} {e.confidence.value:.2f})" if e.confidence else ""
            if e.derived_from:
                how = f" via {e.method}" if e.method else ""
                lines.append(f"{pad}  ← {e.source}{conf}{how} from:")
                for p in e.derived_from:
                    show(p, level + 2, seen | {cid})
            else:
                lines.append(f"{pad}  ← observed in {e.source}" + (f" at {e.locator}" if e.locator else "") + f" via {e.method}{conf}")

    show(claim_id, 0, frozenset())
    return lines
