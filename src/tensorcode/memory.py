"""Memory as several stores with different dynamics over one claim substrate.

    working     the awareness buffer: small, fast to decay, what thinking runs over
    episodic    time-indexed traces of what happened, encoded by salience, decaying
    semantic    generalizations, including what someone simply told us
    spatial     the same claims reached by where they are (``frames.Frames``)
    procedural  skills, recalled by how well a cue fits them
    associative recall by cue similarity, not by exact pattern match

They are not separate databases. An episode is an entity with claims pointing at the claims
it contains, a generalization is a claim in the semantic scope, and forgetting is
``Store.forget``. What differs is the dynamics: episodic memory decays and consolidates,
semantic memory does not, and working memory holds only what is aware.

Forgetting is real and measured. Nothing is evicted while something still rests on it, and
told facts and generalizations are protected, so what goes is stale perception.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

from .awareness import Awareness, AwarenessPolicy
from .cognition import Thought
from .context import shingle_similarity
from .outcomes import Score
from .records import Claim, ClaimRecord, Evidence, Ref, Store

SEMANTIC = Ref("scope:semantic")
EPISODIC = Ref("scope:episodic")


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class MemoryPolicy:
    """The dynamics: how much is held, how fast it fades, what is never dropped."""

    working_budget: int = 64
    episodic_capacity: int = 400  # episodes kept before the least salient are dropped
    half_life: timedelta = timedelta(minutes=30)  # perceptual claims older than this may go
    min_salience: float = 0.15  # below this a stale claim is eligible to be forgotten
    consolidate_after: int = 3  # how many episodes must agree before it becomes semantic
    protect_scopes: tuple[Ref | None, ...] = (SEMANTIC,)
    protect_predicates: frozenset[str] = frozenset({"said", "told", "name"})
    # Testimony is not perception, and the half-life above is a perceptual one. A fact someone told
    # you does not become doubtful because you have not looked at it lately, and it cannot be
    # protected by predicate name: the predicate of a told fact is whatever word the teller used
    # ("my cat is Mackerel" files `cat`), so a list of protected spellings can never contain it.
    # Provenance can: claims whose evidence comes from an utterance are held on those grounds.
    #
    # Either party's utterance. Protecting only what the *user* said looks right and is the same
    # mistake one level down: the record of having said something oneself is an utterance too, and
    # dropping it leaves an assistant that still knows your name and has forgotten that it already
    # told you it does. This list is a convention about source spellings, not the principle — a
    # caller whose utterances are named differently has to extend it.
    protect_sources: tuple[str, ...] = ("utterance:", "person:", "user:", "reply:", "said:")


@dataclass(frozen=True)
class TurnReport:
    """What one turn of memory dynamics did, so the dynamics can be watched rather than trusted."""

    episode: "Episode | None" = None
    consolidated: int = 0
    forgotten: "ForgetReport | None" = None
    ms: float = 0.0


@dataclass(frozen=True)
class Episode:
    """One remembered happening: when, how much it mattered, and what it was made of."""

    ref: Ref
    at: datetime
    salience: float
    summary: str
    claims: tuple[str, ...]


@dataclass(frozen=True)
class Recollection:
    """What recall returned, and why each item came back."""

    episodes: tuple[tuple[Episode, Score], ...] = ()
    claims: tuple[tuple[ClaimRecord, Score], ...] = ()
    cue: str = ""

    def why(self) -> list[str]:
        lines = [f"cue {self.cue!r}"]
        lines += [f"  episode {e.ref} ({e.summary!r}) similarity {s.value:.2f}" for e, s in self.episodes]
        lines += [f"  claim {r.claim.subject} {r.claim.predicate} {getattr(r.claim.object, 'value', r.claim.object)!r} similarity {s.value:.2f}"
                  for r, s in self.claims]
        return lines


@dataclass
class ForgetReport:
    """What forgetting actually did, so it can be measured rather than assumed."""

    claims_forgotten: int = 0
    episodes_dropped: int = 0
    kept_because_depended_on: int = 0
    kept_because_protected: int = 0
    kept_because_testimony: int = 0  # someone said so: not perception, so not on a perceptual clock
    ms: float = 0.0


class Memory:
    """The several memories of one mind."""

    def __init__(self, mind: Store, policy: MemoryPolicy | None = None, *,
                 awareness: Awareness | None = None, frames: Any = None) -> None:
        self.mind = mind
        self.policy = policy or MemoryPolicy()
        self.frames = frames
        self.working = awareness or Awareness(mind, AwarenessPolicy(budget=self.policy.working_budget),
                                              extra_links=frames.links if frames is not None else None)
        self._episodes: dict[Ref, Episode] = {}
        self._counter = 0
        self._turns = 0

    # ---------------------------------------------------------------- episodic

    def encode(self, what: Thought | Sequence[ClaimRecord], *, summary: str = "", salience: float | None = None,
               at: datetime | None = None) -> Episode:
        """Lay down an episode. Salience defaults to how aware its claims were."""
        records = list(what.added) if isinstance(what, Thought) else list(what)
        when = at or _now()
        self._counter += 1
        ref = Ref(f"episode:{self._counter}")
        if salience is None:
            saliences = [self.working.salience(r.id) for r in records]
            salience = max(saliences) if saliences else 0.2
        text = summary or "; ".join(_describe(r) for r in records[:4])
        episode = Episode(ref, when, round(float(salience), 4), text, tuple(sorted(r.id for r in records)))
        self._episodes[ref] = episode
        source = Ref("memory:encode")
        edits = [Claim(ref, "at", when, scope=EPISODIC), Claim(ref, "salience", episode.salience, scope=EPISODIC),
                 Claim(ref, "summary", episode.summary, scope=EPISODIC)]
        edits += [Claim(ref, "contains", cid, scope=EPISODIC) for cid in episode.claims]
        for claim in edits:
            self.mind.tell(claim, Evidence(source, when, method="encode-episode"))
        return episode

    def episodes(self, *, since: datetime | None = None, min_salience: float = 0.0) -> list[Episode]:
        rows = [e for e in self._episodes.values() if e.salience >= min_salience and (since is None or e.at >= since)]
        return sorted(rows, key=lambda e: (e.at, e.ref.id))

    def recall(self, cue: str, k: int = 3, *, min_similarity: float = 0.05) -> Recollection:
        """Associative recall: episodes and claims whose text is nearest the cue."""
        scored_e = []
        for episode in self._episodes.values():
            score = shingle_similarity(cue, f"{episode.summary}", n=2)
            if score.value >= min_similarity:
                scored_e.append((episode, score))
        scored_e.sort(key=lambda pair: (-pair[1].value, -pair[0].salience, pair[0].ref.id))
        scored_c = []
        for rec in self.mind.claims():
            if rec.claim.scope == EPISODIC:
                continue
            score = shingle_similarity(cue, _describe(rec), n=2)
            if score.value >= min_similarity:
                scored_c.append((rec, score))
        scored_c.sort(key=lambda pair: (-pair[1].value, pair[0].id))
        return Recollection(tuple(scored_e[:k]), tuple(scored_c[:k]), cue)

    def consolidate(self) -> list[Claim]:
        """What several episodes agree on becomes semantic: a generalization citing its episodes."""
        counts: dict[tuple[Ref, str, str], list[Ref]] = {}
        for episode in self.episodes():
            for cid in episode.claims:
                rec = self.mind._claims.get(cid)
                if rec is None or rec.retracted or rec.claim.scope == EPISODIC:
                    continue
                key = (rec.claim.subject, rec.claim.predicate, repr(rec.claim.object))
                counts.setdefault(key, []).append(episode.ref)
        made = []
        for (subject, predicate, _), refs in sorted(counts.items(), key=lambda kv: (kv[0][0].id, kv[0][1], kv[0][2])):
            if len(refs) < self.policy.consolidate_after:
                continue
            source_claim = next(r for r in self.mind.claims(subject=subject, predicate=predicate))
            claim = Claim(subject, predicate, source_claim.claim.object, scope=SEMANTIC)
            if self.mind.claims(subject=subject, predicate=predicate, scope=SEMANTIC):
                continue
            self.mind.tell(claim, Evidence(Ref("memory:consolidate"), _now(), method="consolidate",
                                           confidence=Score(min(1.0, len(refs) / (self.policy.consolidate_after * 2)), "uncalibrated"),
                                           derived_from=tuple(sorted(dict.fromkeys(cid for r in refs for cid in self._episodes[r].claims
                                                                                   if self.mind._claims.get(cid) and
                                                                                   self.mind._claims[cid].claim.subject == subject and
                                                                                   self.mind._claims[cid].claim.predicate == predicate)))))
            made.append(claim)
        return made

    # ---------------------------------------------------------------- semantic

    def told(self, subject: Ref, predicate: str, object: Any, *, by: str = "user", at: datetime | None = None) -> ClaimRecord:
        """Someone simply said so. Semantic, protected from forgetting, and provably hearsay."""
        claim = Claim(subject, predicate, object, scope=SEMANTIC)
        return self.mind.tell(claim, Evidence(Ref(f"said:{by}"), at or _now(), method="told"))

    def semantic(self, subject: Ref | None = None, predicate: str | None = None) -> list[ClaimRecord]:
        return self.mind.claims(subject=subject, predicate=predicate, scope=SEMANTIC)

    # ----------------------------------------------------------------- spatial

    def here(self, **where: Any) -> list[ClaimRecord]:
        """What is at a place (see ``frames.Frames.spatial``)."""
        if self.frames is None:
            return []
        return self.frames.spatial(**where)

    # -------------------------------------------------------------- procedural

    def skills(self, cue: str, procedures: Iterable[Any], k: int = 3) -> list[tuple[Any, Score]]:
        """Rank procedures by how well a cue fits their name and what they are for."""
        scored = []
        for proc in procedures:
            text = " ".join(str(getattr(proc, attr, "") or "") for attr in ("id", "act", "why", "summary")).replace("_", " ")
            scored.append((proc, shingle_similarity(cue, text, n=2)))
        scored.sort(key=lambda pair: (-pair[1].value, str(getattr(pair[0], "id", ""))))
        return scored[:k]

    # ------------------------------------------------------------- forgetting

    def turn(self, records: Sequence[ClaimRecord], *, summary: str = "", consolidate_every: int = 5,
             forget_every: int = 5, keep_scopes: tuple[Ref | None, ...] = (), now: datetime | None = None) -> TurnReport:
        """One conversation turn's worth of dynamics: encode, and now and then consolidate and forget.

        Encoding every turn and consolidating every turn are different costs: laying down an episode
        is O(what just happened), while consolidation and forgetting sweep everything held. Doing the
        sweeps on a stride is what keeps per-turn latency flat as a conversation grows long.

        The caller decides what belongs in an episode. Raw perception does not: a retina's worth of
        claims per turn would bury the few that record what actually happened, and the point of an
        episode is to be re-findable later.
        """
        t0 = _time.perf_counter()
        self._turns += 1
        episode = self.encode(records, summary=summary, at=now) if records else None
        consolidated = len(self.consolidate()) if self._turns % max(1, consolidate_every) == 0 else 0
        forgotten = (self.forget_stale(now=now, keep_scopes=keep_scopes)
                     if self._turns % max(1, forget_every) == 0 else None)
        return TurnReport(episode, consolidated, forgotten, (_time.perf_counter() - t0) * 1e3)

    def forget_stale(self, *, now: datetime | None = None, keep_scopes: tuple[Ref | None, ...] = ()) -> ForgetReport:
        """Drop stale, unsalient perception. Nothing that supports something else, nothing protected."""
        t0 = _time.perf_counter()
        when = now or _now()
        report = ForgetReport()
        protected_scopes = set(self.policy.protect_scopes) | set(keep_scopes) | {EPISODIC}
        aware = self.working.aware_ids()
        doomed = []
        for cid, rec in sorted(self.mind._claims.items()):
            if rec.retracted:
                continue
            if rec.claim.scope in protected_scopes or rec.claim.predicate in self.policy.protect_predicates:
                report.kept_because_protected += 1
                continue
            if any(e.source.id.startswith(prefix) for e in rec.evidence for prefix in self.policy.protect_sources):
                report.kept_because_testimony += 1
                continue
            newest = max((e.observed_at for e in rec.evidence), default=when)
            age = when - newest
            salience = self.working.salience(cid) if cid in aware else 0.0
            if age < self.policy.half_life or salience >= self.policy.min_salience:
                continue
            if self.mind._dependents.get(cid):
                report.kept_because_depended_on += 1
                continue
            doomed.append(cid)
        self.mind.forget(doomed)
        report.claims_forgotten = len(doomed)
        for episode in self.episodes():
            if len(self._episodes) <= self.policy.episodic_capacity:
                break
            if episode.salience < self.policy.min_salience:
                self._episodes.pop(episode.ref, None)
                self.mind.forget([r.id for r in self.mind.claims(subject=episode.ref)])
                report.episodes_dropped += 1
        report.ms = (_time.perf_counter() - t0) * 1e3
        return report

    # --------------------------------------------------------------- cycle

    def attend(self, seeds: Sequence[Any], *, fade: bool = True) -> Awareness:
        """One cycle of working memory: fade what was aware, seed from what just arrived, spread."""
        if fade:
            self.working.fade()
        self.working.seed(list(seeds))
        self.working.spread()
        return self.working


def _describe(rec: ClaimRecord) -> str:
    c = rec.claim
    obj = getattr(c.object, "value", c.object)
    return f"{c.subject.id.split(':', 1)[-1]} {c.predicate} {obj}"
