"""Social cognition: what two minds share, what the other one believes, and when to ask.

Four structures, each answering a question the flat claim store cannot:

* :class:`CommonGround` — what is *mutually manifest*: I said it, you said it, or we both
  saw it. A reply can then mark what is new and lean on what is already shared, instead of
  re-explaining. Grounding is itself claims, so ``explain`` reaches it.
* :class:`OtherMind` — what I take *you* to believe, including where I think you are wrong.
  A request presupposes things ("delete the report" presupposes a report); a presupposition
  I can check and find false is a belief worth correcting, not just a fact to report.
* :func:`ask_or_act` — clarification as a decision, not a reflex: ask only when the expected
  cost of acting on the wrong reading exceeds the cost of a question. Both failure modes are
  represented, because an assistant that always asks is as useless as one that never does.
* :func:`indirect_reading` — intention inference gated by *affordance*. A statement or
  question may imply a request, but only if its object is something I could act on: "I can't
  find my invoice" is a request to look, "I can't find my keys" is not, and the difference is
  not in the grammar.

Nothing here calls a model. Everything is deterministic and carries provenance.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .cognition import Fragment, Thought, integrate
from .outcomes import Score, Unknown
from .records import Claim, Ref, Store

AGENT = Ref("agent:self")
OTHER = Ref("person:user")

#: how a claim came to be shared
I_SAID = "i_said"
YOU_SAID = "you_said"
BOTH_SAW = "both_saw"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _key(about: str, how: str) -> Ref:
    return Ref(f"ground:{hashlib.sha256(f'{about}|{how}'.encode()).hexdigest()[:12]}")


@dataclass(frozen=True)
class Shared:
    """One piece of common ground: what, how it became shared, when, how often mentioned."""

    about: str  # the claim id it grounds
    how: str  # I_SAID | YOU_SAID | BOTH_SAW
    turn: int
    mentions: int

    @property
    def mine(self) -> bool:
        return self.how == I_SAID


class CommonGround:
    """The shared state between two minds, kept as claims so it can be explained.

    Mutual manifestness is not symmetry of storage: a fact I hold is not shared until it was
    said or jointly seen. ``status`` answers "does the other mind already have this, and how",
    which is what lets a reply say "as I mentioned" rather than repeating itself.
    """

    def __init__(self, store: Store, *, me: Ref = AGENT, other: Ref = OTHER) -> None:
        self.store, self.me, self.other = store, me, other

    def add(self, about: str, how: str, turn: int, *, source: str | None = None) -> Thought:
        """Record that a claim (by id) became shared, or was mentioned again."""
        ref = _key(about, how)
        before = self.mentions(about, how)
        claims = [
            (Claim(ref, "grounds", about), None),
            (Claim(ref, "shared_via", how), None),
            (Claim(ref, "mentioned", before + 1), None),
            (Claim(ref, "turn", turn), None),
        ]
        if before:  # a repeat supersedes the old count rather than piling up
            self.store.forget([r.id for r in self.store.claims(ref, "mentioned")])
            self.store.forget([r.id for r in self.store.claims(ref, "turn")])
        frag = Fragment(Ref(source or f"utterance:{turn}"), tuple(claims), method="grounding", observed_at=_now())
        return integrate(self.store, frag)

    def mentions(self, about: str, how: str | None = None) -> int:
        hows = [how] if how else [I_SAID, YOU_SAID, BOTH_SAW]
        total = 0
        for h in hows:
            found = self.store.claims(_key(about, h), "mentioned")
            total += max((int(r.claim.object) for r in found), default=0)
        return total

    def status(self, about: str) -> Shared | None:
        """How this claim is shared, preferring the strongest ground (jointly seen > said)."""
        for how in (BOTH_SAW, YOU_SAID, I_SAID):
            ref = _key(about, how)
            said = self.store.claims(ref, "mentioned")
            if said:
                turn = max((int(r.claim.object) for r in self.store.claims(ref, "turn")), default=0)
                return Shared(about, how, turn, max(int(r.claim.object) for r in said))
        return None

    def is_shared(self, about: str) -> bool:
        return self.status(about) is not None

    def new_to_other(self, about: str) -> bool:
        """True when the other mind has not been given this, so a reply should state it plainly."""
        return not self.is_shared(about)

    def again(self, about: str) -> bool:
        """True when I am about to say something I have already said: worth marking, not repeating.

        Asks my own ground specifically — that you told me a thing is not a reason for me to
        say "as I mentioned", and the strongest ground is not the relevant one here.
        """
        return self.mentions(about, I_SAID) >= 1

    def told_me(self, turn_from: int = 0) -> list[Shared]:
        """What the other mind told me, oldest first — the answer to "what did I tell you"."""
        out = []
        for rec in self.store.claims(predicate="shared_via", object=YOU_SAID):
            about = next((r.claim.object for r in self.store.claims(rec.claim.subject, "grounds")), None)
            st = self.status(str(about)) if about else None
            if st and st.turn >= turn_from:
                out.append(st)
        return sorted(out, key=lambda s: s.turn)


# --------------------------------------------------------------- other minds


@dataclass(frozen=True)
class Presupposition:
    """Something a request takes for granted, and can therefore be wrong about."""

    kind: str  # "exists" | "location" | "state" | "capability"
    subject: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.kind}({self.subject}{', ' + self.detail if self.detail else ''})"


@dataclass(frozen=True)
class FalseBelief:
    """A presupposition I checked and found false, with what to say instead."""

    presupposition: Presupposition
    truth: str
    near: tuple[str, ...] = ()

    def correction(self) -> str:
        """A correction of the belief, not a bare report of absence."""
        base = f"there's no {self.presupposition.subject}"
        if self.presupposition.detail:
            base += f" {self.presupposition.detail}"
        if self.near:
            joined = self.near[0] if len(self.near) == 1 else ", ".join(self.near[:-1]) + f" or {self.near[-1]}"
            return f"{base} — did you mean {joined}?"
        return f"{base}; {self.truth}" if self.truth else base


class OtherMind:
    """What I take the other mind to believe, see, and take for granted.

    Two asymmetries matter. What they told me, they believe (until they say otherwise). What
    is on a screen we both look at, they can see — but what sits inside a folder they never
    opened, they cannot, so it is not shared context and a reply should not assume it.
    """

    def __init__(self, store: Store, *, who: Ref = OTHER, visible: Callable[[str], bool] | None = None) -> None:
        self.store, self.who = store, who
        self._visible = visible or (lambda _thing: False)

    def believes(self, predicate: str) -> object | None:
        found = self.store.claims(self.who, predicate)
        return found[0].claim.object if found else None

    def beliefs(self) -> dict[str, object]:
        return {r.claim.predicate: r.claim.object for r in self.store.claims(self.who)}

    def can_see(self, thing: str) -> bool:
        return self._visible(thing)

    def presupposes(self, slots: Mapping[str, object], rules: Mapping[str, str]) -> list[Presupposition]:
        """What a request takes for granted, given which slots presuppose what.

        ``rules`` maps a slot name to a presupposition kind, so the domain decides: a
        ``target`` slot presupposes existence, a ``place`` slot presupposes a location.
        """
        out = []
        for slot, kind in rules.items():
            value = slots.get(slot)
            if isinstance(value, str) and value and not value.startswith("@"):
                detail = ""
                if kind == "exists" and isinstance(slots.get("place"), str) and str(slots["place"]).startswith("~"):
                    detail = f"in {slots['place']}"
                out.append(Presupposition(kind, value, detail))
        return out

    def check(self, presupposition: Presupposition, *, exists: Callable[[str], bool],
              neighbours: Callable[[str], Sequence[str]] = lambda _s: ()) -> FalseBelief | None:
        """A presupposition I can check: return a false belief when it does not hold."""
        if presupposition.kind not in ("exists", "location"):
            return None
        if exists(presupposition.subject):
            return None
        near = tuple(neighbours(presupposition.subject))[:3]
        return FalseBelief(presupposition, truth="", near=near)


def near_names(name: str, candidates: Iterable[str], *, limit: int = 3) -> list[str]:
    """Names close enough that the other mind plausibly meant one of them.

    Cognitively this is why a correction beats a bare "not found": a wrong name is usually a
    near miss, and naming the near miss repairs the belief instead of ending the exchange.
    """
    target = name.lower()
    stem = target.rsplit(".", 1)[0]
    scored: list[tuple[float, str]] = []
    for cand in candidates:
        c = cand.lower().rstrip("/")
        if c == target:
            continue
        c_stem = c.rsplit(".", 1)[0]
        score = max(_ratio(target, c), _ratio(stem, c_stem))
        if stem and (stem in c_stem or c_stem in stem):
            score = max(score, 0.75)
        if score >= 0.6:
            scored.append((score, cand))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [c for _, c in scored[:limit]]


def _ratio(a: str, b: str) -> float:
    """Similarity by edit distance, normalized — small and dependency-free on purpose."""
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


# ------------------------------------------------------------- clarification


@dataclass(frozen=True)
class Reading:
    """One way to take an utterance, with how likely it is and what acting on it would cost."""

    name: str
    probability: float
    description: str = ""
    cost_if_wrong: float = 1.0


@dataclass(frozen=True)
class Clarification:
    """A question worth asking, with the options it puts to the other mind."""

    question: str
    options: tuple[str, ...]
    expected_gain: float
    about: str = ""


@dataclass(frozen=True)
class Decision:
    """Ask, act, or admit — and why, in numbers a reader can check."""

    choose: str  # "act" | "ask" | "admit"
    reading: Reading | None = None
    clarification: Clarification | None = None
    expected_gain: float = 0.0
    reason: str = ""


def uncertainty(readings: Sequence[Reading]) -> float:
    """Entropy over readings, in bits: how undetermined the goal is."""
    total = sum(max(r.probability, 0.0) for r in readings) or 1.0
    bits = 0.0
    for r in readings:
        p = max(r.probability, 0.0) / total
        if p > 0:
            bits -= p * math.log2(p)
    return bits


def ask_or_act(readings: Sequence[Reading], *, ask_cost: float = 0.25, confident: float = 0.8,
               question: str | None = None, about: str = "") -> Decision:
    """Decide between acting on the best reading, asking, and admitting the goal is unclear.

    The two failure modes are both priced: acting on the wrong reading costs ``cost_if_wrong``,
    asking costs ``ask_cost`` (a turn of the other mind's patience). Asking wins only when the
    expected cost it avoids exceeds that, which is what keeps a clarifying assistant from
    becoming a tiresome one.
    """
    live = [r for r in readings if r.probability > 0]
    if not live:
        return Decision("admit", reason="no reading at all")
    best = max(live, key=lambda r: r.probability)
    total = sum(r.probability for r in live) or 1.0
    p_best = best.probability / total
    if p_best >= confident or len(live) == 1:
        return Decision("act", reading=best, expected_gain=0.0, reason=f"one reading dominates (p={p_best:.2f})")
    expected_loss = sum((r.probability / total) * r.cost_if_wrong for r in live if r is not best)
    gain = expected_loss - ask_cost
    if gain <= 0:
        return Decision("act", reading=best, expected_gain=gain,
                        reason=f"asking costs more than the risk ({expected_loss:.2f} vs {ask_cost:.2f})")
    options = tuple(r.description or r.name for r in sorted(live, key=lambda r: -r.probability))
    text = question or "Which did you mean?"
    return Decision("ask", reading=best,
                    clarification=Clarification(text, options, gain, about),
                    expected_gain=gain, reason=f"{uncertainty(live):.2f} bits undetermined, {len(live)} readings")


# ------------------------------------------------------ indirect and implied


@dataclass(frozen=True)
class Indirect:
    """A request recovered from an utterance whose surface form is something else."""

    surface: str  # "statement" | "question" | "complaint" | "wish"
    act: str
    slots: dict
    strength: float
    object_word: str = ""

    def score(self) -> Score:
        return Score(self.strength, "implicature", basis=f"{self.surface} form implying {self.act}")


@dataclass(frozen=True)
class Implication:
    """A surface form that may imply an act, and what it needs to be believed."""

    pattern: str  # a regex with an ``obj`` group where the thing acted on appears
    surface: str
    act: str
    strength: float
    slot: str = "target"


def indirect_reading(text: str, implications: Sequence[Implication], *,
                     in_domain: Callable[[str], bool], floor: float = 0.5) -> Indirect | Unknown:
    """Read an indirect request, but only where the object is something I could act on.

    The gate is the point. Form alone cannot tell "I can't find my invoice" (a request to
    look) from "I can't find my keys" (not my business); what separates them is whether the
    object falls inside what I can act on at all. So intention inference is constrained by
    affordance, and an utterance about the world beyond my reach stays a remark.
    """
    import re

    for imp in implications:
        m = re.search(imp.pattern, text, re.I)
        if not m:
            continue
        obj = (m.groupdict().get("obj") or "").strip(" .!?\"'")
        if not obj:
            continue
        if not in_domain(obj):
            return Unknown("not_my_domain", f"“{obj}” is not something I can act on, so I read that as a remark")
        if imp.strength < floor:
            continue
        return Indirect(imp.surface, imp.act, {imp.slot: obj}, imp.strength, obj)
    return Unknown("no_implicature", "no indirect reading fits this utterance")
