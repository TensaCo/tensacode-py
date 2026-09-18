"""The cognitive core: vision and language in, action and language out, one claim store between.

    agent = Agent([DesktopPlugin(...)])
    turn = agent.turn("make a folder called recipes on my desktop")
    turn.reply          # what it says back, once per message
    turn.outcomes       # what each request, question and statement came to
    turn.events         # everything a viewer may show, as data

One turn:

1. **perceive** — every plugin reports what is true now, as claims;
2. **read** — the message is split into sentences and parsed (``understand.py``);
3. for each act, in order:
   * a **statement** is integrated as claims scoped to the user (what they said, not
     what is thereby true of the world);
   * a **question** is matched against the store; if nothing answers it, a capability
     that *informs* on that predicate is run and the question is asked again;
   * a **request** becomes a goal state (``language/verbnet.py``), and a capability
     whose effects achieve it — with arguments whose kinds fit — is run through
     ``tc.invoke`` and then checked against a fresh observation;
4. **reply** — one message, generated from what happened.

Nothing in this module knows any plugin, any verb, or any phrase.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from ..actions import invoke
from ..language import ENGLISH, Context, Entity, Frame, Grammar, Question, Request, resolve
from ..language import verbnet, wordnet
from ..language.semantics import default_ref, to_claims
from ..outcomes import Receipt, Unknown
from ..records import Claim, Evidence, Ref, Store
from ..runtime import Runtime, use
from .plugin import Call, Capability, Plugin
from .understand import Act, Sentence, read

USER = Ref("agent:user")
SELF = Ref("agent:self")

#: The grammar's own predicates for being somewhere, against VerbNet's. Like the role
#: correspondence in ``verbnet.py``, this aligns two vocabularies; it knows no domain.
PREDICATE_OF = {"located": "has_location", "be": "be", "have": "has_possession"}

#: The claim predicate a wh-word asks for. ``to_claims`` files a frame's roles under their
#: own names, so most of these are the role itself.
ASKED_PREDICATE = {"theme": "object", "time": "time", "location": "location"}

#: WordNet kinds that make a noun a time rather than a place, so "on Tuesday" is when and
#: "on my desktop" is where, without a list of time words.
TIME_KINDS = frozenset({"time period", "clock time", "time unit", "calendar day", "calendar week",
                        "calendar month", "date", "season"})


@dataclass(frozen=True)
class Outcome:
    """What one act came to."""

    act: Act
    status: str                       # done | failed | unverified | answered | unknown | noted | declined | not_understood
    goal: Any = None                  # verbnet.Goal | Unknown
    plan: Any = None                  # (plugin, capability, args) | Unknown
    receipt: Receipt | None = None
    verified: Any = None              # True | False | Unknown
    answer: Any = None
    reason: str = ""


@dataclass
class Turn:
    text: str
    sentences: list[Sentence]
    outcomes: list[Outcome]
    reply: str
    events: list[dict] = field(default_factory=list)
    seconds: float = 0.0


class Agent:
    def __init__(self, plugins: Sequence[Plugin] = (), *, grammar: Grammar | None = None, runtime: Runtime | None = None) -> None:
        self.plugins = list(plugins)
        base = grammar or ENGLISH
        lexicon = wordnet.seed_lexicon(base.lexicon)
        extra = [e for p in self.plugins for e in p.lexicon]
        if extra:
            lexicon = lexicon.extend(*extra)
        self.grammar = replace(base, lexicon=lexicon)
        self.verbs = verbnet.load()
        self.taxonomy = wordnet.taxonomy()
        self.store = Store()
        self.context = Context()
        self.runtime = runtime or Runtime()
        self.turns: list[Turn] = []
        self._calls = 0
        self._guessed: set[str] = set()
        self._images = 0
        self.last_image: Ref | None = None

    # ------------------------------------------------------------------ kinds

    def kinds(self, noun: str) -> frozenset[str]:
        """What ``noun`` is a kind of: WordNet's hierarchy joined with every plugin's links."""
        noun = noun.lower()
        out = set(self.taxonomy.kinds(noun)) if self.taxonomy else {noun}
        frontier = list(out)
        while frontier:
            n = frontier.pop()
            for p in self.plugins:
                for k in p.kinds.get(n, ()):
                    if k not in out:
                        out.add(k)
                        frontier.append(k)
                        if self.taxonomy:
                            for kk in self.taxonomy.kinds(k):
                                if kk not in out:
                                    out.add(kk)
                                    frontier.append(kk)
        return frozenset(out)

    def fits(self, filler: Any, kind: str) -> bool:
        noun = noun_of(filler)
        if noun is None:
            return True
        said = filler.text.lower() if isinstance(filler, Entity) and filler.kind == "description" else noun
        return kind.lower() in self.kinds(noun) or kind.lower() in self.kinds(said)

    # ------------------------------------------------------------------ the turn

    def turn(self, text: str, images: Sequence[Any] = ()) -> Turn:
        """One message: optional images first (seen by every plugin that sees), then the text."""
        t0 = time.perf_counter()
        events: list[dict] = []
        with use(self.runtime):
            self.perceive(events)
            for image in images:
                self._images += 1
                ref = Ref(f"image:{self._images}")
                self.last_image = ref
                n = 0
                for p in self.plugins:
                    for claim in p.see(image, ref):
                        self.store.tell(claim, Evidence(source=Ref(f"plugin:{p.name}"), observed_at=datetime.now(timezone.utc), method="vision"))
                        n += 1
                events.append({"type": "seen", "image": ref.id, "claims": n})
            sents = read(self.grammar, text)
            for s in sents:
                events.append({"type": "parsed", "sentence": s.text, "coverage": s.coverage, "skipped": list(s.skipped),
                               "guessed": [list(g) for g in s.guessed], "acts": [a.describe() for a in s.acts], "ms": s.parse_ms})
            outcomes = []
            requests_in_message = sum(1 for s in sents for a in s.acts if a.kind == "request")
            for s in sents:
                for a in s.acts:
                    a = replace(a, meaning=resolve(a.meaning, self.context)) if a.frame is not None else a
                    o = self.handle(s, a, events, requests_in_message=requests_in_message)
                    outcomes.append(o)
                    if a.frame is not None:
                        self.context.observe(a.meaning)
            from .reply import compose

            reply = compose(self, sents, outcomes)
        turn = Turn(text, sents, outcomes, reply, events, round(time.perf_counter() - t0, 3))
        self.turns.append(turn)
        return turn

    def perceive(self, events: list[dict]) -> None:
        for p in self.plugins:
            n = 0
            for claim in p.perceive():
                self.store.tell(claim, Evidence(source=Ref(f"plugin:{p.name}"), observed_at=datetime.now(timezone.utc), method="perception"))
                n += 1
            events.append({"type": "perceived", "plugin": p.name, "claims": n})

    def deixis(self, value: Any) -> Any:
        """Bind the speech participants: I/me/my is the user, you/your is this agent."""
        if isinstance(value, Entity):
            person = value.features.get("person")
            if value.kind == "pronoun" and value.ref is None and person in (1, 2):
                return value.with_ref(USER if person == 1 else SELF)
            feats = {k: self.deixis(v) for k, v in value.features.items()}
            return Entity(value.kind, value.text, feats, value.ref, value.candidates)
        if isinstance(value, Frame):
            return Frame(value.predicate, {k: self.deixis(v) for k, v in value.roles.items()}, value.features)
        if isinstance(value, tuple):
            return tuple(self.deixis(v) for v in value)
        return value

    def when_not_where(self, frame: Frame) -> Frame:
        """A locative phrase whose object is a time is a time ("on Tuesday" vs "on my desktop")."""
        filler = frame.roles.get("location")
        noun = noun_of(filler)
        if noun and TIME_KINDS & self.kinds(noun):
            roles = {k: v for k, v in frame.roles.items() if k != "location"}
            roles["time"] = filler
            return Frame(frame.predicate, roles, frame.features)
        return frame

    def handle(self, s: Sentence, act: Act, events: list[dict], *, requests_in_message: int) -> Outcome:
        if act.frame is not None:
            frame = self.when_not_where(self.deixis(act.frame))
            meaning = act.meaning
            if isinstance(meaning, Request):
                meaning = Request(frame)
            elif isinstance(meaning, Question):
                meaning = Question(frame, meaning.asked)
            else:
                meaning = frame
            act = replace(act, meaning=meaning, frame=frame)
        if act.kind == "tell":
            return self.tell(s, act, events)
        if act.kind == "question":
            return self.ask(s, act, events)
        if act.kind == "request":
            return self.request(s, act, events)
        if act.kind == "mention":
            return Outcome(act, "mentioned")
        return Outcome(act, "not_understood", reason="a phrase that is not a statement, question or request")

    # ------------------------------------------------------------------ statements

    def tell(self, s: Sentence, act: Act, events: list[dict]) -> Outcome:
        got = to_claims(act.frame, source=USER, scope=USER)
        if isinstance(got, Unknown):
            return Outcome(act, "not_understood", reason=got.reason)
        for claim, ev in got:
            self.store.tell(claim, ev)
        events.append({"type": "noted", "claims": len(got), "frame": act.frame.describe()})
        return Outcome(act, "noted", answer=len(got))

    # ------------------------------------------------------------------ questions

    def ask(self, s: Sentence, act: Act, events: list[dict]) -> Outcome:
        """Look if it can be looked at; otherwise answer from what it was told or saw before."""
        q: Question = act.meaning
        pred = PREDICATE_OF.get(q.frame.predicate, q.frame.predicate)
        looked = self._look(q, pred, act, events)
        if looked is not None:
            return looked
        found = self.lookup(q)
        if found:
            return Outcome(act, "answered", answer=found)
        if any(self._ref_of(v) is not None and str(self._ref_of(v).id).startswith("image:") for v in q.frame.roles.values()):
            return Outcome(act, "unknown", reason="I couldn't recognise anything in it confidently enough to say")
        return Outcome(act, "unknown", reason="nothing I know or can look up answers it")

    def _look(self, q: Question, pred: str, act: Act, events: list[dict]) -> Outcome | None:
        for p in self.plugins:
            for cap in p.capabilities():
                for inf in cap.informs:
                    if inf.pred != pred:
                        continue
                    role_filler = self._filler_for_role(q.frame, inf.role)
                    if role_filler is None:
                        continue
                    param = cap.param(inf.param)
                    arg = p.refer(role_filler, param, context={"store": self.store})
                    if isinstance(arg, Unknown):
                        continue
                    receipt = self._invoke(p, cap, {inf.param: arg}, events)
                    fresh = list(p.reveal(cap, {inf.param: arg}, receipt))
                    if receipt.status == "applied":
                        # a new look replaces the old one: what is no longer seen there is forgotten
                        stale = [r.id for r in self.store.claims(predicate=inf.pred)
                                 if (r.claim.object == arg if inf.role != "undergoer" else r.claim.subject == arg)
                                 and r.claim not in fresh]
                        self.store.forget(stale)
                    for claim in fresh:
                        self.store.tell(claim, Evidence(source=Ref(f"plugin:{p.name}"), observed_at=datetime.now(timezone.utc), method=cap.name))
                    found = self.lookup(q)
                    if receipt.status == "applied":
                        # it looked, and this is what is there — possibly nothing at all
                        return Outcome(act, "answered", plan=(p.name, cap.name, {inf.param: arg}), receipt=receipt, answer=found)
                    return Outcome(act, "unknown", plan=(p.name, cap.name, {inf.param: arg}), receipt=receipt,
                                   reason=receipt.error or f"{cap.name} did not run")
        return None

    def _filler_for_role(self, frame: Frame, role: str) -> Any:
        """The frame's filler for a role class (``goal``: where; ``undergoer``: what)."""
        for grammar_role, value in frame.roles.items():
            classes = {verbnet.role_class(r) for r in verbnet.ROLE_OF_PREPOSITION_ROLE.get(grammar_role, ())}
            if grammar_role in ("subject", "object") and role == "undergoer":
                return value
            if role in classes:
                return value
        return None

    def lookup(self, q: Question) -> list[tuple[Claim, str]]:
        """Claims that answer ``q``, each with the side of it that is the answer.

        The question binds the roles it states ("my name", "the meeting"); a claim answers
        only if every bound side matches it. The answer is the side the question left
        open — never one it already gave, which is how "what is my name?" answered "name".
        """
        pred = PREDICATE_OF.get(q.frame.predicate, q.frame.predicate)
        bound = {role: self._ref_of(v) for role, v in q.frame.roles.items() if role != q.asked}
        bound = {r: v for r, v in bound.items() if v is not None}
        if not bound:
            return []
        taken = set(bound.values())
        out: list[tuple[Claim, str]] = []
        for rec in self.store.claims(predicate=pred):
            c = rec.claim
            if not taken <= {c.subject, c.object}:
                continue
            if c.object not in taken:
                out.append((c, "object"))
            elif c.subject not in taken:
                out.append((c, "subject"))
        return out + self._through_events(q, taken)

    def _through_events(self, q: Question, taken: set) -> list[tuple[Claim, str]]:
        """Answers one hop away, through an event the store reified.

        "the meeting is on Tuesday" is stored as an event with a subject and a time, so
        "when is the meeting?" is: the event whose subject is the meeting, then its time.
        """
        wanted = ASKED_PREDICATE.get(q.asked, q.asked)
        events = {rec.claim.subject for rec in self.store.claims()
                  if rec.claim.object in taken and str(rec.claim.subject.id).startswith("event:")}
        return [(rec.claim, "object") for rec in self.store.claims(predicate=wanted) if rec.claim.subject in events]

    def _ref_of(self, value: Any) -> Ref | None:
        """What a description picks out: a resolved reference, an image, a plugin's
        entity, or the identity the claim store mints for that same description."""
        if not isinstance(value, Entity):
            return None
        if value.ref is not None:
            return value.ref
        noun = noun_of(value)
        if self.last_image is not None and noun and "representation" in self.kinds(noun):
            return self.last_image
        for p in self.plugins:
            got = p.denote(value)
            if isinstance(got, Ref):
                return got
        minted = default_ref(value)
        return minted if isinstance(minted, Ref) else None

    # ------------------------------------------------------------------ requests

    def request(self, s: Sentence, act: Act, events: list[dict]) -> Outcome:
        missed = [w for w in s.skipped if any(c.isalnum() for c in w)]
        if missed:
            # acting on part of a sentence is how "processes" became a process listing
            return Outcome(act, "not_understood", reason=f"I didn't follow {' '.join(repr(w) for w in missed)}")
        self._guessed = {w.lower() for w, _ in s.guessed}
        goal = verbnet.goal_of(act.frame, self.verbs)
        events.append({"type": "goal", "frame": act.frame.describe(),
                       "goal": goal.describe() if hasattr(goal, "describe") else f"unknown: {goal.reason}"})
        if isinstance(goal, Unknown):
            return Outcome(act, "unknown", goal=goal, reason=goal.detail or goal.reason)
        plan = self.plan(goal)
        events.append({"type": "plan", "goal": goal.describe(),
                       "plan": {"unknown": plan.reason, "detail": plan.detail} if isinstance(plan, Unknown)
                       else {"plugin": plan[0].name, "capability": plan[1].name, "args": {k: str(v) for k, v in plan[2].items()}}})
        if isinstance(plan, Unknown):
            return Outcome(act, "declined", goal=goal, plan=plan, reason=plan.detail or plan.reason)
        plugin, cap, args = plan
        receipt = self._invoke(plugin, cap, args, events)
        if receipt.status in ("rejected", "failed"):
            return Outcome(act, "failed", goal, (plugin.name, cap.name, args), receipt, False, reason=receipt.error or receipt.status)
        self.perceive(events)
        verified = plugin.holds(cap, args)
        events.append({"type": "verified", "capability": cap.name, "holds": verified if isinstance(verified, bool) else f"unknown: {verified.reason}"})
        status = "done" if verified is True else "failed" if verified is False else "unverified"
        return Outcome(act, status, goal, (plugin.name, cap.name, args), receipt, verified,
                       reason="" if verified is True else "the effect was not observed afterwards" if verified is False else verified.reason)

    def plan(self, goal: verbnet.Goal) -> tuple[Plugin, Capability, dict] | Unknown:
        """The capability whose effects achieve the most goal conditions, with arguments that fit.

        A condition is achieved by an effect with the same predicate and polarity whose
        every thematic role is either open in the goal or filled by something whose kind
        fits the parameter and that the plugin can refer to.
        """
        best: tuple[int, int, Plugin, Capability, dict] | None = None
        nearest: list[str] = []
        for p in self.plugins:
            for cap in p.capabilities():
                args: dict[str, Any] = {}
                met = 0
                ok = True
                for cond in goal.conditions:
                    filled = {verbnet.role_class(r): v for r, v in cond.args.items() if v is not None and v != "addressee"}
                    effect = next((e for e in cap.effects if e.pred == cond.pred and e.negated == cond.negated
                                   and set(filled) <= set(e.roles)), None)
                    if effect is None:
                        continue
                    for role, pname in effect.roles.items():
                        filler = filled.get(role)
                        if filler is None or filler == "addressee":
                            continue
                        param = cap.param(pname)
                        if param is None:
                            continue
                        if not self.fits(filler, param.kind):
                            ok = False
                            # a type inferred for a word the parser only guessed at is not
                            # evidence; then the plugin's own account is the better reason
                            said = p.refer(filler, param, context={"store": self.store, "args": args}) \
                                if text_of(filler).lower() in self._guessed else None
                            if isinstance(said, Unknown) and said.detail:
                                nearest.insert(0, said.detail)
                            else:
                                nearest.append(f"{cap.name} wants a {param.kind} for {role}, and {text_of(filler)} is not one")
                            break
                        ref = p.refer(filler, param, context={"store": self.store, "args": args})
                        if isinstance(ref, Unknown):
                            ok = False
                            nearest.append(f"{cap.name}: {ref.detail or ref.reason}")
                            break
                        args[pname] = ref
                    if not ok:
                        break
                    met += 1
                required = [c for c in goal.conditions if is_specified(c)]
                if ok and required and not all(self._achieves(cap, c) for c in required):
                    # never act on part of what was asked: a condition the speaker fully
                    # specified ("move it *to documents*") must be among the effects
                    nearest.append(f"{cap.name} would do only part of it")
                    ok = False
                if ok and met and all(p_.name in args for p_ in cap.params):
                    key = (met, -len(cap.params))
                    if best is None or key > (best[0], best[1]):
                        best = (met, -len(cap.params), p, cap, args)
        if best is None:
            return Unknown("no_capability", self._why_not(goal, nearest))
        return best[2], best[3], best[4]

    def _achieves(self, cap: Capability, cond: verbnet.Condition) -> bool:
        filled = {verbnet.role_class(r) for r, v in cond.args.items() if v is not None and v != "addressee"}
        return any(e.pred == cond.pred and e.negated == cond.negated and filled <= set(e.roles) for e in cap.effects)

    def _why_not(self, goal: verbnet.Goal, nearest: list[str]) -> str:
        """Why nothing achieves ``goal``, from the capabilities themselves.

        Either no capability brings about any of the goal's predicates at all, or some
        do but only for kinds of things the request's objects are not.
        """
        by_pred: dict[tuple[str, bool], list[tuple[str, list[str]]]] = {}
        for p in self.plugins:
            for cap in p.capabilities():
                for e in cap.effects:
                    kinds = [x.kind for x in cap.params if x.name in e.roles.values()]
                    by_pred.setdefault((e.pred, e.negated), []).append((cap.name, kinds))
        for cond in goal.conditions:
            fillers = [v for v in cond.args.values() if v is not None and v != "addressee"]
            who = ", ".join(text_of(v) for v in fillers) or "it"
            makers = by_pred.get((cond.pred, cond.negated))
            if not makers:
                continue
            kinds = sorted({k for _, ks in makers for k in ks})
            if nearest and "only part" not in nearest[0] and "wants a" not in nearest[0] and \
                    any(text_of(v).lower() in self._guessed for v in fillers):
                return nearest[0]  # the plugin's own account of why (e.g. no such app here)
            return (f"what I can do brings that about only for {' or '.join(kinds)}"
                    f"{'s' if len(kinds) == 1 else ''}, and {who} is not one" if all(not self.fits(v, k) for v in fillers for k in kinds)
                    else nearest[0] if nearest else f"I could not work out which {' or '.join(kinds)} you mean")
        preds = ", ".join(sorted({("not " if c.negated else "") + c.pred for c in goal.conditions}))
        return f"nothing I can do brings about {preds}" + (f" ({nearest[0]})" if nearest else "")


    def _invoke(self, plugin: Plugin, cap: Capability, args: Mapping[str, Any], events: list[dict]) -> Receipt:
        self._calls += 1
        act = Call(plugin.name, cap.name, tuple(sorted(args.items(), key=lambda kv: kv[0])))
        events.append({"type": "act", "plugin": plugin.name, "capability": cap.name, "args": {k: str(v) for k, v in args.items()}})
        receipt = invoke(act, executor=plugin, key=f"{plugin.name}:{self._calls}")
        events.append({"type": "receipt", "capability": cap.name, "status": receipt.status, "error": receipt.error})
        return receipt


def is_specified(cond: verbnet.Condition) -> bool:
    """Every role of the condition (other than the doer) was filled by what the speaker said."""
    roles = {r: v for r, v in cond.args.items() if verbnet.role_class(r) != "actor"}
    return bool(roles) and all(v is not None and v != "addressee" for v in roles.values())


def noun_of(filler: Any) -> str | None:
    if isinstance(filler, Entity):
        return filler.features.get("noun") or (filler.text if filler.kind == "description" else None)
    return None


def text_of(filler: Any) -> str:
    return getattr(filler, "text", str(filler))

