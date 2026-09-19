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
from ..language.semantics import SYMMETRIC_PREDICATES, default_ref, to_propositions
from ..outcomes import Receipt, Unknown
from ..records import Evidence, Proposition, Ref, Store, Var
from ..runtime import Runtime, use
from .. import ops
from .operations import Plan, Transcript, agent_runtime, install_learned_reader
from .plugin import Call, Capability, Plugin
from .understand import Act, Sentence

USER = Ref("agent:user")
SELF = Ref("agent:self")

#: The grammar's own predicates for being somewhere, against VerbNet's. Like the role
#: correspondence in ``verbnet.py``, this aligns two vocabularies; it knows no domain.
PREDICATE_OF = {"located": "has_location", "be": "be", "have": "has_possession"}

#: The grammar roles that carry a core participant, as against an adjunct. ``_filler_for_role``
#: uses the same convention: a subject or object is the thing the predication is about.
CORE_ROLES = ("object", "complement", "subject")

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
    def __init__(self, plugins: Sequence[Plugin] = (), *, grammar: Grammar | None = None, runtime: Runtime | None = None,
                 reader: Any = None) -> None:
        """``reader`` names which registered ``parse`` implementation to prefer.

        The default is the hand-written grammar and ``"learned"`` is the treebank one. An
        already-built :class:`~tensorcode.agent.understand.LearnedReader` may be handed in
        instead, in which case it is reused rather than loaded again. Either way the reading
        happens through :func:`tensorcode.ops.parse`, so a runtime whose policy orders
        implementations differently switches readers without touching this class.
        """
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
        self.reader = reader
        self.prefer_reader = None
        if isinstance(reader, str):
            self.prefer_reader = reader
        elif reader is not None:
            install_learned_reader(reader)
            self.prefer_reader = "learned"
        self.runtime = runtime or agent_runtime(prefer_reader=self.prefer_reader)
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
            transcript = ops.parse(text, Transcript, grammar=self.grammar, prefer=self.prefer_reader)
            if isinstance(transcript, Unknown):
                # no reader could be used here; say so rather than acting on nothing
                events.append({"type": "unread", "reason": transcript.reason, "detail": transcript.detail})
                transcript = Transcript()
            events.append({"type": "read", "by": transcript.by, "sentences": len(transcript)})
            sents = list(transcript)
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
        """Record what was said, as it was said.

        One proposition per predication, with the sentence's own roles. Nothing is reified
        into invented event nodes, so nothing downstream has to guess what they meant.
        """
        got, dropped = to_propositions(act.frame, source=USER, scope=USER,
                                       method=f"told:{self.prefer_reader or 'grammar'}")
        if not got:
            events.append({"type": "unrecorded", "frame": act.frame.describe(), "why": dropped})
            return Outcome(act, "not_understood",
                           reason=dropped[0] if dropped else "nothing in it was a statement I could record")
        for proposition, ev in got:
            self.store.assert_(proposition, ev)
        events.append({"type": "noted", "propositions": len(got), "frame": act.frame.describe(),
                       **({"dropped": dropped} if dropped else {})})
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

    def lookup(self, q: Question) -> list[Any]:
        """What answers ``q``: the fillers its hole binds to.

        The question is a proposition with a hole where the wh-word stood and the roles it
        states filled in. A recorded proposition answers when everything stated agrees,
        and the answer is what the hole bound to — never a role the question itself
        supplied, which is how "what is my name?" once answered "name".

        Two kinds of question, told apart by VerbNet's own role classes rather than by a
        table here:

        * one that **names the role** it wants — where, when, why, how — becomes that role
          filled with a hole;
        * one that asks for a **participant** ("what", "who"), whose role classes as an
          undergoer, matches on what the asker stated and answers with the core role left
          open. "How many" asks for a quantity, which classes as itself, so it gets no
          participant and stays unanswered instead of reaching for whatever is stored.

        There is no second hop through a reified event. "The meeting is on Tuesday" is one
        proposition with a ``time`` role, so "when is the meeting?" is that proposition
        with ``time`` left open.
        """
        bound = {role: self._ref_of(v) for role, v in q.frame.roles.items() if role != q.asked}
        bound = {role: v for role, v in bound.items() if v is not None}
        if not bound:
            return []
        symmetric = q.frame.predicate in SYMMETRIC_PREDICATES
        # for a symmetric predicate the side a filler sits on says nothing, so it is asked
        # for by presence rather than by role
        stated = {r: v for r, v in bound.items() if not (symmetric and r in CORE_ROLES)}
        among = {v for r, v in bound.items() if symmetric and r in CORE_ROLES}
        taken = set(bound.values())
        wants_participant = verbnet.role_class(q.asked.title()) == "undergoer"
        roles = dict(stated) if wants_participant else {**stated, q.asked: Var(q.asked)}
        found: list[tuple[Any, Any]] = []
        for match in self.store.find(Proposition(q.frame.predicate, roles)):
            fillers = match.record.proposition.roles
            if among and not among <= {fillers[r] for r in CORE_ROLES if r in fillers}:
                continue
            if not wants_participant:
                found.append((match.bindings[q.asked], match.record))
                continue
            open_ = [fillers[r] for r in CORE_ROLES if r in fillers and fillers[r] not in taken]
            if open_:
                found.append((open_[0], match.record))
        # nothing the question itself supplied is an answer to it. A hole can still bind to
        # one — "who is the meeting?" fills the subject the asker already named — and the
        # answer would be the question read back.
        found = [pair for pair in found if pair[0] not in taken]
        # which to say first is a ranking, not an accident of storage order: what was
        # observed most recently and stated most confidently comes first
        if len(found) > 1:
            ranked = ops.rank(q.frame.describe(), found)
            if not isinstance(ranked, Unknown):
                found = [pair for pair, _ in ranked]
        return [value for value, _ in found] + self._from_claims(q, taken)

    def _from_claims(self, q: Question, taken: set) -> list[Any]:
        """The same question against what plugins revealed, which is still binary.

        A plugin reports what it sees as subject/predicate/object claims. Until they speak
        propositions too, a look's results are matched the same way: every side the
        question bound must appear, and the answer is the side it left open.
        """
        pred = PREDICATE_OF.get(q.frame.predicate, q.frame.predicate)
        out: list[Any] = []
        for rec in self.store.claims(predicate=pred):
            c = rec.claim
            if not taken <= {c.subject, c.object}:
                continue
            if c.object not in taken:
                out.append(c.object)
            elif c.subject not in taken:
                out.append(c.subject)
        return out

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
        plan = self.choose_plan(goal)
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
        # the receipt is the executor's report; what the plugin can still see afterwards is
        # the observation. ops.verify keeps the two apart and records both in the trace.
        verdict = ops.verify(receipt, observe=lambda: plugin.holds(cap, args), expect=lambda seen: seen)
        verified = True if verdict.status == "holds" else False if verdict.status == "fails" \
            else Unknown("unverified", "; ".join(verdict.because))
        events.append({"type": "verified", "capability": cap.name,
                       "holds": verified if isinstance(verified, bool) else f"unknown: {verified.reason}"})
        status = "done" if verified is True else "failed" if verified is False else "unverified"
        return Outcome(act, status, goal, (plugin.name, cap.name, args), receipt, verified,
                       reason="" if verified is True else "the effect was not observed afterwards" if verified is False else verified.reason)

    def choose_plan(self, goal: verbnet.Goal) -> tuple[Plugin, Capability, dict] | Unknown:
        """Which capability to invoke, as a choice among the ones that could serve.

        :func:`tensorcode.ops.choose` takes the options, an objective, and the hard
        constraints. The two constraints are checked by the library itself before any
        implementation sees the options, which is where they belong: *do all of what was
        asked* (a capability that achieves half of a fully-specified request — delete for
        move — must not be reachable by scoring well on the other half) and *use the whole
        capability* (every parameter has an argument). The objective then prefers the option
        achieving the most conditions, and the simpler capability when two tie.
        """
        options, nearest = self.plans(goal)
        chosen = ops.choose(
            options,
            objective=ops.Objective("conditions met", "achieve as much of the goal as possible, simply",
                                    utility=lambda plan, _: plan.met - len(plan.capability.params) / 1000),
            constraints=(ops.Constraint("does all of what was asked", lambda plan, _: plan.achieves_all_specified),
                         ops.Constraint("every parameter has an argument", lambda plan, _: plan.fully_applied)),
        )
        if isinstance(chosen, Unknown):
            for plan in options:
                if not plan.achieves_all_specified:
                    nearest.append(f"{plan.capability.name} would do only part of it")
            return Unknown("no_capability", self._why_not(goal, nearest))
        return chosen.plugin, chosen.capability, chosen.args

    def plans(self, goal: verbnet.Goal) -> tuple[list[Plan], list[str]]:
        """Every capability that could serve ``goal``, with arguments that fit it.

        A condition is achieved by an effect with the same predicate and polarity whose
        every thematic role is either open in the goal or filled by something whose kind
        fits the parameter and that the plugin can refer to. What is *not* decided here is
        which of them to use, or whether a partial match is acceptable: those are the
        objective and the constraints of the choice above.
        """
        options: list[Plan] = []
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
                if not ok or not met:
                    continue
                required = [c for c in goal.conditions if is_specified(c)]
                options.append(Plan(p, cap, args, met,
                                    achieves_all_specified=not required or all(self._achieves(cap, c) for c in required),
                                    fully_applied=all(p_.name in args for p_ in cap.params)))
        return options, nearest

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
    """The word to look kinds up by: the head noun, or the name itself ("Tuesday")."""
    if isinstance(filler, Entity):
        return filler.features.get("noun") or (filler.text if filler.kind in ("description", "name") else None)
    return None


def text_of(filler: Any) -> str:
    return getattr(filler, "text", str(filler))

