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
from copy import deepcopy
from datetime import datetime, timezone
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from ..actions import invoke, plan_order
from ..goals import Condition, GoalSpec
from ..language import ENGLISH, Context, Entity, Frame, Grammar, Question, Request, resolve
from ..language import conventions, verbnet, wordnet
from ..language.semantics import SYMMETRIC_PREDICATES, default_ref, to_propositions
from ..outcomes import Receipt, Unknown, Verdict
from ..records import Evidence, Proposition, Ref, Store, Var
from ..runtime import Runtime, use
from .. import ops
from .operations import Plan, Transcript, agent_runtime, install_learned_reader
from .plugin import Call, Capability, Plugin
from .understand import Act, Sentence
from .tasks import StepAttempt, TaskLedger
from .planning import plan_goal

USER = Ref("agent:user")
SELF = Ref("agent:self")

#: The grammar's own predicates for being somewhere, against VerbNet's. Like the role
#: correspondence in ``verbnet.py``, this aligns two vocabularies; it knows no domain.
PREDICATE_OF = {"located": "has_location", "be": "be", "have": "has_possession",
                "know": "has_information", "say": "has_information"}


def sought_predicate(frame: Frame) -> str:
    """The world predicate a question is about.

    A copula with a locative says where something is — "what is *on my desktop*" and "what
    is *located* on my desktop" ask one question, and only the second has a verb for it. The
    two readers differ here (the grammar says ``located``, the treebank parser says ``be``
    with a location role), and a plugin that reports ``has_location`` should answer either.
    """
    if frame.predicate == "be" and ("location" in frame.roles or "goal" in frame.roles):
        return "has_location"
    return PREDICATE_OF.get(frame.predicate, frame.predicate)

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
    task_id: str | None = None
    steps: tuple[StepAttempt, ...] = ()


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
        self.context = Context(speaker=USER, addressee=SELF)
        self.reader = reader
        self.prefer_reader = None
        if isinstance(reader, str):
            self.prefer_reader = reader
        elif reader is not None:
            install_learned_reader(reader)
            self.prefer_reader = "learned"
        self.runtime = runtime or agent_runtime(prefer_reader=self.prefer_reader)
        self.turns: list[Turn] = []
        self.tasks = TaskLedger()
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
                    if a.frame is not None:
                        # the resolved reading has to replace the *frame* too. Only the meaning
                        # was being updated, and `handle` rebuilds the meaning from the frame —
                        # so every pronoun the discourse had just resolved was thrown away on
                        # the next line, and "delete it" went looking for a file called "it".
                        settled = resolve(a.meaning, self.context)
                        inner = settled.frame if isinstance(settled, (Request, Question)) else settled
                        a = replace(a, meaning=settled, frame=inner if isinstance(inner, Frame) else a.frame)
                    if a.interpretation is not None:
                        events.append({"type": "interpretation", "convention": a.interpretation.convention_id,
                                       "source": a.interpretation.source})
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
        if (move := self.conversational_move(s)) is not None:
            formula = conventions.pairs().get(move)
            events.append({"type": "convention", "move": move, "answers": formula})
            return Outcome(act, "reciprocated", goal=move, answer=formula)
        return Outcome(act, "not_understood", reason="a phrase that is not a statement, question or request")

    def attend(self, frame: Frame | None) -> None:
        """The thing just acted on becomes what "it" means.

        Salience in a conversation is not recency. After "create a file called draft.txt on my
        desktop", the last phrase mentioned is *my desktop*, so "delete it" resolved to the
        desktop — while the thing under discussion is plainly the file. What was acted upon is
        the focus, which is what :class:`~tensorcode.language.Context` keeps that field for and
        what nothing had been setting.
        """
        if frame is None:
            return
        acted_on = self._filler_for_role(frame, "undergoer")
        if isinstance(acted_on, Entity):
            self.context.focus = acted_on

    def conversational_move(self, s: Sentence) -> str | None:
        """Is this whole utterance a conversational formula — a greeting, thanks, a farewell?

        Not every sentence without a predicate is a failure to parse. "hello" is a complete
        move in a conversation and it used to come back as *a phrase that is not a statement,
        question or request*, which is the difference between an agent you can talk to and one
        you can only issue commands to.

        What kind of move it is comes from WordNet — ``hello`` is a ``greeting``, ``thanks`` an
        ``acknowledgement`` — so the words are open: anything WordNet files under those classes
        works, and nothing here lists them.
        """
        words = [w for w in s.text.replace("!", " ").replace(".", " ").replace(",", " ").split() if w.strip()]
        if not words or len(words) > 3:
            return None
        for candidate in (" ".join(words), words[0]):
            move = conventions.move_of(self.kinds(candidate.lower()))
            if move is not None:
                return move
        return None

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
        pred = sought_predicate(q.frame)
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
                    fresh = list(p.reveal(cap, {inf.param: arg}, receipt)) if receipt.status == "applied" else []
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
        core = None
        for grammar_role, value in frame.roles.items():
            classes = {verbnet.role_class(r) for r in verbnet.ROLE_OF_PREPOSITION_ROLE.get(grammar_role, ())}
            if role in classes:
                return value
            if core is None and grammar_role in ("subject", "object") and role == "undergoer":
                core = value
        # a role the preposition named wins over the bare subject or object: "what do you know
        # about Austin" is about Austin, and the short-circuit answered "you"
        return core

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
        pred = sought_predicate(q.frame)
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
        """Retain the interpreted task separately from its chosen action and outcome.

        Natural-language task correction is not inferred here: each request creates
        a task. Structured callers can revise and retry a named task with ``pursue``.
        """
        outcome = self._request(s, act, events)
        task = self.tasks.create(s.text, outcome.goal)
        outcome = replace(outcome, task_id=task.id)
        task = self.tasks.record(task.id, outcome)
        events.append({"type": "task", "task_id": task.id, "revision": task.revision, "status": task.status})
        return outcome

    def pursue(self, goal: GoalSpec | None = None, *, task_id: str | None = None,
               source: str = "structured", events: list[dict] | None = None,
               max_steps: int | None = None) -> Outcome:
        """Attempt an explicit desired outcome without asking VerbNet to interpret it.

        To retry, supply only ``task_id``. To change the goal first call
        ``agent.tasks.revise(task_id, new_goal, reason=...)``. A completed task
        cannot execute again without an explicit revision. State is in-memory.
        Plugins with explicit action models use bounded multi-step planning.
        ``max_steps`` suspends a plan after verified steps; resumption replans from
        fresh observations rather than replaying a cached sequence.
        """
        if (goal is None) == (task_id is None):
            raise ValueError("supply either a goal or a task_id")
        if goal is not None and not isinstance(goal, GoalSpec):
            raise TypeError("structured goals must be GoalSpec values")
        if max_steps is not None and (not isinstance(max_steps, int) or max_steps < 1):
            raise ValueError("max_steps must be a positive integer")
        task = self.tasks.create(source, goal) if goal is not None else self.tasks.get(task_id)
        if task.status == "done":
            raise ValueError("completed task requires an explicit revision before another attempt")
        if task.attempts:
            receipts = [receipt for previous in task.attempts if previous.revision == task.revision
                        for receipt in (previous.receipt, *(step.receipt for step in previous.steps))
                        if receipt is not None]
            if task.status != "suspended" and any(receipt.status != "rejected" for receipt in receipts):
                raise ValueError("previous attempt may have changed the world; revise explicitly before retrying")
        if not isinstance(task.goal, (GoalSpec, verbnet.Goal)):
            raise ValueError("task needs an interpreted goal before it can be attempted")
        events = events if events is not None else []
        act = Act("request", task.goal, None)
        with use(self.runtime):
            self.perceive(events)
            self._guessed = set()
            outcome = self._execute_goal(task.goal, act, events, max_steps=max_steps)
        outcome = replace(outcome, task_id=task.id)
        task = self.tasks.record(task.id, outcome)
        events.append({"type": "task", "task_id": task.id, "revision": task.revision, "status": task.status})
        return outcome

    def _request(self, s: Sentence, act: Act, events: list[dict]) -> Outcome:
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
        refined = []
        refinement_errors = []
        for plugin in self.plugins:
            candidate = plugin.refine_goal(goal)
            if isinstance(candidate, GoalSpec):
                if candidate not in refined:
                    refined.append(candidate)
            elif isinstance(candidate, Unknown) and candidate.reason != "no_refinement":
                refinement_errors.append(candidate)
                events.append({"type": "refinement_unavailable", "plugin": plugin.name,
                               "reason": candidate.reason, "detail": candidate.detail})
        if refinement_errors:
            return Outcome(act, "declined", goal=goal,
                           reason="; ".join(error.detail or error.reason for error in refinement_errors))
        if len(refined) > 1:
            return Outcome(act, "declined", goal=goal, reason="multiple domain refinements disagree about the desired outcome")
        if refined:
            goal = refined[0]
            events.append({"type": "refined", "goal": goal.describe(), "basis": list(goal.basis)})
        return self._execute_goal(goal, act, events)

    def _execute_goal(self, goal: GoalSpec | verbnet.Goal, act: Act, events: list[dict],
                      *, max_steps: int | None = None) -> Outcome:
        if isinstance(goal, GoalSpec):
            conditions = (*goal.conditions, *goal.invariants)
            if any(a.pred == b.pred and a.args == b.args and a.negated != b.negated
                   for i, a in enumerate(conditions) for b in conditions[i + 1:]):
                return Outcome(act, "declined", goal=goal, reason="contradictory explicit conditions")
        providers = tuple(p for p in self.plugins if p.planning_enabled)
        if isinstance(goal, GoalSpec) and providers:
            return self._execute_modeled_goal(goal, act, events, providers, max_steps=max_steps)
        if isinstance(goal, GoalSpec) and goal.invariants:
            return Outcome(act, "declined", goal=goal, reason="held conditions require an explicit action model")
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
            else Unknown("unverified", "; ".join(verdict.reasons))
        events.append({"type": "verified", "capability": cap.name,
                       "holds": verified if isinstance(verified, bool) else f"unknown: {verified.reason}"})
        status = "done" if verified is True else "failed" if verified is False else "unverified"
        if status != "failed":
            self.attend(act.frame)
        told: list[Any] = []
        if cap.informs and status != "failed":
            # a capability whose product is *information* has to be able to say it. Only the
            # question path reached `reveal`, so "explain your reasoning" came back as
            # "I explained my reasoning, and checked that it worked" — correct, verified and
            # contentless.
            for claim in plugin.reveal(cap, args, receipt):
                self.store.tell(claim, Evidence(source=Ref(f"plugin:{plugin.name}"),
                                                observed_at=datetime.now(timezone.utc), method=cap.name))
                told.append(claim.object)
        return Outcome(act, status, goal, (plugin.name, cap.name, args), receipt, verified, answer=told or None,
                       reason="" if verified is True else "the effect was not observed afterwards" if verified is False else verified.reason)

    def _observe_condition(self, condition: Condition, providers: Sequence[Plugin]) -> bool | Unknown:
        evidence = set()
        for plugin in providers:
            got = plugin.observe_condition(condition)
            if got is True or got is False:
                evidence.add(got)
            elif not isinstance(got, Unknown):
                return Unknown("invalid_observation", plugin.name)
        if len(evidence) == 1:
            return next(iter(evidence))
        return Unknown("conflicting_observations" if evidence else "unobserved_condition", condition.describe())

    def _check_conditions(self, conditions: Sequence[Condition], providers: Sequence[Plugin],
                          events: list[dict], *, stage: str) -> bool | Unknown:
        unknown = None
        failed = False
        for condition in conditions:
            got = self._observe_condition(condition, providers)
            events.append({"type": "condition", "stage": stage, "condition": condition.describe(),
                           "status": "holds" if got is True else "fails" if got is False else "unknown"})
            if got is False:
                failed = True
            elif isinstance(got, Unknown):
                unknown = got
        return False if failed else unknown if unknown is not None else True

    def _execute_modeled_goal(self, goal: GoalSpec, act: Act, events: list[dict],
                              providers: Sequence[Plugin], *, max_steps: int | None) -> Outcome:
        models = {p.name: deepcopy(tuple(p.capabilities())) for p in providers}
        plan = plan_goal(goal, providers, capability_models=models)
        if isinstance(plan, Unknown):
            events.append({"type": "plan", "goal": goal.describe(),
                           "plan": {"unknown": plan.reason, "detail": plan.detail}})
            return Outcome(act, "declined", goal=goal, plan=plan, reason=plan.detail or plan.reason)
        runnable = plan_order(plan)
        if isinstance(runnable, Verdict):
            return Outcome(act, "declined", goal=goal, plan=plan, reason="; ".join(runnable.reasons))
        events.append({"type": "plan", "goal": goal.describe(), "basis": list(goal.basis),
                       "plan": {"steps": [{"id": step.id, "plugin": step.action.plugin,
                                            "capability": step.action.capability,
                                            "args": {k: str(v) for k, v in step.action.args},
                                            "needs": list(step.needs)} for step in plan.steps],
                                "rationale": plan.rationale}})
        plugins = {p.name: p for p in providers}
        by_id = {step.id: step for step in plan.steps}
        attempts = []
        last_receipt = None
        for step_id in runnable.order:
            if max_steps is not None and len(attempts) >= max_steps:
                return Outcome(act, "suspended", goal=goal, plan=plan, receipt=last_receipt,
                               reason="execution step budget reached; resume from fresh observations",
                               steps=tuple(attempts))
            invariant = self._check_conditions(goal.invariants, providers, events, stage="before_step")
            if invariant is not True:
                return Outcome(act, "failed" if invariant is False else "unverified", goal=goal, plan=plan,
                               receipt=last_receipt, verified=invariant, reason="held condition no longer established",
                               steps=tuple(attempts))
            step = by_id[step_id]
            call = step.action
            plugin = plugins[call.plugin]
            cap = next((c for c in plugin.capabilities() if c.name == call.capability), None)
            searched = next((c for c in models[plugin.name] if c.name == call.capability), None)
            if cap is None or cap != searched:
                return Outcome(act, "failed", goal=goal, plan=plan, receipt=last_receipt,
                               reason="planned capability model changed or is no longer available", steps=tuple(attempts))
            args = dict(call.args)
            events.append({"type": "step", "step_id": step_id})
            receipt = self._invoke(plugin, cap, args, events, observers=providers)
            last_receipt = receipt
            if receipt.status in ("failed", "rejected"):
                # A failure report does not establish that nothing changed.
                self.perceive(events)
                invariant = self._check_conditions(goal.invariants, providers, events, stage="after_step")
                attempts.append(StepAttempt(step_id, call, receipt, False))
                reason = receipt.error or receipt.status
                if invariant is not True:
                    reason += "; held conditions were not established after the attempt"
                return Outcome(act, "failed", goal=goal, plan=plan, receipt=receipt, verified=False,
                               reason=reason, steps=tuple(attempts))
            self.perceive(events)
            effects = tuple(Condition(e.pred, {role: args[name] for role, name in e.roles.items()}, e.negated)
                            for e in cap.effects)
            verdict = ops.verify(receipt,
                                 observe=lambda: self._check_conditions(effects, providers, events, stage="step_effect"),
                                 expect=lambda seen: seen)
            verified = True if verdict.status == "holds" else False if verdict.status == "fails" else Unknown("unverified", "; ".join(verdict.reasons))
            attempts.append(StepAttempt(step_id, call, receipt, verified))
            events.append({"type": "verified", "capability": cap.name,
                           "holds": verified if isinstance(verified, bool) else verified.reason})
            invariant = self._check_conditions(goal.invariants, providers, events, stage="after_step")
            if verified is not True or invariant is not True:
                result = False if verified is False or invariant is False else Unknown("unverified", "step or held condition could not be established")
                return Outcome(act, "failed" if result is False else "unverified", goal=goal, plan=plan,
                               receipt=receipt, verified=result, reason="step effects or held conditions were not established",
                               steps=tuple(attempts))
        complete = self._check_conditions((*goal.conditions, *goal.invariants), providers, events, stage="task_complete")
        status = "done" if complete is True else "failed" if complete is False else "unverified"
        if complete is True:
            self.attend(act.frame)
        return Outcome(act, status, goal=goal, plan=plan, receipt=last_receipt, verified=complete,
                       reason="" if complete is True else "the complete task specification was not observed",
                       steps=tuple(attempts))

    def choose_plan(self, goal: GoalSpec | verbnet.Goal) -> tuple[Plugin, Capability, dict] | Unknown:
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

    def plans(self, goal: GoalSpec | verbnet.Goal) -> tuple[list[Plan], list[str]]:
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
                if any(a.pred == b.pred and a.roles == b.roles and a.negated != b.negated
                       for i, a in enumerate(cap.effects) for b in cap.effects[i + 1:]):
                    nearest.append(f"{cap.name} declares contradictory effects")
                    continue
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
                        # Ask the plugin first. Whether a description picks out one of its
                        # things is the plugin's to know, and a plugin that hands back a
                        # reference has *shown* that the filler fits — the taxonomy is a prior,
                        # not an authority. Checking WordNet first vetoed "readme-first.txt"
                        # on every capability, because no lexicon vouches for a file name, and
                        # that single ordering was seven of the twelve failures on the graded
                        # desktop jobs.
                        ref = p.refer(filler, param, context={"store": self.store, "args": args})
                        if isinstance(ref, Unknown):
                            ok = False
                            if self.fits(filler, param.kind):
                                nearest.append(f"{cap.name}: {ref.detail or ref.reason}")
                            elif ref.detail and text_of(filler).lower() in self._guessed:
                                # a kind inferred for a word the parser only guessed at is not
                                # evidence; the plugin's own account is the better reason
                                nearest.insert(0, ref.detail)
                            else:
                                nearest.append(f"{cap.name} wants a {param.kind} for {role}, and {text_of(filler)} is not one")
                            break
                        if pname in args and args[pname] != ref:
                            ok = False
                            nearest.append(f"{cap.name} needs incompatible bindings for {pname}")
                            break
                        args[pname] = ref
                    if not ok:
                        break
                    met += 1
                if not ok or not met:
                    continue
                # Explicit specifications are conjunctions, including nullary
                # conditions. Only lexical interpretations can contain implicit,
                # unspecified result roles that were not requested by the caller.
                required = goal.conditions if isinstance(goal, GoalSpec) else [c for c in goal.conditions if is_specified(c)]
                options.append(Plan(p, cap, args, met,
                                    achieves_all_specified=not required or all(self._achieves(cap, c) for c in required),
                                    fully_applied=all(p_.name in args for p_ in cap.params)))
        return options, nearest

    def _achieves(self, cap: Capability, cond: Condition) -> bool:
        filled = {verbnet.role_class(r) for r, v in cond.args.items() if v is not None and v != "addressee"}
        return any(e.pred == cond.pred and e.negated == cond.negated and filled <= set(e.roles) for e in cap.effects)

    def _why_not(self, goal: GoalSpec | verbnet.Goal, nearest: list[str]) -> str:
        """Report candidate failures without interpreting their English wording.

        Grounding/model checks know why a candidate failed. Guessing a different
        explanation from taxonomy or substrings of those reports can conceal the
        actual failure, especially for conditions with no participant roles.
        """
        if nearest:
            return "; ".join(dict.fromkeys(nearest))
        return f"no complete grounded capability achieves {goal.describe()}"

    def _invoke(self, plugin: Plugin, cap: Capability, args: Mapping[str, Any], events: list[dict],
                *, observers: Sequence[Plugin] = ()) -> Receipt:
        self._calls += 1
        act = Call(plugin.name, cap.name, tuple(sorted(args.items(), key=lambda kv: kv[0])))
        for condition in cap.preconditions:
            missing = set(condition.roles.values()) - set(args)
            holds = Unknown("unbound_precondition", ", ".join(sorted(missing))) if missing else plugin.precondition_holds(condition, args)
            if observers and not missing:
                shared = self._observe_condition(Condition(condition.pred,
                                                          {role: args[name] for role, name in condition.roles.items()},
                                                          condition.negated), observers)
                if isinstance(holds, Unknown):
                    holds = shared
                elif (shared is True or shared is False) and shared is not holds:
                    holds = Unknown("conflicting_observations", condition.pred)
                elif isinstance(shared, Unknown) and shared.reason in ("conflicting_observations", "invalid_observation"):
                    holds = shared
            # Only observed True licenses dispatch. Unknown is neither False nor success.
            status = "holds" if holds is True else "fails" if holds is False else "unknown"
            detail = holds.detail or holds.reason if isinstance(holds, Unknown) else ""
            events.append({"type": "precondition", "capability": cap.name, "predicate": condition.pred,
                           "negated": condition.negated, "status": status, "detail": detail})
            if holds is not True:
                reason = f"precondition {condition.pred}: {status}" + (f" ({detail})" if detail else "")
                receipt = Receipt(act, "rejected", error=reason)
                events.append({"type": "receipt", "capability": cap.name, "status": receipt.status, "error": reason})
                return receipt
        events.append({"type": "act", "plugin": plugin.name, "capability": cap.name, "args": {k: str(v) for k, v in args.items()}})
        receipt = invoke(act, executor=plugin, key=f"{plugin.name}:{self._calls}")
        events.append({"type": "receipt", "capability": cap.name, "status": receipt.status, "error": receipt.error})
        return receipt


def is_specified(cond: Condition) -> bool:
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
