"""Capabilities whose effect is that someone *knows* something: the agent talking about itself.

Before this module the agent could act and it could answer questions about the world its
plugins perceive, but a request whose only product is language was unreachable — not because
the language layer failed, but because nothing in the plugin registry had an informational
effect. Measured on a fresh ``Agent([])``:

* "explain your reasoning" reads cleanly (coverage 1.0) and VerbNet's ``transfer_mesg-37.1.1``
  gives exactly the right goal, ``has_information(Recipient=None, Topic=reasoning)`` — and
  then ``choose_plan`` returns ``Unknown("no_capability", "nothing I can do brings about
  has_information")``, because no capability anywhere advertised that predicate;
* "what are your capabilities?" reaches ``ask`` with ``sought_predicate`` ``be``, finds no
  capability that *informs* on ``be``, finds nothing in the store, and answers
  "I don't know (nothing I know or can look up answers it)".

So the gap was a missing *capability*, and this is that capability — three of them, one per
place the agent actually has something true to say about itself:

``explain_what_i_did``
    introspection over :class:`tensorcode.runtime.Span` and the previous turn's events. Which
    operations ran, which implementation answered each, which plugin capability was invoked,
    and whether the effect was observed afterwards. Nothing is narrated: every line is fields
    off a recorded span or event, joined.
``say_what_i_can_do``
    :func:`tensorcode.agent.plugin.describe_capabilities` over the agent's own plugin list, so
    the answer is the registry and changes when a plugin is added or a command is learned.
``say_what_i_know_about``
    the propositions and claims in :class:`tensorcode.records.Store` that mention the thing
    asked about.

Both directions are declared, because the two halves of the turn reach a plugin differently.
``effects`` is what ``core.plans`` matches a *request* against ("explain your reasoning"), and
``informs`` is what ``core._look`` matches a *question* against ("what is your reasoning?").
Only the question path voices the content — ``reply.answer_text`` renders what ``reveal``
revealed — which is a limitation of ``reply.py``, not of this module: see :meth:`reveal`.

Nothing here matches the user's words. Which report a topic reaches is decided by WordNet's
hypernyms of the topic's head noun, by whether the grammar says the topic is the addressee's
(:data:`REPORTS`), and by whether the source has anything in it; a topic the sources are
silent about gets :class:`Unknown` from :meth:`refer`, which is the agent's ordinary
abstention path, so an empty trace declines instead of inventing a story.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..language.semantics import default_ref
from ..outcomes import Receipt, Unknown
from ..records import Claim, Proposition, Ref
from .plugin import Capability, Call, Effect, Informs, Param, Plugin, describe_capabilities

@dataclass(frozen=True)
class Report:
    """One self-report: which capability offers it, and what a topic must be for it to apply."""

    capability: str
    #: the parameter its topic fills. ``refer`` is handed a
    #: :class:`~tensorcode.agent.plugin.Param` and not a capability, so the parameter *name* is
    #: what tells it which report is wanted; three capabilities sharing the name "topic" would
    #: be indistinguishable there, and the planner would then choose between them by
    #: declaration order rather than by what the topic is.
    param: str
    #: WordNet noun classes the topic's head noun must fall under; empty means any noun. This
    #: is the ordinary :attr:`~tensorcode.agent.plugin.Param.kind` declaration, as a set rather
    #: than one string, because a topic can be named from either end: what the agent did is
    #: "your reasoning" (WordNet: ``process``) or "your last action" (``act``, ``event``), and
    #: no single WordNet class covers both without covering everything. The closure is
    #: WordNet's own, through ``Agent.kinds``, so any noun WordNet files under these works and
    #: no list of words appears here.
    about: frozenset[str]
    #: whether the topic must be the *addressee's*. WordNet knows "reasoning" is a process and
    #: "a meeting" is an event; it cannot know that one of them is this agent's own. The grammar
    #: can: "**your** reasoning" carries ``possessor=2``, the person being spoken to, which is
    #: the agent. Without this, "explain the meeting" reached the trace report — an event, by
    #: WordNet, and so a plausible thing for the agent to have introspected about.
    of_addressee: bool = False


#: The reports, most specific first. ``say_what_i_know_about`` anchors on nothing and belongs
#: last: what a store happens to hold is not a kind of thing, so its gate is the store itself
#: (:meth:`DiscoursePlugin.refer`), which is stricter than any hypernym test. The first two
#: are disjoint by WordNet's own filing — nothing under ``ability`` is under ``process``,
#: ``act`` or ``event`` — so the order between them settles nothing but which reason the agent
#: gives when neither applies.
REPORTS: tuple[Report, ...] = (
    Report("explain_what_i_did", "doing", frozenset({"process", "act", "event"}), of_addressee=True),
    Report("say_what_i_can_do", "ability", frozenset({"ability"}), of_addressee=True),
    Report("say_what_i_know_about", "subject", frozenset()),
)

REPORT_OF_PARAM: Mapping[str, Report] = {r.param: r for r in REPORTS}
REPORT_OF_CAPABILITY: Mapping[str, Report] = {r.capability: r for r in REPORTS}


class DiscoursePlugin(Plugin):
    """The agent's capabilities for saying something true about itself.

    Constructed with the agent, or with a callable returning it — the plugin needs the agent
    and the agent is built from its plugins, so a zero-argument callable is the way out of
    that circle::

        plugin = DiscoursePlugin(lambda: agent)
        agent = Agent([desktop, plugin])

    With no agent reachable every method abstains rather than raising: a plugin that throws
    during ``perceive`` takes the whole turn down, and "I cannot reach my own record" is an
    honest :class:`Unknown`.
    """

    def __init__(self, agent: Any = None, *, name: str = "discourse") -> None:
        super().__init__(name=name)
        self.reach = agent
        # what was emitted, per (capability, topic): the text this plugin has actually said,
        # so `display` can hand a line back verbatim instead of letting `reply.answer_text`
        # split it on its first colon
        self._said: dict[tuple[str, Ref], tuple[str, ...]] = {}
        # span index bookkeeping, maintained in `perceive` — see there for why
        self._turns_seen = -1
        self._this_start: int | None = None
        self._last_range: tuple[int, int] | None = None

    def attach(self, agent: Any) -> None:
        """Bind the agent after construction, for callers that would rather not pass a lambda."""
        self.reach = agent

    def agent(self) -> Any:
        got = self.reach() if callable(self.reach) else self.reach
        return got

    # ------------------------------------------------------------------ what it offers

    def capabilities(self) -> Sequence[Capability]:
        """One capability per report, each advertising ``has_information`` both ways.

        The effect's roles are VerbNet role *classes*: ``undergoer`` is the Topic and ``goal``
        is the Recipient (:data:`tensorcode.language.verbnet.ROLE_CLASS`). ``goal`` is declared
        but has no parameter behind it, and that is deliberate rather than an oversight. A
        fully specified request — "tell me your capabilities" — yields
        ``has_information(Recipient=me, Topic=capabilities)``, and ``core._achieves`` requires
        an effect mentioning *both* filled roles or the plan is rejected as doing only part of
        what was asked. But the recipient of what this agent says is never a choice: it is
        whoever it is talking to. Were it a parameter, ``Plan.fully_applied`` would then fail
        on the commoner request "explain your reasoning", where VerbNet leaves the Recipient
        open, and both phrasings cannot be served by one capability any other way.
        """
        return tuple(Capability(
            name=r.capability,
            params=(Param(r.param, kind="entity", role="Topic"),),
            effects=(Effect("has_information", {"undergoer": r.param, "goal": "recipient"}),),
            informs=(Informs("has_information", "undergoer", r.param),
                     Informs("be", "undergoer", r.param)),
            effect_kind="read",
            description=f"{r.capability} from {self.name}",
        ) for r in REPORTS)

    # ------------------------------------------------------------------ turn boundaries

    def perceive(self) -> Iterable[Claim]:
        """Nothing is perceived; what this does is notice where one turn's trace ends.

        "Explain what you just did" is about *one* turn, and ``Trace.spans`` is a single
        growing list with no turn boundaries in it. Reporting the whole list would grow the
        answer without bound and mix in the reasoning of every earlier message; reporting the
        current turn would report the agent explaining, which is circular.

        ``Agent.perceive`` is called once at the top of every turn and again after each
        successful action, and ``Agent.turns`` only grows when a turn has finished — so a
        change in its length is exactly a turn boundary, and the span index at that moment is
        exactly where the finished turn's spans end. Before the first boundary there is no
        finished turn, which is how a first-message "explain your reasoning" reaches
        :meth:`refer` with nothing to report.
        """
        agent = self.agent()
        if agent is None:
            return ()
        done = len(agent.turns)
        if done != self._turns_seen:
            now = len(agent.runtime.trace.spans)
            if self._this_start is not None:
                self._last_range = (self._this_start, now)
            self._this_start = now
            self._turns_seen = done
        return ()

    # ------------------------------------------------------------------ the reports

    def report(self, capability: str, topic: Ref) -> tuple[str, ...]:
        """The lines this capability would say about ``topic``, read from the source now.

        Called from :meth:`refer` (to decide whether there is anything to say), from
        :meth:`execute` (to say it) and from :meth:`holds` (to check it is still true), which
        is the point: the check is a fresh read of the record, not a memory of the receipt.
        """
        agent = self.agent()
        if agent is None:
            return ()
        source = {"say_what_i_can_do": lambda: self._registry(agent),
                  "explain_what_i_did": lambda: self._last_turn(agent),
                  "say_what_i_know_about": lambda: self._stored(agent, topic)}.get(capability)
        return source() if source is not None else ()

    def _registry(self, agent: Any) -> tuple[str, ...]:
        """Every capability every mounted plugin offers, as the registry states it."""
        lines = []
        for entry in describe_capabilities(agent.plugins):
            params = ",".join(name for name, _ in entry["params"])
            lines.append(f"{entry['plugin']}.{entry['name']}({params})")
        return tuple(dict.fromkeys(lines))

    def _last_turn(self, agent: Any) -> tuple[str, ...]:
        """The finished turn, from its spans and its events: field values, joined.

        Three things a person asking "what did you just do?" is owed, and all three are
        recorded rather than reconstructed: which operations ran and which implementation
        answered each (``Span.op`` / ``Span.answered_by``, with the abstentions that came
        first), which capability of which plugin was invoked and what its executor said
        (``act`` and ``receipt`` events), and whether the effect was observed afterwards
        (``verified``). ``Span.notes`` is included because a span with no implementation at
        all says so only there.
        """
        if self._last_range is None or not agent.turns:
            return ()
        start, end = self._last_range
        lines: list[str] = []
        for span in agent.runtime.trace.spans[start:end]:
            if not span.attempts and not span.notes:
                continue
            if span.attempts:
                lines.append(f"{span.op}={span.answered_by or span.outcome}")
            for attempt in span.attempts:
                if attempt.outcome not in ("answer", "cache_hit"):
                    lines.append(f"{span.op}/{attempt.implementation}={attempt.outcome}"
                                 + (f"({attempt.reason})" if attempt.reason else ""))
            for note in span.notes:
                lines.append(f"{span.op}/note={note}")
        for event in agent.turns[-1].events:
            kind = event.get("type")
            if kind == "act":
                lines.append(f"invoked={event['plugin']}.{event['capability']}"
                             f"({','.join(sorted(event.get('args') or {}))})")
            elif kind == "receipt":
                lines.append(f"{event['capability']}={event['status']}"
                             + (f"({event['error']})" if event.get("error") else ""))
            elif kind == "verified":
                lines.append(f"{event['capability']}/verified={event['holds']}")
            elif kind == "goal":
                lines.append(f"goal={event['goal']}")
            elif kind == "plan":
                plan = event.get("plan") or {}
                if "steps" in plan:
                    lines.append("plan=" + " -> ".join(f"{s['plugin']}.{s['capability']}" for s in plan["steps"]))
                    if plan.get("rationale"):
                        lines.append(f"plan/reason={plan['rationale']}")
                    lines.extend(f"plan/basis={basis}" for basis in event.get("basis", ()))
                else:
                    lines.append("plan=" + (f"{plan['plugin']}.{plan['capability']}" if "capability" in plan
                                            else f"none({plan.get('unknown')})"))
            elif kind == "condition":
                lines.append(f"{event['stage']}/{event['condition']}={event['status']}")
            elif kind == "interpretation":
                lines.append(f"interpretation={event['convention']} ({event['source']})")
        return tuple(dict.fromkeys(lines))

    def _stored(self, agent: Any, topic: Ref) -> tuple[str, ...]:
        """Everything on record that mentions ``topic``, as the store's own rendering of it.

        Propositions are n-ary and nestable, so "mentions" is a walk over role fillers rather
        than a subject lookup: "Casey said Austin is hot" holds Austin two levels down, and a
        query on the subject role alone answers nothing about Austin. Binary claims (what a
        plugin revealed by looking) are matched from both sides for the same reason.

        What this plugin itself put there is left out, and that is not tidiness. Answering
        "what is your reasoning?" reveals the report as ``be`` claims about the topic, which
        land in the same store; without this the next reading of the store report quoted the
        agent's own last answer back as something it knows, and ``holds`` then compared a
        report against a store the report had changed.

        Nothing is paraphrased: ``Proposition.describe`` is the store's own form, and a frame
        shown as data is the honest thing to show when the alternative is a template that
        would look like understanding.
        """
        mine = Ref(f"plugin:{self.name}")
        lines: list[str] = []
        for record in agent.store.propositions():
            if _mentions(record.proposition, topic) and not _only_from(record, mine):
                lines.append(record.proposition.describe())
        for record in agent.store.claims(subject=topic):
            if not _only_from(record, mine):
                lines.append(f"{record.claim.predicate}({topic.id},{_short(record.claim.object)})")
        for record in agent.store.claims(object=topic):
            if not _only_from(record, mine):
                lines.append(f"{record.claim.predicate}({_short(record.claim.subject)},{topic.id})")
        return tuple(dict.fromkeys(lines))

    # ------------------------------------------------------------------ referring

    def refer(self, description: Any, param: Param, *, context: Mapping[str, Any]) -> Any | Unknown:
        """Which report a topic picks out, or ``Unknown`` when there is nothing to say.

        Three gates, in order, all of them over data rather than over the user's words. What the
        topic *is*: WordNet's hypernyms of its head noun against the report's anchors. Whose it
        is: a self-report wants the addressee's own (:attr:`Report.of_addressee`). And whether
        the source has anything in it at all — which is the gate that matters most, because it
        is why an agent asked to explain itself on its very first message declines ("I can't
        explain your reasoning: …") instead of reporting the parse of the request that is
        asking. Returning a reference here and an empty report later would leave an agent that
        has done nothing having to say something, which is where invented stories come from.
        """
        agent = self.agent()
        if agent is None:
            return Unknown("no_agent", f"{self.name} has no agent to introspect")
        report = REPORT_OF_PARAM.get(param.name)
        if report is None:
            return Unknown("cannot_refer", f"{self.name} has no report filling {param.name}")
        features = getattr(description, "features", None)
        said = getattr(description, "text", description)
        if not isinstance(features, Mapping):
            return Unknown("cannot_refer", f"{self.name} cannot name {description!r} to report on")
        noun = features.get("noun")
        if report.about:
            if not noun:
                return Unknown("cannot_refer", f"{report.capability} is about "
                                               f"{' or '.join(sorted(report.about))}, and {said!r} names no kind of thing")
            if not report.about & agent.kinds(noun):
                return Unknown("cannot_refer", f"{report.capability} is about "
                                               f"{' or '.join(sorted(report.about))}, and {noun} is not one")
        if report.of_addressee and features.get("possessor") != 2:
            return Unknown("cannot_refer", f"{report.capability} reports on what is mine, "
                                           f"and {said!r} is not said to be")
        topic = default_ref(description)
        if not isinstance(topic, Ref):
            return Unknown("cannot_refer", f"{self.name} cannot name {description!r} to report on")
        if not self.report(report.capability, topic):
            return Unknown("nothing_to_report", f"nothing I have on record says anything about {said}")
        return topic

    def denote(self, description: Any) -> Any | Unknown:
        """The same reference, for a question that has to be matched against the store.

        ``core.lookup`` re-resolves the question's own words through ``denote`` while
        ``core._look`` resolved them through ``refer``; if the two disagreed the answer this
        plugin revealed would not be found by the question that asked for it. So this is
        ``refer``'s gate again, over every report, and deliberately nothing else: a ``denote``
        that said yes to any noun would mint references for words no source knows.
        """
        for report in REPORTS:
            got = self.refer(description, Param(report.param, kind="entity"), context={})
            if isinstance(got, Ref):
                return got
        return Unknown("cannot_refer", f"{self.name} has nothing on record about {description!r}")

    def display(self, ref: Any) -> str | None:
        """A line this plugin said, handed back as it was said.

        ``reply.answer_text`` falls back to ``str(value).split(":", 1)[-1]`` for anything no
        plugin displays, which turns ``parse=reader:grammar@1`` into ``grammar@1`` — the
        implementation's name eaten by a namespace convention meant for refs. Only lines this
        plugin has actually emitted are claimed, so nothing else's values are captured.
        """
        if isinstance(ref, str) and any(ref in said for said in self._said.values()):
            return ref
        return None

    # ------------------------------------------------------------------ doing and checking

    def execute(self, act: Call, *, key: str | None) -> Receipt:
        """Produce the report. An empty one is a failure, never an empty success."""
        report = REPORT_OF_CAPABILITY.get(act.capability)
        if report is None:
            return Receipt(act, "rejected", idempotency_key=key,
                           error=f"{self.name} does not implement {act.capability}")
        topic = act.arg(report.param)
        if not isinstance(topic, Ref):
            return Receipt(act, "rejected", idempotency_key=key,
                           error=f"{act.capability} needs a {report.param} to report on")
        lines = self.report(act.capability, topic)
        if not lines:
            # the source went empty between planning and acting; say so rather than applying
            return Receipt(act, "failed", idempotency_key=key,
                           error=f"nothing on record about {topic.id}")
        self._said[(act.capability, topic)] = lines
        return Receipt(act, "applied", idempotency_key=key, effect_id=f"{act.capability}:{topic.id}")

    def holds(self, cap: Capability, args: Mapping[str, Any]) -> bool | Unknown:
        """Whether the information is there to be had, re-read from the source.

        What is checkable is the agent's half: the report exists, is not empty, and still says
        what it said when it was produced. Whether the person it was said to now knows it is
        not observable from here, and ``has_information`` asserts exactly that — so this
        returns the observation it can make and the docstring, rather than the status, carries
        the part it cannot.
        """
        report = REPORT_OF_CAPABILITY.get(cap.name)
        topic = args.get(report.param) if report else None
        if not isinstance(topic, Ref):
            return Unknown("no_check", f"{cap.name} was not given a topic to check")
        said = self._said.get((cap.name, topic))
        if said is None:
            return False
        return bool(said) and self.report(cap.name, topic) == said

    def reveal(self, cap: Capability, args: Mapping[str, Any], receipt: Receipt) -> Iterable[Claim]:
        """The report as claims, which is the only way its content reaches the person.

        A capability invoked to satisfy a *request* leaves nothing a reply can say: ``reply.clause``
        re-says the request's own frame ("I explained your reasoning, and checked that it
        worked") and a ``Receipt`` carries no payload, so the explanation itself goes nowhere.
        A capability run to answer a *question* has this hook, and ``reply.answer_text`` says
        what came back. Until ``reply.py`` can voice a capability's product, asking is the
        phrasing that gets the content and telling is the phrasing that only gets it done.

        The claims are ``be`` because that is what ``core.sought_predicate`` asks a copular
        question with ("what is your reasoning?"), and ``core._from_claims`` answers it with
        the side the question left open.
        """
        report = REPORT_OF_CAPABILITY.get(cap.name)
        topic = args.get(report.param) if report else None
        if receipt.status != "applied" or not isinstance(topic, Ref):
            return ()
        lines = self._said.get((cap.name, topic), ())
        return tuple(Claim(subject=topic, predicate="be", object=line) for line in lines)


def _mentions(proposition: Proposition, ref: Ref) -> bool:
    """Is ``ref`` a filler anywhere in this proposition, however deeply nested?"""
    for filler in proposition.roles.values():
        for one in (filler if isinstance(filler, (list, tuple)) else (filler,)):
            if one == ref:
                return True
            if isinstance(one, Proposition) and _mentions(one, ref):
                return True
    return False


def _short(value: Any) -> str:
    return value.id if isinstance(value, Ref) else str(value)


def _only_from(record: Any, source: Ref) -> bool:
    """Was every piece of evidence for this record put there by one source?"""
    evidence = getattr(record, "evidence", ()) or ()
    return bool(evidence) and all(e.source == source for e in evidence)


__all__ = ["REPORTS", "Report", "DiscoursePlugin"]
