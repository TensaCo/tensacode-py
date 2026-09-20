"""The agent loop: retained language interpretations, beliefs, tasks, and execution.

    agent = Agent([DesktopPlugin(...)])
    turn = agent.turn("make a folder called recipes on my desktop")
    turn.reply          # what it says back, once per message
    turn.outcomes       # what each request, question and statement came to
    turn.events         # everything a viewer may show, as data

One turn:

1. **perceive** — retain raw provider evidence; separately admit explicit claim reports;
2. **interpret** — retain source text and alternative readings; an explicit policy
   chooses or defers each sentence before its acts are dispatched;
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
from uuid import uuid4
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Any, Callable, Mapping, Sequence

from ..actions import invoke, plan_order
from ..goals import Condition, GoalSpec, MeasuredActionGoal, normalize_goal_value
from ..language import ENGLISH, Context, Entity, Frame, Grammar, Question, Request
from ..language import conventions, verbnet, wordnet
from ..language.semantics import explicit_ref, to_propositions
from ..outcomes import Receipt, Unknown, Verdict
from ..records import Claim, Evidence, Proposition, Ref, Store, Var, matches
from ..runtime import Runtime, use
from .. import ops
from .operations import Plan, Transcript, agent_runtime, install_learned_reader
from .plugin import Call, Capability, Plugin
from .understand import Act, Sentence, SentenceAlternative
from .interpretation import InterpretationExpansion, InterpretationGroup, InterpretationWorkspace
from .scene import SceneProposal
from .investigation import CandidateHypothesis, InvestigationResult, investigate
from .tasks import StepAttempt, TaskLedger
from .planning import plan_goal

USER = Ref("agent:user")
SELF = Ref("agent:self")


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
    interpretation_id: str | None = None
    candidate_id: str | None = None
    goal_interpretation_id: str | None = None


@dataclass(frozen=True)
class InterpretationDecision:
    """A procedural choice with its basis; None keeps the sentence unresolved."""
    candidate_id: str | None
    reason: str
    evidence_ids: tuple[str, ...] = ()
    compared_revision: int | None = None
    compared_candidate_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class InvestigatedInterpretation:
    group_id: str
    evidence_source_id: str
    result: InvestigationResult
    decision: InterpretationDecision


@dataclass(frozen=True)
class InterpretationResolution:
    """Bounded search followed by investigation; resolution may remain unknown."""
    expansion: InterpretationExpansion | None
    investigation: InvestigatedInterpretation


@dataclass(frozen=True)
class InterpretedMessage:
    """An unexecuted reading backed by retained source and candidate records."""
    transcript: Transcript
    source_id: str
    group_ids: tuple[str, ...]
    unavailable: Unknown | None = None


@dataclass(frozen=True)
class InterpretedImage:
    """Retained image evidence and uncommitted source-bound scene proposals."""
    image: Ref
    source_id: str
    group_ids: tuple[str, ...]


@dataclass
class Turn:
    text: str
    sentences: list[Sentence]
    outcomes: list[Outcome]
    reply: str
    events: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    interpretation_ids: tuple[str, ...] = ()
    visual_interpretation_ids: tuple[str, ...] = ()


class Agent:
    def __init__(self, plugins: Sequence[Plugin] = (), *, grammar: Grammar | None = None, runtime: Runtime | None = None,
                 reader: Any = None,
                 interpretation_selector: Callable[[InterpretationGroup], InterpretationDecision] | None = None,
                 interpretation_hypotheses: Callable[[InterpretationGroup], Sequence[CandidateHypothesis]] | None = None,
                 interpretation_probe_budget: int = 8,
                 interpretation_expansion_budget: int = 64,
                 interpretation_candidate_budget: int = 16,
                 goal_selector: Callable[[InterpretationGroup], InterpretationDecision] | None = None,
                 goal_derivation_budget: int = 256,
                 goal_model: Any = None, speech_act_model: Any = None,
                 informing_model: Any = None, informing_selector=None,
                 store_query_model: Any = None, store_query_selector=None) -> None:
        """``reader`` names which registered ``parse`` implementation to prefer.

        The default is the hand-written grammar and ``"learned"`` is the treebank one. An
        already-built :class:`~tensorcode.agent.understand.LearnedReader` may be handed in
        instead, in which case it is reused rather than loaded again. Either way the reading
        happens through :func:`tensorcode.ops.parse`, so a runtime whose policy orders
        implementations differently switches readers without touching this class.

        ``interpretation_selector`` receives a detached candidate group and returns
        an InterpretationDecision. None as its candidate defers handling that
        sentence. Without a selector, language interpretation stays unresolved
        and no candidate acts are dispatched. Reader order is not authorization.

        A supplied ``interpretation_hypotheses`` producer instead receives the
        candidate set after one bounded continuation advance. Expansion, output,
        and observation budgets apply per sentence group per selection call.
        Remaining pending work prevents investigation-based commitment.

        ``speech_act_model`` optionally names an explicitly admitted model over
        retained neutral syntax. Its proposals remain unselected alternatives;
        no model or unsupported syntax leaves communicative intent unresolved.

        ``goal_model`` optionally names an explicitly admitted, local learned
        correspondence model. Its predictions replace lexical goal projection;
        unavailable models and unsupported inputs do not fall back to VerbNet.
        ``goal_selector`` separately selects a retained goal proposal.
        Selecting a language reading does not authorize a lexical sense or role
        mapping. Without this policy, even a singleton goal stays unresolved.
        """
        if interpretation_selector is not None and interpretation_hypotheses is not None:
            raise ValueError("supply either an interpretation selector or hypothesis producer")
        for name, value in (("interpretation_probe_budget", interpretation_probe_budget),
                            ("interpretation_expansion_budget", interpretation_expansion_budget),
                            ("interpretation_candidate_budget", interpretation_candidate_budget),
                            ("goal_derivation_budget", goal_derivation_budget)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        self.interpretation_hypotheses = interpretation_hypotheses
        self.interpretation_probe_budget = interpretation_probe_budget
        self.interpretation_expansion_budget = interpretation_expansion_budget
        self.interpretation_candidate_budget = interpretation_candidate_budget
        self.goal_model = goal_model
        self.speech_act_model = speech_act_model
        self.store_query_model = store_query_model
        self.store_query_selector = store_query_selector
        self.informing_model = informing_model
        self.informing_selector = informing_selector
        self.goal_selector = goal_selector
        self.goal_derivation_budget = goal_derivation_budget
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
        self.interpretations = InterpretationWorkspace()
        self.interpretation_selector = interpretation_selector
        self._experience_plans = {}
        self._experience_investigations = {}
        self._empirical_task_locks = {}
        self._structured_task_locks = {}
        self._calls = 0
        self._images = 0
        self.last_image: Ref | None = None
        # Plugin context is a lifecycle dependency, not a side effect of guessing
        # which entity a description names. Attach only after all state exists.
        for plugin in self.plugins:
            attach = getattr(plugin, "attach", None)
            if callable(attach):
                attach(self)

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

    # ------------------------------------------------------------------ the turn

    def interpret(self, text: str) -> InterpretedMessage:
        """Retain input and alternative readings without perceiving or executing.

        Reader order is not calibrated confidence. No candidate is selected here
        and no proposed statement enters the belief store. The original message
        survives reader preprocessing and multiline composition. Group provenance
        identifies sentence index/text; readers that provide exact source spans
        retain those anchors in each alternative's metadata.
        """
        with use(self.runtime):
            parsed = ops.parse(text, Transcript, grammar=self.grammar, prefer=self.prefer_reader)
        unavailable = parsed if isinstance(parsed, Unknown) else None
        transcript = Transcript() if unavailable is not None else parsed
        source = self.interpretations.add_source(text, provider=transcript.by)
        groups = []
        for index, sentence in enumerate(transcript):
            group = self.interpretations.create_group(
                source.id, provenance=(f"sentence-index:{index}", sentence.text))
            alternatives = sentence.alternatives or (SentenceAlternative(
                sentence.reading, sentence.acts, sentence.skipped, sentence.guessed,
                "reader-single-proposal"),)
            for alternative in alternatives:
                self.interpretations.propose(group.id, alternative,
                                             provenance=(transcript.by, alternative.provenance))
            if sentence.continuation is not None:
                self.interpretations.attach_continuation(group.id, sentence.continuation)
            if self.speech_act_model is not None:
                from .speech_act_learning import project_speech_act_group
                project_speech_act_group(self, self.speech_act_model, group.id)
            groups.append(group.id)
        return InterpretedMessage(transcript, source.id, tuple(groups), unavailable)

    def interpret_image(self, image: Any) -> InterpretedImage:
        """Retain a visual source and proposed scene graphs without asserting them.

        Graphs from one provider form alternative accounts. Different providers
        get separate groups because their proposals may cover different aspects;
        no fusion or cross-provider mutual exclusivity is inferred. Paths are
        snapshotted as bytes so later file changes cannot rewrite evidence.
        """
        self._images += 1
        ref = Ref(f"image:{self._images}")
        metadata: dict[str, Any] = {"image_ref": ref.id}
        payload = image
        if isinstance(image, (str, Path)):
            try:
                is_file = Path(image).is_file()
            except OSError:
                is_file = False
            if is_file:
                metadata["original_path"] = str(image)
                payload = Path(image).read_bytes()
        source = self.interpretations.add_source(
            ref.id, modality="image", metadata=metadata, payload=payload)
        groups = []
        for plugin in self.plugins:
            proposals = plugin.interpret_image(deepcopy(source.payload), ref)
            if proposals is None:
                raise TypeError("interpret_image must return proposals or an empty iterable")
            proposals = tuple(proposals)
            # Validate before adding any candidate from this provider.
            for proposal in proposals:
                if not isinstance(proposal, SceneProposal):
                    raise TypeError("interpret_image providers must return SceneProposal values")
                proposal.validate()
                if proposal.graph.image != ref:
                    raise ValueError("scene graph belongs to a different image source")
            group = self.interpretations.create_group(
                source.id, provenance=(f"plugin:{plugin.name}", "visual-scene-proposals"))
            for proposal in proposals:
                self.interpretations.propose(group.id, proposal,
                                             provenance=(f"plugin:{plugin.name}", *proposal.provenance))
            groups.append(group.id)
        return InterpretedImage(ref, source.id, tuple(groups))

    def expand_interpretation(
        self, group_id: str, *, max_expansions: int = 64, max_candidates: int = 16,
    ):
        """Resume retained interpretation work without selecting or executing it.

        Newly published alternatives invalidate an earlier selection, preserving
        its history and already executed task receipts. No text is reparsed.
        """
        result = self.interpretations.expand(
            group_id, max_expansions=max_expansions, max_candidates=max_candidates)
        if self.speech_act_model is not None:
            from .speech_act_learning import project_speech_act_group
            project_speech_act_group(self, self.speech_act_model, group_id)
        return result

    def investigate_interpretation(
        self, group_id: str, hypotheses: Sequence[CandidateHypothesis], *, max_probes: int = 8,
    ) -> InvestigatedInterpretation:
        """Test supplied interpretation predictions against fresh observations.

        Every non-rejected candidate must be represented, including candidates
        with no usable predictions. Evidence can distinguish supplied accounts;
        it does not establish that the candidate set covers the user's meaning.
        Reinvestigation can withdraw a prior selection without replaying actions.
        A winner among materialized hypotheses is retained as diagnostic evidence,
        but pending continuation work prevents a public selection. Exhausting a
        continuation does not prove that the reader generated all possible meanings.
        """
        group = self.interpretations.get(group_id)
        frontier = self.interpretations.continuation_status(group_id)
        hypotheses = tuple(deepcopy(hypotheses))
        if any(not isinstance(h, CandidateHypothesis) for h in hypotheses):
            raise TypeError("interpretation hypotheses must be CandidateHypothesis values")
        expected = {c.id for c in group.candidates if not c.rejected}
        supplied = [h.candidate_id for h in hypotheses]
        if len(set(supplied)) != len(supplied) or set(supplied) != expected:
            raise ValueError("hypotheses must cover every non-rejected candidate exactly once")
        result = investigate(hypotheses, self.plugins, max_probes=max_probes)
        # Provider callbacks cannot silently change the question under examination.
        current = self.interpretations.get(group_id)
        if (current.revision != group.revision or
                tuple(c.id for c in current.candidates) != tuple(c.id for c in group.candidates) or
                self.interpretations.continuation_status(group_id) != frontier):
            raise RuntimeError("interpretation group changed during investigation")
        candidate_result = result
        if frontier.pending:
            result = replace(result, selected_id=None, reason="interpretation_search_pending")
        source = self.interpretations.add_source(
            f"Investigation of {group_id}", modality="observation", provider="interpretation-investigation",
            metadata={"group_id": group_id, "input_source_id": group.source_id, "max_probes": max_probes,
                      "continuation_status": frontier},
            payload={"hypotheses": hypotheses, "result": result, "candidate_result": candidate_result})
        decision = InterpretationDecision(result.selected_id, result.reason, (source.id,))
        if decision.candidate_id is None:
            self.interpretations.unset(group_id, reason=decision.reason, evidence_ids=decision.evidence_ids)
        else:
            self.interpretations.select(group_id, decision.candidate_id,
                                        reason=decision.reason, evidence_ids=decision.evidence_ids)
        return InvestigatedInterpretation(group_id, source.id, result, decision)

    def resolve_interpretation(
        self, group_id: str,
        hypotheses: Callable[[InterpretationGroup], Sequence[CandidateHypothesis]], *,
        max_expansions: int = 64, max_candidates: int = 16, max_probes: int = 8,
    ) -> InterpretationResolution:
        """Expand pending work before generating and testing supplied hypotheses.

        Budgets bound one invocation, not the lifetime of an interpretation. The
        supplied hypothesis producer sees the enlarged candidate set. Pending
        branches withhold a decision even when visible candidates have a winner.
        No hypothesis semantics or expansion priority is learned by this method.
        """
        for name, value in (("max_expansions", max_expansions),
                            ("max_candidates", max_candidates), ("max_probes", max_probes)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not callable(hypotheses):
            raise TypeError("hypotheses must be a candidate hypothesis producer")
        frontier = self.interpretations.continuation_status(group_id)
        expansion = None
        if frontier.pending and max_expansions and max_candidates:
            expansion = self.expand_interpretation(
                group_id, max_expansions=max_expansions, max_candidates=max_candidates)
        group = self.interpretations.get(group_id)
        frontier = self.interpretations.continuation_status(group_id)
        proposed = tuple(deepcopy(hypotheses(group)))
        current = self.interpretations.get(group_id)
        if (current.revision != group.revision or
                tuple(c.id for c in current.candidates) != tuple(c.id for c in group.candidates) or
                self.interpretations.continuation_status(group_id) != frontier):
            raise RuntimeError("interpretation group changed during hypothesis generation")
        investigation = self.investigate_interpretation(group_id, proposed, max_probes=max_probes)
        return InterpretationResolution(expansion, investigation)

    def _select_interpretation(self, group_id: str) -> InterpretationDecision:
        group = self.interpretations.get(group_id)
        if self.interpretation_hypotheses is not None:
            return self.resolve_interpretation(
                group_id, self.interpretation_hypotheses,
                max_expansions=self.interpretation_expansion_budget,
                max_candidates=self.interpretation_candidate_budget,
                max_probes=self.interpretation_probe_budget).investigation.decision
        if self.interpretation_selector is None:
            decision = InterpretationDecision(
                None, "no interpretation policy supplied; meaning remains unresolved")
        else:
            decision = self.interpretation_selector(group)
        if not isinstance(decision, InterpretationDecision):
            raise TypeError("interpretation_selector must return InterpretationDecision")
        current = self.interpretations.get(group_id)
        # A policy may deliberately propose grounded alternatives while deciding.
        # It must then acknowledge the exact enlarged comparison set, rather than
        # silently committing a decision made against its older input snapshot.
        explicit_basis = decision.compared_revision is not None or decision.compared_candidate_ids is not None
        expected_revision = decision.compared_revision if explicit_basis else group.revision
        expected_ids = decision.compared_candidate_ids if explicit_basis else tuple(c.id for c in group.candidates)
        if (type(expected_revision) is not int or current.revision != expected_revision or
                tuple(c.id for c in current.candidates) != expected_ids):
            raise RuntimeError("interpretation group changed during selection; exact comparison basis required")
        if decision.candidate_id is None:
            self.interpretations.unset(group_id, reason=decision.reason, evidence_ids=decision.evidence_ids)
        else:
            self.interpretations.select(group_id, decision.candidate_id, reason=decision.reason, evidence_ids=decision.evidence_ids)
        return decision

    def turn(self, text: str, images: Sequence[Any] = ()) -> Turn:
        """One message: optional images first (seen by every plugin that sees), then the text."""
        t0 = time.perf_counter()
        events: list[dict] = []
        with use(self.runtime):
            self.perceive(events)
            visual_groups = []
            for image in images:
                visual = self.interpret_image(image)
                self.last_image = visual.image
                visual_groups.extend(visual.group_ids)
                events.append({"type": "seen", "image": visual.image.id,
                               "source": visual.source_id,
                               "interpretations": list(visual.group_ids)})
            interpreted = self.interpret(text)
            transcript = interpreted.transcript
            if interpreted.unavailable is not None:
                events.append({"type": "unread", "reason": interpreted.unavailable.reason,
                               "detail": interpreted.unavailable.detail})
            events.append({"type": "read", "by": transcript.by, "sentences": len(transcript)})
            sents, decisions, selected_groups, selected_frontiers = [], [], [], []
            unresolved_readings = {}
            # Select before handling any acts in this message.
            for sentence, group_id in zip(transcript, interpreted.group_ids):
                decision = self._select_interpretation(group_id)
                group = self.interpretations.get(group_id)
                candidate = next((c for c in group.candidates if c.id == decision.candidate_id), None)
                if candidate is None:
                    selected = replace(sentence, acts=())
                else:
                    reading = candidate.payload
                    from .scene_grounding import UnresolvedGrounding
                    if isinstance(reading, UnresolvedGrounding):
                        unresolved_readings[len(sents)] = reading.reason
                        selected = replace(sentence, acts=())
                    elif isinstance(reading, SentenceAlternative) and (reading.metadata.get('speech_act_complete') is False
                            or any(a.kind == 'unresolved' for a in reading.acts)):
                        unresolved_readings[len(sents)] = 'speech_act_unresolved'
                        selected = replace(sentence, acts=())
                    else:
                        selected = replace(sentence, reading=reading.reading, acts=reading.acts,
                                           tokens=tuple(reading.metadata.get("tokens", sentence.tokens)),
                                           skipped=reading.skipped, guessed=reading.guessed)
                sents.append(selected)
                decisions.append(decision)
                selected_groups.append(group)
                selected_frontiers.append(self.interpretations.continuation_status(group_id))
                events.append({"type": "interpretation_selection", "group": group_id,
                               "source": group.source_id, "candidate": decision.candidate_id,
                               "alternatives": len(group.candidates), "reason": decision.reason,
                               "revision": group.revision, "evidence": list(decision.evidence_ids)})
            for s in sents:
                events.append({"type": "parsed", "sentence": s.text, "coverage": s.coverage, "skipped": list(s.skipped),
                               "guessed": [list(g) for g in s.guessed], "acts": [a.describe() for a in s.acts], "ms": s.parse_ms})
            outcomes = []
            requests_in_message = sum(1 for s in sents for a in s.acts if a.kind == "request")
            deferred_indices = {i for i, decision in enumerate(decisions) if decision.candidate_id is None}
            deferred_indices.update(unresolved_readings)
            for index, (s, group_id, decision, selected_group, selected_frontier) in enumerate(zip(
                    sents, interpreted.group_ids, decisions, selected_groups, selected_frontiers)):
                if decision.candidate_id is None or index in unresolved_readings:
                    outcomes.append(Outcome(Act("fragment", s.text, None), "unknown",
                                            reason=unresolved_readings.get(index, decision.reason),
                                            interpretation_id=group_id, candidate_id=decision.candidate_id))
                    continue
                for a in s.acts:
                    from .scene_grounding import grounding_dependencies
                    supporting = grounding_dependencies(self, group_id, decision.candidate_id)
                    if isinstance(supporting, Unknown):
                        outcomes.append(Outcome(a, "unknown", reason=supporting.reason,
                                                verified=supporting, interpretation_id=group_id))
                        deferred_indices.add(index)
                        break
                    current = self.interpretations.get(group_id)
                    if (current.revision != selected_group.revision or
                            current.selected_id != decision.candidate_id or
                            tuple(c.id for c in current.candidates) != tuple(c.id for c in selected_group.candidates) or
                            self.interpretations.continuation_status(group_id) != selected_frontier):
                        reason = "interpretation changed before dispatch; reconsideration required"
                        outcomes.append(Outcome(Act("fragment", s.text, None), "unknown",
                                                reason=reason, interpretation_id=group_id))
                        deferred_indices.add(index)
                        events.append({"type": "interpretation_stale", "group": group_id, "reason": reason})
                        break
                    if a.interpretation is not None:
                        events.append({"type": "interpretation", "convention": a.interpretation.convention_id,
                                       "source": a.interpretation.source})
                    dependency = None
                    if a.kind in {"request", "question"}:
                        try:
                            dependency = self.capture_task_dependency(group_id,
                                basis=("Explicitly selected reading supplies the task or informing interpretation",),
                                evidence_ids=decision.evidence_ids)
                            if (dependency.revision != selected_group.revision or
                                    dependency.candidate_id != decision.candidate_id or
                                    dependency.candidate_ids != tuple(c.id for c in selected_group.candidates) or
                                    dependency.continuation != selected_frontier):
                                raise ValueError("selected reading comparison changed")
                        except ValueError as exc:
                            outcomes.append(Outcome(a, "unknown", reason=str(exc), interpretation_id=group_id))
                            deferred_indices.add(index)
                            break
                    o = self.handle(s, a, events, requests_in_message=requests_in_message,
                                    interpretation_dependency=dependency)
                    o = replace(o, interpretation_id=group_id, candidate_id=decision.candidate_id)
                    outcomes.append(o)
                    if a.frame is not None:
                        self.context.observe(a.meaning)
            from .reply import compose

            reply = compose(self, sents, outcomes, deferred_indices=frozenset(deferred_indices))
        turn = Turn(text, sents, outcomes, reply, events, round(time.perf_counter() - t0, 3),
                    interpreted.group_ids, tuple(visual_groups))
        self.turns.append(turn)
        return turn

    def perceive(self, events: list[dict]) -> None:
        self._capture_observations(events, stage="perception", providers=self.plugins)
        for p in self.plugins:
            n = 0
            for claim in p.perceive():
                self.store.tell(claim, Evidence(source=Ref(f"plugin:{p.name}"), observed_at=datetime.now(timezone.utc), method="perception"))
                n += 1
            events.append({"type": "perceived", "plugin": p.name, "claims": n})

    def _capture_observations(
        self, events: list[dict], *, stage: str, providers: Sequence[Plugin],
        action: Call | None = None, receipt: Receipt | None = None,
        attempt_id: str | None = None, retain_unavailable: bool = False,
    ) -> tuple[str, ...]:
        """Retain raw evidence and failed observation attempts, never world claims."""
        sources = []
        seen = set()
        for provider in providers:
            if id(provider) in seen:
                continue
            seen.add(id(provider))
            metadata = {"stage": stage, "attempt_id": attempt_id,
                        "action": action, "receipt": receipt,
                        "observed_at": datetime.now(timezone.utc).isoformat()}
            payload = None
            try:
                payload = provider.observe_evidence()
                if payload is None:
                    if not retain_unavailable:
                        continue
                    metadata["status"] = "unavailable"
                elif isinstance(payload, Unknown):
                    metadata.update(status="unavailable", reason=payload.reason, detail=payload.detail)
                    payload = None
                else:
                    metadata["status"] = "observed"
                source = self.interpretations.add_source(
                    "", modality="observation", provider=f"plugin:{provider.name}",
                    metadata=metadata, payload=payload)
            except Exception as exc:
                # Failure to observe or snapshot says nothing about world state.
                # It does not change an action receipt or suppress other providers.
                metadata.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
                try:
                    source = self.interpretations.add_source(
                        "", modality="observation", provider=f"plugin:{provider.name}",
                        metadata=metadata)
                except Exception as linkage_exc:
                    # The actual Call/Receipt may itself contain an uncopyable
                    # executor value. Retain that loss explicitly; observation
                    # bookkeeping must not block dispatch or replace its receipt.
                    metadata = {"stage": stage, "attempt_id": attempt_id,
                                "observed_at": metadata["observed_at"],
                                "status": "error", "error": metadata["error"],
                                "action": None, "receipt": None,
                                "receipt_status": receipt.status if receipt is not None else None,
                                "linkage_snapshot_error": {
                                    "type": type(linkage_exc).__name__, "message": str(linkage_exc)}}
                    source = self.interpretations.add_source(
                        "", modality="observation", provider=f"plugin:{provider.name}",
                        metadata=metadata)
            sources.append(source.id)
            events.append({"type": "observation", "source_id": source.id,
                           "provider": source.provider, "stage": stage,
                           "status": metadata["status"], "attempt_id": attempt_id})
        return tuple(sources)

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

    def handle(self, s: Sentence, act: Act, events: list[dict], *, requests_in_message: int,
               interpretation_dependency=None) -> Outcome:
        if act.kind == 'unresolved':
            return Outcome(act, 'unknown', verified=Unknown('speech_act_unresolved'),
                           reason='speech_act_unresolved')
        if act.frame is not None:
            frame = self.deixis(act.frame)
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
            return self.ask(s, act, events, interpretation_dependency=interpretation_dependency)
        if act.kind == "request":
            return self.request(s, act, events, interpretation_dependency=interpretation_dependency)
        if act.kind == "mention":
            return Outcome(act, "mentioned")
        if (move := self.conversational_move(s)) is not None:
            formula = conventions.pairs().get(move)
            events.append({"type": "convention", "move": move, "answers": formula})
            return Outcome(act, "reciprocated", goal=move, answer=formula)
        return Outcome(act, "not_understood", reason="a phrase that is not a statement, question or request")

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
        got, dropped = to_propositions(act.frame, source=USER, scope=USER, resolve=explicit_ref,
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

    def ask(self, s: Sentence, act: Act, events: list[dict], *, interpretation_dependency=None) -> Outcome:
        """Look if it can be looked at; otherwise answer from what it was told or saw before."""
        q: Question = act.meaning
        if self._question_bindings(q) is None:
            return Outcome(act, "unknown", reason="stated question roles require explicit grounding")
        looked = self._look(q, act, events, parent_dependency=interpretation_dependency)
        if looked is not None:
            return looked
        from .store_query_learning import answer_store_question
        return answer_store_question(self, q, act, events, parent_dependency=interpretation_dependency)

    def _look(self, q: Question, act: Act, events: list[dict], *, parent_dependency=None) -> Outcome | None:
        from .informing_learning import answer_informing_question
        return answer_informing_question(self, q, act, events, parent_dependency=parent_dependency)

    def lookup(self, q: Question, *, interpretation_dependency=None) -> list[Any] | Unknown:
        """Use an explicitly selected learned store query for this retained question."""
        from .store_query_learning import answer_store_question
        outcome = answer_store_question(self, q, Act("question", q, q.frame), [],
                                        parent_dependency=interpretation_dependency)
        return outcome.answer if outcome.status == "answered" else Unknown("store_answer_unavailable", outcome.reason)

    @staticmethod
    def _ref_of(value: Any) -> Ref | None:
        """Return an explicit identity; descriptions never mint or guess one."""
        if isinstance(value, Ref):
            return value
        return value.ref if isinstance(value, Entity) and isinstance(value.ref, Ref) else None

    @staticmethod
    def _grounded_value(value: Any) -> Any:
        if isinstance(value, Entity):
            return explicit_ref(value)
        if isinstance(value, (Ref, str, int, float, bool)):
            return value
        if isinstance(value, (tuple, list)):
            values = tuple(Agent._grounded_value(v) for v in value)
            if any(v is None or isinstance(v, Unknown) for v in values):
                return Unknown("unresolved_reference", "unresolved collection member")
            return values
        return Unknown("unresolved_reference", "role is not explicitly grounded")

    def _question_bindings(self, q: Question) -> dict[str, Any] | None:
        """All stated roles must resolve; dropping one would broaden the question."""
        bound = {}
        for role, value in q.frame.roles.items():
            resolved = self._grounded_value(value)
            if resolved is None or isinstance(resolved, Unknown):
                return None
            bound[role] = resolved
        return bound

    # ------------------------------------------------------------------ requests

    def request(self, s: Sentence, act: Act, events: list[dict], *, interpretation_dependency=None) -> Outcome:
        """Retain the interpreted task separately from its chosen action and outcome.

        Natural-language task correction is not inferred here: each request creates
        a task. Structured callers can revise and retry a named task with ``pursue``.
        """
        from .task_dependencies import validate_dependencies
        dependencies = () if interpretation_dependency is None else (interpretation_dependency,)
        from .scene_grounding import grounding_dependencies
        supporting = (() if interpretation_dependency is None else grounding_dependencies(
            self, interpretation_dependency.group_id, interpretation_dependency.candidate_id))
        support_error = supporting if isinstance(supporting, Unknown) else None
        if support_error is None:
            dependencies = (*dependencies, *supporting)
        goal_interpretation_id = None
        def retain_resolution(resolution):
            nonlocal dependencies, goal_interpretation_id
            goal_interpretation_id = resolution.group_id
            if resolution.dependency is not None:
                dependencies = (*dependencies, *resolution.supporting_dependencies, resolution.dependency)
        def execution_guard():
            return support_error if support_error is not None else validate_dependencies(self.interpretations, dependencies)
        derived_goal = None
        def retain_goal(goal):
            nonlocal derived_goal
            derived_goal = goal
        event_start = len(events)
        try:
            if support_error is not None:
                outcome = Outcome(act, "unknown", verified=support_error, reason=support_error.reason)
            else:
                outcome = self._request(s, act, events, execution_guard=execution_guard,
                                        on_goal=retain_goal, on_goal_resolution=retain_resolution,
                                        parent_dependency=interpretation_dependency)
        except Exception as exc:
            outcome = self._interrupted_task_outcome(act, derived_goal, events[event_start:], exc)
        authorization = execution_guard()
        if authorization is not True:
            outcome = replace(outcome, status="unknown", verified=authorization, reason=authorization.reason)
        task = self.tasks.create(s.text, outcome.goal, dependencies=dependencies,
                                 goal_interpretation_id=goal_interpretation_id)
        outcome = replace(outcome, task_id=task.id, goal_interpretation_id=goal_interpretation_id)
        task = self.tasks.record(task.id, outcome)
        events.append({"type": "task", "task_id": task.id, "revision": task.revision, "status": task.status})
        return outcome

    def adopt_task_goal(self, task_id: str, decision: InterpretationDecision, *, reason: str):
        """Revise a task from an explicit choice among its retained goal proposals.

        This performs no parsing, new lexical search, perception, or execution.
        Domain refinement remains supplied policy. Existing receipts keep their
        original revisions; the adopted revision becomes ready for an explicit
        subsequent pursuit only while all interpretation dependencies remain valid.
        """
        from .goal_interpretation import select_goal
        from .task_dependencies import validate_dependencies

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("goal adoption requires a nonempty revision reason")
        if not isinstance(decision, InterpretationDecision):
            raise TypeError("goal adoption requires an explicit InterpretationDecision")
        task = self.tasks.get(task_id)
        group_id = task.goal_interpretation_id
        if group_id is None:
            return Unknown("no_retained_goal_interpretation")
        others = tuple(dependency for dependency in task.dependencies if dependency.group_id != group_id)
        valid = validate_dependencies(self.interpretations, others)
        if valid is not True:
            return valid
        if self.tasks.current_revision(task.id) != task.revision:
            return Unknown("task_revision_changed")
        resolution = select_goal(self, group_id, decision=decision)
        if isinstance(resolution.goal, Unknown):
            return resolution.goal
        if resolution.dependency is None:
            return Unknown("goal_interpretation_unresolved")
        try:
            group = self.interpretations.get(group_id)
            source = self.interpretations.get_source(group.source_id)
            parent = source.metadata.get("parent_dependency")
            # A manually created task may link a genuine retained group without
            # copying its parent commitment. Adoption must not lose that lineage.
            if parent is not None and parent not in others:
                others = (*others, parent)
            for supporting in resolution.supporting_dependencies:
                if supporting not in others:
                    others = (*others, supporting)
            dependencies = (*others, resolution.dependency)
            with use(self.runtime):
                refined, failure = self._refine_request_goal(resolution.goal, [])
            if failure is not None:
                return failure
            obligation = self._lexical_goal_obligations(refined)
            if obligation is not None:
                return obligation
            guard_failure = None
            def before_commit():
                nonlocal guard_failure
                if self.tasks.current_revision(task.id) != task.revision:
                    guard_failure = Unknown("task_revision_changed")
                    return False
                guard_failure = validate_dependencies(self.interpretations, dependencies)
                if self.tasks.current_revision(task.id) != task.revision:
                    guard_failure = Unknown("task_revision_changed")
                return guard_failure is True
            if not before_commit():
                return guard_failure
            try:
                return self.tasks.revise(task.id, refined, reason=reason, dependencies=dependencies,
                    goal_interpretation_id=group_id, expected_revision=task.revision,
                    before_commit=before_commit)
            except ValueError as error:
                if self.tasks.current_revision(task.id) != task.revision:
                    return Unknown("task_revision_changed", str(error))
                if guard_failure is not True and guard_failure is not None:
                    return guard_failure
                return Unknown("goal_adoption_error", str(error))
        except Exception as error:
            return Unknown("goal_adoption_error", f"{type(error).__name__}: {error}")

    def pursue(self, goal: GoalSpec | None = None, *, task_id: str | None = None,
               source: str = "structured", events: list[dict] | None = None,
               max_steps: int | None = None, dependencies=()) -> Outcome:
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
        if task_id is not None and dependencies:
            raise ValueError("revise a task explicitly to change its interpretation dependencies")
        if goal is not None and not isinstance(goal, GoalSpec):
            raise TypeError("structured goals must be GoalSpec values")
        if max_steps is not None and (not isinstance(max_steps, int) or max_steps < 1):
            raise ValueError("max_steps must be a positive integer")
        task = self.tasks.create(source, goal, dependencies=dependencies) if goal is not None else self.tasks.get(task_id)
        lock = self._structured_task_locks.setdefault(task.id, Lock())
        if not lock.acquire(blocking=False):
            raise ValueError("task already has an active attempt")
        try:
            return self._pursue_task(task.id, events=events, max_steps=max_steps)
        finally:
            lock.release()

    def _pursue_task(self, task_id: str, *, events, max_steps) -> Outcome:
        # Re-read under admission lock: a completed/revised attempt may have
        # arrived after the public call obtained its initial task snapshot.
        task = self.tasks.get(task_id)
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
        def execution_guard():
            from .task_dependencies import validate_dependencies
            if self.tasks.current_revision(task.id) != task.revision:
                return Unknown("task_revision_changed")
            result = validate_dependencies(self.interpretations, task.dependencies)
            if self.tasks.current_revision(task.id) != task.revision:
                return Unknown("task_revision_changed")
            return result
        event_start = len(events)
        try:
            with use(self.runtime):
                self.perceive(events)
                outcome = self._execute_goal(task.goal, act, events, max_steps=max_steps,
                                             execution_guard=execution_guard)
        except Exception as exc:
            outcome = self._interrupted_task_outcome(act, task.goal, events[event_start:], exc)
        authorization = execution_guard()
        if authorization is not True:
            outcome = replace(outcome, status="suspended" if authorization.reason == "task_revision_changed" else "unknown",
                              verified=authorization, reason=authorization.reason)
        outcome = replace(outcome, task_id=task.id)
        task = self.tasks.record(task.id, outcome, revision=task.revision)
        events.append({"type": "task", "task_id": task.id, "revision": task.revision, "status": task.status})
        return outcome

    def _interrupted_task_outcome(self, act: Act, goal: Any, events: list[dict], error: Exception) -> Outcome:
        """Retain dispatched receipts when task execution fails before returning."""
        interrupted = Unknown("task_attempt_error", f"{type(error).__name__}: {error}")
        steps = []
        for dispatched in (event for event in events if event.get("type") == "act"):
            attempt_id = dispatched["attempt_id"]
            sources = []
            for event in events:
                if event.get("type") == "observation" and event.get("attempt_id") == attempt_id:
                    try:
                        sources.append(self.interpretations.get_source(event["source_id"]))
                    except Exception:
                        # An unreadable evidence snapshot cannot establish that
                        # dispatch did not occur. Keep an indeterminate receipt.
                        continue
            call = next((source.metadata["action"] for source in sources
                         if source.metadata.get("action") is not None),
                        Unknown("unretained_action", attempt_id))
            receipt = next((source.metadata["receipt"] for source in sources
                            if isinstance(source.metadata.get("receipt"), Receipt)),
                           Receipt(call, "indeterminate", error=interrupted.detail))
            steps.append(StepAttempt(attempt_id, call, receipt, interrupted))
        return Outcome(act, "unverified" if steps else "unknown", goal=goal,
                       plan=interrupted, receipt=steps[-1].receipt if steps else None,
                       verified=interrupted, reason=interrupted.detail, steps=tuple(steps))

    def _request(self, s: Sentence, act: Act, events: list[dict], *, execution_guard=None,
                 on_goal=None, on_goal_resolution=None, parent_dependency=None) -> Outcome:
        missed = [w for w in s.skipped if any(c.isalnum() for c in w)]
        if missed:
            # acting on part of a sentence is how "processes" became a process listing
            return Outcome(act, "not_understood", reason=f"I didn't follow {' '.join(repr(w) for w in missed)}")
        from .goal_interpretation import resolve_goal
        resolution = resolve_goal(self, act.frame, s.text, parent_dependency=parent_dependency,
                                  max_derivations=self.goal_derivation_budget)
        if on_goal_resolution is not None:
            on_goal_resolution(resolution)
        events.append({"type": "goal_interpretation", "group_id": resolution.group_id,
                       "parent_group_id": parent_dependency.group_id if parent_dependency else None,
                       "selected": resolution.dependency.candidate_id if resolution.dependency else None})
        goal = resolution.goal
        if on_goal is not None:
            on_goal(goal)
        events.append({"type": "goal", "frame": act.frame.describe(),
                       "goal": goal.describe() if hasattr(goal, "describe") else f"unknown: {goal.reason}"})
        if isinstance(goal, Unknown):
            return Outcome(act, "unknown", goal=goal, reason=goal.detail or goal.reason)
        if isinstance(goal, MeasuredActionGoal):
            return Outcome(act, 'suspended', goal=goal,
                verified=Unknown('measured_goal_requires_materialization'),
                reason='measured_goal_requires_materialization')
        refined, failure = self._refine_request_goal(goal, events, on_goal=on_goal)
        if failure is not None:
            if failure.reason == "unconsumed_request_semantics":
                return Outcome(act, "unknown", goal=goal, plan=failure, verified=failure, reason=failure.detail)
            return Outcome(act, "declined", goal=goal, reason=failure.detail or failure.reason)
        return self._execute_goal(refined, act, events, execution_guard=execution_guard)

    def _refine_request_goal(self, goal: verbnet.Goal | GoalSpec | MeasuredActionGoal, events: list[dict], *, on_goal=None):
        """Apply the same explicit domain refinements for request and later adoption."""
        if isinstance(goal, (GoalSpec, MeasuredActionGoal)):
            # Learned correspondences already produce complete explicit goals.
            # Their whole source shape is checked by the admitted model; do not
            # run authored lexical refiners over these independent predictions.
            return goal, None
        obligation = self._request_frame_obligations(goal.frame)
        if obligation is not None:
            return goal, obligation
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
            return goal, Unknown("goal_refinement_unavailable",
                                 "; ".join(error.detail or error.reason for error in refinement_errors))
        if len(refined) > 1:
            return goal, Unknown("ambiguous_goal_refinement", "multiple domain refinements disagree about the desired outcome")
        if refined:
            goal = refined[0]
            if on_goal is not None:
                on_goal(goal)
            events.append({"type": "refined", "goal": goal.describe(), "basis": list(goal.basis)})
        return goal, None

    @staticmethod
    def _request_frame_obligations(frame: Frame) -> Unknown | None:
        """The lexical result-state adapter has no semantics for these qualifiers.

        Imperative mood is consumed by the request wrapper. In particular, a
        prohibition is not the negation of a verb's resulting state. Preserve
        every other supplied feature until an interpreter can represent it.
        """
        remaining = dict(frame.features)
        if remaining.get("mood") == "imperative":
            remaining.pop("mood")
        paths = [f"frame.features[{key!r}]" for key in remaining]
        if "subject" in frame.roles:
            paths.append("frame.roles['subject']")
        if paths:
            return Unknown("unconsumed_request_semantics", ", ".join(paths))
        return None

    @classmethod
    def _lexical_goal_obligations(cls, goal) -> Unknown | None:
        if not isinstance(goal, verbnet.Goal):
            return None
        frame = cls._request_frame_obligations(goal.frame)
        paths = ([frame.detail] if frame is not None else [])
        paths.extend(f"frame.roles[{role!r}]" for role in goal.unmapped_roles)
        if paths:
            return Unknown("unconsumed_request_semantics", ", ".join(paths))
        return None

    def _execute_goal(self, goal: GoalSpec | verbnet.Goal, act: Act, events: list[dict],
                      *, max_steps: int | None = None, execution_guard=None) -> Outcome:
        obligation = self._lexical_goal_obligations(goal)
        if obligation is not None:
            return Outcome(act, "unknown", goal=goal, plan=obligation,
                           verified=obligation, reason=obligation.detail)
        authorization = execution_guard() if execution_guard is not None else True
        if authorization is not True:
            return Outcome(act, "suspended", goal=goal, verified=authorization,
                           reason="task authorization changed before planning")
        if isinstance(goal, GoalSpec):
            conditions = (*goal.conditions, *goal.invariants)
            if any(a.pred == b.pred and a.args == b.args and a.negated != b.negated
                   for i, a in enumerate(conditions) for b in conditions[i + 1:]):
                return Outcome(act, "declined", goal=goal, reason="contradictory explicit conditions")
        providers = tuple(p for p in self.plugins if p.planning_enabled)
        if isinstance(goal, GoalSpec) and providers:
            return self._execute_modeled_goal(goal, act, events, providers, max_steps=max_steps,
                                              execution_guard=execution_guard)
        if isinstance(goal, GoalSpec) and goal.invariants:
            return Outcome(act, "declined", goal=goal, reason="held conditions require an explicit action model")
        plan = self.choose_plan(goal)
        events.append({"type": "plan", "goal": goal.describe(),
                       "plan": {"unknown": plan.reason, "detail": plan.detail} if isinstance(plan, Unknown)
                       else {"plugin": plan[0].name, "capability": plan[1].name, "args": {k: str(v) for k, v in plan[2].items()}}})
        if isinstance(plan, Unknown):
            return Outcome(act, "declined", goal=goal, plan=plan, reason=plan.detail or plan.reason)
        plugin, cap, args = plan
        receipt = self._invoke(plugin, cap, args, events,
                               before_dispatch=(lambda _: execution_guard()) if execution_guard else None)
        if receipt.status in ("rejected", "failed"):
            return Outcome(act, "failed", goal, (plugin.name, cap.name, args), receipt, False, reason=receipt.error or receipt.status)
        self.perceive(events)
        authorization = execution_guard() if execution_guard is not None else True
        if authorization is not True:
            return Outcome(act, "suspended", goal=goal, plan=(plugin.name, cap.name, args), receipt=receipt,
                           verified=authorization, reason="task authorization changed after action")
        # the receipt is the executor's report; what the plugin can still see afterwards is
        # the observation. ops.verify keeps the two apart and records both in the trace.
        verdict = ops.verify(receipt, observe=lambda: plugin.holds(cap, args), expect=lambda seen: seen)
        verified = True if verdict.status == "holds" else False if verdict.status == "fails" \
            else Unknown("unverified", "; ".join(verdict.reasons))
        events.append({"type": "verified", "capability": cap.name,
                       "holds": verified if isinstance(verified, bool) else f"unknown: {verified.reason}"})
        status = "done" if verified is True else "failed" if verified is False else "unverified"
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
                              providers: Sequence[Plugin], *, max_steps: int | None, execution_guard=None) -> Outcome:
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
            authorization = execution_guard() if execution_guard is not None else True
            if authorization is not True:
                return Outcome(act, "suspended", goal=goal, plan=plan, receipt=last_receipt,
                               verified=authorization, reason="task authorization changed between steps",
                               steps=tuple(attempts))
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
            receipt = self._invoke(plugin, cap, args, events, observers=providers,
                                   before_dispatch=(lambda _: execution_guard()) if execution_guard else None)
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
        capability* (every parameter has an argument). Only a unique complete grounded
        candidate can proceed. Multiple feasible actions require an explicit choice;
        declaration order and parameter count do not establish intent.
        """
        options, nearest = self.plans(goal)
        feasible = [plan for plan in options if plan.achieves_all_specified and plan.fully_applied]
        if len(feasible) > 1:
            return Unknown("ambiguous_capability", "multiple complete grounded actions require an explicit choice: "
                           + ", ".join(f"{plan.plugin.name}.{plan.capability.name}" for plan in feasible))
        chosen = ops.choose(
            options,
            objective=ops.Objective("conditions met", "evaluate the unique complete grounded candidate",
                                    utility=lambda plan, _: plan.met),
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
        """Match declared effects to explicitly grounded desired values.

        Supplied specifications preserve exact predicates and role names. Only
        lexical Goal values cross the explicit VerbNet role adapter. Applicability
        belongs to declared preconditions and executors, never description guesses.
        """
        obligation = self._lexical_goal_obligations(goal)
        if obligation is not None:
            return [], [obligation.detail]
        options: list[Plan] = []
        nearest: list[str] = []
        lexical = not isinstance(goal, GoalSpec)
        filled_conditions = []
        for condition in goal.conditions:
            filled = {}
            for role, value in condition.args.items():
                if lexical and (value is None or isinstance(value, str) and value == "addressee"):
                    continue
                target_role = verbnet.role_class(role) if lexical else role
                try:
                    bound = normalize_goal_value(value, path=f"{condition.pred}.{role}")
                except ValueError as exc:
                    return [], [str(exc)]
                if target_role in filled and filled[target_role] != bound:
                    return [], [f"lexical role mapping gives incompatible values for {target_role}"]
                filled[target_role] = bound
            filled_conditions.append((condition, filled))
        for plugin in self.plugins:
            for cap in plugin.capabilities():
                if any(a.pred == b.pred and a.roles == b.roles and a.negated != b.negated
                       for i, a in enumerate(cap.effects) for b in cap.effects[i + 1:]):
                    nearest.append(f"{cap.name} declares contradictory effects")
                    continue
                args: dict[str, Any] = {}
                met = 0
                ok = True
                for condition, filled in filled_conditions:
                    effect = next((e for e in cap.effects if e.pred == condition.pred
                                   and e.negated == condition.negated
                                   and set(filled) <= set(e.roles)), None)
                    if effect is None:
                        continue
                    for role, bound in filled.items():
                        parameter = effect.roles[role]
                        if cap.param(parameter) is None:
                            ok = False
                            nearest.append(f"{cap.name} effect names undeclared parameter {parameter}")
                            break
                        if parameter in args and args[parameter] != bound:
                            ok = False
                            nearest.append(f"{cap.name} needs incompatible bindings for {parameter}")
                            break
                        args[parameter] = bound
                    if not ok:
                        break
                    met += 1
                if not ok or not met:
                    continue
                required = goal.conditions if not lexical else [c for c in goal.conditions if is_specified(c)]
                options.append(Plan(plugin, cap, args, met,
                                    achieves_all_specified=not required or all(
                                        self._achieves(cap, c, lexical=lexical) for c in required),
                                    fully_applied=all(parameter.name in args for parameter in cap.params)))
        return options, nearest

    def _achieves(self, cap: Capability, cond: Condition, *, lexical: bool = False) -> bool:
        filled = {verbnet.role_class(role) if lexical else role for role, value in cond.args.items()
                  if not lexical or value is not None and not (isinstance(value, str) and value == "addressee")}
        return any(effect.pred == cond.pred and effect.negated == cond.negated
                   and filled <= set(effect.roles) for effect in cap.effects)

    def _why_not(self, goal: GoalSpec | verbnet.Goal, nearest: list[str]) -> str:
        """Report candidate failures without interpreting their English wording.

        Grounding/model checks know why a candidate failed. Guessing a different
        explanation from taxonomy or substrings of those reports can conceal the
        actual failure, especially for conditions with no participant roles.
        """
        if nearest:
            return "; ".join(dict.fromkeys(nearest))
        return f"no complete grounded capability achieves {goal.describe()}"

    def propose_experience(self, model, observation_source_id: str, calls: Sequence[Call], desired_outcome):
        """Compare one-step learned predictions for an explicitly supplied target.

        This retains hypothetical alternatives; it neither executes an action nor
        asserts that a predicted outcome has occurred.
        """
        from .experience_planning import propose
        return propose(self, model, observation_source_id, calls, desired_outcome)

    def execute_experience(self, proposal_id: str):
        """Recheck a retained proposal, invoke its action, and observe the outcome."""
        from .experience_planning import execute
        return execute(self, proposal_id)

    def propose_experience_investigation(
        self, group_id: str, bindings, observation_source_id: str, calls: Sequence[Call],
    ):
        """Retain same-probe predictions under supplied model-applicability hypotheses.

        Transition outcomes come from fitted execution evidence. Candidate/model
        bindings, measurement projection, and allowed probe calls remain supplied.
        Proposal does not execute a probe or select an interpretation.
        """
        from .experience_investigation import propose
        return propose(self, group_id, bindings, observation_source_id, calls)

    def execute_experience_investigation(self, proposal_id: str, *, call: Call | None = None):
        """Run one retained probe and assess applicability without committing a meaning."""
        from .experience_investigation import execute
        return execute(self, proposal_id, call=call)

    def propose_empirical_plan(self, model, observation_source_id: str, calls: Sequence[Call], goal_state, **bounds):
        """Plan contingent paths through supported observed transitions."""
        from .empirical_execution import propose
        return propose(self, model, observation_source_id, calls, goal_state, **bounds)

    def execute_empirical_plan(self, proposal_id: str, *, call: Call | None = None, execution_guard=None):
        """Execute one best first step; further steps require fresh planning."""
        from .empirical_execution import execute
        return execute(self, proposal_id, call=call, execution_guard=execution_guard)

    def pursue_empirical(self, model, goal=None, *, task_id=None, source="empirical",
                         max_steps=1, choose=None, dependencies=(), **bounds):
        """Advance a revision-bound empirical task, replanning after each observed step."""
        from .empirical_tasks import pursue
        return pursue(self, model, goal, task_id=task_id, source=source,
                      max_steps=max_steps, choose=choose, dependencies=dependencies, **bounds)

    def capture_task_dependency(self, group_id: str, *, basis, evidence_ids=()):
        """Bind an explicitly justified task to the exact selected interpretation."""
        from .task_dependencies import capture_dependency
        return capture_dependency(self.interpretations, group_id, basis=basis, evidence_ids=evidence_ids)

    def _invoke(self, plugin: Plugin, cap: Capability, args: Mapping[str, Any], events: list[dict],
                *, observers: Sequence[Plugin] = (),
                before_dispatch: Callable[[tuple[str, ...]], bool | Unknown] | None = None) -> Receipt:
        self._calls += 1
        act = Call(plugin.name, cap.name, tuple(sorted(args.items(), key=lambda kv: kv[0])))
        attempt_id = uuid4().hex
        providers = (*self.plugins, plugin, *observers)
        before_sources = self._capture_observations(
            events, stage="before_action", providers=providers, action=act,
            attempt_id=attempt_id, retain_unavailable=True)

        def finish(receipt: Receipt) -> Receipt:
            self._capture_observations(
                events, stage="after_action", providers=providers, action=act,
                receipt=receipt, attempt_id=attempt_id, retain_unavailable=True)
            events.append({"type": "receipt", "capability": cap.name,
                           "status": receipt.status, "error": receipt.error,
                           "attempt_id": attempt_id})
            return receipt

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
                return finish(receipt)
        if before_dispatch is not None:
            try:
                permitted = before_dispatch(before_sources)
            except Exception as exc:
                permitted = Unknown("dispatch_guard_error", f"{type(exc).__name__}: {exc}")
            if type(permitted) is not bool and not isinstance(permitted, Unknown):
                permitted = Unknown("invalid_dispatch_guard", "guard returned neither bool nor Unknown")
            reason = permitted.reason if isinstance(permitted, Unknown) else ""
            detail = permitted.detail if isinstance(permitted, Unknown) else ""
            events.append({"type": "dispatch_guard", "attempt_id": attempt_id,
                           "source_ids": before_sources, "allowed": permitted is True,
                           "reason": reason, "detail": detail})
            if permitted is not True:
                return finish(Receipt(act, "rejected", error="dispatch guard declined" +
                                      (f": {reason}: {detail}" if reason else "")))
        events.append({"type": "act", "plugin": plugin.name, "capability": cap.name, "args": {k: str(v) for k, v in args.items()}, "attempt_id": attempt_id})
        receipt = invoke(act, executor=plugin, key=f"{plugin.name}:{self._calls}")
        return finish(receipt)


def is_specified(cond: Condition) -> bool:
    """Every role of the condition (other than the doer) was filled by what the speaker said."""
    roles = {r: v for r, v in cond.args.items() if verbnet.role_class(r) != "actor"}
    return bool(roles) and all(v is not None and v != "addressee" for v in roles.values())


def noun_of(filler: Any) -> str | None:
    """The word to look kinds up by: the head noun, or the name itself ("Tuesday")."""
    if isinstance(filler, Entity):
        return filler.features.get("noun") or (filler.text if filler.kind in ("description", "name") else None)
    return None
