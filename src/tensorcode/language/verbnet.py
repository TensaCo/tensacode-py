"""Resource-derived goal proposals with explicit unresolved lexical alternatives.

VerbNet supplies curated class frames and result predicates. The authored adapter
maps input roles to thematic roles and extracts those predicates; it does not
infer the speaker's intended class or authorize an action. Goal proposals preserve
competing classes, bindings, original frame evidence and projection obligations.
An agent must explicitly select a retained proposal before goal execution.

The loader currently omits syntax/selection restrictions, while input Frames lack
full complement occurrence order and literal preposition evidence. Slot mismatch
is therefore an unresolved obligation, not a reliable hard exclusion. Role
correspondence and result-state extraction remain authored projection conventions.
Exact duplicate grouping establishes structural equality, not semantic equivalence.

The resource is not shipped. It is read from ``$TENSORCODE_VERBNET`` or
``~/.cache/tensorcode/verbnet/verbnet3.4``. Without it, proposals retain explicit
unknown-verb status. Structured callers can supply ``goals.GoalSpec`` directly.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from copy import deepcopy
from collections import deque
from dataclasses import dataclass, field, fields, is_dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from ..goals import Condition
from ..outcomes import Unknown
from .semantics import Frame

#: Predicates that describe the *doing*, not the state it leaves: never a goal.
PROCESS_PREDICATES = frozenset({"do", "cause", "motion", "body_motion", "exert_force", "contact", "manner", "path_rel",
                                "utilize", "take_in", "emit", "rotational_motion", "apply_heat", "apply_material"})

#: The grammar's role for a prepositional phrase (``english.py`` gives each preposition
#: the role it marks) against the VerbNet thematic roles that fill that slot.
ROLE_OF_PREPOSITION_ROLE: Mapping[str, frozenset[str]] = {
    "destination": frozenset({"Destination", "Goal", "Recipient", "Beneficiary"}),
    "location": frozenset({"Location", "Destination", "Goal"}),
    "source": frozenset({"Source", "Initial_Location", "Material"}),
    "recipient": frozenset({"Recipient", "Beneficiary", "Goal"}),
    "instrument": frozenset({"Instrument", "Co-Agent", "Co-Theme"}),
    "content": frozenset({"Topic", "Theme", "Stimulus", "Proposition"}),
    "topic": frozenset({"Topic", "Theme", "Stimulus"}),
}


#: VerbNet's thematic roles, grouped by the part they play in a result state. Classes
#: differ in what they call the thing brought about (Result in *create*, Product in
#: *build*, Patient in *engender*) or the place something ends up (Destination, Goal,
#: Location); a plugin states its effects over these groups so it does not have to
#: know every class's naming. Like the table above, this is about VerbNet's role
#: inventory, not about any verb.
ROLE_CLASS: Mapping[str, str] = {
    **{r: "undergoer" for r in ("Theme", "Patient", "Result", "Product", "Stimulus", "Topic", "Material",
                                  "Eventuality", "Co-Theme", "Co-Patient", "Attribute", "Value")},
    **{r: "goal" for r in ("Destination", "Goal", "Location", "Recipient", "Beneficiary")},
    **{r: "source" for r in ("Source", "Initial_Location", "Initial_State")},
    **{r: "actor" for r in ("Agent", "Experiencer", "Co-Agent", "Causer", "Pivot")},
    **{r: "instrument" for r in ("Instrument",)},
}


def role_class(role: str) -> str:
    return ROLE_CLASS.get(role, role.lower())


@dataclass(frozen=True)
class Pred:
    name: str
    args: tuple[tuple[str, str], ...]  # (arg type, value): ("ThemRole", "Result"), ("Event", "e3"), ...
    negated: bool = False

    @property
    def event(self) -> str | None:
        return next((v for t, v in self.args if t == "Event"), None)

    @property
    def roles(self) -> tuple[str, ...]:
        """Thematic roles the predicate relates, named as the frame's syntax names them."""
        return tuple(v.strip().lstrip("?") for t, v in self.args if t == "ThemRole")

    @property
    def implicit(self) -> tuple[str, ...]:
        """Arguments VerbNet marks as not expressed in this frame (``?Role``, or a
        predicate-specific argument like the Destination of a removal): the result
        involves them, but the sentence does not say what they are."""
        out = [v.strip()[1:] for t, v in self.args if t == "ThemRole" and v.strip().startswith("?")]
        out += [v.strip() for t, v in self.args if t in ("PredSpecific", "VerbSpecific") and v.strip()[:1].isupper()]
        return tuple(dict.fromkeys(out))


@dataclass(frozen=True)
class VFrame:
    primary: str                      # e.g. "NP V NP PP.destination"
    syntax: tuple[tuple[str, str], ...]  # (category, thematic role or preposition value)
    semantics: tuple[Pred, ...]

    @property
    def object_role(self) -> str | None:
        """The thematic role of the first noun phrase after the verb."""
        seen_verb = False
        for cat, value in self.syntax:
            if cat == "VERB":
                seen_verb = True
            elif seen_verb and cat == "NP" and value:
                return value
            elif seen_verb and cat == "PREP":
                return None
        return None

    @property
    def subject_role(self) -> str | None:
        """The role of the noun phrase directly before the verb (the doer, in an active frame)."""
        last = None
        for cat, value in self.syntax:
            if cat == "VERB":
                return last
            if cat == "NP" and value:
                last = value
        return None

    def after_verb(self) -> tuple[str, ...]:
        """The frame's complement slots after the verb: ``NP``, ``PP`` (a preposition and
        its noun phrase count once), ``ADJ``, ``S``... — what a request must fill to fit."""
        out, seen_verb, i = [], False, 0
        items = list(self.syntax)
        while i < len(items):
            cat, _ = items[i]
            if cat == "VERB":
                seen_verb = True
            elif seen_verb:
                if cat == "PREP":
                    out.append("PP")
                    if i + 1 < len(items) and items[i + 1][0] == "NP":
                        i += 1
                elif cat in ("NP", "ADJ", "ADV", "S", "S_INF", "S_ING"):
                    out.append(cat)
            i += 1
        return tuple(out)

    def pp_roles(self) -> tuple[str, ...]:
        """Thematic roles reached through a preposition in this frame."""
        out, after_prep = [], False
        for cat, value in self.syntax:
            if cat == "PREP":
                after_prep = True
            elif cat == "NP" and after_prep and value:
                out.append(value)
                after_prep = False
        return tuple(out)

    def result_state(self) -> tuple[Pred, ...]:
        """What holds once the event is over.

        VerbNet writes an event as a pre-state (the first event), the doing, and one or
        more later events. Every non-process predicate on an event after the first is
        part of the result — "the Theme is no longer at its initial location" and "the
        Theme is at the destination" both — and so is the negation of a pre-state
        predicate that a later event does not restate.
        """
        events = [p.event for p in self.semantics if p.event and not p.event.startswith("E")]
        if not events:
            return ()
        order = sorted(set(events), key=lambda e: (len(e), e.replace("ë", "e")))
        first = order[0]
        later = [p for p in self.semantics if p.event and p.event != first and not p.event.startswith("E")
                 and p.name not in PROCESS_PREDICATES]
        return tuple(dict.fromkeys(later))


@dataclass(frozen=True)
class VerbClass:
    id: str
    members: tuple[str, ...]
    frames: tuple[VFrame, ...]
    senses: Mapping[str, tuple[str, ...]] = field(default_factory=dict)  # member -> WordNet sense keys


@dataclass(frozen=True)
class Goal:
    verb: str
    verb_class: str
    conditions: tuple[Condition, ...]
    frame: Frame
    unmapped_roles: tuple[str, ...] = field(default=())  # grammar roles no thematic role took

    def describe(self) -> str:
        return f"{self.verb} [{self.verb_class}]: " + "; ".join(c.describe() for c in self.conditions)


def find_verbnet() -> Path | None:
    for c in (os.environ.get("TENSORCODE_VERBNET"), "~/.cache/tensorcode/verbnet/verbnet3.4"):
        if c and Path(c).expanduser().is_dir():
            return Path(c).expanduser()
    return None


def _frames(node: ET.Element) -> list[VFrame]:
    out = []
    for fr in node.findall("FRAMES/FRAME"):
        desc = fr.find("DESCRIPTION")
        syntax = []
        for el in fr.find("SYNTAX") if fr.find("SYNTAX") is not None else ():
            value = el.get("value", "") or ""
            syntax.append((el.tag, value.strip()))
        sems = []
        for p in fr.findall("SEMANTICS/PRED"):
            args = tuple((a.get("type", ""), (a.get("value", "") or "").strip()) for a in p.findall("ARGS/ARG"))
            sems.append(Pred(p.get("value", ""), args, p.get("bool") == "!"))
        out.append(VFrame(desc.get("primary", "") if desc is not None else "", tuple(syntax), tuple(sems)))
    return out


def _classes(node: ET.Element, inherited: list[VFrame]) -> list[VerbClass]:
    frames = inherited + _frames(node)
    members = tuple(m.get("name", "") for m in node.findall("MEMBERS/MEMBER"))
    senses = {m.get("name", ""): tuple((m.get("wn", "") or "").split()) for m in node.findall("MEMBERS/MEMBER")}
    out = [VerbClass(node.get("ID", ""), members, tuple(frames), senses)]
    for sub in node.findall("SUBCLASSES/VNSUBCLASS"):
        out.extend(_classes(sub, frames))
    return out


@lru_cache(maxsize=4)
def load(root: Path | None = None) -> Mapping[str, tuple[VerbClass, ...]]:
    """lemma -> the VerbNet classes it belongs to."""
    root = root or find_verbnet()
    if root is None:
        return {}
    by_lemma: dict[str, list[VerbClass]] = {}
    for f in sorted(root.glob("*.xml")):
        try:
            tree = ET.parse(f)
        except ET.ParseError:
            continue
        for vc in _classes(tree.getroot(), []):
            for m in vc.members:
                by_lemma.setdefault(m.replace("_", " ").lower(), []).append(vc)
    return {k: tuple(v) for k, v in by_lemma.items()}


@dataclass(frozen=True)
class GoalDerivation:
    verb_class: str
    frame_index: int
    syntax: tuple[tuple[str, str], ...]
    bindings: tuple[tuple[str, str], ...]
    obligations: tuple[str, ...] = ()


@dataclass(frozen=True)
class GoalProposal:
    goal: Goal
    derivations: tuple[GoalDerivation, ...]


@dataclass(frozen=True)
class GoalSearchUnresolved:
    reason: str
    verb_class: str = ""
    frame_index: int | None = None


@dataclass(frozen=True)
class GoalCandidates:
    proposals: tuple[GoalProposal, ...]
    unresolved: tuple[GoalSearchUnresolved, ...] = ()
    complete: bool = True


def _exact(left, right):
    """Typed structural identity, not thematic or semantic equivalence."""
    if type(left) is not type(right):
        return False
    if is_dataclass(left):
        return all(_exact(getattr(left, f.name), getattr(right, f.name)) for f in fields(left))
    if isinstance(left, Mapping):
        return len(left) == len(right) and all(any(_exact(k, rk) and _exact(v, rv)
            for rk, rv in right.items()) for k, v in left.items())
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_exact(a, b) for a, b in zip(left, right))
    if isinstance(left, (set, frozenset)):
        return len(left) == len(right) and all(any(_exact(a, b) for b in right) for a in left)
    try:
        result = left == right
        return type(result) is bool and result
    except Exception:
        return False


def goal_candidates(frame: Frame, lexicon: Mapping[str, tuple[VerbClass, ...]] | None = None,
                    *, max_derivations: int = 256) -> GoalCandidates:
    """Retain resource-derived goal alternatives without authorizing one.

    Every result-bearing resource frame participates. Role correspondence is an
    authored adapter; compatible PP assignments are enumerated rather than chosen
    by occurrence order. Only maximal injective assignments are emitted: an input
    is left unmapped when all compatible slots are occupied or unavailable. This
    does not prefer the largest matching over other maximal matchings.

    ``max_derivations`` bounds explored binding-search states, including partial
    assignments. Exhaustion is explicit and supplies no resumable cursor. Slot
    mismatches remain obligations because the loaded syntax and input frame do
    not preserve enough construction evidence for reliable hard exclusions.
    """
    if not isinstance(frame, Frame):
        raise TypeError("goal candidates require a Frame")
    if type(max_derivations) is not int or max_derivations < 0:
        raise ValueError("max_derivations must be a nonnegative integer")
    original = deepcopy(frame)
    lexicon = load() if lexicon is None else lexicon
    verb = original.predicate.lower()
    classes = lexicon.get(verb, ())
    if not classes:
        return GoalCandidates((), (GoalSearchUnresolved("unknown_verb"),))
    filled = {key: value for key, value in original.roles.items() if key != "subject"}
    roles = tuple(sorted(filled))
    frontier = deque()
    for vc in classes:
        for index, vf in enumerate(vc.frames):
            if vf.result_state():
                options = tuple((vf.object_role,) if role == "object" and vf.object_role else
                    tuple(dict.fromkeys(target for target in vf.pp_roles()
                          if target in ROLE_OF_PREPOSITION_ROLE.get(role, frozenset())))
                    for role in roles)
                frontier.append((vc.id, index, vf, options, ()))
    if not frontier:
        return GoalCandidates((), (GoalSearchUnresolved("no_result_state"),))
    proposals = []
    explored = 0
    while frontier and explored < max_derivations:
        class_id, index, vf, options, assigned = frontier.popleft()
        explored += 1
        if len(assigned) < len(roles):
            targets = tuple(target for target in options[len(assigned)] if target not in assigned)
            for target in (*targets, None):
                frontier.append((class_id, index, vf, options, (*assigned, target)))
            continue
        # Drop only extendable partial matchings; every maximal conflicting
        # assignment survives even when another matching covers more input roles.
        if any(target is None and any(option not in assigned for option in choices)
               for target, choices in zip(assigned, options)):
            continue
        binding = {target: filled[role] for role, target in zip(roles, assigned) if target is not None}
        unmapped = tuple(role for role, target in zip(roles, assigned) if target is None)
        conditions = []
        for predicate in vf.result_state():
            args = {role: binding[role] if role in binding else
                    "addressee" if role == vf.subject_role else None for role in predicate.roles}
            for role in predicate.implicit:
                args.setdefault(role, binding.get(role))
            conditions.append(Condition(predicate.name, args, predicate.negated))
        goal = Goal(verb, class_id, tuple(conditions), deepcopy(original), unmapped)
        said = (("NP",) if "object" in filled else ()) + tuple("PP" for role in roles if role != "object")
        obligations = tuple("unmapped_input_role:" + role for role in unmapped)
        if sorted(vf.after_verb()) != sorted(said):
            obligations += ("construction_slots_unresolved",)
        derivation = GoalDerivation(class_id, index, vf.syntax,
            tuple((role, target) for role, target in zip(roles, assigned) if target is not None), obligations)
        duplicate = next((i for i, proposal in enumerate(proposals)
            if _exact(proposal.goal.conditions, goal.conditions) and _exact(proposal.goal.frame, goal.frame)
            and _exact(proposal.goal.unmapped_roles, goal.unmapped_roles)), None)
        if duplicate is None:
            proposals.append(GoalProposal(goal, (derivation,)))
        else:
            existing = proposals[duplicate]
            derivations = (*existing.derivations, derivation)
            classes = sorted({item.verb_class for item in derivations})
            # Equivalent conditions do not establish which lexical class applies.
            # Never let source order masquerade as a selected class in the Goal.
            label = classes[0] if len(classes) == 1 else "alternatives:" + "|".join(classes)
            proposals[duplicate] = GoalProposal(replace(existing.goal, verb_class=label), derivations)
    unresolved = tuple(GoalSearchUnresolved("derivation_budget_exhausted", class_id, index)
                       for class_id, index in dict.fromkeys((item[0], item[1]) for item in frontier))
    return GoalCandidates(tuple(proposals), unresolved, not frontier)


def goal_of(frame: Frame, lexicon: Mapping[str, tuple[VerbClass, ...]] | None = None) -> Goal | Unknown:
    """Return a unique fully enumerated goal, never a lexical-prior winner.

    This utility is not an interpretation selector. Agent execution must retain
    and explicitly select goal proposals, including singleton proposal sets.
    """
    batch = goal_candidates(frame, lexicon)
    if not batch.complete:
        return Unknown("incomplete_goal_search", "lexical derivation budget exhausted")
    if len(batch.proposals) == 1:
        proposal = batch.proposals[0]
        if proposal.goal.unmapped_roles or not any(not derivation.obligations for derivation in proposal.derivations):
            return Unknown("unresolved_goal_projection", "the sole goal retains unresolved projection obligations")
        return proposal.goal
    if batch.proposals:
        return Unknown("ambiguous_goal", f"{len(batch.proposals)} distinct lexical goal proposals")
    return Unknown(batch.unresolved[0].reason if batch.unresolved else "no_result_state")
