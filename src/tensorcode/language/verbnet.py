"""What a verb does to the world, from VerbNet: requests become goal states, not intents.

A parsed imperative is a frame: ``make(object=folder{name=recipes, location=desktop})``.
What the speaker wants is the state the verb brings about, and VerbNet (Kipper-Schuler
et al., a public, curated lexicon of about 330 English verb classes) writes that state
down for every class: *create* ends with the Result ``be``-ing, *destroy* with the
Patient ``destroyed``, *put* with the Theme ``has_location`` at the Destination.

So a request is read into :class:`Goal` conditions over thematic roles, and a plugin
advertises what its actions achieve in the same predicates. Matching the two is
planning over effects, not a table from verbs (or keywords) to actions. A verb the
lexicon has never seen yields :class:`Unknown`, not a guess.

The data is not shipped. It is read from ``$TENSORCODE_VERBNET`` or
``~/.cache/tensorcode/verbnet/verbnet3.4`` (a checkout of github.com/cu-clear/verbnet);
without it, :func:`load` returns an empty lexicon and every request is ``Unknown``.
The only thing written by hand here is how this grammar's role names line up with
VerbNet's thematic roles (:data:`ROLE_OF_PREPOSITION_ROLE`), which is a correspondence
between two role inventories, not knowledge of any verb.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

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
class Condition:
    """One state the speaker wants to hold: ``pred(role=value, ...)``, possibly negated."""

    pred: str
    args: Mapping[str, Any]
    negated: bool = False

    def describe(self) -> str:
        inner = ", ".join(f"{k}={getattr(v, 'text', v)}" for k, v in self.args.items())
        return ("not " if self.negated else "") + f"{self.pred}({inner})"


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


@lru_cache(maxsize=2)
def sense_counts(root: Path | None = None) -> Mapping[str, float]:
    """WordNet ``cntlist.rev``: how often each verb sense was used in hand-tagged text."""
    from .wordnet import _Files, find_wordnet

    root = root or find_wordnet()
    if root is None:
        return {}
    out: dict[str, float] = {}
    for line in _Files(root).text("cntlist.rev").splitlines():
        parts = line.split()
        if len(parts) >= 3 and "%2:" in parts[0]:
            key = parts[0].split("::")[0].rstrip(":")
            out[key] = out.get(key, 0.0) + float(parts[2])
    return out


def class_prior(verb: str, vc: VerbClass, counts: Mapping[str, float]) -> float:
    """log P(class | verb) up to a constant: tagged uses of the WordNet senses VerbNet lists."""
    import math

    keys = vc.senses.get(verb, ())
    return math.log(1.0 + sum(counts.get(k, 0.0) for k in keys))


def goal_of(frame: Frame, lexicon: Mapping[str, tuple[VerbClass, ...]] | None = None) -> Goal | Unknown:
    """The end state a request for ``frame`` asks for, per the verb's VerbNet class.

    Picks, among the verb's classes and frames, the frame whose slots cover the most
    of the roles the utterance actually filled; then one whose complements after the
    verb are exactly the ones said (``make NP`` is not ``make NP ADJ``); then the class
    whose WordNet senses are used most in tagged text (:func:`class_prior`). Maps
    grammar roles to thematic roles through the frame's syntax and returns the frame's
    result-state predicates with those fillers.
    """
    lexicon = load() if lexicon is None else lexicon
    verb = frame.predicate.lower()
    classes = lexicon.get(verb)
    if not classes:
        return Unknown("unknown_verb", f"VerbNet has no class for '{verb}'")
    filled = {k: v for k, v in frame.roles.items() if k not in ("subject",)}
    counts = sense_counts()
    best: tuple[tuple, VerbClass, VFrame, dict, tuple] | None = None
    for ci, vc in enumerate(classes):
        for fi, vf in enumerate(vc.frames):
            state = vf.result_state()
            if not state:
                continue
            binding: dict[str, Any] = {}
            unmapped = []
            for role, value in filled.items():
                if role == "object" and vf.object_role:
                    binding[vf.object_role] = value
                    continue
                wanted = ROLE_OF_PREPOSITION_ROLE.get(role, frozenset())
                target = next((r for r in vf.pp_roles() if r in wanted and r not in binding), None)
                if target:
                    binding[target] = value
                else:
                    unmapped.append(role)
            said = (("NP",) if "object" in filled else ()) + tuple("PP" for r in filled if r != "object")
            slots = vf.after_verb()
            # the frame should have exactly the complements that were said: a frame with
            # extra required slots ("make X Y" = render) is a different construction
            extra = max(0, len(slots) - len(said))
            fits = sorted(slots) == sorted(said)
            score = (-len(unmapped), fits, -extra, round(class_prior(verb, vc, counts), 6), -ci, -fi)
            if best is None or score > best[0]:
                best = (score, vc, vf, binding, tuple(unmapped))
    if best is None:
        return Unknown("no_result_state", f"no class of '{verb}' says what state it leaves")
    _, vc, vf, binding, unmapped = best
    agent = vf.subject_role
    conditions = []
    for p in vf.result_state():
        args = {}
        for r in p.roles:
            if r in binding:
                args[r] = binding[r]
            elif r == agent:
                args[r] = "addressee"
            else:
                args[r] = None  # a role the utterance left open
        for r in p.implicit:
            args.setdefault(r, binding.get(r))  # involved, unsaid: open unless the utterance filled it
        conditions.append(Condition(p.name, args, p.negated))
    return Goal(verb, vc.id, tuple(conditions), frame, unmapped)
