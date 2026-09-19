"""What an utterance means, and when that meaning may become a claim.

Two value types carry meaning: an :class:`Entity` (what a referring expression
picks out) and a :class:`Frame` (a predication over entities and other frames).
Frames carry the features that decide whether the sentence asserts anything at
all: mood, polarity, modality, tense.

The conversion to ``tensorcode`` claims is deliberately conservative, because the
cheap version of it is how a reader ends up believing the opposite of what it
read:

* an imperative or a question asserts nothing, and converts to ``Unknown``;
* a negated, modal or non-past-or-present frame is **reified** — the event gets
  a ``Ref`` and the polarity/modality become claims about it — rather than
  flattened into a triple that would state the bare proposition;
* reported speech becomes claims in a scope of their own, sourced to the
  speaker, so "Anem said the field failed" never enters the shared world as
  "the field failed";
* an unresolved reference converts to ``Unknown``, never to a guess.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import cached_property
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping

from ..outcomes import Score, Unknown
from ..records import Claim, Evidence, Proposition, Ref

MOODS = ("declarative", "interrogative", "imperative")

#: WordNet's categories for verbs that take a *reported* argument: what is said, thought
#: or wanted. A clause inside one of these is not thereby a fact about the world.
REPORTING_DOMAINS = frozenset({"verb.communication", "verb.cognition", "verb.emotion"})

#: Predicates whose core participants are interchangeable. The copula states an identity:
#: "my name is Jacob" and "Jacob is my name" are the same proposition, and English inverts
#: the clause to question it ("*what* is my name"), so the phrase that was the subject
#: comes back as the object. Which side a filler landed on therefore cannot be part of
#: matching for these; for every other predicate it must be.
SYMMETRIC_PREDICATES = frozenset({"be"})


def _key_of(value: Any) -> Any:
    """Content identity for a semantic value, without building any strings."""
    key = getattr(value, "key", None)
    if key is not None and isinstance(value, (Entity, Frame)):
        return key
    if isinstance(value, tuple):
        return tuple(_key_of(v) for v in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Ref):
        return value.id
    if isinstance(value, (Request, Question)):
        return value.key
    return str(value)


@dataclass(frozen=True)
class Entity:
    """What a referring expression picks out.

    ``kind`` is how it was named, not what it is: ``name``, ``pronoun``,
    ``path``, ``literal``, ``number``, ``quantified`` or ``skolem`` (introduced
    by the predication itself and given a minted identity).
    """

    kind: str
    text: str
    features: Mapping[str, Any] = field(default_factory=dict)
    ref: Ref | None = None  # set once discourse resolution has settled it
    candidates: tuple["Entity", ...] = ()  # kept when two antecedents are equally good

    @cached_property
    def key(self) -> tuple:
        """A cheap, stable content identity, computed once.

        The chart dedupes distinct meanings per span, which needs content identity for
        thousands of values per parse. ``repr`` was doing that job and cost 520k calls
        on one benchmark run; this is the same distinction without building strings.
        """
        feats = tuple(sorted((k, _key_of(v)) for k, v in self.features.items()))
        return ("e", self.kind, self.text, feats, self.ref.id if self.ref else None,
                tuple(c.key for c in self.candidates))

    def __hash__(self) -> int:
        return hash(self.key)

    @property
    def resolved(self) -> bool:
        return not self.candidates and (self.kind != "pronoun" or self.ref is not None)

    def with_ref(self, ref: Ref) -> "Entity":
        return Entity(self.kind, self.text, self.features, ref, ())


@dataclass(frozen=True)
class Frame:
    """A predication: ``predicate`` over ``roles``, qualified by ``features``."""

    predicate: str
    roles: Mapping[str, Any] = field(default_factory=dict)
    features: Mapping[str, Any] = field(default_factory=dict)

    @cached_property
    def key(self) -> tuple:
        """A cheap, stable content identity, computed once (see :meth:`Entity.key`)."""
        roles = tuple(sorted((k, _key_of(v)) for k, v in self.roles.items()))
        feats = tuple(sorted((k, _key_of(v)) for k, v in self.features.items()))
        return ("f", self.predicate, roles, feats)

    def __hash__(self) -> int:
        return hash(self.key)

    def role(self, name: str, default: Any = None) -> Any:
        return self.roles.get(name, default)

    def feature(self, name: str, default: Any = None) -> Any:
        return self.features.get(name, default)

    @property
    def mood(self) -> str:
        return self.features.get("mood", "declarative")

    @property
    def negated(self) -> bool:
        return self.features.get("polarity") == "negative"

    def added(self, **features: Any) -> "Frame":
        return Frame(self.predicate, self.roles, {**self.features, **features})

    def filled(self, **roles: Any) -> "Frame":
        return Frame(self.predicate, {**self.roles, **roles}, self.features)

    def walk(self) -> Iterable["Frame"]:
        yield self
        for value in self.roles.values():
            if isinstance(value, Frame):
                yield from value.walk()
            elif isinstance(value, tuple):
                for item in value:
                    if isinstance(item, Frame):
                        yield from item.walk()

    def entities(self) -> Iterable[Entity]:
        """Every entity in the frame, including the ones nested inside another's features.

        PP attachment decides *where* a modifier lands, not whether it was said: "make
        a folder in it" may hang "it" on the verb or inside the noun. A caller asking
        what this frame refers to wants the same answer either way, so the walk
        descends into features too.
        """
        for frame in self.walk():
            for value in frame.roles.values():
                yield from _entities_in(value)

    def describe(self) -> str:
        parts = [f"{k}={_short(v)}" for k, v in sorted(self.roles.items())]
        flags = [f"{k}={v}" for k, v in sorted(self.features.items()) if k != "mood"]
        head = f"{self.predicate}({', '.join(parts)})"
        return head + (f" [{', '.join(flags)}]" if flags else "")


def _short(value: Any) -> str:
    if isinstance(value, Entity):
        return value.ref.id if value.ref else f"{value.kind}:{value.text}"
    if isinstance(value, Frame):
        return "{" + value.describe() + "}"
    if isinstance(value, tuple):
        return "[" + ", ".join(_short(v) for v in value) + "]"
    return repr(value)


def _entities_in(value: Any, depth: int = 0) -> Iterable[Entity]:
    if depth > 8:  # a guard, not a limit: nothing this deep is a reference anyone means
        return
    if isinstance(value, Entity):
        yield value
        for inner in value.features.values():
            yield from _entities_in(inner, depth + 1)
    elif isinstance(value, Frame):
        yield from value.entities()
    elif isinstance(value, tuple):
        for item in value:
            yield from _entities_in(item, depth + 1)


# --------------------------------------------------------------------- claims

ASSERTIVE_FEATURES = {"mood", "tense", "person", "number"}


def _mint(frame: Frame, prefix: str = "event") -> Ref:
    """Content identity for a reified event, so the same reading is the same event."""
    payload = json.dumps(_canonical(frame), sort_keys=True, separators=(",", ":"))
    return Ref(f"{prefix}:{hashlib.sha256(payload.encode()).hexdigest()[:12]}")


def _canonical(value: Any) -> Any:
    if isinstance(value, Frame):
        return {"p": value.predicate, "r": {k: _canonical(v) for k, v in sorted(value.roles.items())},
                "f": {k: _canonical(v) for k, v in sorted(value.features.items()) if k != "mood"}}
    if isinstance(value, Entity):
        return value.ref.id if value.ref else f"{value.kind}:{value.text}"
    if isinstance(value, (tuple, list)):
        return [_canonical(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    return value


def default_ref(entity: Entity) -> Ref | Any:
    """Namespace an entity by how it was named; callers usually override this."""
    if entity.ref is not None:
        return entity.ref
    if entity.kind in ("number", "literal"):
        return entity.text
    prefix = {"name": "entity", "path": "path", "skolem": "thing", "quantified": "set"}.get(entity.kind, "entity")
    return Ref(f"{prefix}:{entity.text}")


def explicit_ref(entity: Entity) -> Ref | Any:
    """Resolve only explicitly supplied identity or an explicit literal value.

    Descriptions, names, paths, and pronouns do not establish world identity.
    This resolver is used by the active agent; ``default_ref`` remains a legacy
    converter default pending migration of library callers.
    """
    if entity.ref is not None:
        return entity.ref if isinstance(entity.ref, Ref) else Unknown("invalid_reference", "identity must be an explicit Ref")
    if entity.kind in ("number", "literal"):
        value = entity.features.get("value", entity.text)
        if isinstance(value, (str, int, float, bool)):
            return value
        return Unknown("unresolved_literal", "literal value is not an explicit scalar")
    return Unknown("unresolved_reference", f"no explicit identity for {entity.text!r}")


def to_claims(
    frame: Frame,
    *,
    source: Ref,
    observed_at: datetime | None = None,
    method: str = "grammar",
    confidence: Score | None = None,
    resolve: Callable[[Entity], Any] = default_ref,
    subject_roles: tuple[str, ...] = ("subject", "agent", "theme"),
    object_roles: tuple[str, ...] = ("object", "patient", "value", "complement"),
    scope: Ref | None = None,
) -> list[tuple[Claim, Evidence]] | Unknown:
    """Claims for one frame, with the evidence that records where they came from.

    Returns ``Unknown`` when the frame asserts nothing (a request or a question)
    or when a reference did not resolve — never a guess.
    """
    at = observed_at or datetime.now(timezone.utc)
    if frame.mood != "declarative":
        return Unknown("not_an_assertion", f"{frame.mood} frames carry no claims")
    for entity in frame.entities():
        if not entity.resolved:
            detail = ", ".join(c.text for c in entity.candidates) or entity.text
            return Unknown("unresolved_reference", f"“{entity.text}” could be {detail}")
    out: list[tuple[Claim, Evidence]] = []
    _emit(frame, out, source=source, at=at, method=method, confidence=confidence, resolve=resolve,
          subject_roles=subject_roles, object_roles=object_roles, scope=scope)
    return out


def _emit(frame: Frame, out: list, *, source: Ref, at: datetime, method: str, confidence: Score | None,
          resolve: Callable[[Entity], Any], subject_roles, object_roles, scope: Ref | None) -> Any:
    evidence = Evidence(source=source, observed_at=at, method=method, confidence=confidence)

    def say(claim: Claim) -> None:
        out.append((claim, evidence))

    # reported speech: the content becomes its own scope, sourced to the speaker
    content = frame.role("content")
    if isinstance(content, Frame) and frame.predicate in ("say", "ask", "think", "believe", "promise", "claim"):
        speaker = frame.role("speaker") or frame.role("subject")
        speaker_ref = resolve(speaker) if isinstance(speaker, Entity) else speaker
        said = _mint(frame, "utterance")
        inner_scope = Ref(f"scope:{said.id.split(':', 1)[1]}")
        say(Claim(_as_ref(speaker_ref), frame.predicate, said, scope=scope))
        say(Claim(said, "in_scope", inner_scope, scope=scope))
        inner_source = _as_ref(speaker_ref) if isinstance(speaker_ref, Ref) else source
        nested = _emit(content, out, source=inner_source, at=at, method=f"{method}:reported", confidence=confidence,
                       resolve=resolve, subject_roles=subject_roles, object_roles=object_roles, scope=inner_scope)
        say(Claim(said, "content", nested, scope=scope))
        return said

    simple = (not frame.negated
              and not frame.features.get("modality")
              and not frame.features.get("degree")
              and all(k in ASSERTIVE_FEATURES for k in frame.features))
    subject = next((frame.role(r) for r in subject_roles if frame.role(r) is not None), None)
    obj = next((frame.role(r) for r in object_roles if frame.role(r) is not None), None)
    others = {k: v for k, v in frame.roles.items() if k not in subject_roles + object_roles}

    if simple and subject is not None and obj is not None and not others:
        subject_ref = _as_ref(resolve(subject) if isinstance(subject, Entity) else subject)
        value = _value(obj, out, source=source, at=at, method=method, confidence=confidence, resolve=resolve,
                       subject_roles=subject_roles, object_roles=object_roles, scope=scope)
        say(Claim(subject_ref, frame.predicate, value, scope=scope))
        return subject_ref

    # anything qualified (negated, modal, comparative, extra roles) is reified:
    # the event gets an identity and its qualifications are claims about it
    event = _mint(frame)
    say(Claim(event, "is_a", frame.predicate, scope=scope))
    for role, value in sorted(frame.roles.items()):
        filled = _value(value, out, source=source, at=at, method=method, confidence=confidence, resolve=resolve,
                        subject_roles=subject_roles, object_roles=object_roles, scope=scope)
        say(Claim(event, role, filled, scope=scope))
    for key, value in sorted(frame.features.items()):
        if key != "mood":
            say(Claim(event, key, value, scope=scope))
    return event


def _value(value: Any, out: list, **kw: Any) -> Any:
    if isinstance(value, Frame):
        return _emit(value, out, **kw)
    if isinstance(value, Entity):
        resolved = kw["resolve"](value)
        return resolved
    if isinstance(value, tuple):
        return tuple(_value(v, out, **kw) for v in value)
    return value


def _as_ref(value: Any) -> Ref:
    return value if isinstance(value, Ref) else Ref(f"entity:{value}")


# ------------------------------------------------------------------- requests


@dataclass(frozen=True)
class Request:
    """An imperative reading: what the speaker wants done."""

    frame: Frame

    @cached_property
    def key(self) -> tuple:
        return ("r", self.frame.key)

    def __hash__(self) -> int:
        return hash(self.key)

    @property
    def act(self) -> str:
        return self.frame.predicate

    def describe(self) -> str:
        return self.frame.describe()


@dataclass(frozen=True)
class Question:
    """An interrogative reading: which role is being asked about."""

    frame: Frame
    asked: str  # the role the wh-word or polarity question targets ("polarity" for yes/no)

    @cached_property
    def key(self) -> tuple:
        return ("q", self.asked, self.frame.key)

    def __hash__(self) -> int:
        return hash(self.key)

    def describe(self) -> str:
        return f"?{self.asked} in {self.frame.describe()}"


def to_propositions(
    frame: Frame,
    *,
    source: Ref,
    observed_at: datetime | None = None,
    method: str = "grammar",
    confidence: Score | None = None,
    resolve: Callable[[Entity], Any] = default_ref,
    scope: Ref | None = None,
) -> tuple[list[tuple[Proposition, Evidence]], list[str]]:
    """A frame as n-ary propositions, with what could not be represented listed beside them.

    One proposition per predication, roles kept as roles, and a frame inside a role stays a
    proposition inside a role ("Anem said the field failed" is ``say(content=fail(...))``).
    Nothing is reified into ``event:… subject …`` triples, so nothing downstream has to
    agree about invented nodes.

    The second return value is the **discard record**: parts of the frame this conversion
    could not carry. A converter that silently drops what it cannot represent returns
    something shaped like a full reading of the sentence (symbolic-ai-models, projections).

    Entity resolution establishes identity; it does not establish that the entity's
    features have been projected. Those features are conservatively reported with their
    source paths, including features of nested entities. A custom resolver may retain
    information elsewhere, but this interface has no preservation receipt from it.
    A resolver returning None or Unknown drops the affected clause, including its
    enclosing clause for nested content; the exact unresolved role path is retained.
    """


    at = observed_at or datetime.now(timezone.utc)
    dropped: list[str] = []

    def entity_features(value: Any, path: str) -> None:
        """Report feature attachments without inventing assertions from them.

        A relative clause may also be emitted by ``clauses``, but its attachment to
        this entity is still absent from the resolved role filler.
        """
        if isinstance(value, Entity):
            for name, feature in value.features.items():
                feature_path = f"{path}.features.{name}"
                dropped.append(f"entity feature not explicitly projected: {feature_path}")
                entity_features(feature, feature_path)
        elif isinstance(value, Frame):
            for role, inner in value.roles.items():
                entity_features(inner, f"{path}.roles.{role}")
        elif isinstance(value, Mapping):
            for key, inner in value.items():
                entity_features(inner, f"{path}[{key!r}]")
        elif isinstance(value, (list, tuple)):
            for index, inner in enumerate(value):
                entity_features(inner, f"{path}[{index}]")

    unresolved = object()

    def filler(value: Any, path: str) -> Any:
        if isinstance(value, Frame):
            return build(value, path)
        if isinstance(value, Entity):
            entity_features(value, path)
            got = resolve(value)
            if got is None or isinstance(got, Unknown):
                dropped.append(f"unresolved entity identity: {path}")
                return unresolved
            return got
        if isinstance(value, (list, tuple)):
            values = tuple(filler(v, f"{path}[{index}]") for index, v in enumerate(value))
            return unresolved if any(v is unresolved for v in values) else values
        if isinstance(value, Unknown):
            dropped.append(f"unresolved role filler: {path}")
            return unresolved
        if isinstance(value, (Ref, str, int, float, bool)) or value is None:
            return value
        dropped.append(f"role filler of type {type(value).__name__}")
        return str(value)

    def build(f: Frame, path: str) -> Any:
        roles = {role: filler(value, f"{path}.roles.{role}")
                 for role, value in f.roles.items() if role != "_conj"}
        if any(value is unresolved for value in roles.values()):
            return unresolved
        modality = "asserted"
        if f.features.get("modality"):
            modality = "possible" if f.features["modality"] in ("can", "may", "might") else "obliged" \
                if f.features["modality"] in ("should", "must") else "asserted"
        if f.mood == "interrogative":
            modality = "questioned"
        for feature in ("tense", "aspect", "degree"):
            if f.features.get(feature):
                roles.setdefault(feature, f.features[feature])
        return Proposition(f.predicate, roles, polarity=not f.negated, modality=modality, scope=scope)

    evidence = Evidence(source=source, observed_at=at, method=method, confidence=confidence)
    out: list[tuple[Proposition, Evidence]] = []
    for index, (clause, reported) in enumerate(clauses(frame)):
        if reported:
            # what someone said, thought or wanted is not thereby the case. It stays inside
            # the proposition that reports it, where a nested pattern can still find it.
            continue
        unread = _swallowed(clause)
        if unread:
            # a reading that turned a clause into a name asserts nothing: the name would
            # enter the store as a thing in the world, and any later answer could cite it.
            # Only *this* clause is dropped — the others in the sentence are judged alone.
            dropped.append(f"a phrase that swallowed a clause: {unread}")
            continue
        proposition = build(clause, f"clause[{index}]")
        if proposition is not unresolved:
            out.append((proposition, evidence))
    return out, dropped


def clauses(frame: Frame, *, reported: bool = False) -> list[tuple[Frame, bool]]:
    """Every predication in a sentence, each with whether it is only *reported*.

    The unit matters more than the parser does. Measured on UD English-EWT dev with the
    treebank parser (``eval/parsing/whole_vs_clause.py``): for sentences of 21-35 tokens the
    whole tree is correct **2%** of the time while an individual clause's own arguments are
    correct **47%** of the time, and beyond 35 tokens no sentence is entirely right at all
    while a third of clauses are. A converter that needs the whole tree therefore learns
    almost nothing from a long sentence; one that reads clause by clause keeps what was
    understood and drops what was not.

    ``reported`` marks a clause sitting in a role of a verb of communication, cognition or
    emotion — said, believed, wanted. WordNet's lexicographer category for the verb's
    commonest sense decides that, so it is curated data rather than a list of verbs here. The
    test over-suppresses (a factive "I know that X" is held back too) and does so on purpose:
    missing an answer costs a question, while asserting what someone merely hoped for puts a
    falsehood in the store.
    """
    from . import wordnet

    out = [(frame, reported)]
    domains = wordnet.verb_domains()
    inner = reported or (domains.get(frame.predicate, ("",))[0] in REPORTING_DOMAINS)
    for value in frame.roles.values():
        # a role may hold one filler or, under coordination, several
        for one in (value if isinstance(value, (list, tuple)) else (value,)):
            if isinstance(one, Frame):
                out.extend(clauses(one, reported=inner))
            elif isinstance(one, Entity):
                out.extend(_clauses_under(one, reported))
    return out


def _clauses_under(entity: Entity, reported: bool) -> list[tuple[Frame, bool]]:
    """The predications hanging off a phrase, however deep.

    A phrase carries its modifiers as features — a possessor, an apposition, a prepositional
    phrase, a relative clause — and each of those is itself a phrase that may carry a clause
    of its own. "The RAM upgrade I was getting 6 hours from" hides its clause two levels
    down, and a walk over roles alone never reaches it.
    """
    out: list[tuple[Frame, bool]] = []
    clause = entity.features.get("restriction")
    if isinstance(clause, Frame):
        out.extend(clauses(_with_gap_filled(clause, entity), reported=reported))
    for name, value in entity.features.items():
        if name == "restriction":
            continue
        for one in (value if isinstance(value, (list, tuple)) else (value,)):
            if isinstance(one, Entity):
                out.extend(_clauses_under(one, reported))
            elif isinstance(one, Frame):
                out.extend(clauses(one, reported=reported))
    return out


def _with_gap_filled(clause: Frame, head: Entity) -> Frame:
    """A relative clause with the phrase it modifies put back where the gap is.

    The relativized argument is missing from the clause by definition, so it goes into the
    first core role the clause left empty; a clause with no empty core role is left as it is
    rather than guessed at.
    """
    bare = Entity(head.kind, head.text, {k: v for k, v in head.features.items() if k != "restriction"},
                  head.ref, head.candidates)
    gap = next((role for role in ("object", "subject") if role not in clause.roles), None)
    return clause if gap is None else Frame(clause.predicate, {**clause.roles, gap: bare}, clause.features)


def _swallowed(frame: Frame) -> str:
    """The text of a role filler the reader marked as having swallowed a clause, if any.

    Coordination puts a *tuple* of fillers in a role ("hardware and software"), so the walk
    has to look inside one: a conjunct that swallowed a clause is how a bad reading slipped
    past this check.
    """
    for value in frame.roles.values():
        if deeper := _swallowed_in(value):
            return deeper
    return ""


def _swallowed_in(value: Any) -> str:
    if isinstance(value, Entity):
        return value.text if value.features.get("contains_predicate") else ""
    if isinstance(value, Frame):
        return _swallowed(value)
    if isinstance(value, (list, tuple)):
        return next((found for v in value if (found := _swallowed_in(v))), "")
    return ""
