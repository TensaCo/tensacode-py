"""Conversational formulas: what "hello" is, and what answers it.

Two different things live here, and only one of them is knowledge we had to write down.

**Which class an expression belongs to is WordNet's.** "hello", "hi" and "welcome" are a
``greeting``; "goodbye" is a ``farewell``; "thanks" is an ``acknowledgement``; "yes" is an
``affirmation``; "ok" is an ``approval``. Nothing here lists those words — the taxonomy is
asked, so "howdy" and "hullo" work for the same reason "hello" does, and so does any word
WordNet files under those classes.

**What answers what is convention, and conventions have to be seeded.** A greeting is
answered with a greeting, thanks with a formula that acknowledges it, a farewell with a
farewell. No corpus we have states that, and it differs between languages and registers, so
it is data with a source and a way to replace it — not a branch in the agent. ``PAIRS`` is
what a speaker of this register does; another register supplies another table.

What this deliberately does *not* do is guess. An expression whose class WordNet does not
give ("please", "sorry" — neither is a noun of the right kind) falls through and the agent
says it did not follow it, which is true.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .semantics import Frame, Question, Request

#: The WordNet classes that make an utterance a conversational move rather than a statement
#: about the world. Reading the class from WordNet is what lets the words themselves be open.
CONVERSATIONAL = ("greeting", "farewell", "acknowledgement", "affirmation", "approval")

#: The reply a move of each class calls for — the second half of an adjacency pair. ``None``
#: means the move closes itself and no formula is owed ("ok" wants nothing back).
#:
#: Seeded, English, neutral register. Replaceable: set ``$TENSORCODE_CONVENTIONS`` to a JSON
#: file of the same shape, or hand ``pairs()`` a mapping of your own.
PAIRS: Mapping[str, str | None] = {
    "greeting": "hello",
    "farewell": "goodbye",
    "acknowledgement": "you are welcome",
    "affirmation": None,
    "approval": None,
}


def pairs(override: Mapping[str, str | None] | None = None) -> Mapping[str, str | None]:
    """The adjacency pairs in force: the seeded ones, a file's, or a caller's."""
    if override is not None:
        return dict(override)
    path = os.environ.get("TENSORCODE_CONVENTIONS")
    if path and Path(path).expanduser().is_file():
        return json.loads(Path(path).expanduser().read_text("utf-8"))
    return dict(PAIRS)


def move_of(kinds: frozenset[str] | set[str]) -> str | None:
    """Which conversational move an expression is, given what WordNet says it is a kind of.

    The most specific class wins, because ``greeting`` and ``farewell`` are both kinds of
    ``acknowledgement`` and answering "goodbye" with "hello" would be worse than saying
    nothing.
    """
    for name in CONVERSATIONAL:
        if name in kinds:
            return name
    return None


# Indirect requests are defeasible interpretations. The seed describes a register's
# default, not certainty about a speaker's intention (an ability question can be literal).
# Keep its authority and exact matching/transformation data available to callers.
@dataclass(frozen=True)
class RequestConvention:
    id: str
    source: str
    question: Mapping[str, Sequence[Any]]
    frame: Mapping[str, Sequence[Any]]
    subject: Mapping[str, Sequence[Any]]
    remove_roles: tuple[str, ...] = ()
    remove_features: tuple[str, ...] = ()
    add_features: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip() or not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("request conventions require nonempty id and source")
        for name in ("question", "frame", "subject"):
            criteria = getattr(self, name)
            if not isinstance(criteria, Mapping):
                raise ValueError(f"{name} must be a feature-to-allowed-values mapping")
            for key, values in criteria.items():
                if not isinstance(key, str) or not isinstance(values, (list, tuple)) or not values:
                    raise ValueError(f"{name} criteria require string keys and nonempty value lists")
            object.__setattr__(self, name, {k: tuple(v) for k, v in criteria.items()})
        for name in ("remove_roles", "remove_features"):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or not all(isinstance(v, str) for v in values):
                raise ValueError(f"{name} must be a list of feature names")
            object.__setattr__(self, name, tuple(values))
        if not isinstance(self.add_features, Mapping) or not all(isinstance(k, str) for k in self.add_features):
            raise ValueError("add_features must be a feature mapping")
        object.__setattr__(self, "add_features", dict(self.add_features or {}))


@dataclass(frozen=True)
class RequestInterpretation:
    request: Request
    convention_id: str
    source: str


RequestConventions = Sequence[RequestConvention | Mapping[str, Any]]


def request_conventions(override: RequestConventions | None = None) -> tuple[RequestConvention, ...]:
    """Replaceable request conventions; an empty override disables this inference.

    A JSON list in ``TENSORCODE_REQUEST_CONVENTIONS`` replaces the packaged seed.
    Invalid or missing explicitly configured files fail visibly, never silently restoring
    defaults that may authorize an unwanted interpretation. Earlier matching entries win.
    """
    if override is None:
        configured = os.environ.get("TENSORCODE_REQUEST_CONVENTIONS")
        path = Path(configured).expanduser() if configured else Path(__file__).with_name("data") / "request_conventions.json"
        override = json.loads(path.read_text("utf-8"))
    if not isinstance(override, (tuple, list)):
        raise ValueError("request conventions must be a list")
    result = []
    for entry in override:
        if isinstance(entry, RequestConvention):
            result.append(entry)
        elif isinstance(entry, Mapping):
            try:
                result.append(RequestConvention(**entry))
            except TypeError as exc:
                raise ValueError(f"invalid request convention: {exc}") from exc
        else:
            raise ValueError("each request convention must be a mapping")
    if len({entry.id for entry in result}) != len(result):
        raise ValueError("request convention ids must be unique")
    return tuple(result)


def interpret_request(question: Question, conventions: RequestConventions | None = None) -> RequestInterpretation | None:
    """Apply an inspectable convention to a question, retaining its stated authority."""
    frame = question.frame
    subject = frame.roles.get("subject")
    observed = (
        {"asked": question.asked},
        {**frame.features, "predicate": frame.predicate, "negated": frame.negated},
        {**getattr(subject, "features", {}), "kind": getattr(subject, "kind", None)},
    )
    for convention in request_conventions(conventions):
        criteria = (convention.question, convention.frame, convention.subject)
        if not all(all(key in values and values[key] in allowed for key, allowed in required.items())
                   for values, required in zip(observed, criteria)):
            continue
        roles = {k: v for k, v in frame.roles.items() if k not in convention.remove_roles}
        features = {k: v for k, v in frame.features.items() if k not in convention.remove_features}
        request = Request(Frame(frame.predicate, roles, {**features, **convention.add_features}))
        return RequestInterpretation(request, convention.id, convention.source)
    return None
