"""Resolving what "it", "there" and "she" pick out, against a salience list.

The rule is ordinary centering: candidates are the entities the conversation has
touched, most recent first, filtered by what the pronoun agrees with. Two things
make it honest rather than convenient:

* a pronoun with **no** compatible candidate stays unresolved, and
* a pronoun with **two equally recent** compatible candidates stays unresolved
  with both recorded.

Unresolved references then block claim conversion (:func:`semantics.to_claims`
returns ``Unknown``), so the failure surfaces where a caller must handle it
rather than as a wrong belief. ``symbolic-ai-models``'s learned reader does the
same thing with an XOR factor over the claims the antecedents would produce
(``symbolic_ai_core/reader/learned.py``); this keeps the candidates on the
entity instead, which is the same refusal in a smaller shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..records import Ref
from .semantics import Entity, Frame, Question, Request

#: Which entity features a pronoun demands of its antecedent.
AGREEMENT: Mapping[str, Mapping[str, Any]] = {
    "it": {"animate": False},
    "they": {},
    "them": {},
    "he": {"animate": True, "gender": "m"},
    "him": {"animate": True, "gender": "m"},
    "she": {"animate": True, "gender": "f"},
    "her": {"animate": True, "gender": "f"},
    "there": {"place": True},
}


@dataclass
class Context:
    """What the conversation has touched, most salient last."""

    mentions: list[Entity] = field(default_factory=list)
    speaker: Ref | None = None
    addressee: Ref | None = None
    #: the caller's own notion of the current object ("it" after an action)
    focus: Entity | None = None
    limit: int = 12

    def mention(self, entity: Entity) -> None:
        if entity.kind == "pronoun":
            return
        self.mentions = [m for m in self.mentions if (m.ref, m.text) != (entity.ref, entity.text)][-self.limit:]
        self.mentions.append(entity)

    def observe(self, meaning: Any) -> None:
        """Record the entities of a reading, so the next utterance can refer back."""
        frame = meaning.frame if isinstance(meaning, (Request, Question)) else meaning
        if isinstance(frame, Frame):
            for entity in frame.entities():
                self.mention(entity)
        elif isinstance(frame, Entity):
            self.mention(frame)

    def candidates(self, pronoun: Entity) -> list[Entity]:
        wanted = dict(AGREEMENT.get(pronoun.text.lower(), {}))
        ordered = list(reversed(self.mentions))
        if self.focus is not None:
            ordered = [self.focus] + [m for m in ordered if m is not self.focus]
        out = []
        for entity in ordered:
            if all(entity.features.get(k) == v for k, v in wanted.items() if k in entity.features):
                if wanted.get("place") and not entity.features.get("place"):
                    continue
                out.append(entity)
        return out


def resolve(meaning: Any, context: Context) -> Any:
    """Replace resolvable pronouns; leave the rest unresolved, with candidates."""
    if isinstance(meaning, Request):
        return Request(resolve(meaning.frame, context))
    if isinstance(meaning, Question):
        return Question(resolve(meaning.frame, context), meaning.asked)
    if isinstance(meaning, tuple):
        return tuple(resolve(m, context) for m in meaning)
    if isinstance(meaning, Entity):
        # resolve what is nested first: "put it in a folder called it" has two
        # pronouns at different depths, and only descending reaches the inner one
        inner = {k: resolve(v, context) if isinstance(v, (Entity, Frame, tuple)) else v
                 for k, v in meaning.features.items()}
        if inner != meaning.features:
            meaning = Entity(meaning.kind, meaning.text, inner, meaning.ref, meaning.candidates)
        return _resolve_entity(meaning, context)
    if isinstance(meaning, Frame):
        roles = {k: resolve(v, context) for k, v in meaning.roles.items()}
        features = {k: resolve(v, context) if isinstance(v, (Entity, Frame)) else v for k, v in meaning.features.items()}
        return Frame(meaning.predicate, roles, features)
    return meaning


def _resolve_entity(entity: Entity, context: Context) -> Entity:
    if entity.kind != "pronoun":
        return entity
    word = entity.text.lower()
    if word in ("i", "me", "we") and context.speaker is not None:
        return entity.with_ref(context.speaker)
    if word == "you" and context.addressee is not None:
        return entity.with_ref(context.addressee)
    found = context.candidates(entity)
    if not found:
        return Entity(entity.kind, entity.text, entity.features, None, ())
    if len(found) > 1 and _tied(found[0], found[1]):
        return Entity(entity.kind, entity.text, entity.features, None, tuple(found[:3]))
    best = found[0]
    return Entity("resolved", best.text, {**best.features, "via": entity.text}, best.ref, ())


def _tied(a: Entity, b: Entity) -> bool:
    """Two antecedents are tied when nothing in the discourse separates them."""
    return a.features.get("noun") == b.features.get("noun") and a.ref is None and b.ref is None and a.text != b.text


def unresolved(meaning: Any) -> list[Entity]:
    """Every reference the discourse could not settle, for a caller to ask about."""
    frame = meaning.frame if isinstance(meaning, (Request, Question)) else meaning
    if isinstance(frame, Entity):
        return [] if frame.resolved else [frame]
    if not isinstance(frame, Frame):
        return []
    return [e for e in frame.entities() if not e.resolved]
