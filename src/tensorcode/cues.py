"""Recall by cue, when the cue is not worded the way the memory was.

``Memory.recall`` compares the cue with a rendered claim as bags of word n-grams. That works when
the wording matches and falls off a cliff when it does not: "what is my cat" against
``person:user cat 'Mackerel'`` shares a word, while "what did I name the animal" shares none, and
the claim that answers it scores zero.

The fix here is deliberately *not* a learned vector space. A sibling measurement found embeddings
losing to plain token overlap on this repository's own recall tasks (0.005 vs 0.106 within a cycle,
0.741 vs 0.944 within a subject), so the interesting question is not "can a model do better" but
"what structure is token overlap throwing away". Three things, it turns out:

**Roles.** A claim is not a sentence, it is a subject, a predicate and an object. A cue word that
matches the predicate is worth far more than one matching some substring of the object, and a bag
of n-grams cannot tell the difference.

**Morphology.** "notes" and "note", "reading" and "read" are the same cue. Suffix stripping is
crude and costs nothing.

**The mind's own links.** Synonymy does not have to come from a hand-written list (which would only
encode the answers to whatever test one was running) or a trained model. It can come from what the
agent already believes: if the store holds ``fluffy is_a cat``, then a cue saying "cat" reaches a
claim about Fluffy in one hop. Expansion is discounted per hop, so a direct hit always wins.

What this cannot do is bridge words the agent has never seen related. That is the honest ceiling,
and it is measured rather than argued: see ``eval/temporal_perception/paraphrase_recall.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .outcomes import Score
from .records import ClaimRecord, Ref, Store

STOP = frozenset("""a an the my our your his her their its is are was were be been am do does did
what which who whom whose when where why how that this these those of in on at to for from with
and or not no you i me we they it there here again please tell told say said know remember about
had has have can could would should will shall may might must thing things stuff""".split())

# claims whose object names the same thing as their subject: the hops worth taking
LINKING = ("is_a", "label", "same_as", "means", "kind_of", "in", "part_of", "called", "aka")


def words(text: str) -> list[str]:
    return [w for w in re.split(r"[^\w']+", str(text).lower()) if w]


def lemma(word: str) -> str:
    """Enough morphology to make "notes" and "note" the same cue. Crude on purpose."""
    for suffix, keep in (("ies", "y"), ("sses", "ss"), ("ches", "ch"), ("shes", "sh"), ("xes", "x"),
                         ("ing", ""), ("ed", ""), ("s", "")):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)] + keep
    return word


def content(text: str) -> set[str]:
    """The cue-bearing lemmas of a phrase: no stop words, no punctuation, no inflection.

    Adjacent words are also joined, because a compound is written both ways and a predicate is one
    token: "time zone" has to reach ``timezone``, and "notes folder" has to reach itself.
    """
    kept = [lemma(w) for w in words(text) if w not in STOP and len(w) > 1]
    joined = {a + b for a, b in zip(kept, kept[1:])}
    return (set(kept) | joined) - {""}


def _ref_words(value: Any) -> set[str]:
    """A ref carries words too: ``person:user`` is about a user, ``ui:Files/button/Open#1`` about Files."""
    text = value.id if isinstance(value, Ref) else str(value)
    return content(re.sub(r"[:/#]", " ", text))


FIRST_PERSON = frozenset({"my", "mine", "our", "ours", "i", "me", "myself", "we", "us"})
SELF_WORDS = frozenset({"user", "person", "me", "self", "you"})


@dataclass(frozen=True)
class RoleWeights:
    """How much a match in each role is worth. Chosen on the training half only."""

    predicate: float = 3.0
    object: float = 2.0
    subject: float = 1.0
    hop_discount: float = 0.45  # one link away is worth less than half a direct hit
    first_person: float = 1.5   # "my deadline" is a question about the asker, not about a label
    off_topic_penalty: float = 0.6  # ... and a claim about something else answers it less well


@dataclass
class Hit:
    record: ClaimRecord
    score: Score
    matched: tuple[str, ...] = ()
    via: tuple[str, ...] = ()  # cue words reached through the store's own links

    def why(self) -> str:
        reason = f"matched {', '.join(self.matched) or 'nothing'}"
        return reason + (f" (via {', '.join(self.via)})" if self.via else "")


class Cues:
    """A structural index over claims: role-wise lemma overlap, widened by the store's own links.

    Built once per query batch rather than maintained: the cost is one pass over the claims, and a
    live store changes under any index that tries to be clever about incremental updates.
    """

    def __init__(self, mind: Store, *, weights: RoleWeights | None = None,
                 skip_scopes: Sequence[Ref | None] = (), skip_predicates: Iterable[str] = ()) -> None:
        self.mind = mind
        self.weights = weights or RoleWeights()
        self.skip_scopes = tuple(skip_scopes)
        self.skip_predicates = frozenset(skip_predicates)
        self._synonyms: dict[str, set[str]] = {}
        self._build_links()

    def _build_links(self) -> None:
        """Words the store itself says name the same thing, in one hop."""
        for rec in self.mind.claims():
            if rec.retracted or rec.claim.predicate not in LINKING:
                continue
            left, right = _ref_words(rec.claim.subject), _ref_words(rec.claim.object)
            for a in left:
                self._synonyms.setdefault(a, set()).update(right - {a})
            for b in right:
                self._synonyms.setdefault(b, set()).update(left - {b})

    def expand(self, cue: set[str]) -> dict[str, str]:
        """Cue lemmas one hop out, each remembering which cue word it came from."""
        out: dict[str, str] = {}
        for word in cue:
            for near in self._synonyms.get(word, ()):  # noqa: B007 - small sets
                if near not in cue:
                    out.setdefault(near, word)
        return out

    def find(self, cue: str, k: int = 3, *, min_score: float = 0.05) -> list[Hit]:
        direct = content(cue)
        if not direct:
            return []
        # "what is my deadline" constrains the subject as surely as it names the predicate, and the
        # stop-word list throws that away. Without it, a cue about "my time zone" is answered by a
        # column header reading "Time", because a heading is a perfectly good match for one word.
        about_me = bool(FIRST_PERSON & set(words(cue)))
        indirect = self.expand(direct)
        w = self.weights
        best = w.predicate + w.object + w.subject
        hits: list[Hit] = []
        for rec in self.mind.claims():
            if rec.retracted or rec.claim.scope in self.skip_scopes or rec.claim.predicate in self.skip_predicates:
                continue
            roles = ((w.predicate, content(rec.claim.predicate)),
                     (w.object, _ref_words(rec.claim.object)),
                     (w.subject, _ref_words(rec.claim.subject)))
            total, matched, via = 0.0, set(), set()
            for weight, bag in roles:
                overlap = bag & direct
                if overlap:
                    total += weight
                    matched |= overlap
                    continue
                reached = bag & set(indirect)
                if reached:
                    total += weight * w.hop_discount
                    matched |= reached
                    via |= {indirect[r] for r in reached}
            if total <= 0:
                continue
            if about_me:
                mine = bool(_ref_words(rec.claim.subject) & SELF_WORDS)
                total += w.first_person if mine else -w.off_topic_penalty
            # a claim that answers more of the cue outranks one that happens to match its commonest word
            coverage = len(matched & (direct | set(indirect))) / len(direct)
            value = (total / best) * (0.5 + 0.5 * min(1.0, coverage))
            if value >= min_score:
                hits.append(Hit(rec, Score(round(min(1.0, value), 4), "similarity"),
                                tuple(sorted(matched)), tuple(sorted(via))))
        hits.sort(key=lambda h: (-h.score.value, h.record.id))
        return hits[:k]


def exact_find(mind: Store, cue: str, k: int = 3) -> list[Hit]:
    """The baseline that has no tolerance at all: the cue must contain the predicate verbatim."""
    low = f" {cue.lower()} "
    hits = [Hit(rec, Score(1.0, "similarity"), (rec.claim.predicate,))
            for rec in mind.claims()
            if not rec.retracted and rec.claim.predicate and f" {rec.claim.predicate.lower()} " in low]
    return hits[:k]
