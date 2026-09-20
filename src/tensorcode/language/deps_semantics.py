"""Dependency parses to provisional frames without communicative authority.

The learned parser supplies grammatical relations. This authored adapter maps
relations to frame roles and counts preposition-role alternatives from STREUSLE
training annotations. Those correspondences, entity projections and lexical naming
conventions are not learned semantic understanding.

Clause outputs are :class:`ProvisionalMeaning` records retaining their syntax and
frame. Punctuation, word prefixes and missing subjects do not decide whether an
input is a question, assertion or request. Communicative interpretations require
separate evidence-backed proposals; raw provisional frames are never tell acts.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict, deque
from copy import copy, deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from pathlib import Path
from typing import Any, Mapping, Sequence

from .semantics import Entity, Frame

#: A grammatical relation and the role it fills in the meaning.
ROLE_OF_DEPREL = {
    "nsubj": "subject", "nsubj:pass": "subject", "csubj": "subject",
    "obj": "object", "dobj": "object", "iobj": "recipient",
    "xcomp": "content", "ccomp": "content", "advcl": "purpose",
    "advmod": "manner", "obl:tmod": "time", "obl:npmod": "manner",
}

#: STREUSLE's role labels, in this grammar's names. Roles it distinguishes more finely than
#: we do (Locus/Direction, StartTime/EndTime) fold together.
ROLE_OF_SNACS = {
    "Locus": "location", "Direction": "destination", "Goal": "destination", "Source": "source",
    "Originator": "source", "Time": "time", "StartTime": "time", "EndTime": "time", "Duration": "time",
    "Frequency": "time", "Recipient": "recipient", "Beneficiary": "recipient", "Instrument": "instrument",
    "Means": "instrument", "Purpose": "purpose", "Topic": "content", "Theme": "object", "Stimulus": "object",
    "Possessor": "possessor", "Whole": "possessor", "Manner": "manner", "Explanation": "reason",
}

#: Relations that make their dependent part of the noun phrase rather than a role of the verb.
INSIDE_NP = {"det", "amod", "compound", "nummod", "nmod:poss", "flat", "flat:name", "case", "punct",
             "nmod", "acl", "acl:relcl", "appos", "conj", "cc", "advmod", "aux", "cop", "mark", "obl"}


def find_streusle() -> Path | None:
    for c in (os.environ.get("TENSORCODE_STREUSLE"), "~/.cache/tensorcode/seeds/streusle"):
        if c and Path(c).expanduser().is_dir():
            return Path(c).expanduser()
    return None


def preposition_roles(root: Path | None = None) -> dict[str, list[tuple[str, float]]]:
    """Each preposition's roles with log P(role | preposition), counted over the STREUSLE training split only."""
    root = root or find_streusle()
    if root is None:
        return {}
    counts: dict[str, Counter] = defaultdict(Counter)
    for split in ("train",):
        path = root / split / f"streusle.ud_{split}.json"
        if not path.exists():
            continue
        for sent in json.loads(path.read_text()):
            for unit in list(sent.get("swes", {}).values()) + list(sent.get("smwes", {}).values()):
                ss = unit.get("ss") or ""
                if not ss.startswith("p."):
                    continue
                role = ROLE_OF_SNACS.get(ss[2:])
                if role:
                    counts[str(unit.get("lexlemma", "")).lower()][role] += 1
    out = {}
    for word, c in counts.items():
        total = sum(c.values())
        out[word] = sorted(((role, round(math.log(n / total), 3)) for role, n in c.items()), key=lambda kv: -kv[1])
    return out


@dataclass(frozen=True)
class ProvisionalMeaning:
    """An authored frame projection, not a request, question or assertion.

    Token indexes are one-based. ``root`` identifies the source dependency-tree
    root; ``frame_index`` distinguishes its projected root/coordinated frames.
    It does not claim a finer clause span or independent semantic head alignment.
    Exact character anchors remain in the owning reader alternative metadata.
    """

    frame: Frame
    words: tuple[str, ...]
    tags: tuple[str, ...]
    lemmas: tuple[str, ...]
    heads: tuple[tuple[int, int], ...]
    labels: tuple[tuple[int, str], ...]
    root: int
    frame_index: int = 0


@dataclass(frozen=True)
class PrepositionChoice:
    """One authored-inventory role proposal at a 1-based dependent token anchor."""

    dependent_token: int
    preposition: str
    role: str
    log_prior: float
    provenance: str


@dataclass(frozen=True)
class UnresolvedPreposition:
    dependent_token: int
    preposition: str
    reason: str


@dataclass(frozen=True)
class SemanticProjectionIssue:
    role: str
    dependent_tokens: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class SemanticReadCandidate:
    meanings: tuple[Any, ...]
    choices: tuple[PrepositionChoice, ...] = ()
    unresolved: tuple[UnresolvedPreposition | SemanticProjectionIssue, ...] = ()


@dataclass(frozen=True)
class SemanticReadCandidates:
    candidates: tuple[SemanticReadCandidate, ...]
    truncated: bool
    explored: int
    pending: int
    reason: str | None = None

    @property
    def complete(self) -> bool:
        """Whether enumeration finished, not whether meanings are correct or complete."""
        return not self.truncated


class _NeedPrepositionChoice(Exception):
    def __init__(self, token: int, word: str, options: tuple[tuple[str, float], ...]):
        self.token, self.word, self.options = token, word, options
        super().__init__(f"unresolved preposition {word!r} at dependent token {token}")


class _RoleCollision(Exception):
    def __init__(self, role: str):
        self.role = role
        super().__init__(f"multiple occurrences target role {role!r}; composition unresolved")


def _detach_reader(reader: Reader) -> Reader:
    """Capture configuration without sharing mutable caches or mapping values."""
    detached = copy(reader)
    for name, value in vars(reader).items():
        if isinstance(value, Mapping):
            value = MappingProxyType(deepcopy(dict(value)))
        else:
            value = deepcopy(value)
        setattr(detached, name, value)
    return detached


@dataclass(frozen=True)
class SemanticFrontierSnapshot:
    """Opaque, detached in-memory checkpoint of one projection search.

    Restore uses the captured adapter configuration, even if the originating
    Reader changes. Private payloads include mutable semantic values and are
    copied on capture and every restore; callers must not mutate private fields.
    This is not a disk format, a pickle contract, or a model-independent reading.
    """

    _reader: Reader
    _source: tuple
    _bindings: tuple
    _ready: tuple[SemanticReadCandidate, ...]
    explored: int

    def restore(self) -> SemanticFrontier:
        frontier = SemanticFrontier(self._reader, *self._source)
        frontier._frontier = deque(dict(branch) for branch in self._bindings)
        frontier._ready = deque(deepcopy(self._ready))
        frontier._explored = self.explored
        return frontier

    def __deepcopy__(self, memo):
        result = self.restore().snapshot()
        memo[id(self)] = result
        return result


class SemanticFrontier:
    """Resumable projection work, never an assertion or a completed reading.

    ``advance`` returns only previously undelivered candidates. ``explored`` is
    cumulative; ``pending`` counts retained partial branches plus completed but
    undelivered candidates, not an estimate of unseen complete meanings. Completed
    prefixes survive projection failures and can be delivered with zero expansions.
    Budgets apply to each call, including zero-budget inspection. Distinct binding
    branches rerun the adapter; previously explored branches are never replayed.
    """

    def __init__(self, reader: Reader, words: Sequence[str], tags: Sequence[str],
                 lemmas: Sequence[str], heads: Mapping[int, int], labels: Mapping[int, str]):
        self._reader = _detach_reader(reader)
        self._source = (tuple(words), tuple(tags), tuple(lemmas),
                        MappingProxyType(dict(heads)), MappingProxyType(dict(labels)))
        self._frontier: deque[dict[tuple[int, str], PrepositionChoice]] = deque([{}])
        self._explored = 0
        self._ready: deque[SemanticReadCandidate] = deque()

    def snapshot(self) -> SemanticFrontierSnapshot:
        """Detach all pending and undelivered work without evaluating a branch."""
        words, tags, lemmas, heads, labels = self._source
        return SemanticFrontierSnapshot(
            _detach_reader(self._reader),
            (words, tags, lemmas, dict(heads), dict(labels)),
            tuple(tuple(branch.items()) for branch in self._frontier),
            deepcopy(tuple(self._ready)), self._explored)

    def __deepcopy__(self, memo):
        result = self.snapshot().restore()
        memo[id(self)] = result
        return result

    def advance(self, *, max_expansions: int = 256, max_candidates: int = 32) -> SemanticReadCandidates:
        for value in (max_expansions, max_candidates):
            if type(value) is not int or value < 0:
                raise ValueError("semantic advance budgets must be nonnegative integers")
        explored_before = self._explored
        while (self._frontier and self._explored - explored_before < max_expansions
               and len(self._ready) < max_candidates):
            bindings = self._frontier.popleft()
            branch = copy(self._reader)
            branch._role_bindings = MappingProxyType(bindings)
            branch._branching = True
            self._explored += 1
            try:
                meanings = branch._read(*self._source) if self._source[0] else []
            except _RoleCollision as collision:
                issue = SemanticProjectionIssue(
                    collision.role,
                    tuple(choice.dependent_token for choice in bindings.values()
                          if choice.role == collision.role),
                    "multiple occurrences target one role; composition unresolved")
                self._ready.append(SemanticReadCandidate((), tuple(bindings.values()), (issue,)))
            except _NeedPrepositionChoice as need:
                if not need.options:
                    unresolved = UnresolvedPreposition(need.token, need.word, "no supplied role prior")
                    self._ready.append(SemanticReadCandidate((), tuple(bindings.values()), (unresolved,)))
                    continue
                for role, score in need.options:
                    choice = PrepositionChoice(need.token, need.word, role, score,
                                               self._reader.preposition_provenance)
                    self._frontier.append({**bindings, (need.token, need.word): choice})
            except BaseException:
                # A projection failure is not evidence that this branch was exhausted.
                self._frontier.appendleft(bindings)
                self._explored -= 1
                raise
            else:
                self._ready.append(SemanticReadCandidate(tuple(meanings), tuple(bindings.values())))
        candidates = tuple(self._ready.popleft() for _ in range(min(max_candidates, len(self._ready))))
        pending = len(self._frontier) + len(self._ready)
        reason = "semantic candidate or expansion budget exhausted" if pending else None
        return SemanticReadCandidates(tuple(candidates), bool(pending), self._explored, pending, reason)


class Reader:
    """Turns one parsed sentence into meanings."""

    #: VerbNet's class for the verbs that give something a name — call, name, label, dub,
    #: term, christen. "A folder called notes" is not a claim that anyone called anything;
    #: it is how the folder is named, and the class is what says which verbs do that.
    NAMING_CLASS = "dub-"

    def __init__(self, prepositions: Mapping[str, list[tuple[str, float]]] | None = None,
                 *, preposition_provenance: str | None = None) -> None:
        supplied = prepositions is not None
        priors = prepositions if supplied else preposition_roles()
        normalized = {}
        for word, options in priors.items():
            options = tuple((role, float(score)) for role, score in options)
            if any(not role or not math.isfinite(score) or score > 0 for role, score in options):
                raise ValueError("preposition priors require nonempty roles and finite nonpositive log priors")
            if len({role for role, _ in options}) != len(options):
                raise ValueError("duplicate role in preposition priors")
            normalized[word.lower()] = options
        self.prepositions = MappingProxyType(normalized)
        self.preposition_provenance = preposition_provenance or (
            "authored-preposition-priors" if supplied else "STREUSLE:train; authored ROLE_OF_SNACS projection")
        self._role_bindings = MappingProxyType({})
        self._branching = False
        self._verbs: Mapping[str, tuple] | None = None

    def __deepcopy__(self, memo):
        result = _detach_reader(self)
        memo[id(self)] = result
        return result

    def names_something(self, lemma: str) -> bool:
        """Is this the verb of "a folder *called* notes"?"""
        if self._verbs is None:
            from . import verbnet

            self._verbs = verbnet.load()
        return any(vc.id.startswith(self.NAMING_CLASS) for vc in self._verbs.get(lemma, ()))

    # -------------------------------------------------------------- structure

    def role_of_preposition(self, word: str, dependent_token: int) -> str:
        """Require an occurrence-specific choice; frequency never settles meaning."""
        word = word.lower()
        key = (dependent_token, word)
        if key in self._role_bindings:
            return self._role_bindings[key].role
        options = self.prepositions.get(word, ())
        if not self._branching and len(options) == 1:
            return options[0][0]  # singleton in the configured inventory; not proof of meaning
        if self._branching:
            raise _NeedPrepositionChoice(dependent_token, word, options)
        raise ValueError(f"preposition {word!r} at token {dependent_token} needs read_candidates")

    @staticmethod
    def _put_role(roles: dict, role: str, value: Any) -> None:
        """A repeated slot needs an explicit composition interpretation."""
        if role in roles:
            raise _RoleCollision(role)
        roles[role] = value

    def children(self, heads: Mapping[int, int]) -> dict[int, list[int]]:
        kids: dict[int, list[int]] = defaultdict(list)
        for dep, head in sorted(heads.items()):
            kids[head].append(dep)
        return kids

    def phrase(self, i: int, kids: Mapping[int, list[int]], words: Sequence[str], labels: Mapping[int, str],
               exclude: frozenset[int] = frozenset()) -> str:
        """The words of a subtree as they were said, without punctuation.

        ``exclude`` drops children whose meaning is taken elsewhere (a copular clause's
        subject belongs to the frame, not to the phrase that completes it).
        """
        inside = self.phrase_span(i, kids, words, labels, exclude)
        if not inside:
            return words[i - 1]
        return " ".join(words[j - 1] for j in inside)

    def phrase_span(self, i: int, kids: Mapping[int, list[int]], words: Sequence[str],
                    labels: Mapping[int, str], exclude: frozenset[int] = frozenset()) -> list[int]:
        """The token indices a phrase is made of.

        The phrase is the span between its first and last surviving descendant, so anything
        the parse placed in the middle comes along — which is what makes a wrong parse glue
        foreign material into a name. Whoever wants to judge the phrase has to judge these
        tokens, not the subtree they were supposed to be.
        """
        dropped = set()
        for k in exclude:
            dropped.add(k)
            dropped.update(self._descendants(k, kids))
        # Whatever the reader records as a role of its own is not part of what the phrase
        # *names*. A relative clause becomes a ``restriction``, and a prepositional phrase
        # becomes the role its preposition marks — so "the file scratch.txt from my desktop"
        # names the file, and the desktop is where it is. Left in, those words came along in
        # the name and no plugin could resolve it.
        for k in self._descendants(i, kids):
            label = labels.get(k, "").split(":")[0]
            if label == "acl" or (label in ("nmod", "obl") and self._has_case(k, kids, labels)):
                dropped.add(k)
                dropped.update(self._descendants(k, kids))
        span = sorted(j for j in [i, *self._descendants(i, kids)] if j not in dropped)
        if not span:
            return []
        return [j for j in range(span[0], span[-1] + 1) if j not in dropped and words[j - 1] not in ",.;:!?"]

    def _has_case(self, i: int, kids: Mapping[int, list[int]], labels: Mapping[int, str]) -> bool:
        """Is this modifier introduced by a preposition, so the reader gave it a role of its own?

        A bare ``nmod`` without a case marker is part of the name ("Dell Inspiron 15"); one with
        a preposition is a separate role ("... from my desktop").
        """
        return any(labels.get(c, "").split(":")[0] == "case" for c in kids.get(i, ()))

    def _descendants(self, i: int, kids: Mapping[int, list[int]]) -> list[int]:
        out: list[int] = []
        stack = list(kids.get(i, ()))
        while stack:
            j = stack.pop()
            out.append(j)
            stack.extend(kids.get(j, ()))
        return out

    # -------------------------------------------------------------- meanings

    def entity(self, i: int, words, tags, lemmas, heads, labels, kids, exclude: frozenset[int] = frozenset()) -> Entity:
        features: dict[str, Any] = {}
        modifiers: list[tuple[str, Any]] = []
        kind = "name" if tags[i - 1] == "PROPN" else "number" if tags[i - 1] == "NUM" else "description"
        for k in kids.get(i, ()):
            rel, word, tag = labels.get(k, ""), words[k - 1], tags[k - 1]
            if rel == "det":
                low = word.lower()
                features["definite"] = low not in ("a", "an")
                if tag == "PRON" or low in ("my", "your", "our", "his", "her", "their", "its"):
                    features["possessive"] = True
            elif rel in ("nmod:poss",):
                features["possessive"] = True
                features["possessor"] = 1 if word.lower() in ("my", "our") else 2 if word.lower() == "your" else 3
            elif rel == "amod":
                modifiers.append((rel, lemmas[k - 1]))
            elif rel == "nummod":
                features["count"] = word
            elif rel == "compound":
                # Preserve the modifier subtree, including compounds of compounds.
                # A compound relation is not evidence that this is a proper name.
                modifiers.append((rel, self.entity(k, words, tags, lemmas, heads, labels, kids)))
            elif rel in ("nmod", "obl"):
                case = next((words[c - 1] for c in kids.get(k, ()) if labels.get(c) == "case"), None)
                role = self.role_of_preposition(case or "", k)
                self._put_role(features, role, self.entity(k, words, tags, lemmas, heads, labels, kids))
            elif rel in ("acl:relcl", "acl"):
                if self.names_something(lemmas[k - 1]):
                    self._fold_naming(k, features, words, tags, lemmas, heads, labels, kids)
                else:
                    features["restriction"] = self.frame(k, words, tags, lemmas, heads, labels, kids)
            elif rel == "appos":
                features.setdefault("name", self.entity(k, words, tags, lemmas, heads, labels, kids))
        if modifiers:
            features["modifiers"] = tuple(modifiers)
            qualities = [value for relation, value in modifiers if relation == "amod"]
            compounds = [value for relation, value in modifiers if relation == "compound"]
            if len(qualities) == 1:
                features["quality"] = qualities[0]
            # Keep the historical shorthand for one compound, but never select
            # an arbitrary member of a multi-compound expression as its name.
            if len(compounds) == 1:
                features.setdefault("name", compounds[0])
        if tags[i - 1] == "NOUN":
            features["noun"] = lemmas[i - 1]
            features["number"] = "plural" if words[i - 1].lower() != lemmas[i - 1].lower() else "singular"
        elif tags[i - 1] == "PRON":
            kind = "pronoun"
            features["person"] = 1 if words[i - 1].lower() in ("i", "me", "we", "us", "my", "our") else \
                2 if words[i - 1].lower() in ("you", "your") else 3
        # the preposition marks the role; it is not part of what the phrase names
        cases = frozenset(k for k in kids.get(i, ()) if labels.get(k, "").split(":")[0] in ("case", "mark"))
        # a relative clause is kept as ``restriction``; it restricts the phrase but is not
        # part of what the phrase names, so "the dinner I volunteered at" names the dinner
        cases = cases | frozenset(k for k in kids.get(i, ()) if labels.get(k, "").split(":")[0] == "acl")
        if self._swallowed_a_clause(i, words, tags, labels, kids, exclude | cases):
            # a noun phrase does not contain a finite verb. One that does is not a phrase the
            # parse understood — it is a clause the parse gave up on and glommed into a name,
            # and everything downstream would treat that name as a thing in the world. Saying
            # so here is the only place the tags are still around to see it.
            features["contains_predicate"] = True
        return Entity(kind, self.phrase(i, kids, words, labels, exclude | cases), features)

    def _fold_naming(self, k: int, features: dict, words, tags, lemmas, heads, labels, kids) -> None:
        """"A folder called projects on my desktop": the name is *projects*, and the desktop
        is where the folder goes.

        Taking the whole of the clause's object as the name made a directory called
        ``projects on my desktop`` in the home folder — and the agent then verified it,
        correctly, because that directory did exist. So the name is the object without its
        prepositional phrases, and each of those phrases is attached to the phrase being
        named, under the role its preposition marks.
        """
        named = next((j for j in kids.get(k, ()) if labels.get(j, "").split(":")[0] == "obj"), None)
        if named is None:
            features["restriction"] = self.frame(k, words, tags, lemmas, heads, labels, kids)
            return
        modifiers = frozenset(j for j in kids.get(named, ())
                              if labels.get(j, "").split(":")[0] in ("nmod", "obl"))
        features.setdefault("name", self.entity(named, words, tags, lemmas, heads, labels, kids,
                                                exclude=modifiers))
        for j in modifiers:
            case = next((words[c - 1] for c in kids.get(j, ()) if labels.get(c) == "case"), None)
            role = self.role_of_preposition(case or "", j)
            self._put_role(features, role, self.entity(j, words, tags, lemmas, heads, labels, kids))
        # where the clause itself says the thing goes ("called notes *on my desktop*")
        for role, value in self.frame(k, words, tags, lemmas, heads, labels, kids).roles.items():
            if role not in ("object", "subject"):
                features.setdefault(role, value)

    def _swallowed_a_clause(self, i: int, words, tags, labels, kids, exclude: frozenset[int]) -> bool:
        """Is there a verb among the tokens this phrase is made of?

        A relative clause ("the dinner I volunteered at") is a predication the reader keeps
        separately as ``restriction``, so its verb is accounted for and excluded here. Any
        other verb inside a noun phrase means the phrase boundary is wrong — and the tokens
        to look at are the ones the phrase's *text* is built from, not the subtree, because a
        wrong parse pulls in material that was never a descendant.
        """
        dropped = set(exclude)
        for k in exclude:
            dropped.update(self._descendants(k, kids))
        for k in self._descendants(i, kids):
            if labels.get(k, "").split(":")[0] == "acl":
                dropped.add(k)
                dropped.update(self._descendants(k, kids))
        return any(tags[k - 1] == "VERB" for k in self.phrase_span(i, kids, words, labels, frozenset(dropped)))

    def frame(self, i: int, words, tags, lemmas, heads, labels, kids, coordinated: list | None = None) -> Frame:
        roles: dict[str, Any] = {}
        features: dict[str, Any] = {}
        coordinated = [] if coordinated is None else coordinated
        for k in kids.get(i, ()):
            rel = labels.get(k, "")
            base = rel.split(":")[0]
            if rel in ROLE_OF_DEPREL or base in ROLE_OF_DEPREL:
                role = ROLE_OF_DEPREL.get(rel) or ROLE_OF_DEPREL[base]
                if base == "advmod" and tags[k - 1] == "PART":
                    features["polarity"] = "negative"
                    continue
                value = self.entity(k, words, tags, lemmas, heads, labels, kids) if tags[k - 1] in ("NOUN", "PROPN", "PRON", "NUM") \
                    else self.frame(k, words, tags, lemmas, heads, labels, kids) if tags[k - 1] in ("VERB", "AUX") \
                    else lemmas[k - 1]
                self._put_role(roles, role, value)
            elif base == "obl":
                case = next((words[c - 1] for c in kids.get(k, ()) if labels.get(c) == "case"), None)
                role = self.role_of_preposition(case or "", k)
                self._put_role(roles, role, self.entity(k, words, tags, lemmas, heads, labels, kids))
            elif base == "aux":
                low = words[k - 1].lower()
                if tags[k - 1] == "AUX" and low in ("can", "could", "would", "will", "should", "may", "might", "must"):
                    features["modality"] = {"could": "can", "would": "will"}.get(low, low)
            elif base in ("neg",) or (base == "advmod" and words[k - 1].lower() in ("not", "n't", "never")):
                features["polarity"] = "negative"
            elif base == "conj":
                # "make x and make y": a coordinate clause of its own, carried out of the
                # frame by the caller rather than left inside it as a role
                coordinated.append(self.frame(k, words, tags, lemmas, heads, labels, kids))
        return Frame(lemmas[i - 1], roles, features)

    def copular(self, i: int, cop: int, words, tags, lemmas, heads, labels, kids) -> Frame:
        """"my name is Jacob", "the meeting is on Tuesday": UD makes the complement the root
        and hangs the copula off it, so the frame is ``be(subject, <complement>)``."""
        roles: dict[str, Any] = {}
        features: dict[str, Any] = {}
        for k in kids.get(i, ()):
            base = labels.get(k, "").split(":")[0]
            if base == "nsubj":
                self._put_role(roles, "subject", self.entity(k, words, tags, lemmas, heads, labels, kids))
            elif base == "obl":
                case = next((words[c - 1] for c in kids.get(k, ()) if labels.get(c) == "case"), None)
                self._put_role(roles, self.role_of_preposition(case or "", k),
                               self.entity(k, words, tags, lemmas, heads, labels, kids))
            elif base == "advmod" and words[k - 1].lower() in ("not", "n't", "never"):
                features["polarity"] = "negative"
        if tags[i - 1] in ("NOUN", "PROPN", "PRON", "NUM", "ADJ"):
            taken = frozenset(k for k in kids.get(i, ()) if labels.get(k, "").split(":")[0] in ("nsubj", "cop", "obl", "punct", "advmod"))
            complement = self.entity(i, words, tags, lemmas, heads, labels, kids, taken) if tags[i - 1] != "ADJ" else lemmas[i - 1]
            # "is on Tuesday": the complement carries its own preposition, and that marks the role
            case = next((words[c - 1] for c in kids.get(i, ()) if labels.get(c) == "case"), None)
            self._put_role(roles, self.role_of_preposition(case, i) if case else "object", complement)
        return Frame("be", roles, features)

    def start_candidates(self, words: Sequence[str], tags: Sequence[str], lemmas: Sequence[str],
                         heads: Mapping[int, int], labels: Mapping[int, str]) -> SemanticFrontier:
        """Detach this source and defer all projection work until advance()."""
        return SemanticFrontier(self, words, tags, lemmas, heads, labels)

    def read_candidates(self, words: Sequence[str], tags: Sequence[str], lemmas: Sequence[str],
                        heads: Mapping[int, int], labels: Mapping[int, str], *,
                        max_candidates: int = 32, max_expansions: int = 256) -> SemanticReadCandidates:
        """One bounded advance over occurrence-specific preposition alternatives.

        Use start_candidates() to retain and resume pending branches. Priors are
        training frequencies, not evidence that a proposed role is intended.
        Other grammatical mappings remain authored assumptions of this adapter.
        """
        if any(type(value) is not int or value < 1 for value in (max_candidates, max_expansions)):
            raise ValueError("semantic search budgets must be positive integers")
        return self.start_candidates(words, tags, lemmas, heads, labels).advance(
            max_candidates=max_candidates, max_expansions=max_expansions)

    def read(self, words: Sequence[str], tags: Sequence[str], lemmas: Sequence[str],
             heads: Mapping[int, int], labels: Mapping[int, str]) -> list[Any]:
        """Read only an unambiguous resolved adapter result; otherwise retain alternatives."""
        result = self.read_candidates(words, tags, lemmas, heads, labels, max_candidates=2)
        if result.truncated or len(result.candidates) != 1 or result.candidates[0].unresolved:
            raise ValueError("semantic reading unresolved or ambiguous; use read_candidates")
        return list(result.candidates[0].meanings)

    def _read(self, words: Sequence[str], tags: Sequence[str], lemmas: Sequence[str],
             heads: Mapping[int, int], labels: Mapping[int, str]) -> list[Any]:
        """The meanings of one parsed sentence, in order."""
        kids = self.children(heads)
        roots = [i for i in range(1, len(words) + 1) if heads.get(i) == 0]
        out: list[Any] = []
        for r in roots:
            copula = next((k for k in kids.get(r, ()) if labels.get(k) == "cop"), None)
            if tags[r - 1] not in ("VERB", "AUX") and copula is None:
                out.append(self.entity(r, words, tags, lemmas, heads, labels, kids))
                continue
            extra: list = []
            frame = self.copular(r, copula, words, tags, lemmas, heads, labels, kids) if copula is not None \
                else self.frame(r, words, tags, lemmas, heads, labels, kids, extra)
            for frame_index, projected in enumerate((frame, *extra)):
                out.append(ProvisionalMeaning(deepcopy(projected), tuple(words), tuple(tags), tuple(lemmas),
                    tuple(sorted(heads.items())), tuple(sorted(labels.items())), r, frame_index))
        return out
