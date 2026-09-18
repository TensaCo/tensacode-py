"""From a dependency parse to the meanings the agent already speaks.

The learned parser (:mod:`.learned_parser`) gives a tree of grammatical relations. The
core wants :class:`Request`, :class:`Question` and :class:`Frame` values. This module is
the bridge, and it is deliberately thin: a relation names a role, a preposition names a
role, and the rest is reading the tree.

* which role a relation fills is :data:`ROLE_OF_DEPREL` — the same kind of alignment as
  ``verbnet.ROLE_OF_PREPOSITION_ROLE``, between two inventories for the same thing;
* which role a *preposition* marks is counted from STREUSLE's annotations of real usage
  (:func:`preposition_roles`), not chosen by hand: "in" is a place, "to" is a goal or a
  purpose, "from" is a source, each with how often;
* mood comes from the tree's shape (no subject and a bare verb is an imperative; an
  interrogative word, or an auxiliary before the subject, is a question).
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .semantics import Entity, Frame, Question, Request

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
    """Each preposition's roles with log P(role | preposition), counted over STREUSLE."""
    root = root or find_streusle()
    if root is None:
        return {}
    counts: dict[str, Counter] = defaultdict(Counter)
    for split in ("train", "dev"):
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


class Reader:
    """Turns one parsed sentence into meanings."""

    def __init__(self, prepositions: Mapping[str, list[tuple[str, float]]] | None = None) -> None:
        self.prepositions = dict(prepositions if prepositions is not None else preposition_roles())

    # -------------------------------------------------------------- structure

    def role_of_preposition(self, word: str) -> str:
        options = self.prepositions.get(word.lower())
        return options[0][0] if options else "location"

    def children(self, heads: Mapping[int, int]) -> dict[int, list[int]]:
        kids: dict[int, list[int]] = defaultdict(list)
        for dep, head in sorted(heads.items()):
            kids[head].append(dep)
        return kids

    def phrase(self, i: int, kids: Mapping[int, list[int]], words: Sequence[str], labels: Mapping[int, str]) -> str:
        span = [i] + [k for k in self._descendants(i, kids) if labels.get(k) in INSIDE_NP]
        return " ".join(words[j - 1] for j in sorted(span))

    def _descendants(self, i: int, kids: Mapping[int, list[int]]) -> list[int]:
        out: list[int] = []
        stack = list(kids.get(i, ()))
        while stack:
            j = stack.pop()
            out.append(j)
            stack.extend(kids.get(j, ()))
        return out

    # -------------------------------------------------------------- meanings

    def entity(self, i: int, words, tags, lemmas, heads, labels, kids) -> Entity:
        features: dict[str, Any] = {}
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
                features["quality"] = lemmas[k - 1]
            elif rel == "nummod":
                features["count"] = word
            elif rel == "compound":
                features.setdefault("name", Entity("description", word, {"noun": lemmas[k - 1]}))
            elif rel in ("nmod", "obl"):
                case = next((words[c - 1] for c in kids.get(k, ()) if labels.get(c) == "case"), None)
                role = self.role_of_preposition(case) if case else "possessor"
                features[role] = self.entity(k, words, tags, lemmas, heads, labels, kids)
            elif rel in ("acl:relcl", "acl"):
                features["restriction"] = self.frame(k, words, tags, lemmas, heads, labels, kids)
            elif rel == "appos":
                features.setdefault("name", self.entity(k, words, tags, lemmas, heads, labels, kids))
        if tags[i - 1] == "NOUN":
            features["noun"] = lemmas[i - 1]
            features["number"] = "plural" if words[i - 1].lower() != lemmas[i - 1].lower() else "singular"
        elif tags[i - 1] == "PRON":
            kind = "pronoun"
            features["person"] = 1 if words[i - 1].lower() in ("i", "me", "we", "us", "my", "our") else \
                2 if words[i - 1].lower() in ("you", "your") else 3
        return Entity(kind, self.phrase(i, kids, words, labels), features)

    def frame(self, i: int, words, tags, lemmas, heads, labels, kids) -> Frame:
        roles: dict[str, Any] = {}
        features: dict[str, Any] = {}
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
                roles[role] = value
            elif base == "obl":
                case = next((words[c - 1] for c in kids.get(k, ()) if labels.get(c) == "case"), None)
                role = self.role_of_preposition(case) if case else ROLE_OF_DEPREL.get(rel, "location")
                roles[role] = self.entity(k, words, tags, lemmas, heads, labels, kids)
            elif base == "aux":
                low = words[k - 1].lower()
                if tags[k - 1] == "AUX" and low in ("can", "could", "would", "will", "should", "may", "might", "must"):
                    features["modality"] = {"could": "can", "would": "will"}.get(low, low)
            elif base in ("neg",) or (base == "advmod" and words[k - 1].lower() in ("not", "n't", "never")):
                features["polarity"] = "negative"
            elif base == "conj":
                roles.setdefault("_conj", []).append(self.frame(k, words, tags, lemmas, heads, labels, kids))
        return Frame(lemmas[i - 1], roles, features)

    def read(self, words: Sequence[str], tags: Sequence[str], lemmas: Sequence[str],
             heads: Mapping[int, int], labels: Mapping[int, str]) -> list[Any]:
        """The meanings of one parsed sentence, in order."""
        kids = self.children(heads)
        roots = [i for i in range(1, len(words) + 1) if heads.get(i) == 0]
        out: list[Any] = []
        question = words[-1] == "?" or any(tags[i - 1] == "PRON" and words[i - 1].lower().startswith(("what", "which", "who", "where", "when", "why", "how")) for i in range(1, len(words) + 1))
        for r in roots:
            if tags[r - 1] not in ("VERB", "AUX"):
                out.append(self.entity(r, words, tags, lemmas, heads, labels, kids))
                continue
            frame = self.frame(r, words, tags, lemmas, heads, labels, kids)
            extra = frame.roles.pop("_conj", []) if isinstance(frame.roles, dict) else []
            for f in [frame, *extra]:
                out.append(self.speech_act(f, words, tags, question))
        return out

    def speech_act(self, frame: Frame, words: Sequence[str], tags: Sequence[str], question: bool) -> Any:
        """Imperative, interrogative or declarative, from the tree rather than from wording."""
        wh = next((w.lower() for w, t in zip(words, tags) if t in ("PRON", "ADV", "DET") and w.lower().startswith(
            ("what", "which", "who", "whom", "whose", "where", "when", "why", "how"))), None)
        if question or wh:
            asked = {"where": "location", "when": "time", "why": "reason", "how": "manner"}.get(wh or "", "theme")
            return Question(Frame(frame.predicate, frame.roles, {**frame.features, "mood": "interrogative"}), asked)
        if "subject" not in frame.roles:
            return Request(Frame(frame.predicate, frame.roles, {**frame.features, "mood": "imperative"}))
        return Frame(frame.predicate, frame.roles, {**frame.features, "mood": "declarative"})
