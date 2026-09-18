"""Small deterministic implementations with no dependencies beyond the standard library.

These are rules and classic algorithms. They are labeled as such in traces;
none of them is a learned model.
"""

from __future__ import annotations

import enum
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ..outcomes import Score, Unknown
from ..runtime import Output, Profile, Request, Traits

RULES = Traits(locality="in_process", egress=False, deterministic=True)
IN_PROCESS = Profile(source="declared: in-process code, no metered spend", usd_per_call=0.0)


@dataclass
class KeywordClassifier:
    """Regex rules per label. Answers only when exactly one label's rules match."""

    labels: type[enum.Enum]
    rules: Mapping[Any, Sequence[str]]
    name: str = "keyword-rules"
    version: str = "1"
    op: str = "classify"
    traits: Traits = RULES
    profile: Profile = IN_PROCESS

    def __post_init__(self) -> None:
        self._compiled = {label: [re.compile(p, re.I) for p in pats] for label, pats in self.rules.items()}

    def accepts(self, request: Request) -> bool:
        return request.op == "classify" and request.target is self.labels and isinstance(request.subject, str)

    def run(self, requests: Sequence[Request]) -> list[Output]:
        outs = []
        for r in requests:
            hits = [label for label, pats in self._compiled.items() if any(p.search(r.subject) for p in pats)]
            if len(hits) == 1:
                outs.append(Output(hits[0]))
            elif not hits:
                outs.append(Output(Unknown("no_rule_matched")))
            else:
                outs.append(Output(Unknown("rules_disagree", candidates=tuple((h, Score(1.0, "vote_share")) for h in hits))))
        return outs


@dataclass
class UtilityChooser:
    """Argmax of ``Objective.utility``. Abstains on ties within ``margin`` or when utility is undefined."""

    margin: float = 0.0
    name: str = "utility-argmax"
    version: str = "1"
    op: str = "choose"
    traits: Traits = RULES
    profile: Profile = IN_PROCESS

    def accepts(self, request: Request) -> bool:
        return request.op == "choose" and getattr(request.target, "utility", None) is not None

    def run(self, requests: Sequence[Request]) -> list[Output]:
        outs = []
        for r in requests:
            given = r.params.get("given")
            scored = sorted(((o, r.target.utility(o, given)) for o in r.subject), key=lambda p: p[1], reverse=True)
            cands = tuple((o, Score(u, "utility")) for o, u in scored)
            if len(scored) > 1 and scored[0][1] - scored[1][1] <= self.margin:
                outs.append(Output(Unknown("tie_within_margin", f"top utilities {scored[0][1]:.3g} vs {scored[1][1]:.3g}", cands)))
            else:
                outs.append(Output(scored[0][0], Score(scored[0][1], "utility")))
        return outs


_WORD = re.compile(r"[a-z0-9]+")


def _terms(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass
class BM25Ranker:
    """Okapi BM25 over each candidate's text. Scores are relevance, not probabilities."""

    text_of: Callable[[Any], str] = str
    k1: float = 1.2
    b: float = 0.75
    name: str = "bm25"
    version: str = "1"
    op: str = "rank"
    traits: Traits = RULES
    profile: Profile = IN_PROCESS

    def accepts(self, request: Request) -> bool:
        return request.op == "rank" and isinstance(request.subject, str)

    def run(self, requests: Sequence[Request]) -> list[Output]:
        return [Output(self._rank(r.subject, r.params["candidates"])) for r in requests]

    def _rank(self, query: str, candidates: Sequence[Any]) -> list[tuple[Any, Score]]:
        docs = [Counter(_terms(self.text_of(c))) for c in candidates]
        n = len(docs)
        if n == 0:
            return []
        avg = sum(sum(d.values()) for d in docs) / n or 1.0
        df = Counter(t for d in docs for t in d)
        q = _terms(query)
        scored = []
        for cand, d in zip(candidates, docs):
            length = sum(d.values())
            s = 0.0
            for t in q:
                if t in d:
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    s += idf * d[t] * (self.k1 + 1) / (d[t] + self.k1 * (1 - self.b + self.b * length / avg))
            scored.append((cand, Score(s, "relevance")))
        return sorted(scored, key=lambda p: p[1].value, reverse=True)


@dataclass
class StoreFactCheck:
    """Checks a ``Claim`` against the claims recorded in a ``Store``. No inference beyond the records.

    holds:   every live claim overlapping the proposition's interval agrees, and at least one exists
    fails:   the predicate is functional and every overlapping claim asserts a different object
    unknown: no overlapping claims, or they disagree with each other
    """

    store: Any
    name: str = "store-fact-check"
    version: str = "1"
    op: str = "check"
    traits: Traits = RULES
    profile: Profile = IN_PROCESS

    def accepts(self, request: Request) -> bool:
        from ..records import Claim

        return request.op == "check" and isinstance(request.subject, Claim)

    def run(self, requests: Sequence[Request]) -> list[Output]:
        from ..outcomes import Verdict

        outs = []
        for r in requests:
            p = r.subject
            overlapping = [
                rec for rec in self.store.claims(p.subject, p.predicate, scope=p.scope) if rec.claim.valid.overlap(p.valid) is not None
            ]
            objects = {rec.claim.object for rec in overlapping}
            cited = tuple(e.source for rec in overlapping for e in rec.evidence)
            if not overlapping:
                outs.append(Output(Verdict("unknown", ("no recorded claim covers the interval",))))
            elif objects == {p.object}:
                outs.append(Output(Verdict("holds", (f"{len(overlapping)} agreeing claim(s)",), cited)))
            elif len(objects) == 1 and p.predicate in self.store.functional:
                outs.append(Output(Verdict("fails", (f"recorded {objects.pop()!r} for a functional predicate",), cited)))
            else:
                outs.append(Output(Verdict("unknown", (f"contested: {sorted(map(str, objects))}",), cited)))
        return outs
