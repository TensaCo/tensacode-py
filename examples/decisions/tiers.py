"""Bindings: which implementations can answer the typed questions, in what order.

The five decisions in ``decisions.py`` do not change between configurations. What changes
is this file — a rules tier, a learned tier, an optional model tier, all satisfying the same
``classify`` question, plus a BM25 reranker for ``rank`` and an overlap check for ``check``.

This is the framework's actual claim, and the one thing here that a plain HTTP decision API
cannot offer: the caller's code is written once against the *question*, and the deployment
decides what answers it. The trace then says which tier did.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

import tensorcode as tc
from tensorcode.backends.builtin import IN_PROCESS, KeywordClassifier
from tensorcode.outcomes import Score, Unknown, Verdict
from tensorcode.runtime import Output, Profile, Request, Traits

from ..support_router.config import KEYWORD_RULES, learned_classifier
from ..support_router.domain import Intent
from .domain import Passage

# ------------------------------------------------------------------- rank


_WORD = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class BM25Reranker:
    """Okapi BM25 over the candidate set. A relevance score, and it says so.

    ``Score(kind="relevance")`` is the honest label: these numbers order this ranking and
    mean nothing outside it. The money gate refuses to threshold them, which is the point —
    a reranker score is not a probability that an action is right.
    """

    name = "bm25-rerank"
    version = "1"
    op = "rank"
    traits = Traits(locality="in_process", egress=False, deterministic=True)
    profile = Profile(source="declared: in-process, no metered spend", usd_per_call=0.0)

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b

    def accepts(self, request: Request) -> bool:
        candidates = request.params.get("candidates", ())
        return request.op == "rank" and bool(candidates) and all(isinstance(c, Passage) for c in candidates)

    def run(self, requests: Sequence[Request]) -> list[Output]:
        outs: list[Output] = []
        for request in requests:
            passages: tuple[Passage, ...] = tuple(request.params["candidates"])
            docs = [_tokens(f"{p.title} {p.text}") for p in passages]
            lengths = [len(d) for d in docs]
            avg = sum(lengths) / len(lengths) if lengths else 0.0
            df = Counter(term for d in docs for term in set(d))
            n = len(docs)
            query = _tokens(str(request.subject))
            scored: list[tuple[Passage, Score]] = []
            for passage, doc, length in zip(passages, docs, lengths):
                counts = Counter(doc)
                total = 0.0
                for term in query:
                    if term not in counts:
                        continue
                    idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                    tf = counts[term]
                    total += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / (avg or 1)))
                scored.append((passage, Score(round(total, 6), "relevance", "bm25 over this candidate set")))
            scored.sort(key=lambda pair: (-pair[1].value, pair[0].id))
            outs.append(Output(scored))
        return outs


# ------------------------------------------------------------------ check


class OverlapSupport:
    """Does a passage support a claim? Content-word overlap, with an explicit unknown band.

    Deliberately weak and deliberately three-valued. Between the two thresholds it returns
    ``unknown`` rather than guessing, which is what a citation checker should do when the
    evidence is thin — and what a boolean API cannot express.
    """

    name = "overlap-support"
    version = "1"
    op = "check"
    traits = Traits(locality="in_process", egress=False, deterministic=True)
    profile = Profile(source="declared: in-process, no metered spend", usd_per_call=0.0)

    STOP = frozenset("a an the of and or to in on for with is are was were be been it its this that from as at by".split())

    def __init__(self, *, holds_at: float = 0.5, fails_below: float = 0.2) -> None:
        self.holds_at, self.fails_below = holds_at, fails_below

    def accepts(self, request: Request) -> bool:
        subject = request.subject
        evidence = request.params.get("evidence", ())
        return (
            request.op == "check"
            and isinstance(subject, tuple)
            and len(subject) == 2
            and subject[0] == "supports"
            and bool(evidence)
            and all(isinstance(e, Passage) for e in evidence)
        )

    def run(self, requests: Sequence[Request]) -> list[Output]:
        outs = []
        for request in requests:
            claim = {t for t in _tokens(str(request.subject[1])) if t not in self.STOP}
            passage: Passage = request.params["evidence"][0]
            body = {t for t in _tokens(f"{passage.title} {passage.text}") if t not in self.STOP}
            if not claim:
                outs.append(Output(Verdict("unknown", ("the claim has no content words",))))
                continue
            share = len(claim & body) / len(claim)
            shared = ", ".join(sorted(claim & body)[:6]) or "nothing"
            if share >= self.holds_at:
                verdict = Verdict("holds", (f"{share:.0%} of the claim's content words appear in {passage.id} ({shared})",), (passage.id,))
            elif share < self.fails_below:
                verdict = Verdict("fails", (f"only {share:.0%} of the claim's content words appear in {passage.id}",), (passage.id,))
            else:
                verdict = Verdict("unknown", (f"{share:.0%} overlap with {passage.id} is between {self.fails_below:.0%} and {self.holds_at:.0%}",), (passage.id,))
            outs.append(Output(verdict))
        return outs


# ------------------------------------------------- giving rules a confidence


class ScoredKeywords:
    """A keyword tier that reports a *measured* probability, not bare certainty.

    This exists because of a measured failure. The keyword tier answers 173 of 3,080
    Banking77 test messages at 97.1% precision — better than the learned tier's 94% — and
    ``Output.score`` is ``None``, so a confidence gate escalates every one of them. In a
    cascade it is worse than absent: it captures cases the learned tier would have answered
    with a usable confidence, turning 173 auto-decisions into escalations (191 -> 362).

    The fix is not to trust rules more. It is to measure them: per-label precision on a
    held-out split becomes the probability the tier reports, with the split named in
    ``Score.basis``. A label that was never right on validation reports a low number and the
    gate escalates it, which is the correct outcome.
    """

    version = "1"
    op = "classify"

    def __init__(self, inner: KeywordClassifier, precision: dict[str, float], *, basis: str, floor: float = 0.05) -> None:
        self.inner, self.precision, self.basis, self.floor = inner, precision, basis, floor
        self.name = f"{inner.name}-scored"
        self.traits = inner.traits
        self.profile = inner.profile

    @classmethod
    def measure(cls, inner: KeywordClassifier, rows: Sequence[tuple[str, Intent]], *, basis: str) -> "ScoredKeywords":
        """Per-label precision of the rules on ``rows``. Must be a validation split, not test."""
        hits: dict[str, list[bool]] = {}
        for text, gold in rows:
            out = inner.run([Request("classify", text, Intent, {})])[0]
            value = getattr(out, "value", None)
            if value is None or isinstance(value, Unknown):
                continue
            hits.setdefault(value.name, []).append(value is gold)
        precision = {label: sum(seen) / len(seen) for label, seen in hits.items()}
        counts = {label: len(seen) for label, seen in hits.items()}
        return cls(inner, precision, basis=f"{basis}; per-label precision over {sum(counts.values())} rule firings {counts}")

    def accepts(self, request: Request) -> bool:
        return self.inner.accepts(request)

    def run(self, requests: Sequence[Request]) -> list[Output]:
        outs = []
        for out in self.inner.run(requests):
            value = getattr(out, "value", None)
            if value is None or isinstance(value, Unknown):
                outs.append(out if isinstance(out, Output) else Output(value))
                continue
            p = self.precision.get(value.name)
            if p is None:  # a rule that never fired on validation has no measured precision
                outs.append(Output(Unknown("unmeasured_rule", f"{value.name} never fired on the validation split")))
                continue
            outs.append(Output(value, Score(max(p, self.floor), "probability", self.basis)))
        return outs


# --------------------------------------------------------------- assembly


def rules_only() -> list:
    """Keyword rules for intent, BM25 for rank, overlap for check. No learning, no model."""
    return [KEYWORD_RULES, BM25Reranker(), OverlapSupport()]


def learned_only(train_csv: Path, *, target_accuracy: float = 0.95) -> list:
    """The TF-IDF + logistic-regression tier alone, with its calibrated threshold."""
    learned, report = learned_classifier(train_csv, target_accuracy=target_accuracy)
    return [learned, BM25Reranker(), OverlapSupport()], report


def cascade(train_csv: Path, *, target_accuracy: float = 0.95) -> list:
    """Rules first, then the learned tier on whatever the rules abstain from."""
    learned, report = learned_classifier(train_csv, target_accuracy=target_accuracy)
    return [KEYWORD_RULES, learned, BM25Reranker(), OverlapSupport()], report


def cascade_scored(train_csv: Path, *, target_accuracy: float = 0.95) -> list:
    """The same cascade, with the rules tier reporting measured per-label precision.

    Same order, same rules, same learned tier — the only change is that the cheap tier now
    says how often it is right, so the gate can use its answers instead of escalating them.
    """
    from ..support_router.config import load_banking77, split_train_validation

    learned, report = learned_classifier(train_csv, target_accuracy=target_accuracy)
    _, validation = split_train_validation(load_banking77(train_csv))
    scored = ScoredKeywords.measure(KEYWORD_RULES, validation, basis=f"banking77 train-holdout, n={len(validation)}")
    return [scored, learned, BM25Reranker(), OverlapSupport()], report


def runtime(bindings: Sequence[object], *, cache: bool = True) -> tc.Runtime:
    """A runtime for a request handler: in-process only, no egress, cheap to construct."""
    policy = tc.Policy(localities=frozenset({"in_process"}), allow_egress=False, available=frozenset({"sklearn"}), cache=cache)
    return tc.Runtime(list(bindings), policy=policy)
