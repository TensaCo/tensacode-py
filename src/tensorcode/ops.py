"""Typed operation facades.

Each facade fixes the *meaning* of an operation and validates outputs against it.
Which code actually runs (rules, a parser, a classifier, a solver, a model) is
decided by the bound ``Runtime``; with nothing bound, inferential facades return
``Unknown(reason="no_implementation")``. None of them implies a model call.

Families (see docs/revival/03-operations-and-backends.md):
    infer:   parse, classify, choose, rank
    check:   check, verify
    rewrite: propose            (``Store.apply`` commits)
    invoke:  invoke             (actions.py)
    convert: to_records, from_records, pack   (records.py, context.py)
    query:   Store.claims / match / neighborhood (records.py)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Callable, Sequence, TypeVar

from .outcomes import Receipt, Score, Unknown, Verdict
from .records import Patch
from .runtime import Request, current

T = TypeVar("T")
A = TypeVar("A")


def _is_instance_of(target: Any) -> Callable[[Any], bool]:
    def check(value: Any) -> bool:
        try:
            return isinstance(value, target)
        except TypeError:  # parametrized generics, Literal, ...: the implementation owns validation
            return True

    return check


# ------------------------------------------------------------------- infer


def parse(source: Any, into: type[T], **params: Any) -> T | Unknown:
    """Interpret ``source`` (text, a document, an observation) as a value of type ``into``.

    Fails closed: an output that is not an ``into`` is treated as a failed attempt.
    """
    return current().call(Request("parse", source, into, params), validate=_is_instance_of(into)).value


def _parse_many(sources: Sequence[Any], into: type[T], **params: Any) -> list[T | Unknown]:
    outs = current().call_many([Request("parse", s, into, params) for s in sources], validate=_is_instance_of(into))
    return [o.value for o in outs]


parse.many = _parse_many  # type: ignore[attr-defined]


def classify(item: Any, labels: type[T], **params: Any) -> T | Unknown:
    """Estimate which label *is true* of ``item``. Not a decision about what to do."""
    return current().call(Request("classify", item, labels, params), validate=_label_check(labels)).value


def _classify_many(items: Sequence[Any], labels: type[T], **params: Any) -> list[T | Unknown]:
    outs = current().call_many([Request("classify", x, labels, params) for x in items], validate=_label_check(labels))
    return [o.value for o in outs]


def _classify_scored(item: Any, labels: type[T], **params: Any) -> tuple[T | Unknown, Score | None]:
    """The label *and* the confidence its implementation reported, for callers that gate on it.

    An implementation that reports no confidence yields ``None`` — which a gate must treat as
    "unknown confidence", not as zero, or an accurate tier with no score silently becomes an
    escalation. Measured: a keyword tier answering 173/3,080 at 97.1% precision made a cascade
    *worse* than the learned tier alone until its precision was reported as a probability.
    """
    out = current().call(Request("classify", item, labels, params), validate=_label_check(labels))
    return out.value, out.score


def _classify_many_scored(items: Sequence[Any], labels: type[T], **params: Any) -> list[tuple[T | Unknown, Score | None]]:
    outs = current().call_many([Request("classify", x, labels, params) for x in items], validate=_label_check(labels))
    return [(o.value, o.score) for o in outs]


classify.many = _classify_many  # type: ignore[attr-defined]
classify.scored = _classify_scored  # type: ignore[attr-defined]
classify.many_scored = _classify_many_scored  # type: ignore[attr-defined]


def _label_check(labels: Any) -> Callable[[Any], bool]:
    if isinstance(labels, type) and issubclass(labels, enum.Enum):
        return lambda v: isinstance(v, labels)
    return lambda v: v in labels


@dataclass(frozen=True)
class Objective:
    """What makes one feasible option better than another."""

    name: str
    description: str
    utility: Callable[[Any, Any], float] | None = None  # (option, given) -> utility, when computable


@dataclass(frozen=True)
class Constraint:
    """A hard requirement checked by TensorCode itself, never delegated to a backend."""

    name: str
    test: Callable[[Any, Any], bool | Unknown]  # (option, given)


def choose(options: Sequence[A], *, objective: Objective, given: Any = None, constraints: Sequence[Constraint] = ()) -> A | Unknown:
    """Select an action. Unlike ``classify``, this involves an objective and hard constraints.

    Constraints are evaluated here, before any backend sees the options. An
    undetermined constraint (``Unknown``) excludes the option. The backend can only
    return one of the feasible options; anything else is an invalid attempt.
    """
    feasible, notes = [], []
    for option in options:
        failed = []
        for c in constraints:
            ok = c.test(option, given)
            if isinstance(ok, Unknown):
                failed.append(f"{c.name}=unknown({ok.reason})")
            elif not ok:
                failed.append(c.name)
        if failed:
            notes.append(f"excluded {option!r}: {', '.join(failed)}")
        else:
            feasible.append(option)
    rt = current()
    if not feasible:
        span = rt.trace.open("choose", target=objective.name, input=tuple(options))
        span.notes.extend(notes or ["no options were proposed"])
        return span.close(Unknown("no_feasible_option", "; ".join(notes)), "unknown")
    if len(feasible) == 1:
        span = rt.trace.open("choose", target=objective.name, input=tuple(options))
        span.notes.extend(notes + ["only one feasible option; no backend consulted"])
        return span.close(feasible[0], "answer")
    request = Request("choose", tuple(feasible), objective, {"given": given})
    return rt.call(request, validate=lambda v: any(v is o or v == o for o in feasible), notes=notes).value


def rank(query: Any, candidates: Sequence[A], *, limit: int | None = None, **params: Any) -> list[tuple[A, Score]] | Unknown:
    """Order candidates by relevance to ``query``. Scores are comparable only within this ranking."""

    def valid(v: Any) -> bool:
        return isinstance(v, list) and all(
            isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[1], Score) and pair[1].kind in ("relevance", "similarity")
            for pair in v
        )

    out = current().call(Request("rank", query, None, {"candidates": tuple(candidates), **params}), validate=valid).value
    return out if isinstance(out, Unknown) or limit is None else out[:limit]


# ------------------------------------------------------------------- check


def check(proposition: Any, *, evidence: Sequence[Any] = (), **params: Any) -> Verdict:
    """Evaluate a claim, candidate, or constraint against evidence. Unknown is not false."""
    out = current().call(Request("check", proposition, None, {"evidence": tuple(evidence), **params}), validate=lambda v: isinstance(v, Verdict)).value
    return Verdict("unknown", (f"{out.reason}: {out.detail}",)) if isinstance(out, Unknown) else out


def _check_many(propositions: Sequence[Any], *, evidence: Sequence[Any] = (), **params: Any) -> list[Verdict]:
    requests = [Request("check", p, None, {"evidence": tuple(evidence), **params}) for p in propositions]
    outs = current().call_many(requests, validate=lambda v: isinstance(v, Verdict))
    return [Verdict("unknown", (f"{o.value.reason}: {o.value.detail}",)) if isinstance(o.value, Unknown) else o.value for o in outs]


check.many = _check_many  # type: ignore[attr-defined]


def verify(receipt: Receipt, *, observe: Callable[[], Any], expect: Callable[[Any], bool | Unknown]) -> Verdict:
    """Did an invoked action have its intended effect, according to a fresh observation?

    A receipt saying ``applied`` is the executor's report, not verification.
    """
    rt = current()
    span = rt.trace.open("verify", target=type(receipt.action).__name__, input=receipt.status)
    if receipt.status in ("rejected", "failed"):
        verdict = Verdict("fails", (f"receipt status {receipt.status}: {receipt.error}",))
    else:
        observation = observe()
        if isinstance(observation, Unknown):
            verdict = Verdict("unknown", (f"observation unavailable: {observation.reason}",))
        else:
            ok = expect(observation)
            if isinstance(ok, Unknown):
                verdict = Verdict("unknown", (ok.reason,), (observation,))
            else:
                verdict = Verdict("holds" if ok else "fails", ("postcondition observed" if ok else "postcondition not observed",), (observation,))
    return span.close(verdict, "answer" if verdict.status != "unknown" else "unknown")


# ----------------------------------------------------------------- rewrite


def propose(state: Any, *, goal: Any, base_revision: int, **params: Any) -> Patch | Unknown:
    """Propose a change to structured state. The result is inert until applied."""
    request = Request("propose", state, goal, {"base_revision": base_revision, **params})
    return current().call(request, validate=lambda v: isinstance(v, Patch) and v.base_revision == base_revision).value
