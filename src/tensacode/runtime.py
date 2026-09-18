"""Binding operations to implementations, and recording what happened.

An operation call is a ``Request``. Policy decides which implementations are
acceptable (hard constraints) and in what order to try them (cascade). Each
implementation may answer, abstain (``Unknown``), or fail. Every attempt is
recorded in the active ``Trace``.

Unknown quantities stay unknown: a missing cost is never treated as zero, a
missing quality estimate is never treated as good, and a cost cap excludes
implementations whose cost is unknown.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Literal, Mapping, Protocol, Sequence, runtime_checkable

from .outcomes import Score, Unknown

# ----------------------------------------------------------------- contracts


@dataclass(frozen=True)
class Request:
    op: str  # "parse", "classify", "choose", "rank", "check", "propose"
    subject: Any  # the primary input
    target: Any = None  # output type, label set, objective, ...
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Output:
    value: Any  # an answer or an Unknown (abstention)
    score: Score | None = None
    usd: float | None = None  # metered cost of producing this output, if the backend knows it


@dataclass(frozen=True)
class Failure:
    error: str


@dataclass(frozen=True)
class Traits:
    """Hard, declared properties used for filtering, never for ranking."""

    locality: Literal["in_process", "local_service", "remote"] = "in_process"
    egress: bool = False  # request data leaves this machine
    deterministic: bool = True  # same request, same output (enables caching)
    requires: frozenset[str] = frozenset()  # e.g. {"cuda"}, {"sklearn"}


@dataclass(frozen=True)
class Profile:
    """Measured properties. ``None`` means unmeasured, not zero and not good."""

    source: str | None = None  # the evaluation artifact these numbers came from
    quality: Mapping[str, float] = field(default_factory=dict)
    latency_ms_p50: float | None = None
    latency_ms_p95: float | None = None
    usd_per_call: float | None = None
    peak_memory_mb: float | None = None


@runtime_checkable
class Implementation(Protocol):
    name: str
    version: str
    op: str
    traits: Traits
    profile: Profile

    def accepts(self, request: Request) -> bool: ...

    def run(self, requests: Sequence[Request]) -> Sequence[Output | Failure]: ...


@dataclass
class FunctionImplementation:
    """Adapts a plain function ``request -> value | Unknown | Output`` (or a batch function)."""

    op: str
    name: str
    version: str
    fn: Callable[..., Any]
    accepts_fn: Callable[[Request], bool] = lambda request: True
    traits: Traits = Traits()
    profile: Profile = Profile()
    batched: bool = False

    def accepts(self, request: Request) -> bool:
        return request.op == self.op and self.accepts_fn(request)

    def run(self, requests: Sequence[Request]) -> list[Output | Failure]:
        raw = self.fn(list(requests)) if self.batched else [self.fn(r) for r in requests]
        return [r if isinstance(r, (Output, Failure)) else Output(r) for r in raw]


def implementation(
    op: str,
    *,
    name: str,
    version: str,
    accepts: Callable[[Request], bool] = lambda request: True,
    traits: Traits = Traits(),
    profile: Profile = Profile(),
    batched: bool = False,
) -> Callable[[Callable[..., Any]], FunctionImplementation]:
    def wrap(fn: Callable[..., Any]) -> FunctionImplementation:
        return FunctionImplementation(op, name, version, fn, accepts, traits, profile, batched)

    return wrap


# -------------------------------------------------------------------- policy


@dataclass(frozen=True)
class Policy:
    localities: frozenset[str] = frozenset({"in_process", "local_service"})
    allow_egress: bool = False
    available: frozenset[str] = frozenset()  # capabilities present on this host
    max_attempts: int = 3  # implementations that may *run* per item
    deadline_ms: float | None = None  # per operation call
    max_usd_per_call: float | None = None
    order: Literal["declared", "cheapest", "fastest"] = "declared"
    cache: bool = True
    record_inputs: Literal["full", "digest"] = "full"


class Budget:
    """A shared allowance across many calls (e.g. one agent episode)."""

    def __init__(self, *, usd: float | None = None, attempts: int | None = None, seconds: float | None = None):
        self.usd, self.attempts = usd, attempts
        self.deadline = time.monotonic() + seconds if seconds is not None else None
        self.spent_usd = 0.0
        self.spent_attempts = 0
        self.unmetered_calls = 0  # calls whose cost is unknown; reported, never summed as zero

    def refusal(self, profile: Profile) -> str | None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            return "budget deadline passed"
        if self.attempts is not None and self.spent_attempts >= self.attempts:
            return "budget attempts exhausted"
        if self.usd is not None:
            if profile.usd_per_call is None:
                return "cost unknown under a cost cap"
            if self.spent_usd + profile.usd_per_call > self.usd:
                return "cost would exceed budget"
        return None


def _hard_constraint_violation(impl: Implementation, policy: Policy) -> str | None:
    t = impl.traits
    if t.locality not in policy.localities:
        return f"locality {t.locality} not permitted"
    if t.egress and not policy.allow_egress:
        return "data egress not permitted"
    if missing := t.requires - policy.available:
        return f"missing capability {sorted(missing)}"
    if policy.max_usd_per_call is not None:
        if impl.profile.usd_per_call is None:
            return "cost unknown under a per-call cost cap"
        if impl.profile.usd_per_call > policy.max_usd_per_call:
            return "per-call cost above cap"
    return None


def _ordered(candidates: list[Implementation], policy: Policy) -> list[Implementation]:
    if policy.order == "declared":
        return candidates
    key = "usd_per_call" if policy.order == "cheapest" else "latency_ms_p50"
    # unknown values sort last; they are not assumed cheap or fast
    return sorted(candidates, key=lambda i: (getattr(i.profile, key) is None, getattr(i.profile, key) or 0.0))


# --------------------------------------------------------------------- trace


@dataclass
class Attempt:
    implementation: str
    version: str
    outcome: Literal["answer", "abstain", "invalid", "error", "skipped", "cache_hit"]
    reason: str = ""
    backend_ms: float | None = None
    usd: float | None = None
    usd_basis: Literal["metered", "profile", "unknown", "none"] = "none"


@dataclass
class Span:
    id: str
    op: str
    target: str
    input: Any
    input_digest: str
    output: Any = None
    outcome: Literal["answer", "unknown", "event"] = "event"
    attempts: list[Attempt] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    started_at: str = ""
    total_ms: float = 0.0
    backend_ms: float = 0.0
    parent: str | None = None
    labels: dict[str, Any] = field(default_factory=dict)
    t0: float = field(default_factory=time.perf_counter, repr=False)

    def close(self, output: Any, outcome: Literal["answer", "unknown", "event"]) -> Any:
        self.output, self.outcome = output, outcome
        self.total_ms = (time.perf_counter() - self.t0) * 1e3
        return output

    @property
    def answered_by(self) -> str | None:
        for a in self.attempts:
            if a.outcome in ("answer", "cache_hit"):
                return f"{a.implementation}@{a.version}"
        return None


class Trace:
    def __init__(self) -> None:
        self.spans: list[Span] = []
        self._parent: contextvars.ContextVar[str | None] = contextvars.ContextVar("parent", default=None)

    @contextlib.contextmanager
    def section(self, name: str, **labels: Any) -> Iterator[Span]:
        span = self.open(name, target="", input=None, labels=labels)
        token = self._parent.set(span.id)
        t0 = time.perf_counter()
        try:
            yield span
        finally:
            span.total_ms = (time.perf_counter() - t0) * 1e3
            self._parent.reset(token)

    def open(self, op: str, *, target: str, input: Any, labels: Mapping[str, Any] | None = None, digest: str = "") -> Span:
        span = Span(
            id=uuid.uuid4().hex[:12],
            op=op,
            target=target,
            input=input,
            input_digest=digest,
            started_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            parent=self._parent.get(),
            labels=dict(labels or {}),
        )
        self.spans.append(span)
        return span

    def of(self, op: str) -> list[Span]:
        return [s for s in self.spans if s.op == op]

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(asdict(s), default=_preview) for s in self.spans)

    def render(self, *, since: int = 0) -> str:
        """One line per span, indented under its section, attempts in order."""
        depth: dict[str | None, int] = {None: 0}
        lines = []
        for s in self.spans[since:]:
            d = depth.get(s.parent, 0)
            depth[s.id] = d + 1
            pad = "  " * d
            if s.outcome == "event" and not s.attempts and s.op not in ("invoke", "verify", "choose"):
                labels = " ".join(f"{k}={v}" for k, v in s.labels.items())
                lines.append(f"{pad}{s.op} {labels}".rstrip())
                continue
            out = _preview(s.output, 70)
            lines.append(f"{pad}{s.op:<8} -> {out}  [{s.total_ms:.2f} ms total, {s.backend_ms:.2f} ms backend]")
            if s.labels:
                lines.append(f"{pad}  · " + " ".join(f"{k}={v}" for k, v in s.labels.items()))
            for a in s.attempts:
                cost = "" if a.usd_basis in ("none", "profile") and not a.usd else f", usd={'?' if a.usd is None else a.usd} ({a.usd_basis})"
                reason = f": {a.reason}" if a.reason else ""
                lines.append(f"{pad}  - {a.implementation}@{a.version} {a.outcome}{reason}{cost}")
            for n in s.notes:
                lines.append(f"{pad}  · {n}")
        return "\n".join(lines)


def _preview(value: Any, limit: int = 200) -> Any:
    if isinstance(value, Unknown):
        return {"unknown": value.reason, "detail": value.detail}
    if is_dataclass(value) and not isinstance(value, type):
        text = repr(value)
    elif isinstance(value, type):
        text = value.__qualname__
    else:
        text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def digest(value: Any) -> str:
    try:
        from .records import _canonical

        data = json.dumps(_canonical(value), sort_keys=True)
    except Exception:  # noqa: BLE001 - unhashable inputs are digested by repr, and flagged
        data = "repr:" + repr(value)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


# ------------------------------------------------------------------- runtime


class Runtime:
    def __init__(
        self,
        implementations: Sequence[Implementation] = (),
        *,
        policy: Policy = Policy(),
        budget: Budget | None = None,
        trace: Trace | None = None,
    ) -> None:
        self.implementations = list(implementations)
        self.policy = policy
        self.budget = budget
        self.trace = trace or Trace()
        self._cache: dict[tuple[str, str, str], Output] = {}

    def call(self, request: Request, *, validate: Callable[[Any], bool] = lambda v: True, notes: Sequence[str] = ()) -> Output:
        return self.call_many([request], validate=validate, notes=notes)[0]

    def call_many(
        self, requests: Sequence[Request], *, validate: Callable[[Any], bool] = lambda v: True, notes: Sequence[str] = ()
    ) -> list[Output]:
        t_start = time.perf_counter()
        n = len(requests)
        digests = [digest((r.op, r.subject, _preview(r.target), r.params)) for r in requests]
        spans = [
            self.trace.open(
                r.op,
                target=_preview(r.target),
                input=r.subject if self.policy.record_inputs == "full" else None,
                digest=d,
            )
            for r, d in zip(requests, digests)
        ]
        for s in spans:
            s.notes.extend(notes)
        results: list[Output | None] = [None] * n
        abstentions: list[list[tuple[Any, Score]]] = [[] for _ in range(n)]
        ran = [0] * n
        backend_ms = [0.0] * n

        candidates = [impl for impl in self.implementations if all(impl.accepts(r) for r in requests)]
        if not candidates:
            for s in spans:
                s.notes.append(f"no implementation accepts op={requests[0].op} target={_preview(requests[0].target)}")

        for impl in _ordered(candidates, self.policy):
            pending = [i for i in range(n) if results[i] is None and ran[i] < self.policy.max_attempts]
            if not pending:
                break
            refusal = _hard_constraint_violation(impl, self.policy)
            if refusal is None and self.policy.deadline_ms is not None:
                if (time.perf_counter() - t_start) * 1e3 >= self.policy.deadline_ms:
                    refusal = "deadline passed"
            if refusal is None and self.budget is not None:
                refusal = self.budget.refusal(impl.profile)
            if refusal:
                for i in pending:
                    spans[i].attempts.append(Attempt(impl.name, impl.version, "skipped", refusal))
                continue

            to_run = []
            for i in pending:
                key = (impl.name, impl.version, digests[i])
                if self.policy.cache and impl.traits.deterministic and key in self._cache:
                    cached = self._cache[key]
                    if isinstance(cached.value, Unknown):
                        spans[i].attempts.append(Attempt(impl.name, impl.version, "abstain", f"cached: {cached.value.reason}"))
                    else:
                        results[i] = cached
                        spans[i].attempts.append(Attempt(impl.name, impl.version, "cache_hit"))
                else:
                    to_run.append(i)
            if not to_run:
                continue

            t0 = time.perf_counter()
            try:
                outs = list(impl.run([requests[i] for i in to_run]))
                if len(outs) != len(to_run):
                    raise RuntimeError(f"{impl.name} returned {len(outs)} results for {len(to_run)} requests")
            except Exception as exc:  # noqa: BLE001 - a crashing backend is an attempt outcome, not a program crash
                outs = [Failure(f"{type(exc).__name__}: {exc}")] * len(to_run)
            per_item_ms = (time.perf_counter() - t0) * 1e3 / len(to_run)
            if self.budget is not None:
                self.budget.spent_attempts += 1

            for i, out in zip(to_run, outs):
                ran[i] += 1
                backend_ms[i] += per_item_ms
                usd, basis = self._charge(impl, out)
                attempt = Attempt(impl.name, impl.version, "error", backend_ms=per_item_ms, usd=usd, usd_basis=basis)
                spans[i].attempts.append(attempt)
                if isinstance(out, Failure):
                    attempt.reason = out.error
                    continue
                if isinstance(out.value, Unknown):
                    attempt.outcome, attempt.reason = "abstain", out.value.reason
                    abstentions[i].extend(out.value.candidates)
                elif not validate(out.value):
                    attempt.outcome, attempt.reason = "invalid", f"output violates contract: {_preview(out.value, 80)}"
                    continue  # invalid outputs are never cached
                else:
                    attempt.outcome = "answer"
                    results[i] = out
                if self.policy.cache and impl.traits.deterministic:
                    self._cache[(impl.name, impl.version, digests[i])] = out

        total_ms = (time.perf_counter() - t_start) * 1e3 / n
        final: list[Output] = []
        for i in range(n):
            out = results[i]
            if out is None:
                reasons = [f"{a.implementation}: {a.outcome} ({a.reason})" for a in spans[i].attempts] or spans[i].notes
                ran = [a for a in spans[i].attempts if a.outcome != "skipped"]
                if ran:  # the last real attempt's reason is the most specific explanation
                    reason = ran[-1].reason.removeprefix("cached: ") if ran[-1].outcome == "abstain" else f"{ran[-1].outcome}"
                else:
                    reason = "not_permitted" if spans[i].attempts else "no_implementation"
                out = Output(Unknown(reason, "; ".join(reasons), tuple(abstentions[i])))
            spans[i].output = out.value
            spans[i].outcome = "unknown" if isinstance(out.value, Unknown) else "answer"
            spans[i].total_ms, spans[i].backend_ms = total_ms, backend_ms[i]
            final.append(out)
        return final

    def _charge(self, impl: Implementation, out: Output | Failure) -> tuple[float | None, Literal["metered", "profile", "unknown"]]:
        if isinstance(out, Output) and out.usd is not None:
            usd, basis = out.usd, "metered"
        elif impl.profile.usd_per_call is not None:
            usd, basis = impl.profile.usd_per_call, "profile"
        else:
            usd, basis = None, "unknown"
        if self.budget is not None:
            if usd is None:
                self.budget.unmetered_calls += 1
            else:
                self.budget.spent_usd += usd
        return usd, basis  # type: ignore[return-value]


_current: contextvars.ContextVar[Runtime | None] = contextvars.ContextVar("tensacode_runtime", default=None)


def current() -> Runtime:
    """The bound runtime, or an empty one: with nothing bound, inferential ops return Unknown."""
    return _current.get() or Runtime()


@contextlib.contextmanager
def use(runtime: Runtime) -> Iterator[Runtime]:
    """Bind a runtime for the enclosed code. Programs never construct one themselves."""
    token = _current.set(runtime)
    try:
        yield runtime
    finally:
        _current.reset(token)
