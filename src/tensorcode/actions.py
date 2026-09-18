"""Effects: registered actions, invocation receipts, and runnable plans.

* An action is a typed value whose class declares its effect semantics.
* ``invoke`` runs one action through an executor and returns a ``Receipt``.
  It does not retry: whether a retry is safe depends on facts the caller must
  weigh (idempotency, observed state), which is the recovery agent's job.
* A ``Plan`` is data. ``plan_order`` checks its structure (ids, registered actions, dependencies) and gives an execution order.

Nothing here makes a physical effect exactly-once. Idempotency keys make
retries safe only when the executor honors them; otherwise, reconcile by
observation before retrying.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, Sequence

from .outcomes import Receipt, Unknown, Verdict
from .runtime import current


@dataclass(frozen=True)
class ActionSpec:
    name: str
    effect: Literal["read", "write", "external"]
    idempotent: bool  # the *target system* treats repeats with the same key as one effect
    reversible: bool = False


def action(*, effect: Literal["read", "write", "external"], idempotent: bool, reversible: bool = False) -> Callable[[type], type]:
    def mark(cls: type) -> type:
        cls.__action__ = ActionSpec(cls.__name__, effect, idempotent, reversible)  # type: ignore[attr-defined]
        return cls

    return mark


def spec_of(act: Any) -> ActionSpec | None:
    return getattr(type(act), "__action__", None)


class Executor(Protocol):
    def execute(self, act: Any, *, key: str | None) -> Receipt: ...


def invoke(act: Any, *, executor: Executor, key: str | None) -> Receipt:
    rt = current()
    spec = spec_of(act)
    span = rt.trace.open("invoke", target=type(act).__name__, input=act)
    span.labels["key"] = key
    if spec is None:
        receipt = Receipt(act, "rejected", error="not a registered action")
    elif spec.effect != "read" and key is None:
        receipt = Receipt(act, "rejected", error="write actions require an idempotency key")
    else:
        try:
            receipt = executor.execute(act, key=key)
        except Exception as exc:  # noqa: BLE001 - an exception after dispatch may still have had an effect
            receipt = Receipt(act, "indeterminate", retryable=False, idempotency_key=key, error=f"{type(exc).__name__}: {exc}")
    return span.close(receipt, "answer")


# ------------------------------------------------------------------ plans


@dataclass(frozen=True)
class Step:
    id: str
    action: Any
    needs: tuple[str, ...] = ()  # execution dependencies, not world relationships


@dataclass(frozen=True)
class Plan:
    steps: tuple[Step, ...]
    rationale: str = ""


@dataclass(frozen=True)
class RunnablePlan:
    """A plan whose structure checks out, with an execution order. No permissions involved."""

    plan: Plan
    order: tuple[str, ...] = field(default=())


def plan_order(plan: Plan) -> RunnablePlan | Verdict:
    """Check only what execution needs: unique ids, registered actions, resolvable dependencies, no cycle."""
    reasons = []
    ids = [s.id for s in plan.steps]
    if len(set(ids)) != len(ids):
        reasons.append("duplicate step ids")
    for s in plan.steps:
        if spec_of(s.action) is None:
            reasons.append(f"{s.id}: not a registered action")
        for need in s.needs:
            if need not in ids:
                reasons.append(f"{s.id}: depends on unknown step {need}")
    order = _topological(plan.steps) if not reasons else None
    if order is None and not reasons:
        reasons.append("dependency cycle")
    if reasons:
        return Verdict("fails", tuple(reasons))
    return RunnablePlan(plan, tuple(order or ()))


def _topological(steps: Sequence[Step]) -> list[str] | None:
    remaining = {s.id: set(s.needs) for s in steps}
    order: list[str] = []
    while remaining:
        ready = sorted(k for k, deps in remaining.items() if not deps)
        if not ready:
            return None
        for k in ready:
            order.append(k)
            del remaining[k]
        for deps in remaining.values():
            deps.difference_update(ready)
    return order


def run_plan(runnable: RunnablePlan | Plan, *, executor: Executor, key_prefix: str) -> dict[str, Receipt | Unknown]:
    """Execute in dependency order. A step runs only if everything it needs was ``applied``."""
    authorized = runnable if isinstance(runnable, RunnablePlan) else plan_order(runnable)
    if isinstance(authorized, Verdict):
        return {s.id: Unknown("not_run", "; ".join(authorized.reasons)) for s in runnable.steps}  # type: ignore[union-attr]
    by_id = {s.id: s for s in authorized.plan.steps}
    results: dict[str, Receipt | Unknown] = {}
    for step_id in authorized.order:
        step = by_id[step_id]
        blocked = [n for n in step.needs if not (isinstance(results[n], Receipt) and results[n].status == "applied")]  # type: ignore[union-attr]
        if blocked:
            results[step_id] = Unknown("not_run", f"dependencies not applied: {blocked}")
            continue
        results[step_id] = invoke(step.action, executor=executor, key=f"{key_prefix}:{step_id}")
    return results
