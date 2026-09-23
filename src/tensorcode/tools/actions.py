"""Explicit action callback records and bounded orchestration construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActionRequest:
    """What a chooser sees: current state, allowed action names, step index and prior receipts."""
    state: Any
    options: tuple[str, ...]
    step: int
    receipts: tuple[ActionReceipt, ...]


@dataclass(frozen=True)
class ActionOutcome:
    """Returned by an action: the next state, an effect receipt, and whether the run is done."""
    state: Any
    receipt: Any
    done: bool = False


@dataclass(frozen=True)
class ActionReceipt:
    """Snapshot of one executed action and the receipt it returned."""
    step: int
    action: str
    effect: Any


@dataclass(frozen=True)
class ActionLoopResult:
    """Final state, receipts, and ``stop_reason`` (completed, abstained or budget_exhausted)."""
    state: Any
    receipts: tuple[ActionReceipt, ...]
    stop_reason: str


def action_loop(*, chooser, actions, max_steps):
    """Construct a bounded loop without invoking the chooser or any action.

    Choosers receive ActionRequest and keyword context and return an action
    name (or a result with ``value``/``abstained``); actions receive state and
    return ActionOutcome. Calling the returned loop as ``loop(state, context=...)``
    authorizes execution and returns an ActionLoopResult.
    """
    from .._internal.execution.action_loop import ActionLoop
    return ActionLoop(chooser=chooser, actions=actions, max_steps=max_steps)


__all__ = ['ActionRequest', 'ActionOutcome', 'ActionReceipt', 'ActionLoopResult', 'action_loop']
