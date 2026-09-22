"""Explicit action callback records and bounded orchestration construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActionRequest:
    state: Any
    options: tuple[str, ...]
    step: int
    receipts: tuple[ActionReceipt, ...]


@dataclass(frozen=True)
class ActionOutcome:
    state: Any
    receipt: Any
    done: bool = False


@dataclass(frozen=True)
class ActionReceipt:
    step: int
    action: str
    effect: Any


@dataclass(frozen=True)
class ActionLoopResult:
    state: Any
    receipts: tuple[ActionReceipt, ...]
    stop_reason: str


def action_loop(*, chooser, actions, max_steps):
    """Construct a bounded loop without invoking the chooser or any action.

    Choosers receive ActionRequest and keyword context; actions receive state
    and return ActionOutcome. Calling the returned loop authorizes execution.
    """
    from .._internal.execution.action_loop import ActionLoop
    return ActionLoop(chooser=chooser, actions=actions, max_steps=max_steps)


__all__ = ['ActionRequest', 'ActionOutcome', 'ActionReceipt', 'ActionLoopResult', 'action_loop']
