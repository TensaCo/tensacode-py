"""A bounded chooser/effect loop with explicit receipts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
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


class ActionLoop:
    """Choose only from supplied names and execute at most ``max_steps``."""

    def __init__(
        self,
        *,
        chooser: Callable[..., Any],
        actions: Mapping[str, Callable[[Any], ActionOutcome]],
        max_steps: int,
    ) -> None:
        if not callable(chooser):
            raise TypeError("chooser must be callable")
        if not isinstance(max_steps, int) or max_steps < 0:
            raise ValueError("max_steps must be a non-negative integer")
        copied_actions = dict(actions)
        if not all(isinstance(name, str) and name for name in copied_actions):
            raise ValueError("action names must be non-empty strings")
        if not all(callable(action) for action in copied_actions.values()):
            raise TypeError("every action must be callable")
        self.chooser = chooser
        self.actions = MappingProxyType(copied_actions)
        self.max_steps = max_steps

    def __call__(self, state: Any, *, context=None) -> ActionLoopResult:
        receipts: list[ActionReceipt] = []
        options = tuple(self.actions)
        for step in range(self.max_steps):
            request = ActionRequest(state, options, step, tuple(receipts))
            choice = self.chooser(request, context=context)
            selected = self._selected_name(choice)
            if selected is None:
                return ActionLoopResult(state, tuple(receipts), "abstained")
            if selected not in self.actions:
                raise ValueError(
                    f"chosen action {selected!r} is not one of the supplied actions"
                )
            outcome = self.actions[selected](state)
            if not isinstance(outcome, ActionOutcome):
                raise TypeError("actions must return ActionOutcome with an effect receipt")
            receipt = ActionReceipt(step, selected, outcome.receipt)
            receipts.append(receipt)
            state = outcome.state
            if outcome.done:
                return ActionLoopResult(state, tuple(receipts), "completed")
        return ActionLoopResult(state, tuple(receipts), "budget_exhausted")

    @staticmethod
    def _selected_name(choice: Any) -> str | None:
        if isinstance(choice, str):
            return choice
        abstained = getattr(choice, "abstained", False)
        if not isinstance(abstained, bool):
            raise TypeError("chooser abstained flag must be boolean")
        selected = getattr(choice, "value", None)
        if abstained:
            return None
        if not isinstance(selected, str):
            raise TypeError("chooser must return a string or a result with string value")
        return selected
