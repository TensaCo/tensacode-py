from dataclasses import dataclass

import pytest

from tensorcode.runtime import ActionLoop, ActionOutcome


@dataclass(frozen=True)
class Choice:
    value: str | None
    abstained: bool = False


def test_action_loop_validates_exact_choice_before_running_an_effect():
    effects = []
    loop = ActionLoop(
        chooser=lambda request, *, context=None: Choice(" SEND "),
        actions={"send": lambda state: effects.append(state)},
        max_steps=2,
    )

    with pytest.raises(ValueError, match="not one of the supplied actions"):
        loop("draft")

    assert effects == []


def test_action_loop_abstention_executes_nothing():
    effects = []
    loop = ActionLoop(
        chooser=lambda request, *, context=None: Choice(None, abstained=True),
        actions={"send": lambda state: effects.append(state)},
        max_steps=2,
    )

    result = loop("draft")

    assert result.stop_reason == "abstained"
    assert result.state == "draft"
    assert result.receipts == ()
    assert effects == []


def test_action_loop_stops_at_budget_and_returns_effect_receipts():
    calls = []

    def advance(state):
        next_state = state + 1
        calls.append(next_state)
        return ActionOutcome(next_state, receipt={"observed_state": next_state})

    loop = ActionLoop(
        chooser=lambda request, *, context=None: Choice("advance"),
        actions={"advance": advance},
        max_steps=2,
    )

    result = loop(0)

    assert result.stop_reason == "budget_exhausted"
    assert result.state == 2
    assert calls == [1, 2]
    assert [receipt.action for receipt in result.receipts] == ["advance", "advance"]
    assert [receipt.effect for receipt in result.receipts] == [
        {"observed_state": 1},
        {"observed_state": 2},
    ]


def test_action_loop_honors_explicit_completion():
    loop = ActionLoop(
        chooser=lambda request, *, context=None: "finish",
        actions={
            "finish": lambda state: ActionOutcome(
                state="done", receipt="effect-17", done=True
            )
        },
        max_steps=3,
    )

    result = loop("ready")

    assert result.stop_reason == "completed"
    assert result.state == "done"
    assert result.receipts[0].effect == "effect-17"
    assert result.receipts[0].step == 0


def test_action_loop_requires_structured_action_outcome():
    loop = ActionLoop(
        chooser=lambda request, *, context=None: "bad",
        actions={"bad": lambda state: "unverifiable effect"},
        max_steps=1,
    )

    with pytest.raises(TypeError, match="ActionOutcome"):
        loop(None)
