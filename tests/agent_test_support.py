"""Explicit interpretation assumptions for downstream integration tests.

These fixtures exercise execution, stores, and realization after a supplied reading
has been chosen. Selecting reader order here is test setup, not a production
interpretation policy or evidence that language understanding is correct. Tests of
unresolved/default interpretation must construct the production Agent directly.
"""
from tensorcode.agent.core import Agent, InterpretationDecision


def select_fixture_reading(group):
    return InterpretationDecision(
        group.candidates[0].id if group.candidates else None,
        "test fixture supplies intended reader candidate",
    )


def selected_agent(*args, **kwargs):
    kwargs.setdefault("interpretation_selector", select_fixture_reading)
    return Agent(*args, **kwargs)
