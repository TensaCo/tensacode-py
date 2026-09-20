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


def select_unique_fixture_goal(group):
    """Explicit test policy for one supplied goal, never lexical order or a prior.

    Tests opt into this separately from sentence selection. Competing goals or
    unresolved search rows remain undecided; this is authored authorization for
    downstream mechanisms, not evidence of inferred intent.
    """
    from tensorcode.language.verbnet import GoalProposal

    proposals = [candidate for candidate in group.candidates
                 if isinstance(candidate.payload, GoalProposal)]
    selected = proposals[0].id if len(proposals) == len(group.candidates) == 1 else None
    return InterpretationDecision(
        selected, 'test explicitly authorizes the sole supplied goal' if selected else
        'test has not supplied a choice among competing or unresolved goals',
        compared_revision=group.revision,
        compared_candidate_ids=tuple(candidate.id for candidate in group.candidates),
    )


def supplied_goal_batch(goal, *, frame):
    """Supply an authored lexical-shaped projection for a known structured goal.

    Tests separately install a refiner when they require a GoalSpec. This helper
    does not broaden the production GoalProposal contract to accept other types.
    """
    from tensorcode.language.verbnet import Goal, GoalCandidates, GoalProposal

    lexical = Goal(frame.predicate, 'authored:test-goal', goal.conditions, frame)
    return GoalCandidates((GoalProposal(lexical, ()),))


def fixture_goal_selector(verb_class, *, frame_index=None):
    """Choose only a unique proposal matching an explicitly named fixture derivation.

    Class/frame indices are authored test expectations, never an execution policy
    inferred from corpus order. Competing matching proposals remain unresolved.
    """
    from tensorcode.language.verbnet import GoalProposal

    def select(group):
        matches = [candidate for candidate in group.candidates
                   if isinstance(candidate.payload, GoalProposal)
                   and any(derivation.verb_class == verb_class
                           and (frame_index is None or derivation.frame_index == frame_index)
                           for derivation in candidate.payload.derivations)]
        return InterpretationDecision(
            matches[0].id if len(matches) == 1 else None,
            f'authored test chooses unique {verb_class} frame {frame_index!r} projection',
            compared_revision=group.revision,
            compared_candidate_ids=tuple(candidate.id for candidate in group.candidates),
        )
    return select
