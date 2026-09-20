"""Complete authenticated task-set snapshots for learned association."""

from copy import deepcopy
from dataclasses import replace
import importlib

from tensorcode.agent import Agent, InterpretationDecision
from tensorcode.agent.goal_interpretation import (
    retain_taught_goal,
    retain_taught_sentence_goal,
    select_goal,
)
from tensorcode.agent.task_dependencies import capture_dependency
from tensorcode.agent.understand import Act, SentenceAlternative
from tensorcode.goals import Condition, GoalSpec
from tensorcode.language import Frame
from tensorcode.language.semantics import Request
from tensorcode.outcomes import Unknown
from tensorcode.records import Ref


def api():
    return importlib.import_module("tensorcode.agent.task_association_context")


def selected_reading(agent, text, frames):
    workspace = agent.interpretations
    source = workspace.add_source(text, provider="authored complete sentence fixture")
    group = workspace.create_group(source.id)
    acts = tuple(Act("request", Request(frame), frame) for frame in frames)
    candidate = workspace.propose(
        group.id,
        SentenceAlternative(None, acts, provenance="supplied structural fixture"),
    )
    workspace.select(group.id, candidate.id, reason="explicit fixture interpretation")
    dependency = capture_dependency(workspace, group.id, basis=("selected complete fixture reading",))
    return group, candidate, dependency


def supported_task(agent, name, *, shared_goal=None):
    item = Ref(f"item:{name}")
    protected = Ref(f"protected:{name}")
    frame = Frame("prepare", {"object": item, "protected": protected})
    goal = shared_goal or GoalSpec(
        (Condition("prepared", {"item": item}),),
        invariants=(Condition("unchanged", {"item": protected}),),
    )
    reading, _, parent = selected_reading(agent, f"prepare {name}", (frame,))
    goal_group_id = retain_taught_goal(
        agent,
        frame,
        goal,
        f"explicit goal for {name}",
        parent_dependency=parent,
        reason="fixture supplies the task goal",
    )
    group = agent.interpretations.get(goal_group_id)
    resolution = select_goal(
        agent,
        goal_group_id,
        decision=InterpretationDecision(
            group.candidates[0].id,
            "explicitly select supplied goal",
            compared_revision=group.revision,
            compared_candidate_ids=tuple(candidate.id for candidate in group.candidates),
        ),
    )
    assert not isinstance(resolution.goal, Unknown)
    dependencies = tuple(dict.fromkeys((parent, *resolution.supporting_dependencies, resolution.dependency)))
    task = agent.tasks.create(
        f"prepare {name}",
        resolution.goal,
        dependencies=dependencies,
        goal_interpretation_id=resolution.group_id,
    )
    return task, reading


def incoming(agent):
    frames = (
        Frame("change", {"destination": Ref("place:new")}),
        Frame("preserve", {"item": Ref("protected:earlier")}, {"qualifier": "existing"}),
    )
    group, candidate, _ = selected_reading(agent, "change it but preserve the existing item", frames)
    return group, candidate, frames


def capture(agent):
    group, candidate, frames = incoming(agent)
    context = api().capture_task_association_context(
        agent,
        group.id,
        candidate.id,
        basis=("explicit association comparison",),
    )
    assert not isinstance(context, Unknown), context
    return context, frames


def test_captures_complete_incoming_reading_and_every_task_identity():
    agent = Agent()
    earlier, _ = supported_task(agent, "earlier")
    later, _ = supported_task(agent, "later")

    context, frames = capture(agent)

    assert context.incoming_frames == frames
    assert context.ledger_basis == ((earlier.id, 1), (later.id, 1))
    assert tuple(member.task_id for member in context.members) == (earlier.id, later.id)
    assert all(member.unresolved_reason is None for member in context.members)
    assert tuple(member.origin_frames[0].roles["object"] for member in context.members) == (
        Ref("item:earlier"),
        Ref("item:later"),
    )
    assert api().validate_task_association_context(agent, context) is True


def test_missing_origin_stays_an_explicit_unresolved_rival():
    agent = Agent()
    supported, _ = supported_task(agent, "supported")
    unsupported = agent.tasks.create(
        "caller-authored task without retained origin",
        GoalSpec((Condition("prepared", {"item": Ref("item:unsupported")}),)),
    )

    context, _ = capture(agent)

    assert tuple(member.task_id for member in context.members) == (supported.id, unsupported.id)
    unresolved = context.members[1]
    assert unresolved.origin is None and unresolved.origin_dependency is None
    assert unresolved.origin_frames == ()
    assert "origin" in unresolved.unresolved_reason
    assert api().validate_task_association_context(agent, context) is True


def test_authentic_dependencies_cannot_link_a_different_originating_goal():
    agent = Agent()
    authentic, _ = supported_task(agent, "authentic")
    forged = agent.tasks.create(
        "caller reuses authentic provenance for a different goal",
        GoalSpec((Condition("different", {"item": Ref("item:forged")}),)),
        dependencies=authentic.dependencies,
        goal_interpretation_id=authentic.goal_interpretation_id,
    )

    context, _ = capture(agent)
    member = next(member for member in context.members if member.task_id == forged.id)

    assert member.origin is None
    assert "goal" in member.unresolved_reason


def test_identical_task_contents_do_not_collapse_distinct_identities():
    agent = Agent()
    shared = GoalSpec((Condition("prepared", {"item": Ref("item:shared")}),))
    first, _ = supported_task(agent, "first", shared_goal=shared)
    second, _ = supported_task(agent, "second", shared_goal=shared)

    context, _ = capture(agent)

    assert len(context.members) == 2
    assert context.members[0].goal == context.members[1].goal
    assert context.members[0].task_id == first.id
    assert context.members[1].task_id == second.id


def test_new_task_invalidates_live_context_but_not_registered_history():
    agent = Agent()
    supported_task(agent, "original")
    context, _ = capture(agent)

    supported_task(agent, "later teaching episode")

    assert isinstance(api().validate_task_association_context(agent, context), Unknown)
    assert api().validate_task_association_context(agent, context, historical=True) is True


def test_rival_revision_invalidates_live_context_but_preserves_historical_revision():
    agent = Agent()
    supported_task(agent, "target")
    rival, _ = supported_task(agent, "rival")
    context, _ = capture(agent)

    agent.tasks.revise(
        rival.id,
        GoalSpec((Condition("prepared-elsewhere", {"item": Ref("item:rival")}),)),
        reason="new explicit rival constraint",
    )

    assert isinstance(api().validate_task_association_context(agent, context), Unknown)
    assert api().validate_task_association_context(agent, context, historical=True) is True


def test_withdrawn_origin_support_invalidates_live_and_historical_context():
    agent = Agent()
    task, origin_group = supported_task(agent, "target")
    context, _ = capture(agent)

    agent.interpretations.unset(origin_group.id, reason="withdraw originating reading")

    assert isinstance(api().validate_task_association_context(agent, context), Unknown)
    assert isinstance(api().validate_task_association_context(agent, context, historical=True), Unknown)
    assert agent.tasks.current_revision(task.id) == 1


def test_historical_flag_does_not_authenticate_a_caller_created_context():
    agent = Agent()
    supported_task(agent, "target")
    context, _ = capture(agent)
    forged = replace(context, context_id="association-context:caller-created")

    assert isinstance(api().validate_task_association_context(agent, forged, historical=True), Unknown)


class _ReviseDuringCopy:
    def __init__(self, callback):
        self.callback = callback
        self.armed = False

    def __deepcopy__(self, memo):
        if self.armed:
            self.armed = False
            self.callback()
        return self


def test_rival_revision_during_payload_copy_rejects_capture():
    agent = Agent()
    supported_task(agent, "target")
    holder = {}
    trigger = _ReviseDuringCopy(lambda: agent.tasks.revise(
        holder["rival"].id,
        GoalSpec((Condition("changed-during-copy", {"item": Ref("item:rival")}),)),
        reason="reentrant rival revision while copying its payload",
    ))
    rival_goal = GoalSpec((Condition("prepared", {"item": trigger}),))
    holder["rival"] = agent.tasks.create("unsupported callback-bearing rival", rival_goal)
    trigger.armed = True
    group, candidate, _ = incoming(agent)

    result = api().capture_task_association_context(
        agent,
        group.id,
        candidate.id,
        basis=("explicit association comparison",),
    )

    assert isinstance(result, Unknown)
    assert agent.tasks.current_revision(holder["rival"].id) == 2


def test_current_goal_constraints_keep_the_historical_originating_reading():
    agent = Agent()
    task, _ = supported_task(agent, "target")
    revised = GoalSpec(
        (Condition("prepared", {"item": Ref("item:target"), "destination": Ref("place:new")}),),
        invariants=task.goal.invariants,
    )
    agent.tasks.revise(task.id, revised, reason="explicit current constraint", goal_interpretation_id=None)

    context, _ = capture(agent)
    member = context.members[0]

    assert member.task_revision == 2 and member.goal == revised
    assert member.origin_frames[0].roles["object"] == Ref("item:target")
    assert member.unresolved_reason is None


def test_historical_validation_authenticates_the_origin_revision_goal():
    agent = Agent()
    task, _ = supported_task(agent, "target")
    agent.tasks.revise(
        task.id,
        GoalSpec((Condition("current", {"item": Ref("item:target")}),)),
        reason="make the originating revision historical before capture",
    )
    context, _ = capture(agent)
    stored = agent.tasks._tasks[task.id]
    changed_origin = replace(
        stored.revisions[0],
        goal=GoalSpec((Condition("forged-origin", {"item": Ref("item:other")}),)),
    )
    agent.tasks._tasks[task.id] = replace(
        stored,
        revisions=(changed_origin, *stored.revisions[1:]),
    )

    assert isinstance(
        api().validate_task_association_context(agent, context, historical=True),
        Unknown,
    )


def test_origin_comes_from_complete_parent_not_goal_projection_envelope():
    agent = Agent()
    frames = (
        Frame("prepare", {"object": Ref("item:whole")}),
        Frame("preserve", {"object": Ref("protected:whole")}),
    )
    _, _, parent = selected_reading(agent, "prepare and preserve", frames)
    goal = GoalSpec(
        (Condition("prepared", {"item": Ref("item:whole")}),),
        invariants=(Condition("unchanged", {"item": Ref("protected:whole")}),),
    )
    group_id = retain_taught_sentence_goal(
        agent,
        goal,
        "supplied complete goal",
        parent_dependency=parent,
        reason="fixture supplies the whole-request goal",
    )
    group = agent.interpretations.get(group_id)
    resolution = select_goal(agent, group_id, decision=InterpretationDecision(
        group.candidates[0].id,
        "select complete supplied goal",
        compared_revision=group.revision,
        compared_candidate_ids=tuple(candidate.id for candidate in group.candidates),
    ))
    agent.tasks.create(
        "prepare and preserve",
        resolution.goal,
        dependencies=(parent, *resolution.supporting_dependencies, resolution.dependency),
        goal_interpretation_id=group_id,
    )

    context, _ = capture(agent)

    assert context.members[0].origin_frames == frames
    goal_source = agent.interpretations.get_source(group.source_id)
    assert goal_source.payload["frame"].predicate == "request-sequence"


def test_comparison_basis_is_identity_sensitive_and_payload_callback_free():
    agent = Agent()
    first, _ = supported_task(agent, "first")
    fired = []
    trigger = _ReviseDuringCopy(lambda: fired.append(True))
    task = agent.tasks.create(
        "callback-bearing task",
        GoalSpec((Condition("prepared", {"item": trigger}),)),
    )
    trigger.armed = True

    basis = agent.tasks.comparison_basis()

    assert basis == ((first.id, 1), (task.id, 1))
    assert not fired
    assert trigger.armed
