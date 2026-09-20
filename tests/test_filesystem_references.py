"""Explicit resource grounding remains live through filesystem dispatch."""
from datetime import datetime, timezone

import pytest

from tensorcode.actions import Plan
from tensorcode.agent.filesystem import FileSystemPlugin
from tensorcode.agent.planning import plan_goal
from tensorcode.goals import Condition, GoalSpec
from tensorcode.outcomes import Unknown
from tensorcode.records import Evidence, Proposition, Ref


def bound(tmp_path):
    plugin = FileSystemPlugin(tmp_path)
    resource = Ref('resource:opaque')
    evidence = Evidence(Ref('source:explicit'), datetime.now(timezone.utc))
    proposition = plugin.bind_resource(resource, 'actual/destination', binding=Ref('binding:one'), evidence=evidence)
    return plugin, resource, proposition, evidence


def test_symbolic_goal_preserves_arguments_and_creates_real_content(tmp_path):
    plugin, resource, proposition, _ = bound(tmp_path)
    path = {'root': resource, 'relative': 'nested/file.txt'}
    goal = GoalSpec((Condition('directory_exists', {'path': resource}),
                     Condition('content', {'path': path, 'text': 'hello'})))
    plan = plan_goal(goal, [plugin])
    assert isinstance(plan, Plan), plan
    assert any(dict(step.action.args)['path'] == resource for step in plan.steps)
    assert any(dict(step.action.args)['path'] == path for step in plan.steps)
    for step in plan.steps:
        assert plugin.execute(step.action, key=step.id).status == 'applied'
    assert (tmp_path / 'actual/destination/nested/file.txt').read_text() == 'hello'
    assert all(plugin.observe_condition(c) is True for c in goal.conditions)
    assert proposition.roles['resource'] == resource


def test_withdrawal_blocks_retained_call_and_rebinding(tmp_path):
    plugin, resource, proposition, evidence = bound(tmp_path)
    (tmp_path / 'actual').mkdir()
    goal = GoalSpec((Condition('directory_exists', {'path': resource}),))
    plan = plan_goal(goal, [plugin])
    assert isinstance(plan, Plan)
    plugin.resources.supersede(proposition, 'withdrawn')
    assert isinstance(plugin.observe_condition(goal.conditions[0]), Unknown)
    assert isinstance(plan_goal(goal, [plugin]), Unknown)
    assert plugin.execute(plan.steps[-1].action, key='retained').status == 'rejected'
    with pytest.raises(ValueError):
        plugin.bind_resource(resource, 'elsewhere', binding=Ref('binding:two'), evidence=evidence)
    assert not (tmp_path / 'actual/destination').exists()


@pytest.mark.parametrize('change', [
    {'path': 'other'}, {'binding': Ref('binding:rival')},
])
def test_competing_record_prevents_resolution(tmp_path, change):
    plugin, resource, proposition, evidence = bound(tmp_path)
    plugin.resources.assert_(Proposition(proposition.predicate, dict(proposition.roles) | change), evidence)
    assert isinstance(plugin.observe_condition(Condition('directory_exists', {'path': resource})), Unknown)


@pytest.mark.parametrize('value', ['../escape', '/absolute', 12])
def test_structural_relative_paths_cannot_escape(tmp_path, value):
    plugin, resource, _, _ = bound(tmp_path)
    condition = Condition('file_exists', {'path': {'root': resource, 'relative': value}})
    assert isinstance(plugin.observe_condition(condition), Unknown)
    assert list(plugin.enumerate_actions(GoalSpec((condition,)))) == []


def test_symlink_change_after_binding_is_rejected(tmp_path):
    plugin, resource, _, _ = bound(tmp_path)
    (tmp_path / 'real').mkdir()
    (tmp_path / 'actual').symlink_to(tmp_path / 'real', target_is_directory=True)
    assert isinstance(plugin.observe_condition(Condition('directory_exists', {'path': resource})), Unknown)


def test_no_reference_spelling_fallback(tmp_path):
    plugin = FileSystemPlugin(tmp_path)
    (tmp_path / 'resource:existing').mkdir()
    assert isinstance(plugin.observe_condition(Condition('directory_exists', {'path': Ref('resource:existing')})), Unknown)


@pytest.mark.parametrize('qualifiers', [{'scope': Ref('scope:other')}, {'polarity': False}, {'modality': 'possible'}])
def test_qualified_binding_cannot_replace_withdrawn_binding(tmp_path, qualifiers):
    plugin, resource, proposition, evidence = bound(tmp_path)
    plugin.resources.supersede(proposition)
    plugin.resources.assert_(Proposition(proposition.predicate, proposition.roles, **qualifiers), evidence)
    assert isinstance(plugin.observe_condition(Condition('directory_exists', {'path': resource})), Unknown)


def test_unverified_derived_binding_does_not_establish_path(tmp_path):
    plugin = FileSystemPlugin(tmp_path)
    evidence = Evidence(Ref('source:derived'), datetime.now(timezone.utc), derived_from=('prop:missing',))
    with pytest.raises(ValueError):
        plugin.bind_resource(Ref('resource:opaque'), 'target', binding=Ref('binding:one'), evidence=evidence)


def test_support_withdrawn_after_plan_blocks_dispatch(tmp_path):
    plugin, resource, proposition, evidence = bound(tmp_path)
    (tmp_path / 'actual').mkdir()
    goal = GoalSpec((Condition('directory_exists', {'path': resource}),))
    plan = plan_goal(goal, [plugin])
    assert isinstance(plan, Plan)
    record = plugin.resources.propositions()[0]
    record.evidence[:] = [Evidence(evidence.source, evidence.observed_at, derived_from=('prop:missing',))]
    assert isinstance(plugin.observe_condition(goal.conditions[0]), Unknown)
    assert plugin.execute(plan.steps[-1].action, key='invalid-support').status == 'rejected'
    assert not (tmp_path / 'actual/destination').exists()


@pytest.mark.parametrize('value', [
    {'root': Ref('resource:opaque'), 'relative': 'file', 'extra': True},
    {'root': 'resource:opaque', 'relative': 'file'},
    {'root': Ref('resource:opaque')},
])
def test_malformed_symbolic_paths_are_unknown(tmp_path, value):
    plugin, _, _, _ = bound(tmp_path)
    condition = Condition('file_exists', {'path': value})
    assert isinstance(plugin.observe_condition(condition), Unknown)
    assert list(plugin.enumerate_actions(GoalSpec((condition,)))) == []


def test_withdrawal_blocks_every_ancestor_action_in_retained_plan(tmp_path):
    plugin, resource, proposition, _ = bound(tmp_path)
    goal = GoalSpec((Condition('directory_exists', {'path': resource}),))
    plan = plan_goal(goal, [plugin])
    assert isinstance(plan, Plan)
    assert len(plan.steps) == 2
    plugin.resources.supersede(proposition, 'withdrawn before any dispatch')
    for step in plan.steps:
        assert plugin.execute(step.action, key=step.id).status == 'rejected'
    assert not (tmp_path / 'actual').exists()


@pytest.mark.parametrize('kind', ['directory_exists', 'content'])
def test_withdrawal_during_final_precondition_prevents_mutation(tmp_path, kind):
    plugin, resource, proposition, _ = bound(tmp_path)
    (tmp_path / 'actual').mkdir()
    args = {'path': resource}
    if kind == 'content':
        args['text'] = 'must not be written'
    plan = plan_goal(GoalSpec((Condition(kind, args),)), [plugin])
    assert isinstance(plan, Plan)
    check = plugin.precondition_holds

    def withdraw_after_check(condition, arguments):
        result = check(condition, arguments)
        if condition.pred == 'path_exists' and condition.negated:
            plugin.resources.supersede(proposition, 'withdrawn in final precondition')
        return result

    plugin.precondition_holds = withdraw_after_check
    assert plugin.execute(plan.steps[-1].action, key='late-withdrawal').status == 'rejected'
    assert not (tmp_path / 'actual/destination').exists()


def test_final_parent_validation_cannot_withdraw_already_validated_target(tmp_path, monkeypatch):
    from tensorcode.agent import filesystem
    from tensorcode.agent.plugin import Call

    plugin, target, target_binding, evidence = bound(tmp_path)
    (tmp_path / 'actual').mkdir()
    parent = Ref('resource:parent')
    parent_binding = plugin.bind_resource(parent, 'actual', binding=Ref('binding:parent'), evidence=evidence)
    validate = filesystem.validate_record_support
    parent_checks = 0

    def revoke_target_during_last_parent_check(store, record_id):
        nonlocal parent_checks
        result = validate(store, record_id)
        if record_id == parent_binding.id:
            parent_checks += 1
            if parent_checks == 3:
                store.supersede(target_binding, 'withdrawn while validating parent')
        return result

    monkeypatch.setattr(filesystem, 'validate_record_support', revoke_target_during_last_parent_check)
    action = Call(plugin.name, 'mkdir', (('path', target), ('parent', parent)))
    assert plugin.execute(action, key='cross-binding-change').status == 'rejected'
    assert not (tmp_path / 'actual/destination').exists()
