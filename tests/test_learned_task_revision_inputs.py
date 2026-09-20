"""Actual correction syntax revises a taught task and governs real file effects.

Task association, speech labels, occurrence identities, initial/revised teaching
goals, resource bindings, and selections are supplied. Only whole-context
reference correspondence is learned here; no paraphrase or unit grounding claim.
"""
from datetime import datetime, timezone
import pytest

from tensorcode.agent import Agent, InterpretationDecision
from tensorcode.agent.filesystem import FileSystemPlugin
from tensorcode.agent.grounding import MentionBinding, propose_grounding
from tensorcode.agent.goal_interpretation import retain_taught_goal, select_goal
from tensorcode.agent.task_dependencies import capture_dependency, validate_dependencies
from tensorcode.agent.task_revision_learning import (
    retain_task_revision_example, fit_task_revision_model, admit_task_revision_model,
)
from tensorcode.goals import Condition, GoalSpec
from tensorcode.language import Request
from tensorcode.language.deps_semantics import ProvisionalMeaning
from tensorcode.outcomes import Unknown
from tensorcode.records import Evidence, Ref

from test_learned_informing_inputs import actual_reader


INITIAL = 'Make a hello-world project in scratch.'
CORRECTION = 'Actually, use Documents, but keep the existing README.'
CONTENT = 'print("Hello, world!")\n'
README = 'Existing user documentation.\n'


def chosen(group, correction, *, provisional=False, destination='Documents'):
    matches = []
    for candidate in group.candidates:
        acts = candidate.payload.acts
        expected = ProvisionalMeaning if provisional else Request
        if not acts or not all(type(a.meaning) is expected for a in acts):
            continue
        frames = tuple(a.meaning.frame for a in acts)
        if correction:
            good = (len(frames) == 2 and tuple(f.predicate for f in frames) == ('use', 'keep')
                and getattr(frames[0].roles.get('object'), 'kind', None) == 'name'
                and frames[0].roles['object'].text == destination
                and getattr(frames[1].roles.get('object'), 'text', None) == 'the existing README')
        else:
            good = (len(frames) == 1 and frames[0].predicate == 'make'
                and getattr(frames[0].roles.get('destination'), 'text', None) == 'scratch')
        if good:
            matches.append(candidate)
    assert len(matches) == 1, [(c.id, c.payload.acts) for c in matches]
    return matches[0]


def read(agent, correction, text=None):
    message = agent.interpret(text or (CORRECTION if correction else INITIAL))
    assert message.unavailable is None
    group, = [agent.interpretations.get(g) for g in message.group_ids]
    return group


def teach_speech(agent):
    from tensorcode.agent.speech_act_learning import (
        retain_speech_act_example, fit_speech_act_model, admit_speech_act_model,
    )
    from tensorcode.learning.speech_act import SpeechActLabel
    batches = []
    for object_name, destination in (('universe', 'Pictures'), ('galaxy', 'Downloads'), ('world', 'Documents')):
        records = []
        for correction in (False, True):
            text = (CORRECTION.replace('Documents', destination) if correction
                    else INITIAL.replace('hello-world', 'hello-' + object_name))
            group = read(agent, correction, text)
            candidate = chosen(group, correction, provisional=True, destination=destination)
            for index in range(len(candidate.payload.acts)):
                record = retain_speech_act_example(agent, group.id, candidate.id, index,
                    SpeechActLabel('request'), basis=('Explicit labels for every retained act',))
                assert not isinstance(record, Unknown), record
                records.append(record)
        batches.append(records)
    fitted = fit_speech_act_model(agent, (*batches[0], *batches[1]), batches[2])
    assert not isinstance(fitted, Unknown), fitted
    handle = admit_speech_act_model(agent, fitted, reason='Explicit complete-utterance teaching')
    assert not isinstance(handle, Unknown), handle
    agent.speech_act_model = handle
    return handle


def ground(agent, correction, bindings):
    group = read(agent, correction)
    if agent.interpretations.continuation_status(group.id).pending:
        agent.expand_interpretation(group.id, max_expansions=512, max_candidates=128)
    assert not agent.interpretations.continuation_status(group.id).pending
    group = agent.interpretations.get(group.id)
    candidate = chosen(group, correction)
    evidence = agent.interpretations.add_source('Supplied occurrence identities',
        provider='integration fixture', payload=bindings)
    child = propose_grounding(agent.interpretations, group.id, candidate.id, tuple(
        MentionBinding(('acts', index, 'frame', 'roles', role), reference,
            (evidence.id,), 'Explicit source-to-resource correspondence')
        for index, role, reference in bindings))
    agent.interpretations.select(group.id, child.id, reason='Explicit full-reading selection')
    return agent.interpretations.get(group.id), child


def choice(agent, group_id, candidate_id=None):
    group = agent.interpretations.get(group_id)
    return InterpretationDecision(candidate_id or group.candidates[0].id,
        'Explicit contextual proposal selection', compared_revision=group.revision,
        compared_candidate_ids=tuple(c.id for c in group.candidates))


def context(agent, name, *, retain_correction=True):
    old, new, protected, project = (Ref(name + ':' + r) for r in ('old', 'new', 'readme', 'project'))
    invariant = (Condition('content', {'path': protected, 'text': README}),)
    def goal(root):
        return GoalSpec((Condition('content', {
            'path': {'root': root, 'relative': 'hello-world/main.py'}, 'text': CONTENT}),),
            invariants=invariant)
    group, candidate = ground(agent, False, ((0, 'destination', old), (0, 'object', project)))
    parent = capture_dependency(agent.interpretations, group.id, basis=('Supplied original request choice',))
    goal_group = retain_taught_goal(agent, candidate.payload.acts[0].frame, goal(old), INITIAL,
        parent_dependency=parent, reason='Explicit initial goal and preservation requirement')
    assert not isinstance(goal_group, Unknown), goal_group
    resolution = select_goal(agent, goal_group, decision=choice(agent, goal_group))
    assert not isinstance(resolution.goal, Unknown), resolution.goal
    task = agent.tasks.create(INITIAL, resolution.goal, goal_interpretation_id=goal_group,
        dependencies=(parent, *resolution.supporting_dependencies, resolution.dependency))
    correction_group, correction = (ground(agent, True, ((0, 'object', new), (1, 'object', protected)))
                                    if retain_correction else (None, None))
    return task, correction_group, correction, goal(new), (old, new, protected)


@pytest.mark.parametrize('withdraw_speech', [False, True], ids=['execute', 'withdraw'])
def test_real_correction_preserves_readme_and_completed_receipt(actual_reader, tmp_path, withdraw_speech):
    plugin = FileSystemPlugin(tmp_path)
    assert callable(getattr(plugin, 'bind_resource', None)), 'symbolic filesystem bindings missing'
    from tensorcode.agent.task_revision import propose_task_revision, adopt_task_revision
    agent = Agent([plugin], reader=actual_reader)
    speech = teach_speech(agent)
    teaching = []
    for name in ('alpha', 'beta', 'heldout'):
        task, group, candidate, revised, _ = context(agent, name)
        example = retain_task_revision_example(agent, task.id, group.id, candidate.id, revised,
            basis=('Explicit contextual correction teaching, including every clause',))
        assert not isinstance(example, Unknown), example
        teaching.append(example)
    model = fit_task_revision_model(agent, teaching[:2], teaching[2:])
    assert not isinstance(model, Unknown), model
    model = admit_task_revision_model(agent, model, reason='Independent grounded context validation')
    assert not isinstance(model, Unknown), model
    task, _, _, revised, refs = context(agent, 'fresh', retain_correction=False)
    (tmp_path / 'scratch').mkdir()
    (tmp_path / 'Documents').mkdir()
    (tmp_path / 'Documents/README.md').write_text(README)
    for reference, path in zip(refs, ('scratch', 'Documents', 'Documents/README.md')):
        plugin.bind_resource(reference, path, binding=Ref('binding:' + reference.id),
            evidence=Evidence(Ref('source:supplied-resource-map'), datetime(2026, 9, 20, tzinfo=timezone.utc)))
    calls = []
    execute = plugin.execute
    def recorded(call, **kwargs):
        calls.append(call)
        return execute(call, **kwargs)
    plugin.execute = recorded
    first = agent.pursue(task_id=task.id, max_steps=1)
    assert first.status == 'suspended' and len(calls) == 1
    previous = agent.tasks.get(task.id)
    group, candidate = ground(agent, True, ((0, 'object', refs[1]), (1, 'object', refs[2])))
    proposal = propose_task_revision(agent, task.id, group.id, candidate.id, model=model,
        basis=('Explicit association of this full correction with the existing task',))
    assert not isinstance(proposal, Unknown), proposal
    adopted = adopt_task_revision(agent, task.id, proposal, choice(agent, proposal),
        reason='Use the learned contextual correction')
    assert not isinstance(adopted, Unknown), adopted
    assert adopted.id == task.id and adopted.revision == 2
    assert adopted.goal.conditions == revised.conditions and adopted.goal.invariants == revised.invariants
    assert adopted.attempts == previous.attempts and len(calls) == 1
    if withdraw_speech:
        agent.interpretations.unset(speech.group_id, reason='Withdraw communicative model before dispatch')
        blocked = agent.pursue(task_id=task.id)
        assert blocked.status in ('unknown', 'suspended') and isinstance(blocked.verified, Unknown)
        assert len(calls) == 1 and not (tmp_path / 'Documents/hello-world/main.py').exists()
        assert (tmp_path / 'Documents/README.md').read_bytes() == README.encode()
        return
    second = agent.pursue(task_id=task.id)
    assert second.status == 'done', second
    assert (tmp_path / 'Documents/hello-world/main.py').read_text() == CONTENT
    assert (tmp_path / 'Documents/README.md').read_bytes() == README.encode()
    assert not (tmp_path / 'scratch/hello-world/main.py').exists()
    assert sum(call == calls[0] for call in calls) == 1
    assert [a.revision for a in agent.tasks.get(task.id).attempts] == [1, 2]
    assert validate_dependencies(agent.interpretations, adopted.dependencies) is True
    agent.interpretations.unset(speech.group_id, reason='Withdraw supplied communicative interpretation')
    assert isinstance(validate_dependencies(agent.interpretations, adopted.dependencies), Unknown)
