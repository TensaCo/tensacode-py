"""Supplied visual graphs and teaching labels; learned query choice changes actions."""
from tensorcode.agent import Agent
from tensorcode.agent.scene import SceneGraph, SceneProposal
from tensorcode.agent.understand import Act, SentenceAlternative
from tensorcode.agent.scene_grounding import (
    retain_grounding_example, fit_grounding_model, admit_grounding_model,
    propose_scene_groundings, get_grounding_model,
)
from tensorcode.agent.grounding_investigation import (
    prepare_grounding_investigation, record_grounding_feedback,
    refit_grounding_from_feedback,
)
from tensorcode.language import Entity, Frame, Request, verbnet
from tensorcode.goals import GoalSpec, Condition
from tensorcode.records import Ref, Proposition
from tensorcode.outcomes import Unknown
from test_agent_tasks import Devices
from test_agent_scene_grounding import select_bound_request
from agent_test_support import select_unique_fixture_goal, supplied_goal_batch

PATH = ('acts', 0, 'frame', 'roles', 'object')


def case(agent, name, *, color=0, shape=0):
    workspace = agent.interpretations
    image = Ref('image:' + name)
    nodes = tuple(Ref(f'device:{name}:{i}') for i in range(2))
    graph = SceneGraph(image, nodes, tuple(
        Proposition(predicate, {'entity': node, 'value': i == marked})
        for predicate, marked in (('opaque-color', color), ('opaque-shape', shape))
        for i, node in enumerate(nodes)))
    source = workspace.add_source('authored visual evidence ' + name,
        modality='image', payload=b'fixture, not decoded pixels', metadata={'image_ref': image.id})
    scene_group = workspace.create_group(source.id)
    scene_candidate = workspace.propose(scene_group.id, SceneProposal(graph, ('authored visual graph',)))
    workspace.select(scene_group.id, scene_candidate.id, reason='explicit scene fixture selection')
    frame = Frame('prepare', {'object': Entity('name', 'the marked control')})
    source = workspace.add_source('prepare the marked control', provider='authored structured reading')
    language_group = workspace.create_group(source.id)
    reading = workspace.propose(language_group.id, SentenceAlternative(None, (Act('request', Request(frame), frame),)))
    return language_group.id, reading.id, scene_group.id, scene_candidate.id, graph


def model_with_correlated_teaching(agent):
    teaching = []
    for name, target in (('train-one', 0), ('train-two', 1), ('heldout', 0)):
        lg, lc, sg, sc, graph = case(agent, name, color=target, shape=target)
        record = retain_grounding_example(agent, lg, lc, PATH, sg, sc,
            (graph.nodes[target],), (graph.nodes[1-target],), basis=('explicit teacher target and exclusion',))
        assert not isinstance(record, Unknown), record
        teaching.append(record)
    model = fit_grounding_model(agent, teaching[:2], teaching[2:], max_patterns=2048, max_matches=20000)
    assert not isinstance(model, Unknown), model
    model = admit_grounding_model(agent, model, reason='explicit initial model admission')
    assert not isinstance(model, Unknown), model
    return model


def test_discriminating_teaching_revises_grounding_and_invalidates_old_task(monkeypatch):
    devices = Devices()
    agent = Agent([devices], goal_selector=select_unique_fixture_goal)
    model = model_with_correlated_teaching(agent)
    def desired(frame):
        return GoalSpec((Condition('enabled', {'undergoer': frame.roles['object'].ref}),))
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda frame, *a, **kw:
        supplied_goal_batch(desired(frame), frame=frame))
    monkeypatch.setattr(devices, 'refine_goal', lambda lexical: desired(lexical.frame))

    # All three learned queries agree in this familiar layout.
    lg, lc, sg, sc, graph = case(agent, 'earlier-task')
    report = propose_scene_groundings(agent, model, lg, lc, PATH, sg, sc)
    sentence, act, dependency = select_bound_request(agent, report, lg)
    old = agent.request(sentence, act, [], interpretation_dependency=dependency)
    assert old.status == 'done' and devices.calls == [graph.nodes[0]]
    # A completed task cannot be retried implicitly; explicitly request a recheck
    # while preserving its original model and scene commitments.
    prior_task = agent.tasks.get(old.task_id)
    agent.tasks.revise(old.task_id, prior_task.goal, reason='explicit requested recheck')

    # The agent computes differences between query predictions before teaching.
    redundant = case(agent, 'redundant', color=1, shape=1)
    crossed = case(agent, 'crossed', color=0, shape=1)
    lg, lc, sg, sc, graph = crossed
    proposal = prepare_grounding_investigation(agent, model, lg, lc, PATH,
        ((redundant[2], redundant[3]), (sg, sc)))
    assert not isinstance(proposal, Unknown), proposal
    assert proposal.investigation.best_scene_ids == (graph.image,)
    prior = agent.interpretations.get_source(proposal.evidence_source_id)
    prediction = next(p for p in proposal.investigation.predictions if p.scene_id == graph.image)
    denotations = {frozenset(p.references) for p in prediction.predictions}
    assert denotations == {frozenset(), frozenset((graph.nodes[0],)), frozenset((graph.nodes[1],))}
    assert len(devices.calls) == 1  # Investigation is not an environment action.

    feedback = record_grounding_feedback(agent, proposal, graph.image,
        (graph.nodes[0],), (graph.nodes[1],), basis=('teacher explicitly aligns the color-marked control',))
    assert not isinstance(feedback, Unknown), feedback
    assert feedback.confirmed_query_ids and feedback.contradicted_query_ids
    contradicted_patterns = tuple(q.query for q in get_grounding_model(agent, model).queries
                                 if q.id in feedback.contradicted_query_ids)
    assert agent.interpretations.get_source(proposal.evidence_source_id) == prior
    revised = refit_grounding_from_feedback(agent, feedback)
    assert not isinstance(revised, Unknown), revised
    assert revised.group_id == model.group_id and revised.dependency is None
    assert isinstance(get_grounding_model(agent, model), Unknown)
    assert agent.pursue(task_id=old.task_id).status == 'unknown'
    assert len(devices.calls) == 1  # Withdrawing a model never replays its old action.
    revised = admit_grounding_model(agent, revised, reason='explicit review of counterexample refit')
    assert not isinstance(revised, Unknown), revised

    # Fresh identities and swapped positions: the wrong shape rule no longer binds.
    lg, lc, sg, sc, graph = case(agent, 'transfer', color=1, shape=0)
    report = propose_scene_groundings(agent, revised, lg, lc, PATH, sg, sc)
    sentence, act, dependency = select_bound_request(agent, report, lg)
    assert act.frame.roles['object'].ref == graph.nodes[1]
    outcome = agent.request(sentence, act, [], interpretation_dependency=dependency)
    assert outcome.status == 'done' and devices.calls[-1] == graph.nodes[1]
    assert len(devices.calls) == 2
    task = agent.tasks.get(outcome.task_id)
    assert revised.dependency in task.dependencies
    fitted = get_grounding_model(agent, revised)
    assert len(fitted.training_examples) == 3 and len(fitted.validation_examples) == 1
    assert all(q.query not in contradicted_patterns for q in fitted.queries)
    assert agent.store.propositions() == []


def test_equally_informative_scenes_require_explicit_teacher_choice():
    agent = Agent([])
    model = model_with_correlated_teaching(agent)
    first = case(agent, 'first-crossing', color=0, shape=1)
    second = case(agent, 'second-crossing', color=1, shape=0)
    proposal = prepare_grounding_investigation(agent, model, first[0], first[1], PATH,
        ((first[2], first[3]), (second[2], second[3])))
    assert not isinstance(proposal, Unknown), proposal
    assert set(proposal.investigation.best_scene_ids) == {first[4].image, second[4].image}
    assert agent.interpretations.get(model.group_id).selected_id == model.candidate_id
    feedback = record_grounding_feedback(agent, proposal, second[4].image,
        (second[4].nodes[1],), (second[4].nodes[0],), basis=('explicit choice among tied scenes',))
    assert not isinstance(feedback, Unknown), feedback
    assert feedback.teaching_record.example.scene.image == second[4].image


def test_investigation_ranking_does_not_veto_explicit_teacher_evidence():
    agent = Agent([])
    model = model_with_correlated_teaching(agent)
    redundant = case(agent, 'teacher-chosen', color=0, shape=0)
    discriminating = case(agent, 'recommended', color=0, shape=1)
    proposal = prepare_grounding_investigation(agent, model, redundant[0], redundant[1], PATH,
        ((redundant[2], redundant[3]), (discriminating[2], discriminating[3])))
    assert not isinstance(proposal, Unknown), proposal
    assert proposal.investigation.best_scene_ids == (discriminating[4].image,)
    feedback = record_grounding_feedback(agent, proposal, redundant[4].image,
        (redundant[4].nodes[0],), (redundant[4].nodes[1],),
        basis=('teacher explicitly supplies available evidence instead of suggested experiment',))
    assert not isinstance(feedback, Unknown), feedback
    assert feedback.confirmed_query_ids and not feedback.contradicted_query_ids
    assert agent.interpretations.get(model.group_id).selected_id == model.candidate_id


def test_none_of_these_feedback_learns_conjunction_without_invented_positive():
    agent = Agent([])
    model = model_with_correlated_teaching(agent)
    lg, lc, sg, sc, graph = case(agent, 'none-crossed', color=0, shape=1)
    before = propose_scene_groundings(agent, model, lg, lc, PATH, sg, sc)
    assert not isinstance(before, Unknown), before
    assert before.complete and not before.candidate_ids
    assert any(reason.startswith('query_predicts_no_referent:') for reason in before.unresolved)
    proposal = prepare_grounding_investigation(agent, model, lg, lc, PATH, ((sg, sc),))
    assert not isinstance(proposal, Unknown), proposal
    feedback = record_grounding_feedback(agent, proposal, graph.image, (), graph.nodes,
        basis=('teacher explicitly excludes both visible controls',))
    assert not isinstance(feedback, Unknown), feedback
    assert feedback.teaching_record.example.positive_refs == ()
    assert feedback.confirmed_query_ids and feedback.contradicted_query_ids
    revised = refit_grounding_from_feedback(agent, feedback)
    assert not isinstance(revised, Unknown), revised
    revised = admit_grounding_model(agent, revised, reason='explicit review of none-of-these counterexample')
    assert not isinstance(revised, Unknown), revised
    learned = get_grounding_model(agent, revised)
    assert len(learned.queries) == 1 and len(learned.queries[0].query.atoms) == 2
    still_none = propose_scene_groundings(agent, revised, lg, lc, PATH, sg, sc)
    assert not isinstance(still_none, Unknown) and not still_none.candidate_ids
    lg, lc, sg, sc, new_graph = case(agent, 'both-transfer', color=1, shape=1)
    after = propose_scene_groundings(agent, revised, lg, lc, PATH, sg, sc)
    _, act, _ = select_bound_request(agent, after, lg)
    assert act.frame.roles['object'].ref == new_graph.nodes[1]
