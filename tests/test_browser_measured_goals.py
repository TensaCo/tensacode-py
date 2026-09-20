"""Three learned components compose without a pursuit-time desired value.

Teaching labels, structured input frames, and explicit selections are authored.
Actual DOM graphs and activation episodes ground the supported correspondences.
No lexical bootstrap, checkbox effect rule, or token lookup supplies intent.
"""
from urllib.parse import quote

import pytest

from examples.general_agent.browser_connection import BrowserPlugin
from tensorcode.agent import Agent, InterpretationDecision
from tensorcode.agent.document_actions import capture_browser_document
from tensorcode.agent.document_transition_evidence import browser_transition_projection
from tensorcode.agent.goal_interpretation import retain_goal_proposals, retain_taught_goal
from tensorcode.agent.goal_learning import fit_goal_model, admit_goal_model
from tensorcode.agent.measured_document_tasks import pursue_measured_document_task
from tensorcode.agent.scene_grounding import propose_groundings
from tensorcode.agent.task_dependencies import capture_dependency
from tensorcode.agent.understand import Sentence
from tensorcode.goals import MeasuredActionGoal
from tensorcode.outcomes import Unknown
from test_browser_connection import browser_endpoint
from test_browser_document_tasks import train_outcomes
from test_browser_grounded_actions import trained as train_grounding
from test_scene_grounding_uncertainty import language, PATH


def context(agent, browser, grounding, episode, *, prevented=False):
    prevent = 'event.preventDefault();' if prevented else ''
    html = f'''<!doctype html><title>measured goal episode {episode}</title>
    <section data-context="other"><input type="checkbox"></section>
    <section data-context="target"><input id="intended" type="checkbox"></section>
    <script>window.activations=0;
    document.getElementById('intended').addEventListener('click', event => {{
      window.activations += 1; {prevent}
    }});</script>'''
    browser.page.goto('data:text/html,' + quote(html))
    document = capture_browser_document(agent, browser)
    assert not isinstance(document, Unknown), document
    lg, lc = language(agent)
    agent.interpretations.select(document.group_id, document.candidate_id,
                                 reason='explicit document interpretation')
    report = propose_groundings(agent, grounding, lg, lc, PATH,
                                document.group_id, document.candidate_id)
    assert not isinstance(report, Unknown), report
    choices = [candidate for candidate in agent.interpretations.get(lg).candidates
               if candidate.id in report.candidate_ids]
    assert len(choices) == 1
    chosen = choices[0]
    agent.interpretations.select(lg, chosen.id,
        reason='explicit selection after review of supported and unresolved grounding alternatives')
    parent = capture_dependency(agent.interpretations, lg, basis=('explicit fixture input selection',))
    return document, lg, chosen, parent


def adopt(agent, group_id):
    assert type(group_id) is str, group_id
    group = agent.interpretations.get(group_id)
    assert len(group.candidates) == 1
    task = agent.tasks.create('explicit measured goal adoption', goal_interpretation_id=group_id)
    choice = InterpretationDecision(group.candidates[0].id, 'explicit fixture goal choice',
        compared_revision=group.revision, compared_candidate_ids=tuple(c.id for c in group.candidates))
    selected = agent.adopt_task_goal(task.id, choice, reason='explicit adoption of retained goal')
    assert not isinstance(selected, Unknown), selected
    assert type(selected.goal) is MeasuredActionGoal
    return selected


def teach(agent, browser, grounding):
    examples = []
    for index in range(3):
        _, _, chosen, parent = context(agent, browser, grounding, 'teaching-' + str(index))
        frame = chosen.payload.acts[0].frame
        goal = MeasuredActionGoal(frame.roles['object'].ref, 'activate_node',
            browser_transition_projection().name, True,
            basis=('explicit teacher label; not an inferred browser effect',))
        group_id = retain_taught_goal(agent, frame, goal, 'inspect the marked thing',
            parent_dependency=parent, reason='explicit measured-action teaching example')
        examples.append(adopt(agent, group_id))
        assert browser.page.evaluate('window.activations') == 0
    fitted = fit_goal_model(agent, [task.id for task in examples[:2]], [examples[2].id])
    assert not isinstance(fitted, Unknown), fitted
    handle = admit_goal_model(agent, fitted, reason='explicit admission after independent target validation')
    assert not isinstance(handle, Unknown), handle
    agent.goal_model = handle
    return handle, examples


@pytest.mark.parametrize('case', ['confirmed', 'counterexample', 'withdrawn_goal_model', 'selected_request'])
def test_learned_measured_goal_materializes_and_pursues_without_restatement(browser_endpoint, case):
    browser = BrowserPlugin(browser_endpoint)
    agent = Agent([browser])
    try:
        grounding = train_grounding(agent, browser)
        handle, examples = teach(agent, browser, grounding)
        outcomes = train_outcomes(agent, browser)
        document, lg, chosen, parent = context(agent, browser, grounding, 'fresh-pursuit',
                                               prevented=case == 'counterexample')
        frame = chosen.payload.acts[0].frame
        if case == 'selected_request':
            def choose(group):
                assert len(group.candidates) == 1
                return InterpretationDecision(group.candidates[0].id,
                    'explicit fixture authorization of learned goal', compared_revision=group.revision,
                    compared_candidate_ids=tuple(c.id for c in group.candidates))
            agent.goal_selector = choose
            act = chosen.payload.acts[0]
            sentence = Sentence('inspect the marked thing', ('inspect', 'the', 'marked', 'thing'), None, (act,))
            declaration = agent.request(sentence, act, [], interpretation_dependency=parent)
            assert declaration.status == 'suspended' and declaration.receipt is None
            assert declaration.reason == 'measured_goal_requires_materialization'
            task = agent.tasks.get(declaration.task_id)
            assert len(task.attempts) == 1
        else:
            group_id = retain_goal_proposals(agent, frame, 'inspect the marked thing', parent_dependency=parent)
            task = adopt(agent, group_id)
        assert task.goal.target == frame.roles['object'].ref
        assert task.goal.target not in {example.goal.target for example in examples}
        assert task.goal.desired_outcome is True and task.goal.operation == 'activate_node'
        assert any(dep.group_id == handle.group_id for dep in task.dependencies)
        assert browser.page.evaluate('window.activations') == 0
        if case == 'withdrawn_goal_model':
            agent.interpretations.unset(handle.group_id, reason='withdraw learned goal model authority')
        # No desired value, operation, measurement, token or action proposal is
        # restated here: all intent comes from the selected learned goal.
        result = pursue_measured_document_task(agent, browser, outcomes, task_id=task.id,
            language_group_id=lg, candidate_id=chosen.id, path=PATH,
            document_group_id=document.group_id, document_candidate_id=document.candidate_id)
        retained = agent.tasks.get(task.id)
        assert retained.goal == task.goal and retained.revision == task.revision
        if case == 'withdrawn_goal_model':
            assert isinstance(result, Unknown) or result.status != 'done'
            assert browser.page.evaluate('window.activations') == 0
        else:
            assert not isinstance(result, Unknown), result
            assert result.task_id == task.id and result.receipt.status == 'applied'
            assert browser.page.evaluate('window.activations') == 1
            assert retained.attempts[-1].revision == task.revision
            if case in ('confirmed', 'selected_request'):
                assert result.status == retained.status == 'done' and result.verified is True
                assert browser.page.locator('#intended').is_checked()
                assert not outcomes.history
                if case == 'selected_request':
                    assert len(retained.attempts) == 2
                    assert [attempt.status for attempt in retained.attempts] == ['suspended', 'done']
            else:
                assert result.status == retained.status == 'unverified' and result.verified is False
                assert not browser.page.locator('#intended').is_checked()
                assert len(outcomes.history) == 1
        assert not tuple(agent.store.propositions())
    finally:
        browser.close()
