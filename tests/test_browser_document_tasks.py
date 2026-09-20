"""One pursuit joins learned grounding, learned effects, action and observation.

Language frames, example labels, desired post-activation measurement, operation,
and explicit interpretation selections are supplied. Real CDP evidence provides
the grounding structure and outcome samples; no checkbox effect is authored.
"""
from urllib.parse import quote

import pytest

from examples.general_agent.browser_connection import BrowserPlugin
from tensorcode.agent import Agent
from tensorcode.agent.document_actions import capture_browser_document, prepare_document_action
from tensorcode.agent.document_tasks import create_document_task, pursue_document_task
from tensorcode.agent.document_transition_evidence import (
    retain_document_transition_batch, fit_document_transitions,
)
from tensorcode.agent.scene_grounding import propose_groundings
from tensorcode.outcomes import Unknown
from test_browser_connection import browser_endpoint
from test_browser_grounded_actions import trained as train_grounding
from test_browser_transition_learning import fresh_target, activate
from test_scene_grounding_uncertainty import language, PATH


def train_outcomes(agent, browser):
    training, heldout = [], []
    episode = 0
    for split, repetitions in ((training, 2), (heldout, 1)):
        for kind in ('unchecked', 'checked', 'div'):
            for _ in range(repetitions):
                episode += 1
                token = fresh_target(browser, kind, episode)
                split.append(activate(agent, browser, token))
    batch = retain_document_transition_batch(agent, browser)
    assert not isinstance(batch, Unknown), batch
    model = fit_document_transitions(agent, browser, batch,
        train_attempt_ids=training, evaluation_attempt_ids=heldout)
    assert not isinstance(model, Unknown), model
    return model


def grounded_action(agent, browser, grounding, *, unsupported=False, prevented=False):
    control = 'radio' if unsupported else 'checkbox'
    prevent = 'event.preventDefault();' if prevented else ''
    html = f'''<!doctype html><title>Fresh grounded task</title>
      <main><section data-context="other"><input type="checkbox"></section>
      <section data-context="target"><input id="intended" type="{control}"></section></main>
      <script>window.activations = 0;
      document.getElementById('intended').addEventListener('click', event => {{
        window.activations += 1; {prevent}
      }});</script>'''
    browser.page.goto('data:text/html,' + quote(html))
    document = capture_browser_document(agent, browser)
    assert not isinstance(document, Unknown), document
    lg, lc = language(agent)
    agent.interpretations.select(document.group_id, document.candidate_id,
                                 reason='explicit selection of retained document evidence')
    report = propose_groundings(agent, grounding, lg, lc, PATH,
                                document.group_id, document.candidate_id)
    assert not isinstance(report, Unknown), report
    choices = [candidate for candidate in agent.interpretations.get(lg).candidates
               if candidate.id in report.candidate_ids]
    assert len(choices) == 1
    chosen = choices[0]
    agent.interpretations.select(lg, chosen.id,
        reason='explicit review of supported grounding and unresolved alternatives')
    proposal = prepare_document_action(agent, browser, lg, chosen.id, PATH,
                                       document.group_id, document.candidate_id)
    assert not isinstance(proposal, Unknown), proposal
    return proposal, lg, chosen.id


@pytest.mark.parametrize('case', ['confirmed', 'counterexample', 'unsupported', 'different_goal'])
def test_one_document_task_pursuit_uses_learned_grounding_and_actual_outcomes(browser_endpoint, case):
    browser = BrowserPlugin(browser_endpoint)
    agent = Agent([browser])
    try:
        grounding = train_grounding(agent, browser)
        model = train_outcomes(agent, browser)
        proposal, _, _ = grounded_action(agent, browser, grounding,
            unsupported=case == 'unsupported', prevented=case == 'counterexample')
        desired = case != 'different_goal'
        task = create_document_task(agent, browser, model, proposal, desired)
        assert not isinstance(task, Unknown), task
        assert task.dependencies
        assert browser.page.evaluate('window.activations') == 0
        result = pursue_document_task(agent, browser, model, task_id=task.id)
        assert result.task_id == task.id
        retained = agent.tasks.get(task.id)
        assert len(retained.attempts) == 1
        assert retained.attempts[0].revision == task.revision
        if case == 'confirmed':
            assert result.status == 'done' and result.verified is True
            assert retained.status == 'done'
            assert result.receipt.status == 'applied'
            assert browser.page.locator('#intended').is_checked()
            assert browser.page.evaluate('window.activations') == 1
            assert not model.history
        elif case == 'counterexample':
            assert result.status == 'unverified' and result.verified is False
            assert retained.status == 'unverified'
            assert result.receipt.status == 'applied'
            assert not browser.page.locator('#intended').is_checked()
            assert browser.page.evaluate('window.activations') == 1
            assert len(model.history) == 1
            assert model.history[0].predicted is True and model.history[0].observed is False
        else:
            assert result.status != 'done' and result.verified is not True
            assert result.receipt is None or result.receipt.status == 'rejected'
            assert browser.page.evaluate('window.activations') == 0
            assert not browser.page.locator('#intended').is_checked()
            assert not model.history
        # A completed or declined pursuit cannot become an implicit second click.
        count = browser.page.evaluate('window.activations')
        replay = pursue_document_task(agent, browser, model, task_id=task.id)
        assert replay.status != 'done'
        assert browser.page.evaluate('window.activations') == count
        assert len(agent.tasks.get(task.id).attempts) == 1
        assert not tuple(agent.store.propositions())
    finally:
        browser.close()


@pytest.mark.parametrize('change', ['withdrawn_reading', 'revision_after_activation'])
def test_document_task_changes_preserve_real_action_boundary(browser_endpoint, monkeypatch, change):
    browser = BrowserPlugin(browser_endpoint)
    agent = Agent([browser])
    try:
        grounding = train_grounding(agent, browser)
        model = train_outcomes(agent, browser)
        proposal, language_group, _ = grounded_action(agent, browser, grounding)
        task = create_document_task(agent, browser, model, proposal, True)
        assert not isinstance(task, Unknown), task
        if change == 'withdrawn_reading':
            agent.interpretations.unset(language_group, reason='explicit interpretation withdrawal')
        else:
            observe = browser.observe_evidence
            revised = False

            def observe_and_revise():
                nonlocal revised
                evidence = observe()
                if not revised and browser.page.locator('#intended').is_checked():
                    revised = True
                    agent.tasks.revise(task.id, task.goal,
                        reason='explicit task correction while actual outcome is arriving')
                return evidence

            monkeypatch.setattr(browser, 'observe_evidence', observe_and_revise)
        result = pursue_document_task(agent, browser, model, task_id=task.id)
        assert result.status != 'done' and result.verified is not True
        retained = agent.tasks.get(task.id)
        assert len(retained.attempts) == 1
        assert retained.attempts[0].revision == task.revision
        if change == 'withdrawn_reading':
            assert browser.page.evaluate('window.activations') == 0
            assert not browser.page.locator('#intended').is_checked()
            assert result.receipt is None
        else:
            assert browser.page.evaluate('window.activations') == 1
            assert browser.page.locator('#intended').is_checked()
            assert result.receipt.status == 'applied'
            assert retained.attempts[0].receipt.status == 'applied'
            assert retained.revision == task.revision + 1 and retained.status == 'ready'
            assert not any(attempt.revision == retained.revision for attempt in retained.attempts)
    finally:
        browser.close()
