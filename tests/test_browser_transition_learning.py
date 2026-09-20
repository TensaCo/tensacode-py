"""Real browser outcomes train target-relative predictions, without toggle priors.

HTML, target choice, operation and feature projection are supplied fixtures.
Only outcome associations are induced from actual paired CDP observations.
"""
from urllib.parse import quote

from examples.general_agent.browser_connection import BrowserPlugin
from tensorcode.agent import Agent
from tensorcode.agent.document_transition_evidence import (
    retain_document_transition_batch, fit_document_transitions,
    predict_document_transition, observe_document_transition,
)
from tensorcode.learning.experience import extract_transitions
from tensorcode.outcomes import Unknown
from tensorcode.runtime import use
from test_browser_connection import browser_endpoint


def fresh_target(browser, kind, episode, *, prevent_default=False):
    if kind == 'div':
        element = '<div id="explicit-fixture-target">ordinary element</div>'
        node_name = 'DIV'
    else:
        checked = ' checked' if kind == 'checked' else ''
        element = f'<input id="explicit-fixture-target" type="checkbox"{checked}>'
        node_name = 'INPUT'
    script = ('<script>document.getElementById("explicit-fixture-target").addEventListener("click", '
              'event => event.preventDefault())</script>') if prevent_default else ''
    # Actual navigation gives each episode a new frame-loader identity. Merely
    # changing content in the same document must not create a heldout partition.
    html = f'<!doctype html><title>episode {episode}</title>{element}{script}'
    browser.page.goto('data:text/html,' + quote(html))
    capture = browser.capture_document()
    assert not isinstance(capture, Unknown), capture
    matches = [(di, ni) for di, document in enumerate(capture.snapshot['documents'])
               for ni, name in enumerate(document['nodes']['nodeName'])
               if capture.snapshot['strings'][name] == node_name]
    assert len(matches) == 1  # Authored fixture identification, not learned grounding.
    token = browser.prepare_document_target(capture, *matches[0])
    assert type(token) is str, token
    return token


def activate(agent, browser, token):
    capability = next(cap for cap in browser.capabilities() if cap.name == 'activate_node')
    assert not capability.effects and not capability.preconditions
    events = []
    with use(agent.runtime):
        receipt = agent._invoke(browser, capability, {'target': token}, events)
    assert receipt.status == 'applied', receipt.error
    attempts = {event['attempt_id'] for event in events if event['type'] == 'receipt'}
    assert len(attempts) == 1
    return attempts.pop()


def test_real_browser_transition_learning_predicts_fresh_targets_and_suspends_counterexample(browser_endpoint):
    browser = BrowserPlugin(browser_endpoint)
    agent = Agent([browser])
    try:
        training, heldout = [], []
        episode = 0
        classes = ('unchecked', 'checked', 'div')
        expected_outcomes = {'unchecked': True, 'checked': False, 'div': False}
        for split, repetitions in ((training, 2), (heldout, 1)):
            for kind in classes:
                for _ in range(repetitions):
                    episode += 1
                    token = fresh_target(browser, kind, episode)
                    split.append(activate(agent, browser, token))
        retained = retain_document_transition_batch(agent, browser)
        assert not isinstance(retained, Unknown), retained
        assert len(retained.batch.transitions) == 9 and not retained.batch.exclusions, retained.batch.exclusions
        model = fit_document_transitions(agent, browser, retained,
            train_attempt_ids=training, evaluation_attempt_ids=heldout)
        assert not isinstance(model, Unknown), model
        assert model.evaluation.coverage == 1.0 and model.evaluation.accuracy == 1.0
        assert not model.history
        assert len(model.examples) == 9
        assert all(set(dict(example.facts)) == {'nodeName', 'nodeType', 'type_attribute', 'inputChecked'}
                   for example in model.examples)
        assert {example.attempt_id for example in model.examples if example.split == 'training'} == set(training)

        for kind in classes:
            episode += 1
            token = fresh_target(browser, kind, episode)
            before_attempts = len(extract_transitions(agent.interpretations.sources(),
                                                     provider='plugin:' + browser.name).transitions)
            prediction = predict_document_transition(agent, browser, model, token)
            assert not isinstance(prediction, Unknown), prediction
            assert not isinstance(prediction.prediction, Unknown), prediction.prediction
            assert prediction.prediction.outcome is expected_outcomes[kind]
            source_before_action = agent.interpretations.get_source(prediction.evidence_source_id)
            assert browser.authenticate_document_observation(source_before_action.payload) is True
            assert source_before_action.metadata['action'] == prediction.action
            assert len(extract_transitions(agent.interpretations.sources(),
                provider='plugin:' + browser.name).transitions) == before_attempts
            attempt = activate(agent, browser, token)
            assert agent.interpretations.get_source(prediction.evidence_source_id) == source_before_action
            assert observe_document_transition(agent, browser, model, prediction, attempt) is None
            assert not model.history
            if kind != 'div':
                assert browser.page.locator('#explicit-fixture-target').is_checked() is expected_outcomes[kind]

        # A listener changes the actual result while all supplied projected
        # features remain in-domain. No decoder is allowed to impose a toggle.
        episode += 1
        token = fresh_target(browser, 'unchecked', episode, prevent_default=True)
        prediction = predict_document_transition(agent, browser, model, token)
        assert not isinstance(prediction, Unknown), prediction
        assert prediction.prediction.outcome is True
        attempt = activate(agent, browser, token)
        assert browser.page.locator('#explicit-fixture-target').is_checked() is False
        suspension = observe_document_transition(agent, browser, model, prediction, attempt)
        assert not isinstance(suspension, Unknown), suspension
        assert suspension.predicted is True and suspension.observed is False
        assert suspension.rule_id == prediction.prediction.rule_id
        assert len(model.history) == 1 and suspension.source_ids
        feedback = [source for source in agent.interpretations.sources()
                    if source.modality == 'document-transition-feedback']
        assert feedback[-1].payload['outcome'] is False
        assert feedback[-1].payload['transition'].attempt_id == attempt

        episode += 1
        token = fresh_target(browser, 'unchecked', episode)
        subsequent = predict_document_transition(agent, browser, model, token)
        assert not isinstance(subsequent, Unknown), subsequent
        assert isinstance(subsequent.prediction, Unknown)
        assert not tuple(agent.store.propositions())
    finally:
        browser.close()
