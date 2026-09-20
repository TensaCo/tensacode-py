"""Actual CDP node identity and programmatic activation, without selectors for action."""
from copy import deepcopy

from examples.general_agent.browser_connection import BrowserDocumentCapture, BrowserPlugin
from tensorcode.agent.plugin import Call
from tensorcode.outcomes import Unknown
from test_browser_connection import browser_endpoint


def target_index(capture, marker='target'):
    strings = capture.snapshot['strings']
    for d, document in enumerate(capture.snapshot['documents']):
        for n, attributes in enumerate(document['nodes']['attributes']):
            pairs = [(strings[attributes[i]], strings[attributes[i + 1]])
                     for i in range(0, len(attributes), 2)]
            if ('data-fixture', marker) in pairs:
                return d, n
    raise AssertionError('fixture target missing')


def checked(snapshot):
    return {doc['nodes']['backendNodeId'][i] for doc in snapshot['documents']
            for i in doc['nodes']['inputChecked'].get('index', ())}


def setup(browser):
    browser.page.set_content('<main><input type="checkbox" data-fixture="other">'
                             '<input type="checkbox" data-fixture="target"></main>')
    capture = browser.capture_document()
    d, n = target_index(capture)
    token = browser.prepare_document_target(capture, d, n)
    assert isinstance(token, str), token
    return capture, token


def activate(browser, token):
    return browser.execute(Call(browser.name, 'activate_node', (('target', token),)))


def test_exact_node_activation_and_raw_observed_effect(browser_endpoint):
    browser = BrowserPlugin(browser_endpoint)
    try:
        capture, token = setup(browser)
        assert browser.authenticate_document_capture(capture) is True
        assert browser.validate_document_target(token) is True
        assert checked(capture.snapshot) == set()
        before_html = browser.page.content()
        observation = browser.observe_evidence()
        assert browser.page.content() == before_html
        assert observation['document_snapshot'] == capture.snapshot
        assert browser.validate_document_target(token) is True
        d, n = target_index(capture)
        expected = capture.snapshot['documents'][d]['nodes']['backendNodeId'][n]
        assert activate(browser, token).status == 'applied'
        assert checked(browser.observe_evidence()['document_snapshot']) == {expected}
        assert activate(browser, token).status == 'rejected'
        assert checked(browser.document_snapshot()) == {expected}
        assert not any(cap.effects for cap in browser.capabilities())
    finally:
        browser.close()


def test_capture_mutation_forgery_and_foreign_adapter_are_rejected(browser_endpoint):
    browser = BrowserPlugin(browser_endpoint)
    other = BrowserPlugin(browser_endpoint)
    try:
        capture, token = setup(browser)
        d, n = target_index(capture)
        forged = BrowserDocumentCapture('not-issued', deepcopy(capture.snapshot))
        assert isinstance(browser.prepare_document_target(forged, d, n), Unknown)
        assert isinstance(other.authenticate_document_capture(capture), Unknown)
        assert isinstance(other.validate_document_target(token), Unknown)
        altered = deepcopy(capture.snapshot)
        altered['documents'][d]['nodes']['backendNodeId'][n] += 1000
        assert isinstance(browser.authenticate_document_capture(BrowserDocumentCapture(capture.id, altered)), Unknown)
        capture.snapshot['strings'].append('tampered')
        assert isinstance(browser.authenticate_document_capture(capture), Unknown)
        # Mutating the caller copy cannot change an already prepared target.
        assert browser.validate_document_target(token) is True
        assert checked(browser.document_snapshot()) == set()
    finally:
        other.close()
        browser.close()


def test_identical_replacement_and_navigation_invalidate_targets(browser_endpoint):
    browser = BrowserPlugin(browser_endpoint)
    try:
        _, token = setup(browser)
        browser.page.evaluate("document.querySelector('[data-fixture=target]').replaceWith(document.querySelector('[data-fixture=target]').cloneNode(true))")
        assert isinstance(browser.validate_document_target(token), Unknown)
        assert activate(browser, token).status == 'rejected'
        assert checked(browser.document_snapshot()) == set()
        _, token = setup(browser)
        browser.page.reload()
        assert isinstance(browser.validate_document_target(token), Unknown)
        assert activate(browser, token).status == 'rejected'
    finally:
        browser.close()


def test_no_parent_climb_and_closed_target_rejection(browser_endpoint):
    browser = BrowserPlugin(browser_endpoint)
    try:
        browser.page.set_content('<button>repeat</button>')
        capture = browser.capture_document()
        nodes = capture.snapshot['documents'][0]['nodes']
        text = next(i for i, kind in enumerate(nodes['nodeType']) if kind == 3)
        assert isinstance(browser.prepare_document_target(capture, 0, text), Unknown)
        for d, n in [(-1, 0), (True, 0), (0, -1), (0, True), (99, 0), (0, 999)]:
            assert isinstance(browser.prepare_document_target(capture, d, n), Unknown)
        _, token = setup(browser)
        browser.close()
        assert isinstance(browser.validate_document_target(token), Unknown)
        assert activate(browser, token).status == 'rejected'
    finally:
        browser.close()


def test_replacing_connected_page_never_targets_the_old_cdp_session(browser_endpoint):
    browser = BrowserPlugin(browser_endpoint)
    original = browser.page
    try:
        _, token = setup(browser)
        browser.page = original.context.new_page()
        assert isinstance(browser.validate_document_target(token), Unknown)
        assert activate(browser, token).status == 'rejected'
        assert original.locator('input:checked').count() == 0
        assert browser.page.url == 'about:blank'
    finally:
        browser.close()


def test_authenticated_transition_identity_survives_consumption_but_not_replacement(browser_endpoint):
    browser = BrowserPlugin(browser_endpoint)
    try:
        capture, token = setup(browser)
        evidence = browser.document_target_evidence(token)
        assert browser.authenticate_document_target_evidence(evidence) is True
        assert evidence.action == Call(browser.name, 'activate_node', (('target', token),))
        d, n = target_index(capture)
        assert (evidence.document_index, evidence.node_index) == (d, n)
        before = browser.observe_evidence()
        assert evidence in before['document_targets']
        assert browser.validate_document_target_observation(evidence, before) is True
        assert activate(browser, token).status == 'applied'
        after = browser.observe_evidence()
        assert before['document_observation_id'] != after['document_observation_id']
        assert browser.document_target_evidence(token) == evidence
        assert browser.authenticate_document_observation(after) is True
        assert browser.validate_document_target_observation(evidence, after) is True
        assert checked(before['document_snapshot']) == set()
        assert checked(after['document_snapshot']) == {evidence.backend_node_id}
        assert activate(browser, token).status == 'rejected'
        browser.page.evaluate("document.querySelector('[data-fixture=target]').replaceWith(document.querySelector('[data-fixture=target]').cloneNode(true))")
        replaced = browser.observe_evidence()
        assert isinstance(browser.validate_document_target_observation(evidence, replaced), Unknown)
        # Historical authenticated evidence remains usable, not a fresh permit.
        assert browser.validate_document_target_observation(evidence, after) is True
        browser.page.reload()
        assert isinstance(browser.validate_document_target_observation(evidence, browser.observe_evidence()), Unknown)
    finally:
        browser.close()


def test_transition_evidence_rejects_forgery_foreign_identity_and_preissuance_observation(browser_endpoint):
    from dataclasses import replace

    browser = BrowserPlugin(browser_endpoint)
    foreign = BrowserPlugin(browser_endpoint)
    try:
        browser.page.set_content('<input type="checkbox" data-fixture="target">')
        earlier = browser.observe_evidence()
        capture = browser.capture_document()
        token = browser.prepare_document_target(capture, *target_index(capture))
        evidence = browser.document_target_evidence(token)
        observation = browser.observe_evidence()
        assert isinstance(browser.validate_document_target_observation(evidence, earlier), Unknown)
        assert isinstance(foreign.authenticate_document_target_evidence(evidence), Unknown)
        assert isinstance(foreign.authenticate_document_observation(observation), Unknown)
        changed = replace(evidence, backend_node_id=evidence.backend_node_id + 1)
        assert isinstance(browser.authenticate_document_target_evidence(changed), Unknown)
        for mutate in (
            lambda value: value.update(document_observation_id='forged'),
            lambda value: value['document_identity'].update(connection_id='forged'),
            lambda value: value['document_snapshot']['strings'].append('altered'),
            lambda value: value.update(document_targets=(changed,)),
            lambda value: value.update(html='forged retained content'),
        ):
            altered = deepcopy(observation)
            mutate(altered)
            assert isinstance(browser.authenticate_document_observation(altered), Unknown)
        assert browser.authenticate_document_observation(observation) is True
        assert checked(browser.document_snapshot()) == set()
    finally:
        foreign.close()
        browser.close()
