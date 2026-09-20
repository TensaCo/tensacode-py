"""Learned structural targets drive exact-node programmatic browser activation.

Rendered fixtures, labels, language, operation, and selections are supplied; graph
queries and target bindings are learned. No pixel or free-language claim follows.
"""
import pytest

from examples.general_agent.browser_connection import BrowserPlugin
from tensorcode.agent import Agent
from tensorcode.agent.document_actions import (
    capture_browser_document, prepare_document_action, execute_document_action,
)
from tensorcode.agent.scene_grounding import (
    retain_grounding_example, fit_grounding_model, admit_grounding_model,
    propose_groundings, get_grounding_model,
)
from tensorcode.outcomes import Unknown
from test_browser_connection import browser_endpoint
from test_scene_grounding_uncertainty import language, PATH


def document(agent, browser, index, *, duplicate=False):
    good = f'<section data-context="target" id="fresh-{index}-a"><input type="checkbox"></section>'
    other = f'<section data-context="other" id="fresh-{index}-b"><input type="checkbox"></section>'
    parts = [good, other] if index % 2 else [other, good]
    if duplicate:
        parts.append(f'<section data-context="target" id="fresh-{index}-c"><input type="checkbox"></section>')
    browser.page.set_content('<main>' + ''.join(parts) + '</main>')
    retained = capture_browser_document(agent, browser)
    assert not isinstance(retained, Unknown), retained
    candidate = next(c for c in agent.interpretations.get(retained.group_id).candidates
                     if c.id == retained.candidate_id)
    graph = candidate.payload.graph
    parents = {p.roles['node'] for p in graph.propositions if p.predicate == 'cdp:attribute'
               and p.roles['name'] == 'data-context' and p.roles['value'] == 'target'}
    inputs = {p.roles['node'] for p in graph.propositions if p.predicate == 'cdp:nodeName'
              and p.roles['value'] == 'INPUT'}
    targets = tuple(p.roles['node'] for p in graph.propositions if p.predicate == 'cdp:parentIndex'
                    and p.roles['parent'] in parents and p.roles['node'] in inputs)
    assert len(targets) == (2 if duplicate else 1)
    return retained, graph, targets


def trained(agent, browser):
    records = []
    for index in range(3):
        retained, graph, targets = document(agent, browser, index)
        lg, lc = language(agent)
        record = retain_grounding_example(agent, lg, lc, PATH,
            retained.group_id, retained.candidate_id, targets,
            tuple(r for r in (graph.root, *graph.nodes) if r not in targets),
            basis=('explicit labels for actual rendered checkbox fixtures',))
        assert not isinstance(record, Unknown), record
        records.append(record)
    model = fit_grounding_model(agent, records[:2], records[2:], max_atoms=2,
                               max_patterns=100000, max_matches=100000)
    assert not isinstance(model, Unknown), model
    model = admit_grounding_model(agent, model, reason='explicit heldout browser-model admission')
    assert not isinstance(model, Unknown), model
    assert all(len(q.query.atoms) == 2 for q in get_grounding_model(agent, model).queries)
    return model


def checked_backend_ids(snapshot):
    return {document['nodes']['backendNodeId'][i]
            for document in snapshot['documents']
            for i in document['nodes'].get('inputChecked', {}).get('index', [])}


@pytest.mark.parametrize('duplicate', [False, True])
def test_learned_target_activates_only_explicitly_selected_checkbox_and_retains_outcomes(browser_endpoint, duplicate):
    browser = BrowserPlugin(browser_endpoint)
    agent = Agent([browser])
    try:
        model = trained(agent, browser)
        retained, graph, targets = document(agent, browser, 4, duplicate=duplicate)
        lg, lc = language(agent)
        agent.interpretations.select(retained.group_id, retained.candidate_id,
                                     reason='explicit source interpretation selection')
        report = propose_groundings(agent, model, lg, lc, PATH, retained.group_id, retained.candidate_id)
        assert not isinstance(report, Unknown), report
        group = agent.interpretations.get(lg)
        choices = [c for c in group.candidates if c.id in report.candidate_ids]
        assert {c.payload.acts[0].frame.roles['object'].ref for c in choices} == set(targets)
        assert len(choices) == (2 if duplicate else 1) and group.selected_id is None
        chosen = choices[-1]  # Explicit fixture choice, never a first-match policy.
        declined = prepare_document_action(agent, browser, lg, chosen.id, PATH,
                                           retained.group_id, retained.candidate_id)
        assert isinstance(declined, Unknown)
        assert not checked_backend_ids(browser.document_snapshot())
        agent.interpretations.select(lg, chosen.id, reason='explicit review of all supported and unknown alternatives')
        proposal = prepare_document_action(agent, browser, lg, chosen.id, PATH,
                                           retained.group_id, retained.candidate_id)
        assert not isinstance(proposal, Unknown), proposal
        target = chosen.payload.acts[0].frame.roles['object'].ref
        assert proposal.target == target
        raw = agent.interpretations.get_source(retained.source_id).payload
        # Fixture maps a known retained graph identity to raw capture index for
        # outcome verification. Production must use its authenticated association.
        index = next(i for i in range(len(raw['documents'][0]['nodes']['backendNodeId']))
                     if target.id == f'{retained.source_id}/document:0/node:{i}')
        backend_id = raw['documents'][0]['nodes']['backendNodeId'][index]
        result = execute_document_action(agent, browser, proposal)
        assert not isinstance(result, Unknown), result
        assert result.receipt.status == 'applied', result.receipt.error
        assert checked_backend_ids(browser.document_snapshot()) == {backend_id}
        observations = [s for s in agent.interpretations.sources() if s.modality == 'observation'
                        and s.metadata.get('action') and s.metadata['action'].capability == 'activate_node']
        before = next(s for s in observations if s.metadata.get('stage') == 'before_action')
        after = next(s for s in observations if s.metadata.get('stage') == 'after_action')
        assert before.metadata['attempt_id'] == after.metadata['attempt_id']
        assert not checked_backend_ids(before.payload['document_snapshot'])
        assert checked_backend_ids(after.payload['document_snapshot']) == {backend_id}
        assert after.metadata['receipt'].status == 'applied'
        assert isinstance(execute_document_action(agent, browser, proposal), Unknown)
        assert checked_backend_ids(browser.document_snapshot()) == {backend_id}
        assert agent.store.propositions() == []
    finally:
        browser.close()
