"""Learn relations from actual Chromium snapshots, not hand-constructed scene graphs.

Teaching labels and language are supplied. Literal DOM projection is a transport
interpretation; neither pixel recognition nor natural-language learning is tested.
"""

from examples.general_agent.browser_connection import BrowserPlugin
from tensorcode.agent import Agent
from tensorcode.agent.document_evidence import retain_document_snapshot
from tensorcode.agent.scene_grounding import (
    retain_grounding_example, fit_grounding_model, admit_grounding_model,
    propose_groundings, get_grounding_model, grounding_dependencies,
)
from tensorcode.outcomes import Unknown
from test_browser_connection import browser_endpoint
from test_scene_grounding_uncertainty import language, PATH


def capture(agent, browser, index, *, duplicate=False):
    target = f'<section id="fresh-{index}-a" data-context="target">repeat</section>'
    other = f'<section id="fresh-{index}-b" data-context="other">repeat</section>'
    pieces = [target, other] if index % 2 else [other, target]
    if duplicate:
        pieces.append(f'<section id="fresh-{index}-c" data-context="target">repeat</section>')
    browser.page.set_content('<main>' + ''.join(pieces) + '</main>')
    snapshot = browser.document_snapshot()
    retained = retain_document_snapshot(agent, snapshot, browser.name)
    assert not isinstance(retained, Unknown), retained
    group = agent.interpretations.get(retained.group_id)
    assert group.selected_id is None
    graph = group.candidates[0].payload.graph
    facts = graph.propositions
    # Explicit teacher labels derive from known fixture structure, not a semantic
    # rule in the browser adapter or learned grounding runtime.
    containers = {p.roles['node'] for p in facts if p.predicate == 'cdp:attribute'
                  and p.roles['name'] == 'data-context' and p.roles['value'] == 'target'}
    texts = {p.roles['node'] for p in facts if p.predicate == 'cdp:nodeValue'
             and p.roles['value'] == 'repeat'}
    targets = tuple(p.roles['node'] for p in facts if p.predicate == 'cdp:parentIndex'
                    and p.roles['node'] in texts and p.roles['parent'] in containers)
    assert len(targets) == (2 if duplicate else 1)
    source = agent.interpretations.get_source(retained.source_id)
    assert source.modality == 'document' and source.payload == snapshot
    assert source.metadata['root_ref'] == graph.root.id
    assert group.candidates[0].payload.source_id == source.id
    assert not hasattr(graph, 'image')
    return retained, graph, targets


def test_real_documents_teach_relational_grounding_across_node_identity_and_order_changes(browser_endpoint):
    browser = BrowserPlugin(browser_endpoint)
    agent = Agent([browser])
    try:
        examples = []
        roots = []
        for index in range(3):
            retained, graph, targets = capture(agent, browser, index)
            roots.append(graph.root)
            lg, lc = language(agent)
            record = retain_grounding_example(agent, lg, lc, PATH,
                retained.group_id, retained.candidate_id, targets,
                tuple(r for r in (graph.root, *graph.nodes) if r not in targets),
                basis=('explicit teacher labels for rendered fixture documents',))
            assert not isinstance(record, Unknown), record
            examples.append(record)
        assert len(set(roots)) == 3
        handle = fit_grounding_model(agent, examples[:2], examples[2:],
            max_atoms=2, max_patterns=100000, max_matches=100000)
        assert not isinstance(handle, Unknown), handle
        handle = admit_grounding_model(agent, handle, reason='explicit heldout document model admission')
        assert not isinstance(handle, Unknown), handle
        learned = get_grounding_model(agent, handle)
        assert learned.queries and all(len(q.query.atoms) == 2 for q in learned.queries)
        for index, duplicate in ((3, False), (4, True)):
            retained, graph, targets = capture(agent, browser, index, duplicate=duplicate)
            lg, lc = language(agent)
            agent.interpretations.select(retained.group_id, retained.candidate_id,
                                         reason='explicit document interpretation choice')
            report = propose_groundings(agent, handle, lg, lc, PATH,
                                        retained.group_id, retained.candidate_id)
            assert not isinstance(report, Unknown), report
            assert report.complete and not report.unresolved
            group = agent.interpretations.get(lg)
            assert group.selected_id is None
            candidates = [c for c in group.candidates if c.id in report.candidate_ids]
            assert {c.payload.acts[0].frame.roles['object'].ref for c in candidates} == set(targets)
            assert len(candidates) == (2 if duplicate else 1)
            for candidate in candidates:
                dependencies = grounding_dependencies(agent, lg, candidate.id)
                assert not isinstance(dependencies, Unknown), dependencies
                assert retained.group_id in {d.group_id for d in dependencies}
            evidence = agent.interpretations.get_source(report.source_id)
            assert evidence.payload.matches
            assert all(m.matched_proposition_indices for m in evidence.payload.matches)
        assert agent.store.propositions() == []
        assert browser.page.locator('section').count() == 3
    finally:
        browser.close()
