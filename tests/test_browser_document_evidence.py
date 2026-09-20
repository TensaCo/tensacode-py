"""Actual document transport; browser metadata is not screenshot cognition."""
from copy import deepcopy
from dataclasses import replace

import pytest

from test_browser_connection import browser_endpoint
from examples.general_agent.browser_connection import BrowserPlugin
from tensorcode.agent import Agent
from tensorcode.agent.document_evidence import retain_document_snapshot
from tensorcode.agent.evidence_graph import EvidenceGraph, GraphProposal
from tensorcode.agent.interpretation import InterpretationWorkspace
from tensorcode.outcomes import Unknown
from tensorcode.records import Proposition, Var, matches


def query(graph, predicate, **roles):
    pattern = Proposition(predicate, roles)
    return tuple(binding for fact in graph.propositions if (binding := matches(pattern, fact)) is not None)


def retained(agent, browser):
    raw = browser.document_snapshot()
    document = retain_document_snapshot(agent, raw, "plugin:" + browser.name)
    assert not isinstance(document, Unknown), document
    group = agent.interpretations.get(document.group_id)
    return raw, document, group.candidates[0].payload


@pytest.mark.parametrize("reverse", [False, True])
def test_actual_document_structure_preserves_repeated_text_and_parent_relations(browser_endpoint, reverse, monkeypatch):
    browser = BrowserPlugin(browser_endpoint)
    try:
        sections = ['<section data-context="target">Repeated</section>', '<aside data-context="other">Repeated</aside>']
        if reverse:
            sections.reverse()
        browser.page.set_content('<main>' + ''.join(sections) + '</main>')
        monkeypatch.setattr(browser, "screenshot", lambda: pytest.fail("document capture must not imply pixel perception"))
        agent = Agent([])
        raw, document, proposal = retained(agent, browser)
        source = agent.interpretations.get_source(document.source_id)
        assert source.modality == "document" and source.payload == raw
        assert source.metadata["root_ref"] == proposal.graph.root.id
        assert source.provider == "plugin:" + browser.name
        assert type(proposal) is GraphProposal and type(proposal.graph) is EvidenceGraph
        assert proposal.source_id == source.id
        assert not hasattr(proposal.graph, "image")
        group = agent.interpretations.get(document.group_id)
        assert group.selected_id is None
        assert not tuple(agent.store.claims()) and not tuple(agent.store.propositions())
        text_nodes = query(proposal.graph, "cdp:nodeValue", node=Var("node"), value="Repeated")
        assert len(text_nodes) == 2  # same text is not merged into one identity
        selected = []
        for binding in text_nodes:
            node = binding["node"]
            parents = query(proposal.graph, "cdp:parentIndex", node=node, parent=Var("parent"))
            for parent in parents:
                if query(proposal.graph, "cdp:attribute", node=parent["parent"], name="data-context", value="target"):
                    selected.append(node)
        assert len(selected) == 1
        _, second, second_proposal = retained(agent, browser)
        assert second.root != document.root
        assert set(second_proposal.graph.nodes).isdisjoint(proposal.graph.nodes)
        assert all(fact.predicate.startswith("cdp:") for fact in proposal.graph.propositions)
        raw["strings"].append("external mutation")
        assert agent.interpretations.get_source(document.source_id).payload != raw
    finally:
        browser.close()


def test_malformed_or_over_budget_document_retains_raw_source_without_graph(browser_endpoint):
    browser = BrowserPlugin(browser_endpoint)
    try:
        browser.page.set_content('<article data-context="literal">text</article>')
        raw = browser.document_snapshot()
        malformed = deepcopy(raw)
        malformed["documents"][0]["nodes"]["parentIndex"][1] = 999999
        for snapshot, budget, reason in [(malformed, 5000, "malformed_document_snapshot"),
                                         (raw, 1, "document_projection_budget")]:
            workspace = InterpretationWorkspace()
            result = retain_document_snapshot(workspace, snapshot, "actual-browser", max_nodes=budget)
            assert isinstance(result, Unknown) and result.reason == reason
            sources = workspace.sources()
            assert sources[0].modality == "document" and sources[0].payload == snapshot
            assert sources[-1].modality == "assessment"
            assert not workspace.values()
    finally:
        browser.close()


def test_source_mutation_during_publication_rejects_candidate(browser_endpoint):
    class MutatingWorkspace(InterpretationWorkspace):
        def propose(self, group_id, payload, **kwargs):
            candidate = super().propose(group_id, payload, **kwargs)
            source_id = self.get(group_id).source_id
            source = self._sources[source_id]
            self._sources[source_id] = replace(source, payload={"forged": True})
            return candidate

    browser = BrowserPlugin(browser_endpoint)
    try:
        browser.page.set_content('<div>Original source</div>')
        raw = browser.document_snapshot()
        workspace = MutatingWorkspace()
        result = retain_document_snapshot(workspace, raw, "actual-browser")
        assert isinstance(result, Unknown)
        group = workspace.values()[0]
        assert group.selected_id is None and group.candidates[0].rejected
        assessment = workspace.sources()[-1]
        assert assessment.payload.payload == raw
        assert "changed" in assessment.metadata["reason"]
    finally:
        browser.close()



def test_selection_during_final_source_read_rejects_published_candidate(browser_endpoint):
    class SelectingWorkspace(InterpretationWorkspace):
        def get_source(self, source_id):
            source = super().get_source(source_id)
            for group in tuple(self._groups.values()):
                if group.source_id == source_id and group.candidates:
                    self.select(group.id, group.candidates[0].id, reason="callback changed comparison")
            return source

    browser = BrowserPlugin(browser_endpoint)
    try:
        browser.page.set_content('<section>Retained document</section>')
        workspace = SelectingWorkspace()
        result = retain_document_snapshot(workspace, browser.document_snapshot(), "actual-browser")
        assert isinstance(result, Unknown)
        group = workspace.values()[0]
        assert group.selected_id is None and group.candidates[0].rejected
        assert "final source authentication" in workspace.sources()[-1].metadata["reason"]
    finally:
        browser.close()
