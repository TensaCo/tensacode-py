"""Literal browser document structure retained as an unselected graph proposal.

This is a supplied, inspectable CDP-to-proposition projection, not pixel perception
or learned document semantics. Tags, attributes, and text remain literal data.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from uuid import uuid4

from ..outcomes import Unknown
from ..learning.experience import _same
from ..records import Proposition, Ref
from .evidence_graph import EvidenceGraph, GraphProposal


@dataclass(frozen=True)
class RetainedDocument:
    source_id: str
    group_id: str
    candidate_id: str
    root: Ref


class _Malformed(ValueError):
    pass


def _project(snapshot, source_id, root, max_nodes):
    if type(snapshot) is not dict or type(snapshot.get("documents")) is not list or type(snapshot.get("strings")) is not list:
        raise _Malformed("snapshot requires documents and strings arrays")
    strings, documents = snapshot["strings"], snapshot["documents"]
    if not documents or any(type(value) is not str for value in strings):
        raise _Malformed("snapshot needs a document and a string table")

    def text(index, *, allow_absent=False):
        if type(index) is not int or index < (-1 if allow_absent else 0) or index >= len(strings):
            raise _Malformed("invalid string table index")
        return None if index == -1 else strings[index]

    rows = []
    total = 0
    for document in documents:
        if type(document) is not dict or type(document.get("nodes")) is not dict:
            raise _Malformed("document requires literal nodes")
        nodes = document["nodes"]
        fields = ("parentIndex", "nodeType", "nodeName", "nodeValue", "attributes")
        if any(type(nodes.get(field)) is not list for field in fields):
            raise _Malformed("node field arrays are missing")
        count = len(nodes["nodeType"])
        if not count or any(len(nodes[field]) != count for field in fields):
            raise _Malformed("node field arrays have unequal lengths or are empty")
        total += count
        if total > max_nodes:
            raise OverflowError("document node budget exceeded; no truncated graph admitted")
        rows.append(nodes)
    references, propositions = [], []
    for document_index, nodes in enumerate(rows):
        refs = tuple(Ref(f"{source_id}/document:{document_index}/node:{index}")
                     for index in range(len(nodes["nodeType"])))
        references.extend(refs)
        parents = nodes["parentIndex"]
        if any(type(parent) is not int or parent < -1 or parent >= len(refs) for parent in parents):
            raise _Malformed("parent index is outside this document")
        roots = [index for index, parent in enumerate(parents) if parent == -1]
        if len(roots) != 1:
            raise _Malformed("parent indices must identify one document root")
        complete = set()
        for start in range(len(refs)):
            active, current = set(), start
            while current != -1 and current not in complete:
                if current in active:
                    raise _Malformed("cyclic document parents")
                active.add(current)
                current = parents[current]
            complete.update(active)
        propositions.append(Proposition("cdp:document", {"root": root, "document": refs[roots[0]]}))
        for index, reference in enumerate(refs):
            node_type = nodes["nodeType"][index]
            if type(node_type) is not int or node_type < 1:
                raise _Malformed("node type must be a positive integer")
            propositions.append(Proposition("cdp:nodeType", {"node": reference, "value": node_type}))
            propositions.append(Proposition("cdp:nodeName", {"node": reference, "value": text(nodes["nodeName"][index])}))
            value = text(nodes["nodeValue"][index], allow_absent=True)
            if value is not None:
                propositions.append(Proposition("cdp:nodeValue", {"node": reference, "value": value}))
            if parents[index] >= 0:
                propositions.append(Proposition("cdp:parentIndex", {"node": reference, "parent": refs[parents[index]]}))
            attributes = nodes["attributes"][index]
            if type(attributes) is not list or len(attributes) % 2:
                raise _Malformed("attributes must be paired string table indices")
            for position in range(0, len(attributes), 2):
                propositions.append(Proposition("cdp:attribute", {"node": reference,
                    "name": text(attributes[position]), "value": text(attributes[position + 1])}))
    return EvidenceGraph(root, tuple(references), tuple(propositions), limitations=(
        "Literal CDP document projection; no visual recognition or semantic tag interpretation.",
        "Projects node type/name/value, attributes, and within-document parents; other CDP fields remain in the raw source.",
        "Does not establish visibility, clickable affordances, or identity across captures.",))


def retain_document_snapshot(workspace_or_agent, snapshot, provider: str, *, max_nodes: int = 5000):
    """Retain raw document evidence and an unselected proposal, or a recorded refusal.

    Invalid or over-budget input retains its raw source and failure assessment.
    Missing projected fields never become fabricated negative facts. The raw source
    remains available when CDP supplies fields outside this projection's scope.
    """
    if type(provider) is not str or not provider.strip():
        raise ValueError("an explicit document provider is required")
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError("max_nodes must be a positive integer")
    workspace = getattr(workspace_or_agent, "interpretations", workspace_or_agent)
    root = Ref("document:" + uuid4().hex)
    source = workspace.add_source("CDP document snapshot", modality="document", provider=provider,
        metadata={"root_ref": root.id, "source_kind": "cdp:DOMSnapshot.captureSnapshot", "status": "observed"},
        payload=snapshot)
    expected_source = deepcopy(source)
    group = candidate = None

    def authenticate_source():
        if not _same(workspace.get_source(source.id), expected_source):
            raise _Malformed("retained document source changed during projection")

    try:
        authenticate_source()
        graph = _project(expected_source.payload, source.id, root, max_nodes)
        authenticate_source()
        group = workspace.create_group(source.id, provenance=(provider, "literal-cdp-document-projection"))
        authenticate_source()
        proposal = GraphProposal(graph, source.id, provenance=(provider, "cdp:DOMSnapshot.captureSnapshot",
                                                           "authored literal field projection"))
        expected_proposal = deepcopy(proposal)
        candidate = workspace.propose(group.id, proposal, provenance=proposal.provenance)
        current = workspace.get(group.id)
        actual = next((item for item in current.candidates if item.id == candidate.id), None)
        if (current.source_id != source.id or current.selected_id is not None or actual is None
                or actual.rejected or not _same(actual.payload, expected_proposal)
                or actual.provenance != expected_proposal.provenance):
            raise _Malformed("document proposal changed during publication")
        authenticate_source()
        expected_basis = (source.id, 0, None, (candidate.id,), None, False, 0)
        if workspace.comparison_basis(group.id) != expected_basis:
            raise _Malformed("document comparison changed during final source authentication")
    except (ValueError, TypeError, OverflowError, KeyError) as error:
        if candidate is not None:
            workspace.reject(group.id, candidate.id, reason="document evidence changed during publication")
        reason = "document_projection_budget" if isinstance(error, OverflowError) else "malformed_document_snapshot"
        workspace.add_source("Document projection did not establish a graph", modality="assessment",
            provider="document-projection", metadata={"document_source_id": source.id,
                "status": reason, "reason": str(error)}, payload=expected_source)
        return Unknown(reason, f"{source.id}: {error}")
    return RetainedDocument(source.id, group.id, candidate.id, root)
