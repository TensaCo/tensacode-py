# Neutral evidence graphs

Relational learning and grounding should operate on source-backed evidence graphs,
not require every graph to claim it came from an image. This milestone removes
that coupling while retaining the visual scene specialization. It also connects
a real browser document snapshot to the same learned-query machinery.

The active target is a shared path: a retained browser source supplies a literal
structural graph; supervised examples induce relational queries; those queries
propose grounding alternatives or missing-evidence probes in another snapshot.
The graph operations do not receive a separate set of browser-specific meaning
rules. Source type and observation provenance remain explicit.

## Shared structure, distinct evidence

The new generic `EvidenceGraph` and source-bound `GraphProposal` separate graph
structure from the assumption of visual input. The existing `SceneGraph` remains
a specialization for visual scene evidence, rather than forcing browser, language,
or other observations through an image-shaped wrapper. Both can participate in
the same bounded matching, query learning, evidence assessment, and probe planning.

A generic graph is still an interpretation of a source. Its source identity,
root and declared entities, propositions, and retained limitations must be
inspectable. A browser document source does not become an image simply because
its contents can also be rendered visually. Visual anchors and image-specific
validation remain meaningful only where that evidence exists.

`agent.evidence_graph.EvidenceGraph(root, nodes=(), propositions=(),
limitations=())` validates local references and typed propositions.
`GraphProposal(graph, source_id, provenance=(), score=None)` names the retained
source; construction checks shape, while workspace consumers authenticate that
source ID against the actual source record. `graph_root` returns the neutral
root or the visual scene image identity without renaming document evidence as
visual evidence. `SceneGraph` retains its existing visual anchors and contract.

This milestone does not make every existing connection an
automatic graph provider or silently convert arbitrary files into understood scenes.

## Literal browser evidence

The browser path uses a real Chrome DevTools Protocol document snapshot and
retains it as source evidence before constructing its graph. The adapter exposes
literal document structure and properties, not inferred tag semantics. A tag or
attribute is observed data; it does not automatically establish an affordance,
user intent, importance, ownership, or the meaning of an element.

`agent.document_evidence.retain_document_snapshot(workspace_or_agent, snapshot,
provider, *, max_nodes=5000)` retains the raw CDP source and returns
`RetainedDocument(source_id, group_id, candidate_id, root)` for an unselected
proposal. The authored projection exposes node type, name, value, attributes,
within-document parent links, and document membership as literal `cdp:*`
propositions. Other snapshot fields remain in the raw source; they are not
silently interpreted. Malformed or over-budget snapshots retain a source and
refusal assessment and return `Unknown`, rather than admitting a truncated graph.

The browser integration explicitly calls `BrowserPlugin.document_snapshot()`
and then this retention/projection API. Standard `BrowserPlugin.observe_evidence`
still returns raw evidence; the chat does not automatically select graphs, create
teaching labels, train a model, or choose a grounding.

This is DOM evidence, not pixel inference. Rendering a page in an actual browser
does not establish that the agent understands its visual organization, appearance,
occlusion, or human-facing meaning. CSS effects and visual relationships require
appropriate evidence rather than assumptions derived from tag names.

Node identities are local to the retained snapshot. They must not be described
as persistent identities across navigation, document replacement, or later
snapshots. A later graph is new evidence; correspondence across observations is
a separate inference problem. Source binding makes the exact snapshot used for
a learned match inspectable without solving temporal entity identity.

## Grounding and questions remain guarded

The shared learner still needs structured descriptions and explicit positive or
negative teaching alignments. It induces graph queries from those examples rather
than receiving a hand-authored target query. Training and held-out identity
separation, unvalidated rivals, contradictory facts, unknown roots, and unseen
referent possibilities remain relevant for browser graphs too.

Explicit model admission and graph/reading selection remain commitments. A literal
snapshot does not authorize a browser action, and a single supported target does
not erase unresolved alternatives. Query-derived probes likewise remain questions
about source-backed evidence, with explicit observation and later selection under
the existing boundaries. Switching to a neutral graph representation does not
learn a selection policy or establish the correctness of teaching labels.

## Acceptance evidence and remaining gaps

The actual Chromium integration in `tests/test_browser_learned_grounding.py`
passed **one test in 2.46 seconds**. It captures two training pages and a held-out
page, learns a two-atom relation from supplied text-node alignments, and applies
the model to two further captured pages with fresh snapshot identities and
changed order. One page yields the correct single supported target; another
retains both matching targets. The result groups remain unselected, and source
IDs, document roots, supporting fact indices, and graph dependencies remain
inspectable. No task action or world belief is produced.

A combined run passed **22 new tests in 5.99 seconds**: seven pure graph tests,
ten wrapper tests, four document-projection tests, and the real-browser learning
test. After the final source-read callback guard, the updated five-test document
subset passed in 4.26 seconds. This adds one regression, for 23 new tests overall;
it is not a claim of a single final 23-test run. The full repository suite passed
**2,407 tests, with five skipped**, in 352.56 seconds (exit status 0). These counts
verify mechanisms, not general browser understanding.

The browser page content, descriptions, teaching labels, and explicit choices in
that fixture are authored. The query is learned from real captured structural
evidence. This is stronger evidence for the adapter and shared learning path than
an authored graph alone, while remaining a narrow test of supplied supervision
and structural transfer rather than general web understanding.

Remaining gaps include scene formation from pixels, correspondence across browser
snapshots, semantic affordance learning, interpretation of unfamiliar descriptions,
source reliability, autonomous evidence acquisition, and deciding which evidence
modality is appropriate. Neutral graph types support these future capabilities;
the types alone do not establish them. The measured browser integration and
remaining assumptions must be reported separately from the representation change.

[Retiring pixel semantic rules](70-retiring-pixel-semantic-rules.md) removes the
older pixel-rule fallback. Neutral structural evidence does not replace the still
unimplemented learned pixel-to-scene capability.
