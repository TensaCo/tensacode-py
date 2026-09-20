# Grounded browser actions

[Neutral evidence graphs](69-neutral-evidence-graphs.md) demonstrated learned
relational grounding in actual browser snapshots, but a proposed Ref alone was
not an authenticated live browser target. This milestone connects an explicitly
selected learned grounding to the exact captured DOM node and a separately
chosen operation, without synthesizing a selector or climbing to a guessed
interactive parent.

The operation is programmatic DOM activation. It is not a physical pointer click,
a visibility or hit-testing guarantee, learned affordance recognition, or evidence
of pixel understanding. The caller supplies the operation and makes the model,
graph, and language commitments. The relational query supplies the target proposal.

## Captured identity and execution authority

A retained document capture associates its source-local graph identities with
CDP backend node identities and the exact browser-provider instance. Snapshot
node IDs are observation-local references; they do not become persistent entity
identities just because an action interface can address them.

The action boundary must authenticate the captured source and target, current
provider, selected model/graph/language commitments, and live document identity.
A forged target, different provider, navigation, replaced node, stale comparison,
or replay must not silently resolve to a convenient replacement element. There
is no selector synthesis, positional fallback, or nearest-interactive-ancestor
rule to recover an executable target after identity validation fails.

The caller explicitly chooses the supported operation. Grounding a text node
must not authorize activating its parent merely because that parent looks useful.
To activate a parent control, the learned grounding itself must propose that
control's Ref from the retained relational evidence.

The live API is in `agent.document_actions`:

- `capture_browser_document(agent, provider)` obtains a provider-owned capture,
  authenticates it, and retains its unselected source-bound graph.
- `prepare_document_action(agent, provider, language_group_id, candidate_id,
  path, document_group_id, document_candidate_id)` prepares activation for the
  exact Entity Ref at the selected reading occurrence. This explicit API call
  chooses activation; it does not infer an operation from the reading's words.
- `execute_document_action(agent, provider, proposal)` consumes that proposal
  through the agent's observation/receipt path and returns a retained
  `DocumentActionResult` or `Unknown`.

`BrowserPlugin.capture_document()` creates `BrowserDocumentCapture(id, snapshot)`.
Only its issuing adapter authenticates it. The adapter creates an opaque token
for the exact captured backend node; a Ref string is not parsed into an executable
selector. Non-element targets cannot be promoted to an ancestor.

Live validation is deliberately conservative: changed document identity or any
changed captured snapshot content invalidates the target, including replacement
nodes, form state, or captured layout changes. This can reject a benign page update.
The final comparison and activation are separate CDP operations; no atomic
snapshot-check-and-dispatch guarantee is claimed for the external browser.
Provider/page guards and evidence authentication are covered by the focused
verification below. Screenshot capture preserves the DOM by using the original
caret behavior; it does not weaken snapshot identity comparison to tolerate
self-induced style changes.

## Evidence around the action

The action retains its chosen target and operation before dispatch, then records
fresh before/after observation evidence and the actual outcome. A stale or invalid
commitment must block dispatch rather than produce a fabricated success receipt.
A live browser change is evidence to assess, not proof that the user's intended
goal has been achieved.

The browser fixture reads literal snapshot state such as `inputChecked` after
activation. That observation does not constitute a learned outcome model or a
learned mapping from language to a goal. It verifies a specific browser effect
under an explicit operation contract. Programmatic activation may differ from a
physical click, including event provenance and visibility constraints; those
behaviors must not be conflated.

## Acceptance evidence and remaining gaps

Twelve agent evidence tests passed in 3.07 seconds. A transport/root integration/
existing browser-and-document run passed twenty tests in 15.74 seconds after the
caret fix; the final seven transport/root tests passed in 7.55 seconds after the
page guard. These runs overlap and are not additive. The full repository suite
passed **2,422 tests, with two skipped**, in 369.75 seconds (exit status 0).

Two real-browser integration cases train from two independently captured pages
and a held-out page, inducing a two-atom relation for checkbox parent targets.
Fresh pages produce a single matching target or two retained ambiguous targets.
The action path requires a selected reading even when one supported target is
available. Only the explicitly selected captured backend node becomes checked;
raw `inputChecked` before/after evidence shares the actual action attempt. Replaying
the consumed proposal does not change the browser again.

This uses the learned parent-control Ref itself, not a supplied target lookup,
generated CSS selector, or automatic traversal from a text child. Capture retains
and authenticates the provider-owned raw snapshot before associating it with the
graph. Model and graph dependencies remain part of the selected target's authority.

Page content, structured descriptions, labels, operation choice, and explicit
model/graph/language selections remain authored. The graph-query association is
learned from real captured DOM structure. This is a measured target-to-operation
integration, not general browser intent understanding, learned task outcomes,
visual affordance learning, or an autonomous planning policy.

Remaining work includes temporal identity inference, choosing between physical
and programmatic interaction, learning action applicability and effects,
interpreting failures and side effects, acquiring missing perceptual evidence,
and deciding what the user intended. Source and revision guards preserve a
specific commitment while it is valid; they do not complete those capabilities.

[Browser transition learning](72-browser-transition-learning.md) adds contextual
outcome learning from authenticated target observations and actual action attempts.
Its supplied measurement decoder does not author the learned outcome association.
