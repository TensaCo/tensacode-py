# The big picture: useful pretrained cognition

Pinned 2026-09-21 after the 0.4.0a3 operation-convention milestone (`fac68a3`).
The owner explicitly approved this direction and asked that it stay pinned here.

## Objective

Encode real inputs into a shared cognitive workspace; form, compare and revise
structured interpretations there; reason, investigate, plan and learn before
producing language or taking action. Library users should be able to initialize
opaque tools from configuration or load complete pretrained artifacts and call
them. Hugging Face under `jacob-valdez` is the preferred artifact host.

The workspace must preserve evidence, uncertainty, competing accounts and revision
history. Visual understanding includes relationships and holistic organization,
not merely an inventory of objects. Inherited foundation ability, authored policy,
and TensorCode-learned improvements must remain distinguishable.

This restates the governing objective formerly recorded in
`docs/revival/36-structured-cognitive-workspace.md`, preserved in Git history.
It does not restore the deleted implementation or semantic defaults.

## Where we are

- Coherent public `ops.vec`, `ops.text`, and reserved `ops.graph` contracts;
  JSON construction, owned parameters, explicit foundations and safe artifacts.
- Trace/collect/train/save/reload and supported exact training continuation work.
  These establish software and learning mechanics, not autonomous supervision.
- Real transformer text/ViT encoders and embedding-conditioned text/diffusion
  decoders exist. Native readout and dimensional compatibility do not establish a
  learned shared semantic space; OUTPUT_ENCODING readout remains unimplemented.
- Tools own proposal, verification, realization and retrieval components where
  configured. Evidence revisions, hypothesis assessments, memory and bounded
  action/outcome learning are active mechanisms.
- Recurrent attention slots are implemented, but no consistent useful workspace
  advantage has been demonstrated. Existing ablations sometimes match or improve
  results when the workspace is bypassed.
- Experimental tool checkpoints are available, not dependable general-purpose
  cognitive software. The recorded complete 32-question evaluation produced
  30 abstentions, one correct answer and one circular non-answer with oracle
  supporting passages. This is the bottleneck, not a naming problem.
- Holistic visual grounding and transferable real-world planning remain open.
  Symbolic graph operations intentionally remain stubs.

## Priority order

1. **Make one complete tool useful.** Start with evidence-based cognitive Chatbot
   answering, correction and episodic memory. Diagnose candidate generation,
   evidence screening, relevance/completeness, and realization separately.
2. **Prove the workspace earns its place.** Compare identical foundations and
   data with workspace active/bypassed; measure task quality, cost and failure
   behavior. Change the architecture if controlled evidence shows no benefit.
3. **Expand across modalities and action.** Test relational scene pairs, actively
   chosen observations, executed outcomes and transfer to unfamiliar instances.
   A supplied graph or authored simulator cannot substitute for real-input tests.

Do not substitute API cleanup, new schemas, more slots, or test counts for these
behavioral objectives. Infrastructure work needs a concrete blocker it removes.

## Acceptance gates

Every capability milestone must state:

- What observable behavior improves, through the public tool interface.
- Which models, policies, targets, environments and structure were supplied.
- Train/development/final-evaluation boundaries and known pretraining overlap.
- Baselines, failure cases and ablations relevant to the claimed improvement.
- Correctness, useful coverage, unsupported answers and answer completeness;
  NLI approval, lexical overlap and fluent output are not accuracy labels.
- Revision/omission/conflict behavior, retained source evidence and persistence.
- What failed and what remains unproven. Preserve negative results.

Do not lower screening thresholds just to increase response counts. A changed
verification approach must retain source provenance and test unsupported answers.
A final set becomes development data once used to select another change; freeze
configuration and decisions before a new final evaluation.

## Working constraints

Work on main, commit coherent verified milestones and push to origin/main.
Use GB10 for real-model inference and substantial training; do not crash the host.
No backward-compatibility shims, bundled semantic seeds, first-reader defaults,
image-to-claim APIs, or implicit execution of generated text. Graph stays stubbed.
Update the current milestone note with evidence and next work after each run.
