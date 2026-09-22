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
  learned shared semantic space. Trainable OUTPUT_ENCODING readout is implemented
  for text and ViT, with native-attention, gradient and persistence tests; useful
  learned alignment remains unproven.
- Tools own proposal, verification, realization and retrieval components where
  configured. Evidence revisions, hypothesis assessments, memory and bounded
  action/outcome learning are active mechanisms. Remembered evidence can now be
  corrected under its original logical ID across episodes, preserving revision
  history and replacing its retrieval entry transactionally.
- Recurrent attention slots are implemented, but no consistent useful workspace
  advantage has been demonstrated. Existing ablations sometimes match or improve
  results when the workspace is bypassed.
- Experimental tool checkpoints are available, not dependable general-purpose
  cognitive software. The original complete 32-question evaluation produced
  30 abstentions, one correct answer and one circular non-answer. Reusing those
  now-development questions with FLAN-T5-XL proposals, an owned evidence-QA prompt
  and joint verification yields seven correct answers, one incorrect source
  attribution and 24 abstentions. Other components remain unchanged. This is
  progress on known cases, not final generalization or a qualified release.
- A three-axis response-quality assessor now trains and reloads exactly. Its
  first document-disjoint pilot failed to learn useful support/constraint
  rejection; it is not connected to tools. See [the next milestone](response-quality-training.md).
- Source-wise screening with the same XL proposals returns 18 correct, 2 incorrect,
  1 incomplete, 1 ambiguous and 10 abstentions on those known 32 questions. Neither
  verification scope qualifies the tool. Claim verbalization also fails fidelity;
  NLI accepts 25 nonfaithful rewrites out of 123 reviewed transformations.
- A controlled frozen-foundation workspace adaptation collapses to approving all
  development candidates. Exact continuation/reload works, but useful workspace
  discrimination remains unproven. These weights are not promoted.
- Adapting the native XL foundation jointly with that workspace also collapses:
  all 91 development candidates pass, including 28 known failures. Bypassing the
  adapted workspace rejects all 91. The native foundation changed and exact
  fixed-next-batch continuation passed; neither path qualifies as a verifier.
- The collapse diagnosis identified workspace residuals over 1,000 times native
  token magnitude. Residuals are now normalized and gated relative to native RMS.
  Repeating frozen-foundation adaptation prevents collapse but increases known
  development failure acceptance from 14 to 15; useful learning is still unproven.
- Bounded joint training improves known development failure acceptance from 14
  to seven while retaining 49 good answers. All per-axis criteria pass, but the
  combined gate fails. Active and bypass paths make identical decisions. It also
  wrongly approves 21 of 24 reviewed missing/swapped-evidence controls. Neither
  reliable grounding nor a useful workspace contribution is established.
- A real-foundation example demonstrates connected OUTPUT_ENCODING collection,
  replayed SGD and exact operation reload. Shared semantic alignment is unproven.
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

The latest stage diagnosis identifies question-conditioned verification as the
main blocker: 15 correct first beams are screened out, while ranking makes one
additional confirmed error over a correct eligible alternative. Candidate-ranking
adaptation improves a tiny development subset but degrades calibration; it is
not integrated. Joint native-foundation/workspace verification adaptation also
failed with fixed data/schedule and unchanged admission gates. The diagnosed
unbounded conditioning path is corrected; the bounded joint repeat improves ordinary development errors but fails both
combined admission and evidence-use checks. A native-only training control is
running with the same initial foundation, data and schedule, freezing all
adapters and using explicit bypass. Do not infer workspace learning from the
joint result, or connect these unqualified weights to admission.
See [the execution sequence](autonomous-completion.md) for completed experiments
and pending qualification work.
