# 38 — Scene interpretations in the shared workspace

*2026-09-19. Implementation report extending [37](37-interpretation-workspace-first-slice.md)
under the holistic visual objective in [36](36-structured-cognitive-workspace.md#vision-means-understanding-a-scene).
This checkpoint connects supplied relational visual hypotheses to the agent workspace. It does
not add a model that infers general scene structure from unfamiliar pixels.*

## What the representation means

A scene interpretation can express an account of the whole situation: layout and grouping,
relations among regions or entities, events, and proposed affordances. Its semantics need not
reduce to a list of detected elements. Competing interpretations can disagree about global
organization while referring to the same visible parts.

[scene.py](../../src/tensorcode/agent/scene.py) introduces `SceneGraph`, `VisualAnchor`, and
`SceneProposal`. The graph reuses the existing `Proposition` and `Ref` machinery. There is
no prescribed inventory of node kinds or relation names. A proposition may describe the
image as a whole or connect declared nodes through named roles. Propositions retain the
existing polarity, modality, validity, and scope fields. The representation validates local
references, including nested fillers; this establishes referential integrity, not truth,
completeness, or correspondence with entities in another image. This first graph contract
requires bound role fillers: missing details can be omitted or represented by unnamed
nodes, while query variables are reserved for matching. It is not yet a general constraint
language for partially specified scene hypotheses.

`VisualAnchor(entity, region)` optionally associates a referent with a normalized
`(left, top, right, bottom)` rectangle. `region=None` denotes whole-image support. Anchors
locate evidence; they do not claim that an object detector or segmentation model found it.
A graph may describe a global arrangement without supplying boxes for every participant.
Rectangles are the first supported region geometry, not a claim that all visual organization
is rectangular.

A `SceneProposal` contains a graph, provenance, and an optional `Score` retaining its declared
kind. Graph `limitations` record boundaries supplied by the producer. Arbitrary predicates
are representable, but their names alone provide no inference rules, grounding, or learned
meaning. A consumer must supply usable semantics for any relation on which it relies.

## Image evidence and provider boundary

`Agent.interpret_image(image)` retains the source and calls each plugin's
`interpret_image(image, ref)`. It returns `InterpretedImage` with the image reference,
`source_id`, and `group_ids`. The source's `payload` preserves the supplied
image value using the workspace's detached-copy contract. Existing file paths are read into
bytes, with `original_path` retained in metadata, so a subsequent change to that file does
not rewrite the retained evidence. This is in-memory retention, not durable evidence storage.

Each provider returns an iterable of `SceneProposal` alternatives. Its proposals are
validated against the current image reference before that provider's candidates are inserted.
Different providers receive separate groups: their accounts might describe different aspects
of a scene, so the implementation does not assume cross-provider mutual exclusivity or
perform graph fusion. An empty iterable supplies no proposals. The base implementation returns
an empty iterable. `None` is not an alternate interface or a signal to bypass interpretation.

The old `Plugin.see` interface and its direct visual-claim insertion path are removed. Image
providers must produce explicit proposals through `interpret_image`; there is no fallback
that turns a classifier's preferred label into a stored fact. Calling `interpret_image` does
not assert scene propositions or select a visual candidate. A candidate's existence, score,
and recorded selection are all distinct from a verified belief. Workspace selection and
rejection retain reasons and revision history; they do not automatically retract a derived
belief or replan a task.

`Agent.turn(text, images=...)` retains the visual proposals before interpreting the text.
`Turn.visual_interpretation_ids` links the turn to the visual groups; the `seen` trace includes
source and group identifiers. Each provider receives its own copy of retained evidence, so
file-backed image inputs arrive as captured bytes. Providers requiring a pathname must adapt.
This is an intentional interface change; callers must update rather than opt into the removed
visual path.

## Default interpretation no longer commits by reader order

The default language policy now defers with an explanation that no interpretation policy
was supplied. Candidate acts are retained but not dispatched. An explicit supplied policy is
required to select a language interpretation for execution. A reader that produces only one
structured candidate still contributes that candidate with explicit provenance; a singleton
is not permission to execute it. No production compatibility flag restores automatic
first-candidate selection.

This breaking behavior removes an unsupported commitment, while leaving automatic semantic
resolution unfinished. A caller can supply an authored selection policy, but its assumptions
must remain visible; merely choosing the first candidate in a callback does not establish
understanding. Structured goal execution remains a separate explicit API. Selection no longer initiates
an implicit reference-resolution pass. Needed reference and domain knowledge must be supplied
explicitly; choosing a reader candidate alone does not guarantee an executable interpretation.
The related cleanup is documented in [39](39-removing-implicit-semantic-authority.md).

## A graph can influence behavior

`SceneGraph.match(pattern)` queries the proposal using the ordinary proposition matcher and
returns variable bindings. It does not insert the matched proposition into the belief store
or verify its content. An omitted pattern scope is a wildcard under the existing matching
semantics; consumers requiring a particular scope must specify it.

A caller-supplied language interpretation selector can inspect the retained visual groups,
query a scene relationship, and use the result to choose or defer a language candidate. This
connects visual structure to the existing interpretation and execution path without introducing
a fixed vocabulary of visual relations into the central agent loop. It remains an authored
selection policy; the agent has not learned how every new relation should affect a task.

The integration test distinguishes supplied scenes with the same element inventory but
different relational organization: an authored selector queries the primary-workspace
relationship, and the filesystem planner creates the corresponding project.
Supplied graph fixtures test this integration independently of perception quality. They do
not establish that a visual model can recover those graphs from pixels, or that a generic
policy can resolve arbitrary scene ambiguity.

## The installed classifier remains a narrow adapter

The existing `VisionPlugin` now proposes the classifier's category alternatives, retaining
its full available distribution as explicitly uncalibrated scores. Alternatives share an
entity reference and declare that only whole-image classification is supplied. The normal
proposal path does not turn the highest-scoring class into a belief automatically. The old
thresholded `see` method has been removed along with the direct visual-claim path.

This change preserves uncertainty at an interface. It adds no scene-trained model, layout
inference, relational extraction, event recognition, or scene-level reasoning to that
classifier. A missing model or unusable distribution supplies no category proposals. Do not
present classification proposals wrapped in a scene container as holistic understanding.

## Remaining work and acceptance boundary

Learned scene formation, automatic grounding between visual and linguistic referents, graph
fusion, calibrated scene uncertainty, contradiction-driven revision, and dependency tracking
remain unfinished. Identity over time, changing viewpoints, occlusion reasoning, and active
visual investigation are subsequent behavioral requirements. Text interpretation now defers
by default; an automatic evidence-driven resolver is still needed. Authored grammar rules,
dependency-reader mechanisms remain and need semantic audits. Bundled request conventions,
project recipes, seeded greeting replies, and additional implicit resolution shortcuts are
removed in [39 — Removing implicit semantic authority](39-removing-implicit-semantic-authority.md).
Explicitly supplied knowledge remains authored knowledge; neither cleanup eliminates all
brittle rules or supplies a learned replacement.

The next visual milestone needs a producer that derives relational hypotheses from visual
evidence and an evidence-driven consumer that can resolve or defer a consequential global
ambiguity. Preserve both success and counterexamples, original sources, and the distinction
between supplied knowledge and inferred structure. A new graph type is only useful when it
changes what the agent can inspect, explain, or do correctly.

## Verification

The final combined scene and fallback-removal checkpoint passed **1,485 tests with
5 skipped**. The supplied-scene integration tests change relational organization while
holding category inventory constant and verify different filesystem outcomes under an
explicit test policy and recipe. Source isolation, invalid graph rejection, model score
retention, missing-model abstention, and default noncommitment are covered. The wheel
build passed; see [39](39-removing-implicit-semantic-authority.md#checkpoint-verification)
for the package and removal checks.
