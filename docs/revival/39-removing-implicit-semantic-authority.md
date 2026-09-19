# 39 — Removing implicit semantic authority

*2026-09-19. Implementation direction and breaking changes following
[36](36-structured-cognitive-workspace.md) and [38](38-scene-interpretations.md).
The owner explicitly requested removal of legacy paths rather than compatibility switches.
These changes reduce automatic behavior; they do not introduce a learned replacement.*

## Why removal is necessary

Retaining alternatives is insufficient if downstream code silently resolves references,
relabels predicates, repairs an attachment, or chooses a domain recipe after selection.
Those operations can overwrite the meaning that the interpretation policy actually selected.
A convenient default can acquire semantic authority without evidence simply because a later
component expects a concrete value.

The governing requirement is that substantive interpretation decisions be explicit,
inspectable, and revisable. Grammar and perception may propose structure. An explicit policy
may commit to a proposal. A downstream adapter must preserve that commitment or expose a
missing semantic obligation. It must not quietly manufacture a different reading to keep
execution moving.

## Removed production defaults and shortcuts

| Previous implicit decision | New boundary |
|---|---|
| Execute the first reader candidate without a supplied policy | Language acts defer until an explicit policy selects a candidate |
| Insert visual classifications directly through `Plugin.see` | Image providers return retained scene proposals through `interpret_image` |
| Automatically resolve selected acts through discourse heuristics | Selection no longer triggers a second automatic reference-resolution pass |
| Recast a location-like expression as time using WordNet | Preserve the supplied interpretation; any temporal reinterpretation needs an explicit account |
| Answer through an undirected binary-claim fallback | No fallback that ignores which relation and role were actually requested |
| Translate question or quantity predicates through built-in aliases | Preserve the supplied predicate rather than invent an equivalence |
| Rewrite a parsed named-object attachment after reading | Retain the reader's structure; repair requires an explicit interpretation proposal |
| Load bundled indirect-request conventions | Such conventions must be explicitly supplied; the bundled default resource is removed |
| Load a bundled Python-project recipe in `FileSystemPlugin` | Domain refinement knowledge must be explicitly supplied as a `RefinementLibrary` |
| Seed greeting replies through a built-in adjacency-pair table | No seeded greeting-response table supplies an unrequested default |

The filesystem still supports explicit grounded goals and action models. Removing its bundled
project recipe does not remove `mkdir` or `write_file`; it removes the assumption that the
agent already knows what an English project request should mean. `RefinementLibrary.load(path)`
requires an explicit path. Passing a library is an explicit authored-knowledge dependency,
not evidence that the agent learned that knowledge.

Likewise, an explicit interpretation selector is necessary for language dispatch but may not
be sufficient for a particular task. Its selected payload must contain adequate reference,
intent, and goal information, and any needed domain knowledge must be supplied. The agent
must be allowed to remain unresolved when one of those obligations is unmet.

Question-answering capabilities now declare an `Informs.query` proposition pattern and
an answer variable. The query must bind the supplied parameter to the requested entity;
only matching fresh observations supply an answer. Missing contracts do not invoke a
capability, and observations contradicting the declared projection produce an unknown
outcome rather than a false empty answer. This is an explicit observation contract,
not a restored global predicate-alias or undirected claim lookup.

## Intentional loss of default English automation

Historical examples that sent an English request to a default agent and obtained a project,
a greeting, or a reference-resolved answer may no longer work without explicit dependencies.
This is deliberate. Restoring the old behavior behind a compatibility flag would preserve the
same unsupported semantic commitments and undermine the direction of the work.

Documents [35](35-model-based-planning.md) and [37](37-interpretation-workspace-first-slice.md)
remain historical implementation reports. Their default-policy and default-recipe examples
must not be read as current setup instructions. Current callers should either supply a
structured goal or provide explicit interpretation, reference, and domain-knowledge policies
appropriate to their application. Merely moving the old first-choice behavior into a caller
callback does not satisfy the evidence-driven interpretation objective.

## Test knowledge is evidence about machinery, not agent competence

Removed authored examples can remain under test fixtures where their assumptions are explicit.
A test may supply a project recipe, request convention, or interpretation selector to exercise
planning, semantic preservation, or the dispatch boundary. The test then establishes behavior
conditional on those supplied assumptions. It does not demonstrate default English competence,
learned project knowledge, or general reference resolution.

Production code must not import those test fixtures or silently restore their content through
another resource. Verification should distinguish rejection of unsupported default behavior
from successful execution with explicit supplied knowledge. Regression totals alone cannot
show whether the agent understood a task.

## Remaining semantic commitments to audit

This cleanup is not a claim that every authored rule or fixed schema has disappeared.
Grammar and dependency readers still propose structures using their existing mechanisms.
Deterministic semantic projection, including `to_propositions`, and identity construction such
as `default_ref` still deserve review for lost distinctions and unsupported correspondence.
A string converted into an identifier has not thereby been grounded in an observed entity.

The boundaries of lexical resources, role mappings, reference contexts, answer realization,
and supplied refinement libraries also remain relevant. Some operations provide necessary
mechanics; others may still encode defeasible domain assumptions as unconditional behavior.
Audit each operation by asking what meaning it commits to, what evidence licenses that
commitment, whether alternatives survive, and how a later correction can revise it.

Do not replace removed shortcuts with a larger lookup table and call it learning. The next
positive capability should retain a consequential ambiguity, obtain distinguishing evidence,
and select or defer on that evidence while preserving dependencies and counterexamples.
Holistic scene understanding remains part of that same objective: global layout and relational
hypotheses must influence interpretation without becoming facts merely because a provider
emitted them. There is still no broad learned scene model or general semantic resolver.

## Checkpoint verification

The final regression suite passed with **1,485 passed, 5 skipped**. Tests cover
no-policy deferral, preserved selected structures, absence of the direct visual claim
interface, declared observation direction, retained scene alternatives, and execution
under explicit supplied interpretations and domain knowledge. These conditional tests
do not establish autonomous language or scene understanding.

The wheel build passed. Inspection confirmed that it contains the scene implementation
and excludes the removed request/project resources and all test fixtures. Current
architecture-document links and whitespace checks passed.
