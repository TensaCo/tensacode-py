# 54 — Investigation with pending interpretations

*2026-09-19. This connects the supplied-hypothesis investigation in
[40](40-evidence-driven-interpretation.md) to the retained continuation in
[53](53-resumable-workspace-interpretation.md), under the objective in
[36](36-structured-cognitive-workspace.md). It does not supply a learned hypothesis
producer or establish that world agreement identifies a user's intent.*

## A visible winner can still have an unseen rival

Investigation previously compared every materialized candidate, but pending semantic
work could contain another interpretation that had not reached the workspace yet.
Uniqueness among visible candidates did not justify treating that unfinished comparison
as settled. Doc53 made further candidates available through a separate expansion call;
this checkpoint connects that call to the supplied investigation policy.

`Agent.resolve_interpretation(group_id, hypotheses, max_expansions=64,
max_candidates=16, max_probes=8)` performs at most one bounded expansion phase followed
by one investigation. The supplied `hypotheses(group)` callback receives the enlarged
candidate set. It must cover every non-rejected candidate exactly once, including those
for which it can supply no predictions. A rival with no usable semantic predictions
remains viable; it cannot be silently omitted or treated as contradicted.

When `interpretation_hypotheses` is configured, the automatic text-selection path uses
this resolution method. Constructor settings `interpretation_expansion_budget`,
`interpretation_candidate_budget`, and `interpretation_probe_budget` bound it. There
is no default hypothesis producer. An application supplying only a direct selector does
not thereby receive this automatic expansion policy.

## Pending work blocks an investigation commitment

Both direct investigation and the integrated resolution path inspect continuation status.
If pending work remains after the available expansion, the public investigation result
and decision have no selected candidate and report `interpretation_search_pending`.
The observation source retains the visible-candidate assessment as `candidate_result`,
alongside the gated `result` and the continuation status. This preserves what the evidence
supported without presenting it as a completed interpretation decision.

Zero pending work only exhausts the retained, generated frontier. It does not recover
syntax removed by decoder pruning, prove semantic projection complete, or establish that
all possible meanings were considered. Selection still depends on supplied predictions
and fresh provider observations. Neither this gate nor a successful fixture makes those
predictions learned from the input.

The hypothesis producer and observation providers must not silently change the comparison
basis. Revision, candidate IDs, and continuation status are checked around callbacks.
Continuation generation makes newly attached or changed pending work visible even where
the candidate IDs stay the same. A stale callback cannot publish a decision about the old
set. Dispatch also rechecks continuation status before each act, so a later callback
attaching pending work cannot leave an earlier selection executable merely because its
candidate IDs did not change. The direct-selector acknowledgment mechanism in doc53
remains a separate, explicitly supplied policy route.

## Resource and lifetime boundaries

Caps apply **per group, per invocation**, not to the lifetime of an interpretation or
the entire multi-sentence turn. Multiple groups each receive their configured bounds.
A later explicit resolution call spends new budgets. There is no internal retry loop
that resets the caps until a winner appears. Either zero expansion or zero output budget
skips expansion; investigation can still retain evidence while withholding commitment.

Expansion counts semantic work and new output candidates. A probe counts a distinct
condition atom within that investigation and may call every provider; it is not a bound
on provider-call count or wall time. Repeated later investigations obtain fresh evidence
and spend their probe budget again. All continuation state remains in memory, and the
resource allocation is authored rather than learned information-value scheduling.

## Verification and evidence limits

Six focused tests in `tests/test_pending_investigation.py` passed. They establish that:

- A visible sole winner stays unselected while an unseen rival is pending; its conditional
  assessment remains inspectable in the evidence source.
- Automatic expansion exposes a new candidate before the supplied hypothesis callback;
  an authored filesystem-world model can then select and execute that candidate.
- A newly materialized rival with unknown semantics remains viable, and omitting it fails
  coverage validation.
- Either zero expansion/output cap leaves pending work intact, and later resolution calls
  spend separate bounded work without hidden extra rounds.

The execution test uses explicitly authored language proposals, world predictions, and
project refinements. It demonstrates the integration mechanism and its consequences,
not learned language understanding, grounding, hypothesis formation, or general cognition.
Runtime-wide stale-callback and regression verification is recorded after the final suite.

## Real-input scheduling audit

[The recorded audit](../../eval/results/pending_investigation.json) reuses the exact three
source IDs predeclared in doc53. Initial candidates come from the cached learned reader;
the supplied hypothesis callback deliberately gives every candidate empty predictions,
with the basis “No inferred grounding model; candidate remains unmodeled.” No plugin or
invented prediction supplies the missing semantic model. The repeatable command is:

```sh
.venv/bin/python -m eval.parsing.evaluate_pending_investigation
```

Each group grows from sixteen to twenty candidates before the callback runs. Resolution
performs four semantic expansions, below its bound of 64, and emits four alternatives,
at the output cap. Pending counts change from 16 to 12, 16 to 12, and 80 to 76. All sixty
candidates in the enlarged groups retain valid source anchors; callback and assessment
IDs cover the enlarged sets exactly, and original candidate payloads remain unchanged.
Guarded learned decoder methods receive zero calls during resolution.

All three public results and decisions remain unselected with
`interpretation_search_pending`. There are zero observations, tasks, or action executions.
Even an exhausted frontier would remain semantically unresolved with these empty
predictions; the fixture tests above separately establish the gate against a conditionally
supported visible winner. Resolution took 156.72, 207.82, and 220.48 milliseconds on the
shared verification host. These timings include copying and investigation overhead and
are not a controlled throughput comparison.

Every recorded source, model, cohort, and dataset hash matches its post-run check. This
audit establishes that the actual learned reader's unfinished work reaches the bounded
resolution path without reparsing and without inventing missing hypotheses. It does not
establish learned reference grounding, observational discrimination of real meanings,
semantic accuracy, or generalized task success.

## Final verification and next inference gap

The final full repository suite passed: **1,916 passed, 5 skipped** in 262.90 seconds.
The focused investigation, continuation, and callback suite also passed all 77 tests.
The real-input report's recorded source hashes still matched the current files after
verification. These checks cover the mechanism described here, not generalized cognition.

The next positive inference gap is between learned transition predictions and interpretation
investigation. Existing experience models predict outcomes for supplied actions, while
investigation still requires supplied condition predictions. A bridge should compare
candidate-specific empirical predictions for the **same** probe against its actual outcome,
retaining model applicability assumptions and unknown predictions. Different actions having
different effects does not establish which intent a user expressed. Evidence against a
model's applicability to this case must also remain distinct from falsifying its learned
rule in the context where it was trained and validated.
