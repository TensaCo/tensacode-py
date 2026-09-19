# 53 — Resumable workspace interpretation

*2026-09-19. This checkpoint follows the unfinished semantic work measured in
[52 — Global interpretation retention](52-global-interpretation-retention.md).
It concerns continuing an interpretation in the same live workspace, not choosing
the correct meaning or completing general cognition.*

## The missing capability

Doc52 exposed pending semantic work but discarded the live search cursor when a read
returned. Its counts and deferred syntax metadata could explain incompleteness without
letting the workspace directly ask for the next alternatives. Repeating the input
could repeat learned segmentation, tagging, and dependency search.

The new path retains a detached sentence continuation in the interpretation workspace.
`Agent.expand_interpretation(group_id, max_expansions=..., max_candidates=...)` advances
that retained work and appends newly materialized alternatives to the existing evidence
group. It uses already generated source-anchored syntax and semantic frontiers. It does
not rerun the learned parser or promote a proposal into a decision.

The scheduler uses authored round-robin allocation, with explicit per-call expansion
and output bounds. These are resource policies, not learned confidence or expected
information gain. Pending counts summarize queued branches and unstarted families;
they are not an exhaustive number of possible meanings. All generated syntax families can continue, including those excluded by the initial
retention cap. Syntax eliminated by decoder beam or expansion-budget pruning cannot be
recovered through a semantic continuation.

This is **in-memory across-turn state only**. The workspace owns a detached cursor;
there is no durable serialization, restart recovery, cross-process handoff, or stored
chat-session reconstruction of the frontier. Keeping a group alive is necessary for
later expansion. Calls are explicitly requested through the API; no autonomous policy
yet decides when or which unresolved group to expand. The detached state preserves adapter
configuration but does not version-freeze module code or external lexical resources not
yet loaded. Audit source hashes establish code stability during the recorded run only.
Unstructured vision and generalized learning remain separate gaps.

## Workspace invariants

Existing candidate IDs and source evidence remain stable as candidates are appended.
New alternatives withdraw any previous selection and record a revision, requiring a
new decision over the enlarged set. Search progress without a new alternative records
work but does not itself disprove a previous selection. Expansion neither executes
actions nor rewrites past task receipts.

A supplied selection callback can intentionally ground or expand candidates while making
a decision. It must then identify the exact fresh comparison basis with
`InterpretationDecision.compared_revision` and `compared_candidate_ids`; the agent verifies
both against the current workspace. An unacknowledged stale decision is rejected. This
acknowledgment records what the authored policy claims to have compared; it does not prove
semantic completeness or turn the callback into default selection authority.

Workspace expansion advances a detached cursor and prepares the candidate copies before
publishing its state. Projection or copy failures therefore leave the owned cursor
available for a later attempt. This atomic publication behavior is a mechanism guarantee;
it does not establish that the eventual candidates match user intent.

## Predeclared real-input audit

[`continuation_cohort.json`](../../eval/results/continuation_cohort.json) fixes the first
three ordered IDs from doc52's reused twelve-input diagnostic cohort before continuation
inference. No quality outcome selects these examples. Original CoNLL-U `# text` is used
unchanged. Each input is initially interpreted by the cached learned reader with its
default budgets; each resulting group then receives two separate expansion calls,
each bounded to 64 semantic expansions and four newly returned alternatives. These are
separate calls against the same live agent, not a process restart or a demonstration that
the demo chatbot persists or automatically schedules interpretation work.

The audit records initial and appended candidate IDs, source anchors, model/source
hashes, per-call work and pending counts, and timings. During expansion, guarded learned
segmentation, tagging, and dependency decoder methods count and reject any invocation.
The guard applies to decoder classes, including detached copies. It checks that old
candidate payloads survive unchanged and that expansion creates no selected meaning or
task. This is a reproducible mechanism audit on real inputs, not an attachment-accuracy,
semantic-completeness, or end-to-end task-success benchmark.

The repeatable command is:

```sh
.venv/bin/python -m eval.parsing.evaluate_continuation
```

Two focused measurement tests verify that the decoder guard catches detached model
copies and that malformed source anchors fail validation. Both pass; these fixtures isolate
measurement validity from the real-input audit below.

## Recorded result

[The real-input report](../../eval/results/interpretation_continuation.json) completed all
three predeclared inputs. All source, cohort, dataset, parser-model, and segmentation-model
hashes matched after the run. Each input produced one workspace group with sixteen initial
alternatives, followed by four new alternatives in each of the two explicit expansion calls.

| Source ID | Initial interpretation | Pending before / after call 1 / after call 2 | Expansion call 1 / call 2 |
| --- | ---: | ---: | ---: |
| `email-enronsent18_02-0044` | 3.615 s | 16 / 12 / 8 | 43.84 / 47.46 ms |
| `weblog-blogspot.com_aggressivevoicedaily_20060811122000_ENG_20060811_122000-0029` | 11.065 s | 16 / 12 / 8 | 54.64 / 58.79 ms |
| `email-enronsent28_01-0040` | 3.786 s | 80 / 76 / 72 | 56.41 / 58.64 ms |

Each call performed four semantic expansions, below its bound of 64, and returned four
candidates, at its output cap. Across the three inputs, 24 new alternatives reached the
workspace after 24 additional semantic expansions. All 24 have valid source anchors and
contain acts. Existing candidate IDs and payloads remained unchanged; original source text
was preserved. Every guarded decoder-method count remained zero. No meaning was selected
and no task was created. Pending work remains on all three groups.

The appended alternatives include five, six, and zero span/head/label trees respectively
that were absent from each input's initial set. This demonstrates that the continuation can
materialize some previously deferred syntax families; a new candidate ID does not by itself
mean a novel tree or a correct interpretation. Tag or semantic distinctions can produce
additional candidates for a tree already present. No gold labels were used to score or
choose these outputs.

Initial interpretation took 18.466 seconds in total; the six subsequent expansion calls
took 0.320 seconds, including workspace cursor/payload copying. These are different workloads,
not a controlled throughput speedup benchmark. Concurrent verification shared the host.
The measured mechanism avoids repeated decoding while making more already-bounded work
available to the workspace. It does not prove semantic coverage, appropriate automatic
allocation, grounded identity, reliable intent selection, or task success.

## Verification

The final full repository run passed: **1,886 passed, 5 skipped** in 262.97 seconds.
It includes continuation isolation and rollback, bounded expansion, stable source and
candidate retention, stale callback/dispatch rejection, and fresh investigation of an
expanded candidate set. The authored grounding selector fixtures explicitly acknowledge
their new comparison set; no runtime fallback accepts a stale decision. The separate
real-input audit above verifies source-anchored expansion without decoder replay.
