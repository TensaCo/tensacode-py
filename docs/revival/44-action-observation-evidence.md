# Observations and action attempts enter the cognitive workspace

This checkpoint connects active perception and execution to the source evidence
required by [the governing objective](36-structured-cognitive-workspace.md).
Browser DOM/pixels and Gym transitions previously remained inside their adapters.
The agent could expose previews without retaining the observation on which a later
interpretation or learned model would depend.

## Active behavior

`Plugin.observe_evidence()` supplies raw evidence. Its default is `None`, meaning
this provider does not supply an observation through this interface. It must not
perform the requested action. Browser and Gym adapters expose their existing raw
observation methods through this explicit boundary. The separate method name avoids
confusing raw evidence with existing domain methods that observe interpreted frames.

`Agent.perceive` retains available observations as interpretation sources before
language interpretation. Around an invocation, the agent retains observations
before and after the action attempt. Both phases share an attempt identifier;
source metadata contains the structured call and the after phase records the actual
receipt. Applied, rejected, failed, and indeterminate receipts remain distinct.
Recording a receipt does not upgrade its outcome or retry the action.

The invoking provider, mounted providers, and explicitly supplied observers are
included once each by object identity. Observation failure from one provider does
not prevent observing another. An error after a mutation cannot erase its receipt.
Before and after observations are sequential samples, not an atomic snapshot of a
shared external world. The records alone do not establish that an action caused
every observed difference.

Sources use modality `observation`, provider `plugin:<name>`, and metadata:

- `stage`: `perception`, `before_action`, or `after_action`;
- `status`: `observed`, `unavailable`, or `error`;
- `observed_at`: a UTC timestamp;
- `attempt_id`, `action`, and `receipt` linking action evidence;
- error details or an explicit unknown reason when acquisition did not succeed.

Raw payloads are detached workspace snapshots. Browser payloads retain URL, title,
DOM and screenshot bytes together. Gym payloads retain observation values, reward,
termination, truncation, and transition sequence separately. Unknown or failed
acquisition is not a negative world observation. Silent providers do not add empty
sources during ordinary perception, but missing action observations remain explicit
so a learner cannot mistake incomplete coverage for a complete transition.

If structured call or receipt metadata cannot itself be copied, the source records
`linkage_snapshot_error`, the attempt identifier, and the actual receipt status when
available. It does not invent replacement structured arguments. Such an incomplete
record cannot become a training pair, and evidence retention failure must not prevent
execution or replace the actual receipt returned to its caller.

Live trace events carry source identifiers and acquisition status rather than raw
DOM, arrays, or image bytes. Consumers obtain retained evidence from the workspace.
Selecting an interpretation or fitting a model requires a separate operation.

## What this establishes

Tests exercise the real agent invocation boundary and real disposable Chromium and
Gymnasium environments. They verify before/after linkage, source immutability,
receipt preservation, acquisition failures, and the absence of automatic beliefs
from raw observations. Supplied calls and environment seeds are test controls;
these tests do not establish natural-language planning or a learned control policy.

[Transition learning](45-evidence-backed-transition-learning.md) consumes these
records through explicit projections and separated training/evaluation attempts.
The source layer preserves what happened without deciding what it means.

## Remaining boundaries

Existing explicit `Plugin.perceive()` claim reports remain a separate, older path;
this change does not validate or infer the semantics of those reports. Raw sources
do not themselves produce scene graphs, correspondences, hypotheses, or intentions.
The browser's DOM and screenshot are acquired sequentially and may reflect changes
between reads. Gym rewards retain their environment-defined meaning; the agent does
not silently adopt reward maximization as its user's goal.

Sources currently live in the in-memory workspace. Chat transcript persistence is
not persistence of these observations, learned models, or cognitive state. Retaining
large repeated screenshots also has a memory cost; durable evidence storage and a
retrieval/retention policy remain necessary. A provider name identifies its local
agent route, not authenticated origin or globally unique world identity.
