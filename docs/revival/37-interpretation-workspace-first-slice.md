# 37 — Interpretation workspace: first implementation slice

*2026-09-19. Implementation report for the first step of
[36 — The structured cognitive workspace](36-structured-cognitive-workspace.md).
This adds retained reader alternatives and explicit selection mechanics. It does not
claim broad language understanding, automatic disambiguation, or integrated model learning.*

## What changed

The agent can now retain a text source and the alternative interpretations supplied by its
reader before executing an interpretation. Each sentence has a group of candidate readings.
The workspace records which candidate was selected, deferred, or rejected, together with
a reason and revision history. Candidates and original sources remain available after a
selection changes.

The relevant implementation is in [interpretation.py](../../src/tensorcode/agent/interpretation.py),
[understand.py](../../src/tensorcode/agent/understand.py), and
[core.py](../../src/tensorcode/agent/core.py).

`Agent.interpret(text)` parses and retains the result without plugin perception, executing
acts, or inserting candidate statements into the belief store. It returns an
`InterpretedMessage` with `transcript`, `source_id`, `group_ids`, and an `unavailable` value
when parsing returned `Unknown`. The original message is retained independently of sentence
normalization and composition. Groups retain sentence index/text provenance; exact source
spans are not supplied by this slice.

`Agent.turn(text)` uses the same retention path and makes an explicit selection for each
sentence before handling its acts. A caller can supply
`interpretation_selector(group) -> InterpretationDecision(candidate_id, reason)` to choose
among the supplied candidates. A decision with `candidate_id=None` defers that sentence:
its candidate acts are not dispatched, and an unknown outcome records the reason. Other
selected sentences in the same message can still execute. Normal turn perception and image
handling occur before text selection; deferring a text sentence is not a promise that the
whole turn performed no observation or belief-store update. All sentence selections are
made before any sentence acts are handled, so a selector cannot yet use context changes
caused by an earlier sentence in the same message.

The default remains a **reader-order compatibility policy**: it selects the first candidate
and records that basis. This is an explicit continuation of previous behavior, not an
evidence-driven ambiguity resolver. The full milestone-1 acceptance gate in document 36,
which calls for consequentially unresolved ambiguity to prevent default commitment, is
therefore still unfinished. Callers can opt into deferral now without changing the parser
or executor.

## Inspect without executing

```python
from tensorcode.agent import Agent

agent = Agent()
message = agent.interpret("make a python hello world project")
source = agent.interpretations.get_source(message.source_id)
assert source.text == "make a python hello world project"

for group_id in message.group_ids:
    group = agent.interpretations.get(group_id)
    assert group.selected_id is None
    for candidate in group.candidates:
        print(candidate.id, candidate.provenance)
        print([act.describe() for act in candidate.payload.acts])
```

This makes the reader's proposals available for inspection. It does not establish that a
correct interpretation is among them, or that every interpretation is represented. It also
does not fix the original compound project request.

## Explicitly defer a sentence

```python
from tensorcode.agent import Agent, InterpretationDecision

agent = Agent(
    interpretation_selector=lambda group: InterpretationDecision(
        None, "Interpretation requires review before using these candidate acts"
    )
)
turn = agent.turn("make a python project called hello")
for group_id in turn.interpretation_ids:
    assert agent.interpretations.get(group_id).selected_id is None
```

The callback in this example deliberately defers every sentence. A real interpretation
policy still needs evidence and a criterion for deciding when a commitment is justified.
Counting candidates alone is insufficient: several readings may be semantically equivalent,
and a singleton may simply reflect a reader that emits one guess.

## Selection and revision mechanics

The workspace exposes `add_source`, `get_source`, `sources`, `create_group`, `propose`, `get`,
and `values`. `select`, `unset`, and `reject` require a nonempty reason and append a revision
record. Candidate IDs belong to a group; a selection from another group is invalid. Rejecting
a selected candidate withdraws that selection. Explicitly selecting a rejected candidate
restores it and records the new decision.

Sources and payloads are copied on entry, and workspace access returns detached snapshots.
Mutating a nested value in a returned snapshot therefore does not silently change the stored
interpretation. Payloads and metadata must support `copy.deepcopy`. This is in-memory isolation,
not a persistence or serialization contract.

Selecting or rejecting a candidate through the workspace changes the recorded interpretation
only. It does not execute that candidate, resume a previous turn, retract beliefs, revise a
task, or undo an action already performed. There is no API in this slice for automatically
continuing a previously deferred group. Those behaviors require dependencies linking
interpretations to task revisions, beliefs, and attempts.

`Turn.interpretation_ids` links a turn to its groups. `Outcome.interpretation_id` and
`Outcome.candidate_id` identify the interpretation used for an outcome. Selection trace events
include the source, group, candidate, number of alternatives, reason, and revision. These
links expose a choice; they do not make its reason an independently verified explanation.

## Reader coverage and semantic limits

The grammar path retains the alternatives returned by its bounded reader. Reader ordering
and parse scores are not calibrated confidence in user intent. Learned-reader and composed
paths that still supply a single selected interpretation identify that limitation in candidate
provenance. Legacy readers without alternatives are represented as a single compatibility
candidate.

The slice retains candidate acts and parse details, including skipped or guessed words when
available. It does not add new grammar rules, solve attachment ambiguity, infer missing domain
concepts, or ensure that the candidate set covers the correct meaning. The selection callback
chooses among already proposed interpretations; it does not introduce model induction.

At this checkpoint there was no new image integration. The source record had modality
metadata, but the agent workspace path was text-based. A modality string alone does not
implement holistic scene understanding, relational visual graphs, layout and grouping,
competing scene hypotheses, or shared multimodal grounding. The owner's subsequent correction
in [36](36-structured-cognitive-workspace.md#vision-means-understanding-a-scene) makes this
broader visual objective explicit. Subsequent scene integration must report supplied-graph
behavior separately from the ability to infer scene structure from pixels.
[38 — Scene interpretations](38-scene-interpretations.md) reports that subsequent integration.

The workspace has no disk persistence, calibrated uncertainty, evidence-ranking policy,
automatic contradiction-driven revision, dependency invalidation, semantic equivalence test,
or natural-language correction/resumption behavior. Existing language projection and goal
refinement losses also remain. The default agent can still act on an incorrect first reading;
this report must not be cited as eliminating brittle language understanding.

## Verification scope

The agent integration checks exercise an alternate selection that changes the real filesystem
output from the default `hello` destination to `demo`, the compatibility default, deferral
without action, invalid-candidate refusal before dispatch, source retention when the reader
is unavailable, and revision of an earlier selection without replaying its action. These
checks establish the selection boundary and its links to execution. They do not establish
that an automatic policy can select the correct meaning. Both code examples above were also
run successfully against this implementation.

Verification for this checkpoint: the full suite passed with **1,419 passed, 5 skipped**.
A final focused run passed all **7 agent interpretation tests**, including an additional
deferred-statement check added after the full run started. The wheel build and whitespace
checks passed. These are regression results, not measures of interpretation accuracy.

## What the next checkpoint should establish

The next useful checkpoint should replace a consequential first-reading commitment with an
actual reasoning step. Retain two plausible interpretations, identify a distinguishing
observation, acquire it, and choose or defer on that basis. Preserve the observation and its
relationship to the choice, and demonstrate that later conflicting evidence reopens the
issue. The goal is a change in the agent's behavior supported by evidence, rather than a
larger collection of unused workspace records.
