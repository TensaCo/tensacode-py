# Learned communicative interpretations

A structured frame describes a possible situation or relation. It does not by
itself establish whether the speaker is asserting it, requesting an action, or
asking a question. The previous active reader assigned these commitments through
punctuation, prefix, question-role, and subjectless-frame rules. This milestone
removes that implicit authority and retains neutral source-anchored frames while
learning communicative interpretations from explicit supervision.

This is a bounded learned association, not unrestricted English understanding.
The parser and existing lexical role/naming/frame projection still impose authored
choices. Removing a request heuristic does not prove the resulting frame captured
the user's meaning. The learned communicative proposal must remain attributable
to teaching and source evidence, with explicit admission and selection.

## Neutral structure before intent

Without an admitted communicative model, the reader must not turn a question mark,
a familiar prefix, or absence of a subject into executable request authority.
`language.deps_semantics.ProvisionalMeaning` retains the provisional `frame`,
`words`, `tags`, `lemmas`, head/label pairs, dependency root, and `frame_index`.
Character anchors remain in the owning reader alternative metadata. The root
identifies the source dependency tree, not a newly inferred independent clause
span or semantic alignment. Entity fragments retain their existing representation.

Source spans, syntax hypotheses, and projected frames remain available for
inspection rather than being discarded because the communicative function is
unresolved.

The neutral interpretation is not an instruction to execute, a factual assertion,
or an inferred question. It preserves a candidate structure for later reasoning.
A punctuation change may be evidence a learned model uses under its supported
scope; it is not a built-in decision rule restoring the retired behavior.

## Supervised intent and workspace alternatives

Explicit teaching associates retained source/frame evidence with communicative
intent. Training and held-out validation are source-disjoint. A fitted model must
be explicitly admitted before its proposals enter the active interpretation path.
Competing supported intents remain workspace alternatives; they are not resolved
by first-reader order or a punctuation fallback.

Source identity, teaching evidence, validation support, and model commitment must
remain reviewable with each proposal. Unsupported inputs, insufficient evidence,
or incomplete search must not become guessed requests. An explicitly selected
communicative candidate remains subject to the existing grounding, goal-selection,
semantic-preservation, and action guards.

The pure `learning.speech_act` API includes `SpeechActExample(id, source_id,
text, meaning, label, basis=())`, `SpeechActLabel`, and
`fit_speech_acts(training, validation, max_pairs=256)`. Its pair abstraction requires
matching tags, dependency tree, root, frame index, and taught label. It can abstract
varying aligned words/lemmas and corresponding lexical values in the projected
frame. Nonlexical structure and qualifiers remain literal. This is structural
transfer within a narrow pattern class, not arbitrary paraphrase understanding.

Supported templates require at least two training examples and held-out evidence.
Source IDs and normalized texts must be unique, and the token stream must agree
with the supplied text. Labels are explicit `request`, `statement`, `question`,
or `unresolved`; none comes from a punctuation or missing-subject prior. Competing
supported labels and unvalidated rival IDs remain retained. Question labels require an explicitly taught queried role, `query_path`, and
source token indices. Only that declared slot can be consumed, and only when its
scalar or featureless Entity text matches the contiguous taught token/lemma span
under the specified case/space normalization. Other qualifiers and clause polarity
remain intact. Wrong spans and erasure of qualified entities are refused. The
proposal retains the full original frame separately from the realized Question
frame; no wh-prefix or asked-role heuristic chooses the slot.

The pure learner passed nine focused tests. The agent APIs in
`agent.speech_act_learning` retain teaching with
`retain_speech_act_example(agent, group_id, candidate_id, act_index, label, *, basis)`,
fit with `fit_speech_act_model(agent, training_records, validation_records,
*, group_id=None, max_pairs=256)`, and explicitly admit with
`admit_speech_act_model(agent, handle, *, reason)`. Assigning the admitted handle
to `agent.speech_act_model` enables group projection on newly retained neutral
candidates, including continuation results. It does not select a candidate.

`propose_speech_acts(agent, admitted_handle, group_id, candidate_id,
*, max_alternatives=256)` retains joint alternatives and teaching/model evidence.
Incomplete or unresolved batches, including joint-budget exhaustion, do not
publish committed speech acts. Neutral reader outputs remain `Act("unresolved",
ProvisionalMeaning, None)` until such a projection supplies alternatives. Integration verification is recorded below. This does not establish a general
model of pragmatics or speaker intention.

## Breaking behavior and historical interpretation results

Default reading no longer assigns request/question/assertion authority using the
retired punctuation and grammatical shortcuts. Callers and tests that need a
specific communicative act must supply explicit teaching/model selection or an
honestly authored fixture. A migrated fixture does not count as evidence of
learned intent understanding.

The syntactic measurements in
[learned interpretation alternatives](47-learned-interpretation-alternatives.md)
remain measurements of their recorded decoder versions. Their historical
communicative projection must not be mistaken for the current neutral reading
path or for demonstrated learned speech intent. New intent behavior needs its
own evidence and scope statement.

## Acceptance evidence and remaining work

The neutral dependency-semantic projection passed **23 focused tests** after
removing `speech_act`, punctuation/wh-prefix detection, and the subjectless-request
rule. It assigns no communicative mood and does not delete a wh-role to fabricate
a question. A combined run passed **82 tests in 22.79 seconds**, including four actual-reader
cases, nine pure learner tests, twelve evidence-wrapper tests, and chat/browser/
gym connection regressions. The existing frontend regression harness also passed.
Dead communicative conventions were removed, and the final backend focused run
passed 27 tests. The full repository suite passed **2,556 tests, with two skipped**,
in 614.07 seconds (exit status 0). Focused runs overlap and are not additive.

`tests/test_learned_speech_act_inputs.py` uses locally trained segmentation, POS,
and dependency artifacts on actual text. “open files” and “running errands” have
no default communicative authority. Explicit request teaching on “open files” and
“open folders,” with “open windows” held out, transfers a request proposal to
“open pictures.” Separate taught unresolved running-phrase examples keep “running
checks” unresolved. Withdrawing the admitted model prevents subsequent use.
Conflicting request/statement teaching retains both interpretations of “open
pictures”; no selection policy means no automatic act dispatch. These are small,
chosen examples within a bounded syntax class, not a population accuracy estimate.

Failures and ambiguity are distinct from the availability of a useful candidate.

Teaching labels and initial selection policies remain authored. Lexical role
mapping, entity naming, polarity/modality mapping, and dependency-to-frame
projection remain authored too.
The learned component is the association supported by the new communicative
examples; that does not establish free-language goal formation or understanding
of arbitrary indirect requests, sarcasm, discourse context, or social intent.

A migrated quantity fixture exposes a separate remaining limit. The neutral
reading of “how many plants does Shondra have?” preserves the counted object and
its noun constraint; binding only Shondra correctly abstains rather than deleting
“plants.” Existing informing-role correspondence can still choose the wrong
subject/object filler from frame order after further binding. This milestone does
not fix generic quantity scope or informing-role alignment. A positive fixture
for explicitly taught “how much does Shondra have?” demonstrates narrower
reachability, not a resolution of the counted-noun case.

The next boundary is a learned correspondence from the full Question to an
informing capability call and answer query, preserving every role and feature
under explicit selection and dependencies. `_look` and `attend` still use
`_filler_for_role`, whose first-insertion subject/object choice for `undergoer`
can misbind the question. An owner-only quantity capability cannot honor a
counted-kind restriction. Acceptance should require role-order invariance,
distinct plant/coin restrictions, held-out owners, conflicting correspondences,
and qualifier-preserving abstention when the capability cannot express the query.
This is next work, not functionality added by the communicative learner.

Remaining gaps include richer context, transfer to unfamiliar structures,
question meaning, scope and negation, discourse reference, evidence-driven
selection, and learning the upstream frame projection itself. Neutral structure
removes a premature commitment; only verified learned behavior warrants a claim
of improved interpretation.

[Learned informing correspondences](76-learned-informing-correspondences.md)
develops the concrete next boundary above: taught full-question mappings to
observation calls and answer queries, preserving quantity scope and qualifiers.
