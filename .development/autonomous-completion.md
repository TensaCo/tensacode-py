# Autonomous completion sequence

Owner request: continue autonomously through all pinned objectives. Main only,
verified milestone pushes, real-model work on GB10, artifacts under jacob-valdez
when promotion is justified. This sequence is execution work, not a completion
claim. Graph remains intentionally unimplemented.

## Active dependencies

1. **Evidence-conditioned useful chatbot.** Fix the measured single-source support
   bottleneck with explicit joint-evidence verification. Retain every source's
   contradiction veto, complete coverage, truncation abstention, source revisions
   and artifact identity. Evaluate source/joint with identical weights and policy.
2. **Response quality.** Expand natural reviewed training proposals on independent
   Hotpot training documents. Use 192 questions: 128 train, 32 calibration, 32
   development; seed 20260922. Reserve validation documents by title without
   inspecting reserved final question/answer contents. Keep all variants together.
   Preserve assistant label authorship, ambiguity and exact source evidence.
3. Train the existing three-axis assessor with full inputs, five epochs, AdamW
   2e-5, batch four, max512, seed20260922; no threshold search. Report every
   truncation exclusion. Temperature fitting uses calibration only. Compare
   train-majority, evidence-free and evidence-shuffled predictions. Ablated
   context changes support semantics; do not score them against unchanged labels
   as if they were fully reviewed truth labels.
4. Before integration, require each axis to reject at least half of its labelled
   development negatives while retaining at least 80% of positives; joint
   acceptance must retain at least half of all-known-good candidates with no
   known failures. Small-sample acceptance is provisional. Demonstrate evidence
   sensitivity on explicitly reviewed paired source interventions. Preserve
   failures and revise the approach instead of relaxing thresholds after results.
5. Integrate only qualified owned components, freeze the complete tool and run
   reserved final comparison with omission/revision/conflict controls. Judge
   actual answers using sources, not NLI/lexical approval. Include direct answering
   and identical-foundation workspace bypass.
6. Resolve workspace usefulness based on these controlled results; then implement
   and evaluate missing latent readout, multimodal relationships and action/outcome
   transfer. Schemas, authored scene graphs and simulations alone do not satisfy
   the objective. Pretrained tool releases require documented competent behavior.

## Validation and persistence

Use focused failing tests before code, exact artifact/optimizer continuation,
fresh-process inference, full suite/build/CI per coherent milestone. Store raw
training evidence and failures, update the big picture, and maintain clean main
between milestones. No arbitrary claim of “everything complete” while measured
capability gaps remain.

A separate foundation-capacity diagnostic compares the existing generator with
pinned `google/flan-t5-xl` revision `7d6315df2c2fb742f0f5b556879d730926ca9001`
on historical development questions only. Run direct answering and the same
proposal prompt before deciding whether more adaptation of the smaller model is
warranted. Any improvement is inherited foundation capability until controlled
training/workspace tests establish TensorCode-specific gains. This diagnostic
does not alter the v2 corpus's fixed original generator or data split.

## V2 result and next controlled interventions

The larger natural-proposal run failed all per-axis gates, and removing/shuffling
evidence barely moved support scores. Preserve v2 at `0a150f7`. Next compare only
input encoding (existing JSON sequence versus native tokenizer pair: question and
candidate / full evidence) using identical seed20260922, data and five-epoch
schedule. No threshold tuning. Separately prepare and review up to64 training-only
source-swap and omission pairs with the same questions/candidates. These are
explicit authored evidence interventions, not new natural outputs or human labels;
they cannot establish real-world correctness alone. No final cases are used.

## Current execution checkpoint

Trainable text/ViT OUTPUT_ENCODING now passes native-equivalence, gradient,
foundation and artifact tests, including ALBERT factorized embeddings and RoBERTa
position offsets found during independent review. Full suite: 699 passed, one
skipped; wheel and sdist build. This implements readout mechanics, not learned
shared semantics.

Paired assessor encoding reduced known-bad development approvals from 20 to 9;
adding 128 reviewed evidence interventions produced 12. Neither passes promotion.
The larger FLAN-T5-XL diagnostic completed all 32 historical cases without errors.
Direct answers appear stronger, while the existing declaration prompt often
copies source fragments; source-grounded review is in progress. Next test a
single simpler evidence-QA declaration prompt with the same foundation, then
measure the complete joint-verification tool. Keep prompt changes and foundation
changes separately identifiable. Do not access reserved final questions yet.

The larger foundation comparison is reviewed: base direct 18/32 correct, XL direct
29/32, XL old-prompt proposals 17/32, XL evidence-QA proposals 30/32. Reviews permit
clipped trailing prose when the answer is unambiguous; well-formedness is separate.
XL active/bypass first beams match; the workspace is untrained. Investigator now
owns prompt version 1 or 2, shared by generation and loss and retained in receipts.
Full XL + joint verification evaluation is running on GB10, with unchanged other
component weights and fixed policy, plus eight omission/revision controls.

Next diagnostic: frozen FLAN-T5-XL judgments for the existing reviewed quality
calibration/development candidates. Three fixed instructions assess support,
requested information and restrictions. Score the first decoder token conditional
on native `yes`/`no` token alternatives at threshold .5. No prompt search, fitting
or calibration; exclude and report any full-input overflow. Preserve original
per-axis/joint gates. These are inherited model judgments, not calibrated truth
or learned TensorCode quality. This investigates whether the stronger foundation
can assess question-conditioned quality that source-only NLI misses.

The frozen generative assessor also fails promotion (49 known-good and 14 known-bad
joint acceptances out of 91 development candidates). Next bounded learning test:
freeze FLAN-T5-XL, train only owned workspace/projection/gate on the 367 natural
training candidates' known axis labels as yes/no sequence targets. Three epochs,
batch four, AdamW .001, shuffle seed20260923; no threshold or prompt tuning. Use
ToolTrainer capture/replay and exact optimizer continuation. Compare final active
and bypassed workspace with identical frozen foundation/dtype/attention settings
on calibration/development; preserve failures. Foundation weights must remain
bitwise unchanged. This is a mechanism/capability experiment, not release approval.

Complete XL/joint run: seven correct, one incorrectly attributed answer, 24
abstentions. All eight omission controls abstain; six withdrawal controls echo
the withdrawal notice while not setting the formal abstention flag. Authored
conflict abstains. Oracle-corpus memory hit@1 is 1.0 (lexical .96875), not open
retrieval competence. No promotion. Full report and assistant reviews preserved.

Repeated fresh sessions previously rehashed all investigator weights, including
the multi-billion-parameter generator. The fingerprint cache is now shared per
Investigator under its existing session lock; evidence remains session-owned.
Regression tests verify zero extra tensor reads on fresh sessions, invalidation
on parameter updates and explicit invalidation across sessions.

Independent review found no blockers in the shared fingerprint cache; runtime
tests and manual configuration mutation, parameter replacement and concurrent
independent-session probes passed. Full local suite: 707 passed, one skipped;
wheel/sdist built. Source-only full-tool comparison is queued after workspace
training, using the identical saved XL generator, realizer, ranker and verifier.

Read-only architectural review identified predicate confusion from answer fragments:
February (birth month) conflicts with an October crash; early 1970s (band prominence)
conflicts with a 1984 song release. Next isolated diagnostic verbalizes question +
proposed answer into a self-contained claim using the existing owned XL generator,
without supplying evidence or gold answers to the rewriting pass. Retain originals,
transformation prompt and outputs; review fidelity before any admission. Compare
unchanged joint NLI on original and transformed text for all 32 historical first
beams and all 91 natural quality-development candidates, including known failures.
Literal answer retention is only a formatting guard, not semantic equivalence.

## Reviewed failures and connected readout checkpoint

Workspace-only XL adaptation failed: all 91 development candidates are approved,
including 28 known failures. Frozen-foundation bypass retains the earlier 70
approvals (14 known failures). Foundation weights remained bitwise unchanged;
optimizer continuation and fresh-process active/bypass scores on all 183
calibration/development records reproduce exactly. This is learned adapter change,
not learned useful discrimination. Do not integrate or publish these weights.

Source-wise full-tool screening yields 18 correct, 2 incorrect, 1 incomplete,
1 ambiguous and 10 abstained responses on the same 32 known cases. Joint screening
had 7 correct, 1 incorrect and 24 abstained. Scope alone does not solve completeness,
question restrictions or source attribution. Both remain unqualified.

Question/answer verbalization also failed fidelity review: 58 faithful,
40 incomplete, 13 malformed, 7 changed meaning, 5 ambiguous out of 123 rewrites.
NLI approves 63 rewrites, including 25 nonfaithful transformations. Do not admit
these rewrites merely because their entailment scores improve. Exact inputs,
outputs, assistant reviews and hashes are preserved in foundation-scale artifacts.

The connected OUTPUT_ENCODING example now initializes encoder and decoder,
collects dependency-bearing traces, reloads them for SGD, saves both operations
and reloads exact loss. GB10 FLAN-T5-small run: 16 supplied QA2D training pairs,
16 updates; readout and decoder bridge change, exact weights-only reload.
Frozen foundation preservation and dependency edges are regression-tested.
This establishes the requested training lifecycle, not useful semantic alignment.

Next isolate realization from selection failures before selecting another training
architecture. The stronger proposal generator still feeds the old base realizer;
a stronger realizer must preserve cognitive weights, screening, prompt, source
provenance and controls in a separately identified comparison. Reserved final
questions remain untouched. Overall cognitive objectives remain incomplete.

Realization diagnosis revises that next step: 21 of 22 accepted responses exactly
copy the selected candidate; the single changed answer introduces the magazine
attribution error. Bigger realization is not the principal blocker. The ranker
was trained on supporting-document relevance, whereas production ranks answer
hypotheses. Next bounded adaptation uses the existing reviewed candidate groups:
all axes true means positive; any explicit false means known negative; remaining
unknowns excluded without inferred labels. Train only mixed positive/negative
training groups with uniform positive targets, three epochs AdamW .001 seed
20260924. Reject complete groups with any segment exceeding the existing token
budget. Compare initial and adapted ranking, active/bypassed workspace, on fixed
calibration/development groups. Record all-bad groups explicitly: ranking alone
cannot establish answerability. Require complete-tool comparison and original
quality gates before any integration. No final questions or threshold tuning.

Candidate-ranking adaptation completed on GB10: 34 mixed training groups fit the
existing segment budget, 102 updates. Development mixed top-good rises 3/7 → 4/7,
but calibration falls 7/9 → 4/9; trained active and bypassed workspace give the
same development choices. No promotion. The ranker's frozen encoder remains
unchanged, with exact artifact reload and one fixed next optimizer update checked.
This is not shuffled-epoch continuation. Forty training, fifteen calibration and
five development groups overflow and are excluded in full.

Next verification adaptation will train the native XL foundation as well as its
owned adapters on the same reviewed training labels and fixed instructions. This
separates an insufficient adapter-only approach from supervised adaptation of the
inherited verifier. Three epochs, batches of four, foundation AdamW 2e-5, adapters
.001; no threshold search or final questions. Compare before/after active/bypass
at identical float32-parameter/bfloat16-autocast settings. Recreate an empty
optimizer before checkpoint restoration to avoid duplicate 3B optimizer states
in GB10 memory. Preserve exact next-update digests, foundations changes and all
failures; no learned-workspace benefit can be attributed merely to joint training.

Checkpoint restoration now has one encompassing ToolTrainer transaction rather
than nesting two full model/optimizer snapshots. Private preparation validates
before mutation; private application runs inside the outer modes/RNG transaction.
The standalone public loader keeps its own atomic rollback. Regression tests
measure one snapshot per original model/Adam storage and verify rollback after
optimizer state was applied and interrupted. Incoming checkpoint tensors and one
rollback copy still remain; this is not a measured GB10 peak-memory result. The
currently running foundation experiment retains its original executable code.

The fixed numerical criteria are now executable in
`experiments/assess_quality_gate.py`. It verifies exact report/data provenance and
coverage, accepts only the prescribed .5 threshold and keeps excluded examples
in retention denominators. Unknown labels never become truth labels; any explicit
false axis remains a known joint failure. Numerical passage alone does not
qualify a component: source sensitivity, complete-tool behavior and frozen final
validation are still outstanding. Frozen XL and workspace-only XL both fail.

Saved generative receipts now have an independent fresh-process verifier:
`experiments/verify_generative_quality_reload.py`. It checks exact data coverage,
instructions, token counts, parameter/autocast settings, foundation provenance,
recorded foundation digest where available, and every active/bypass score.
Current artifact hashes identify the files inspected; they do not independently
prove historical artifact identity. Verification is persistence, not qualification.
Local validation: 766 passed, one skipped; wheel/sdist built.

A tiny real-T5 CUDA probe on GB10 confirms that no-gradient capture inside the
same bfloat16 autocast context preserves the subsequent replay update: both losses
are 3.4375, complete state digests match, and all 26 foundation parameter gradients
are present in both paths. This checks the training mechanism, not learned quality.
Probe and log are retained under the cognition-20260921 remote run root as
`check_cuda_capture.py` and `artifacts/quality-v2-data/cuda-capture-check.log`.

Fixed a reproduced public-tool lifecycle failure: evidence remembered in one
episode could be retrieved in the next but could not be corrected under its
original logical ID. Explicit revision lineage now survives episode boundaries;
correction replaces the remembered entry transactionally and preserves immutable
history. Prior outcome feedback is cleared rather than reassigned to new content.
Removed/stale versions cannot return through correction or retrieval. Restoring
conflicting historical/memory content under one ID fails without replacing the
live session. Session schema 2 is required; model weight artifacts are unchanged.
Independent review passed after the restoration consistency fix. Full validation:
778 passed, one skipped; wheel/sdist built. These tests use authored outputs to
isolate revision mechanics, not to establish learned correction judgment.

## Native XL adaptation result

The GB10 run completed all three epochs and 639 updates. Foundation parameters
and all 18 adapter tensors changed; fixed-next-batch optimizer continuation is
exact. Active quality screening nevertheless accepts all 91 development
candidates, including 28 known failures; adapted-foundation bypass rejects all
91. Both fail the fixed gate, as do their before-training baselines. Calibration
also approves all 84 eligible candidates (42 known failures) with active workspace;
eight over-budget candidates are excluded. Preserve the checkpoints and raw
receipts; no admission change, final-set use, Hub promotion or release follows.

This joint run does not isolate native-only learning. Before another expensive
training run, audit teacher-forced alignment, masking, replay and native decoder
parity, then distinguish optimization collapse from implementation error. Useful
question-conditioned verification, learned workspace benefit, holistic visual
grounding and transferable action remain unresolved. A future qualified assessor
must screen both candidate admission and realized answers, with owned artifacts,
question/evidence context, complete coverage and revision invalidation.

Fresh-process reload reproduced all 183 calibration/development records exactly,
including 175 eligible bypass comparisons and eight explicit truncations. The
saved foundation digest also matches. No alignment, masking, decoder scaling or
trace-replay defect was found in a separate tiny-model audit: direct-native and
bypass losses/gradients match with tied/untied heads, EOS, padding and bf16.
Next diagnostic is inference-only on four deterministically selected calibration
rows (two all-known-good and two known-failure): inspect absolute yes/no mass,
token/EOS loss, residual norms and float32 versus bf16. Do not fit thresholds or
select another checkpoint from this diagnostic.

Evidence/memory consistency is enforced on direct session construction and
batched ingestion as well as restoration. A conflicting ID fails before any
batch record commits; identical remembered evidence remains valid. Independent
review passed. With the inference-only diagnostic tests included, local validation
is 786 passed, one skipped; wheel/sdist built.

The four-row diagnostic found a concrete magnitude failure: adapted workspace
residual RMS is 1,068–1,439 times native token RMS in float32 (up to 1,442 under
bf16), despite a raw gate of .004235. Float32 active conditional yes stays near
.7898; direct-native and bypass logits/loss are exact. This is not merely bf16
quantization. Bypass absolute yes/no probability mass is only about 1e-5–1e-4;
conditional scores alone conceal that loss of the prompted answer vocabulary.
See `docs/results/quality-collapse-diagnostic.json` for exact scope and hashes.

Bounded architecture correction: normalize the projected update relative to each
example's unmasked encoder RMS, then apply a tanh-bounded scalar gate. Keep masked
tokens excluded, zero inputs finite, actual gradients, and exact native bypass.
Record the changed conditioning contract in canonical configuration; reject old
artifacts that would acquire changed constructor defaults rather than silently
reinterpreting their weights. No legacy unbounded behavior switch is added.
Historical failed model behavior remains reproducible with its archived source.
Remote source archive: `foundation-xl/runtime-source.tar.gz`, SHA256
`960380d1d762b1705fd726eff920192841b5ef0d448670141bf73248e8c4546c`.

Next controlled run freezes the original XL foundation and repeats workspace-only
adaptation with the bounded residual, same reviewed data, seed, three epochs,
batch four and AdamW .001. Record active/bypass before and after at identical
precision, then exact continuation/reload and the unchanged numerical gate.
A magnitude bound is a mechanism correction, not evidence of useful cognition;
the run must earn any behavioral claim. No native-foundation retraining yet.

The bounded update and exact canonical artifact checks passed independent review.
An absolute norm floor addresses subnormal-gradient overflow; masking, zero
inputs, extreme updates, BF16 rounding, real component gradients and native bypass
are covered. Type-sensitive canonical JSON prevents `true`, `1` and `1.0` from
being treated as the same saved configuration. Full suite: 808 passed, one skipped;
wheel/sdist built. Current generative reports and reload diagnostics now require
the exact workspace configuration. Prior failed artifacts retain their archived
runtime, rather than receiving an implicit unbounded compatibility path.

Two action-lifecycle failures are fixed: generic ActionLoop receipts are copied
when observed and chooser-visible history is independently copied, so reused
mutable action outputs cannot rewrite prior feedback. State remains caller-owned;
external effects are not rolled back. PlanExecutor now rejects blank candidate
and action identifiers before execution, using the same invariant as trajectory
persistence. Independent review passed; full suite 817 passed, one skipped, and
wheel/sdist built. Tests isolate authored callback mechanics, not learned action
selection or real-world transfer.

Pinned Hub manifest audit also found Investigator/Decision Hotpot artifacts
missing three canonical defaults (`verification_scope`, `max_proposals`,
`proposal_template_version`). A metadata-only refresh is staged, preserving the
exact weight files. Verify historical-source versus current-source state and
prediction parity in fresh processes before publishing updated manifests. This
does not qualify the behavior of a newly trained tool; prior scope remains.

Bounded workspace-only adaptation completed: losses .2371/.2334/.2323 over 639
updates; the foundation stayed bitwise unchanged and 18 adapter tensors changed.
The previous all-approval collapse is gone, but development known-failure
acceptance worsens from 14 to 15 with the same 49 good and seven unresolved
acceptances. Calibration failures rise from 13 to 18. No useful workspace gain;
fixed numerical gates fail. Exact next-update continuation and fresh-process
reload of all 183 records / 175 eligible bypass receipts passed. Preserve this
negative result in `docs/results/bounded-workspace-quality-development.json`.

Repeat the already specified native-foundation adaptation with the corrected
bounded residual: fresh original XL foundation, float32 master parameters,
bf16 autocast, same data, seed, three epochs, batch four, foundation AdamW 2e-5,
adapters .001. This isolates the changed conditioning architecture from the
earlier unbounded joint run; do not warm-start from failed adapted weights.
Maintain all admission, source-use, complete-tool and reserved-final gates.

Investigator/Decision configuration-only refreshes are published under
`jacob-valdez`. Historical runtime `6607a8b` and current runtime `8e7c15a`
reproduce all 224 state tensors (13,590,657 parameters), identical raw weight
files and three authored supplied-candidate receipts. Fresh Hub loading at the
new pinned revisions also reproduces those checks; downloaded metadata matches
staged bytes and Hub LFS hashes match the original weight files. See
`docs/results/pretrained-configuration-refresh.json`. No weights were trained or
changed and no new task-performance claim follows. Historical release records
remain unchanged; catalog pins now point to the compatible metadata refreshes.

A frozen evidence-use control pack now selects the first 12 distinct, originally
all-known-good questions in prepared development order. Each keeps its exact
candidate with omitted evidence and with the next selected question's evidence.
Two independent assistant reviewers agree that all 24 variants lack supplied
support; completeness/constraints remain unknown. Exact sources, hashes, train
disjointness and anchor caveats are preserved under
`.development/datasets/quality-evidence-controls/`. No model scores informed
selection or labels. These authored development interventions do not replace
natural failures, complete-tool validation or the untouched final partition.

Evidence-control evaluation now uses the exact native three-axis prompts/scoring,
one owned model, and explicit active/bypass paths. It checks the source training
manifest against the frozen control pack, keeps unreviewed targets unknown, and
reports per-intervention coverage and support errors. GB10 tokenizer-only
verification finds all 36 inputs fit the unchanged 512-token budget (maximum 453).
The evaluator passed independent review, five focused tests, and a full suite of
832 tests with one skip. This is validation machinery, not a behavioral result.

The native-only training control uses an experiment-local, explicitly constructed
ToolTrainer adapter. It owns the complete Chatbot state for safe checkpoints but
trains only foundation parameters through `loss_batch(...,
workspace_ablation="bypass")`; all workspace, projection and gate weights must
remain bitwise unchanged. It starts from the same original XL foundation and
uses the same labels, order, seeds, precision, three epochs, batch four, and
foundation learning rate 2e-5. Active and bypass receipts are both preserved; the
primary control is bypass. The saved artifact remains an ordinary Chatbot, whose
default behavior has not changed. Fixed numerical, evidence-use, complete-tool
and untouched-final gates still apply. No native-only run has completed yet.

The native-only control passed independent code review and a tiny CPU BF16
continuation probe. Full local validation after both experiment harnesses:
837 passed, one skipped. These tests establish isolation and persistence
mechanics; the real GB10 comparison remains pending.

Bounded joint native-foundation/workspace adaptation completed all 639 updates:
losses .17135/.04271/.02296. The foundation and all 18 adapter tensors changed;
exact fixed-next-batch continuation and fresh-process reproduction of all 183
receipts / 175 eligible bypass receipts passed. Development accepts 49 known-good,
seven known-bad and six unresolved candidates, versus 49/14/7 before training.
All per-axis criteria pass; the combined gate fails. Calibration accepts 36 good,
12 bad and three unresolved candidates, with eight over-budget exclusions.
Active and bypass paths make identical threshold decisions for all 175 eligible
cases; maximum score difference is .00660. This is no demonstrated workspace gain.

The frozen, independently reviewed evidence controls also fail: both paths accept
all 12 original positives, all 12 evidence-free variants, and nine of 12 source
swaps. All 36 inputs fit. These are authored development interventions with
assistant labels, including the previously recorded anchor caveats. No promotion,
public-tool integration or reserved-final evaluation follows. Preserve raw
artifacts and exact source archive on GB10; compact records are in
`docs/results/bounded-foundation-quality-development.json` and
`docs/results/bounded-foundation-evidence-controls.json`.

The isolated native-only comparison is now running on GB10 from the same original
foundation, with all adapter weights frozen. Its training objective explicitly
uses bypass; saved Chatbot default behavior is unchanged. In parallel, prepare
TRAIN-only near-correct negative examples for independently reviewed gaps in
restrictions, predicate attachment, category narrowing and unsupported premises.
Do not fit or train on the development controls. Existing natural-label review
also flags ambiguity in typo/metonymy boundaries: preserve those caveats and
existing scores, rather than relabelling selected failures to pass the gate.

Before training, the native-only control reproduces every baseline calibration
and development receipt from the bounded joint run exactly, including scores,
coverage and both paths. Foundation assets, data, instructions, initialization
configuration and precision metadata also match. Its runtime archive is
`foundation-only-xl/runtime-source.tar.gz` on GB10, SHA256
`52d5866976cfd866fdb8aff7df00e3fe63ee103ce66847d23d577709704ad219`.

The separate TRAIN-only near-correct pack is reviewed and adjudicated: 32 variants
from 32 distinct positive training anchors, with exact source evidence retained.
Two assistant reviewers agree on every axis for 22 rows. Four explicit wrong-date
relations are incomplete; six unestablished qualification/alias constraints stay
unknown. Counts: support 8 true/24 false; completeness 28 true/4 false;
constraints 9 true/17 false/6 unknown. Both reviews and all ten resolutions remain
in `.development/datasets/quality-near-correct-negatives/`; independent provenance
and rubric review passed. No previous labels or held-out inputs were changed.

Next supervision comparison, after native-only baseline completion: append these
32 variants and the existing 128 reviewed TRAIN-only source interventions to the
367 natural training candidates. Preserve calibration/development bytes exactly
and validate question/source/text-hash disjointness. Start again from the original
XL foundation, train foundation only with explicit bypass, same seeds, 3 epochs,
batch 4 and 2e-5 learning rate; report the larger update count rather than describing
this as a matched-compute data-only causal comparison. Keep all admission gates
and the same reviewed development evidence controls. Unreviewed or unknown labels
must never become inferred negatives. No final questions are used.

The reviewed augmentation builder now produces 527 training rows, preserving the
367 natural-row prefix and the exact 92 calibration / 91 development rows. Its
manifest binds both training packs, review files, source hashes and label joins;
held-out question, source-ID and text-hash separation is checked again. The
optional evidence evaluator `--data` validates the actual augmented corpus and
unchanged held-out hashes before evaluating. Independent code review and all
18 focused tests pass; the preceding full suite passed 850 tests with one skip.
The prepared manifest SHA256 is
`d78fb0e4e796b2acd965277aa430c4ae565d5b93172cc8c797f1e73547a4021d`.
These are authored supervision and validation mechanisms, not learned results.
