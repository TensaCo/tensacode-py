# Validation and scope

TensorCode provides operation, tool, persistence and training paths described below. Semantics still come from explicitly supplied models, training data, algorithms and policies; importing an operation does not install a cognitive ontology.

## Implemented surfaces

| Area | Active behavior | Important boundary |
|---|---|---|
| Vector operations | Space-tagged native tensors; text and spatial image encoding; supplied transforms/decoders; classification, candidate scoring, decision and retrieval | Random initialization supplies no semantics. Matching dimensions alone do not establish matching spaces. Discrete selection does not make its index differentiable. |
| Message operations | Immutable text/image messages, transforms, validated classification/decision/scoring/retrieval, explicit async and batch calls | Structured answers can fail validation. Missing confidence stays missing; model probabilities are not calibrated truth. |
| Providers | Explicit provider-neutral request/response protocol, HTTP adapters, local Transformers image/text model adapter | Remote credentials and model choice belong to the caller. Local models must be explicitly acquired; image URLs are not fetched by the local adapter. |
| Graph representation and reserved operations | Immutable identities, attributes, relations and source anchors; structural lookups | Symbolic encode/decode, transform, scoring, retrieval, decision and classification are unimplemented stubs that raise `NotImplementedError`. No active neural graph adapter. |
| Owned tools | Configured Investigator generates hypotheses and assesses source support/contradiction; cognitive Chatbot retains interpretations and screens its decoded response; Planner generates or ranks proposals; Scene supports candidate ranking and owned visual language interpretation | Complete artifacts must contain the relevant components. Generation and NLI inherit supplied foundation capabilities. Authored screening is not a truth guarantee. Measurements below limit capability claims. |
| Runtime | Independent sessions, immutable evidence revisions, learned-encoder episodic retrieval, bounded action execution and outcome-driven replanning | Action authority and external effects remain explicit. Generated text is inert. Saving model weights excludes runtime sessions. |
| Tracing | Identity-based dependency capture, explicit scalar handles, native live gradients, async boundaries, external roots and dependency closure | Ordinary Python between operations is not inferred as a differentiable operation. |
| Persistence/training | Data-only experiences, explicit target provenance, named operation/configuration bindings, intermediate release, recorded external boundaries, trainers and checkpoints | No executable-program deserialization. Remote outputs can be recorded constants, never differentiable remote calls. Custom codecs/configurations are trusted caller declarations. |

Start with the [documentation index](README.md) for current API guides and runnable examples.

## 0.4 alpha pretrained vector checks

The [latent foundation report](results/latent-foundations.json) records bounded
GB10 checks using pinned public weights. These validate integration and restoration,
not a shared cognitive space or generalization:

Those measurements retain the pre-0.4.0a2 source revision. The separate
[a2 artifact recreation record](results/latent-foundations-a2.json) documents
new artifacts and old/new weight, output and training-continuation checks on the
GB10. This is restoration validation, not a new pretrained quality evaluation.

| Operation | Measured behavior | Boundary |
|---|---|---|
| FLAN-T5 text encoding/decoding | Final encoder states match native output exactly; native input-embedding decoding produces identical text; trained artifact restores identical output | Final encoder states are not interchangeable with native input embeddings. |
| Text adapter training | Four authored pairs, four optimizer steps: cross-entropy 9.48475 → 4.88764; training progress and weights restore | Outputs remain poor. This is a lifecycle demonstration without held-out evaluation. |
| ViT image encoding | Real photograph produces `[1,196,768]` patch states; native comparison and complete artifact reload both have maximum absolute error 0 | Patch features inherit the supplied ViT; no scene reasoning is established. |
| SD-Turbo image decoding | Four DDIM steps at 512×512, fixed noise: native-pipeline and restored-output maximum absolute error 0 | DDIM replaces the foundation's default sampler. Conditioning is explicitly the same foundation's CLIP space. |

The generated image depicts a red ceramic vessel but has malformed teapot geometry;
native parity does not establish image quality. Arbitrary latent-to-language and
latent-to-image bridges remain untrained until supplied paired supervision.
The [runnable example](../examples/pretrained_latent_lifecycle.py) demonstrates
initialization, collection, training, artifact saving and restoration; the
[operation guide](latent-models.md) specifies supported model families and inputs.

## Pretrained tool measurements

These are small, fixed-split experiments on public data, not general cognitive
benchmarks. Complete model configurations, weights and evaluation records are
[hosted on Hugging Face](results/pretrained-releases.json). Each tool reconstructs
its owned components without requiring a caller-supplied model.

| Model and task | Before → after training | Workspace ablation |
|---|---|---|
| [Chatbot: HotpotQA answers](results/chatbot-hotpot.json), 256 train / 64 held out | Exact match 42.19% → 46.88%; token F1 55.63% → 60.15% | Bypassing slot updates gives identical answer metrics and slightly better cross-entropy. Removing all encoded evidence gives 0% exact match. |
| [Investigator: supporting-document ranking](results/investigator-hotpot.json), 1,024 train / 128 held out | Hit@1 28.13% → 55.47%; supporting-document recall@2 25.00% → 42.58% | Zero workspace gives the same hit@1 and recall@2 43.36%. |
| [Planner: document-read relevance](results/planner-hotpot.json), 1,024 train / 128 held out | Hit@1 28.13% → 55.47%; relevance MSE 0.19878 → 0.14763 | Zero workspace lowers hit@1 to 52.34%, but improves recall@2 from 46.48% to 47.66%. |

Chatbot inherits FLAN-T5-small language/instruction weights and uses **oracle
supporting passages** supplied from annotations. This evaluates answering given
relevant evidence, not retrieval, unrestricted conversation, or autonomous
investigation. A bootstrap weight-alias bug was caught and corrected before this
run; corrected foundation and workspace-bypass outputs were checked for exact
parity before fine-tuning. Earlier invalid artifacts are not distributed.

The rankers inherit a frozen Electra-small encoder. All annotated supporting titles
are treated as valid hypotheses. An authored token-overlap baseline scores 53.13%
hit@1 and 42.97% recall@2. Decision exposes the same trained weights through the
decision interface; it is not independently evaluated. Planner's labels measure
**document relevance**, not observed causal utility of executed plans. A failed
random-embedding pilot motivated this architecture; its evaluation subset was
excluded from the final held-out subset.

These results establish owned pretrained model loading, parameter learning and
fresh-process restoration on narrow tasks. They **do not establish consistent
benefit from the recurrent slot workspace**, autonomous hypothesis discovery,
calibrated uncertainty, or general reasoning. The workspace is an implemented,
trainable research mechanism; schemas and gradient tests alone do not prove those
capabilities.

## Visual experiment: negative result

[Scene VSR results](results/scene-vsr.json) use 512 training and 128 held-out real
COCO photographs with spatial-caption annotations. Image IDs are disjoint across
splits, and the held-out photos exclude the earlier random-encoder pilot's test
photos. The model owns frozen CLIP perception, positional image/text encodings,
the shared workspace and a learned candidate scorer.

Accuracy falls from **55.47% to 50.78%** after the prescribed ten training epochs.
Blank-image accuracy is **53.91%**, and zero-workspace accuracy is **52.34%**.
These results do **not** demonstrate useful visual grounding or scene reasoning.
The checkpoint is an explicitly experimental negative result, not a recommended
pretrained visual assistant. A prior random-patch model also failed (56.25% to
43.75% on a separate 64-photo evaluation).

The complete learned weights reproduce metrics after a fresh process restart.
The full binary training checkpoint also restores optimizer state, RNG state and
5,120 training steps. This verifies the training/software lifecycle independently
of the failed capability evaluation.

## Learning from executed outcomes

[The service-recovery example](../examples/learn_action_outcomes.py) collects 18
actual simulator transitions. Each training trace labels only the candidate that
was executed, using its observed reward. After 324 small supervised updates,
success changes from 0/6 to 6/6 evaluation scenarios and mean reward from -0.5 to
1.5. A fixed-action baseline and a policy that ignores changed state both score
0/6. Model, experience, optimizer continuation, session and trajectory reloads
reproduce the recorded behavior. [Full result](results/action-outcomes.json).

The six scenario IDs are disjoint from training, but use the **same three authored
status classes and deterministic transition rules**. This is a small mechanism
test of feedback, learning and replanning, not a real-world benchmark, causal
discovery result or test of novel action semantics. The production library does
not import the example's environment or domain policy.

## Source-wise verification and calibration

[The verifier run](results/verifier-snli.json) uses a pinned DeBERTa NLI foundation,
1,024 SNLI training pairs, 256 separate validation pairs for temperature fitting,
and 256 test pairs. Training and real-model evaluation ran on the connected GB10.
The foundation was already trained on SNLI/MultiNLI, so these splits establish
isolation for this fine-tune, not previously unseen foundation data.

| Test metric | Pretrained | Fine-tuned | Fine-tuned + temperature |
|---|---|---|---|
| Accuracy | 92.58% | 92.19% | 92.19% |
| NLL | 0.2913 | 0.3401 | 0.2420 |
| Brier | 0.1281 | 0.1343 | 0.1220 |
| ECE | 0.0577 | 0.0649 | 0.0286 |

Fine-tuning lost one correct test prediction. Temperature fitting improved the
reported probability metrics while preserving predictions; its value is 1.9768,
fit on validation scores only. Full checkpoint reload reproduces logits exactly.
Five explicitly authored diagnostic pairs also exercise entailment direction,
contradiction and unrelated evidence. They are mechanism checks, not an additional
statistical benchmark. NLI support is model inference, not a guarantee that a
source is true, complete, current or trustworthy. Recalibrate after weight changes.

## Owned visual language: inherited baseline

[Scene language results](results/scene-language.json) exercise an owned
SmolVLM-256M model on 32 real VSR images. Spatial-caption yes/no accuracy is
20/32 (62.5%). With blank images, agreement with the **original** image labels is
15/32 (46.88%); with different images it is 13/32 (40.63%). Those controls measure
original-label retention, not correctness against newly annotated altered images.
The majority-label baseline is 17/32 (53.13%), just three fewer correct answers.
The images do not overlap the earlier TensorCode visual training images; exposure
during foundation pretraining is unknown.

This model was not fine-tuned. Its workspace residual starts at zero, so these
results describe inherited VLM behavior and complete TensorCode ownership, not
learned workspace improvement. Four additional scene descriptions are preserved
for inspection without assigning truth scores. Three hit the 64-token evaluation
limit, and blank inputs elicited hallucinated content. Outputs therefore remain
explicitly unverified interpretations with full-image anchors and unknown
confidence; they do not create accepted scene facts or symbolic graph structure.

## Hypothesis generation and faithful realization

[Hypothesis training](results/hypotheses-qa2d.json) fine-tunes FLAN-T5-small on
1,024 human QA2D declarations, with 128 development and 128 test examples grouped
by disjoint source articles. Inputs contain only the question and original SQuAD
paragraph; answer annotations and target declarations are excluded. After three
fixed epochs, test declaration exact match rises from **1/128 to 30/128** and
token F1 from **19.97% to 82.92%**. This is improved declaration generation given
relevant evidence, not reliable factual inference: inspected outputs still change
numbers and other facts. Foundation exposure to these datasets is unknown.

A post-hoc NLI audit accepts only 35/128 human references and 29/128 generated
statements under the configured support policy. Even among untruncated pairs,
only 26/94 human references pass. This exposes a verification/domain limitation;
NLI acceptance cannot serve as answer accuracy. The report preserves raw outputs,
truncation counts and the distinction between model support and truth.

[Realization training](results/realization-qa2d.json) addresses a separate failure:
the decoder shortened accepted statements into fragments that failed subsequent
verification. FLAN-T5-base learns to preserve an **already selected statement**
using the production realization prompt. The target is deliberately present in
that input; this is faithful rendering, not target-blind question answering.
Three fixed epochs on 256 statements raise normalized statement exact match from
4.69% to 100% on 64 article-disjoint development examples. Verbatim match reaches
96.875%; the remaining differences remove backtick quotation marks. No separate
test score is claimed for this component.

Both runs execute on the connected GB10 and restore complete model and optimizer
checkpoints. Hypothesis continuation is bit-exact only in the explicitly recorded
deterministic CUDA probe; the default CUDA continuation showed numerical
variation despite equal initial state, batch and loss. None of these component
measurements establishes an end-to-end reasoning improvement by itself. Training
scripts received reporting-only corrections after these runs; recorded script
hashes identify the versions actually executed, while current sources add
configurable card metadata and truncation audits.

## Complete cognitive pipeline: experimental result

The [final cognitive evaluation](results/cognition-hotpot.json) freezes the full
configuration and component hashes before selecting 32 new HotpotQA validation
questions. It combines the trained hypothesis generator and realizer above,
Electra ranking, calibrated source-wise NLI, and an owned MiniLM sentence encoder.
Questions receive oracle supporting passages. The prior 16-question run was used
for diagnosis and component changes and is recorded as
[development evidence](results/cognition-diagnostic.json), not a final test.

On the final 32 questions, the chatbot returns **30 abstentions, one correct
answer and one circular non-answer**. Both non-abstained responses were manually
checked against the original sources and gold answers. The correct response
identifies James Franco; the other repeats “uppermost age range” without supplying
an age. Passing NLI therefore does not establish relevance or answer completeness.
Literal short-answer exact match is zero because the one correct response is a
full sentence; it must not be confused with the source-reviewed 1/32 correct rate.
There were no failed calls or generation truncations. All eight omission and all
eight replacement controls abstained, but high abstention on original evidence
prevents interpreting this alone as successful counterfactual reasoning.

Dedicated MiniLM retrieval finds a supporting passage at rank 1 for all 32
questions on this small oracle-passage corpus; a lexical baseline gets 31/32.
Both reach 32/32 at rank 5. This inherits a pretrained sentence embedding space;
no retrieval fine-tune or open-domain retrieval result is claimed. In a separate
smoke test using the actual developer guide, evidence survived episode boundaries
and exact session save/load and was retrieved again, but the chatbot still
abstained on the question. That verifies memory behavior, not developer-document
answering competence.

These complete artifacts demonstrate owned component execution, revisable source
state, persistent retrieval and response screening. They remain experimental:
source-wise verification rejects many useful statements, generation can invent
facts, and screening admits non-answers. Reliable multi-source inference,
answer completeness, useful response coverage and a demonstrated cognitive
workspace advantage remain unsolved. The symbolic graph path is still a stub.

## Cognitive coverage: development experiments

The [controlled verifier comparison](results/cognitive-coverage-development.json)
reuses the previous 32-question final set as **development data** after inspection.
It is not a new held-out improvement result. The proposal generator, ranker,
workspace, realizer, token budget and screening thresholds remain fixed. Only the
verifier foundation changes, with temperature fitted on 256 separate SNLI
validation pairs. No classifier weights are trained in this comparison.

Assistant review against the supplied sources classified the responses as follows:

| Outcome | Original verifier | Replacement verifier |
|---|---:|---:|
| Correct and responsive | 1 | 6 |
| Incorrect response | 0 | 1 |
| Incomplete or non-answer | 1 | 3 |
| Abstention | 30 | 22 |

The replacement **failed the predeclared promotion gate**: incorrect/incomplete
outputs increased from one to four. Omission, replacement and the authored
conflicting-source controls abstained, but those controls do not establish answer
completeness. Every non-verifier tensor stayed identical; complete artifact and
probe-logit reloads were exact. The experimental artifact remains on GB10 and was
not promoted or published as an improved checkpoint.

A separate question-answerability model rejected two non-answers but still gave
high scores to the wrong surfing location and a magazine-name response where the
question asked for its type. Those are post-hoc development scores, not a tested
combined pipeline. Earlier source segmentation lost the correct James Franco
proposal and admitted a repetitive non-answer. Neither diagnostic became a
production fallback or relaxed threshold.

The next training target is evidence-conditioned completeness and constraint
satisfaction. Source support, topical relevance and fluent realization each leave
important failure modes. Planned fresh final cases remain untested because the
candidate intervention failed development acceptance. All factual outcome labels
here are source-grounded assistant review, not independent human annotation.

## Actual learning across process restarts

[Banking77 restart results](results/banking77-restart.json) were produced by [the executable example](../examples/banking77_restart.py). Four subprocesses terminate in sequence: capture/save, baseline evaluation, reload/train/checkpoint, and final checkpoint evaluation. The run captures 79 experiences containing 9,997 explicitly labeled training rows. Six literal train/test overlaps are excluded. Vocabulary is built from training data only.

On 3,080 official held-out examples across 77 labels, accuracy changes from **0.97% to 89.48%** and cross-entropy from **4.3646 to 0.4566**. The model is a trainable 96-dimensional mean embedding and linear classifier, with fixed seed 7, Adam at 0.01 and 20 epochs. This demonstrates that persisted operation dependencies support actual supervised parameter updates after restart. It does not demonstrate autonomous target discovery, calibrated confidence, or arbitrary-program differentiation.

Dataset source: [PolyAI Banking77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data). The result file records source hashes, process IDs/exit status, settings and command. Data, experiences and weights stay outside the repository.

## Historical graph experiments

The graph neural adapter and graph applications have been retired. The current graph API reserves symbolic operations and does not implement them. The measurements below describe the earlier implementation, not current capability.

[The historical MUTAG example](https://github.com/TensaCo/tensacode-py/blob/d8188ed/examples/mutag.py) read the [official TU dataset](https://chrsmrrs.github.io/datasets/docs/datasets/) with a pinned archive hash. A fixed seed-7 split assigns 150 whole molecule graphs to training and 38 to evaluation. Training-only atom categories become supplied node features; adjacency drives two trainable message-passing steps, followed by mean pooling and classification. No node or graph appears in both splits.

Held-out cross-entropy changes from **0.68665 to 0.29464**. Accuracy changes from **33/38 to 34/38**, against a training-majority baseline of **26/38**. This measured real parameter learning through the now-removed graph adapter, but the random initialization already classifies most test graphs correctly. One additional correct graph on this small split is weak accuracy evidence, not a benchmark result or evidence of inferred chemistry. That neural adapter preserved relation labels but did not use them. [Full results and split IDs](results/mutag.json).

[The retired dependency-impact example](https://github.com/TensaCo/tensacode-py/blob/d8188ed/examples/dependency_impact.py) used Python AST import extraction and explicit graph callbacks. It performed static dependency analysis, not learned symbolic interpretation; there is no replacement graph application while symbolic operations remain unimplemented.

## Actual multimodal model behavior

[The local evaluation](../examples/local_multimodal.py) sends real image bytes and text through `ImageEncoder`, `Message`, `Transform`, and structured message operations. Models are loaded explicitly from locally downloaded, pinned revisions. The public [candy photograph](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG) has SHA256 `fc417c899e94f8df465b7541c5a70f0eebb85c414d06345f0b290c061eccc84c`. It contains five candies and visible fish-like printed symbols. This published model-card example is a smoke test, not unseen benchmark data.

| Supplied pretrained model | Observed behavior |
|---|---|
| [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct), revision `7e3e67edbbed1bf9888184d9df282b700a323964` | Described candies as gems, counted six, denied the visible animal symbols. Both structured classification and retrieval answers were rejected as invalid JSON. |
| [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), revision `89644892e4d85e24eaac8bacfd4f463576704203` | Recognized candies and counted five correctly; described the symbol inconsistently as a leaf/turtle. Both structured answers were rejected. An explicit no-Markdown/no-explanation format-instruction follow-up also failed JSON validation. |

[SmolVLM results](results/multimodal-smolvlm-256m.json) and [Qwen results, including the follow-up](results/multimodal-qwen3-vl-2b.json) preserve the actual outputs and failures. Final runs used the available NVIDIA GB10 device through the same optional local adapter. No semantic answer repair, confidence fabrication, parser relaxation or hidden fallback was applied. Token-budget exhaustion also raises an explicit error rather than returning a partial answer as complete.

The local adapter establishes a working supplied-model vision/language path. These runs **do not establish reliable visual understanding or reliable local structured decisions**. The current local adapter uses prompted JSON plus strict validation; it does not implement grammar-constrained decoding. A stronger or constrained backend is still needed for dependable use of these particular structured tasks. The tiny-model failures remain part of the evidence rather than being replaced by the stronger model's partial successes.

## Verification and operational limits

CI runs the complete suite and package build on Python 3.11, 3.12 and 3.13, plus a separate dependency-free wheel job. Core operations, training, provider adapters and the lazy tools namespace import without PyTorch or Transformers. Importing concrete owned tool classes requires the `tools` extra. See the [test guide](../tests/README.md) for local verification commands.

The HTTP adapters are exercised through local servers, including real request bytes, refusal/truncation, redirects, timeouts and invalid outputs. No OpenAI or TypeSafe credentials were configured; live hosted-provider quality and account-specific behavior remain unverified. The Jev adapter supports only its documented typed choice/score operations, not chat, images or retrieval.

Saved tensor artifacts restore on CPU. Checkpoints cover supported module state
and SGD/Adam/AdamW optimizer state. ToolTrainer additionally saves Python and
PyTorch RNG state, module modes and explicit caller progress; external data-loader,
NumPy RNG and scheduler state require separate handling. Data-only JSON avoids
executable deserialization, but supplied codecs and configuration declarations are
trusted application code, not a sandbox for arbitrary untrusted inputs. Replaying
recorded external outputs holds them constant and does not rerun their effects.

Memory transactions and turn serialization cover one process; distributed locking is not implemented. Cancellation during commit settles that commit before releasing the turn lock. External effects from supplied callbacks cannot be undone by rolling back tool state. Symbolic graph operations remain unimplemented; graph records preserve caller-supplied structure without inferring semantics from free-form evidence.

Tools can own pretrained models and generate proposals internally. Training targets
still require explicit evidence or feedback. Selection/retention policies and
action authority remain explicit; symbolic graph operations remain stubs.

## Earlier measurement

[The earlier in-process Banking77 run](results/banking77-in-process.json) measured held-out accuracy increasing from 0.97% to 89.38%, with cross-entropy decreasing from 4.3646 to 0.4628. It used the same official split and overlap exclusions. Its older `examples/banking77.py` entrypoint is available in source history at `7827ac0`; the current runnable path is [the restart example](../examples/banking77_restart.py). The newer restart measurement above is a separate run, not a replacement of the earlier evidence.

## Response-quality training pilot

The [response-quality report](results/response-quality-pilot.json) records an
experimental owned transformer with separate support, completeness and constraint
heads. It trained on 55 assistant-reviewed natural candidates from 20 questions,
with source-disjoint calibration (17 candidates / 6 questions) and development
(16 / 6). These previously inspected examples are adaptation data, not a fresh
final benchmark. Reference answers and reviewer explanations never enter model
inputs. The five-epoch GB10 run made 70 optimizer updates; encoder and head weights
changed, exact optimizer continuation passed, and a fresh process reproduced all
88 calibrated prediction receipts.

The checkpoint was not promoted. Support and constraints accept every development
candidate; completeness accuracy is 0.50 versus 0.8125 for an all-positive baseline.
Joint screening accepts five known-good development candidates and no known
failures, but accepts two bad calibration candidates. It acts as a completeness
filter on those partitions, with no demonstrated source-sensitive support or
constraint rejection. This does not establish an improvement to the complete
chatbot. No final questions were accessed and no tool defaults changed.

`examples/train_response_quality.py` reproduces preparation, explicit training,
calibration and persistence checks. It is an experimental training runner using
an internal assessor, not a supported pretrained response-quality tool. Its local
foundation must have pinned Hugging Face download metadata and matching asset
hashes. Further work needs broader reviewed supervision and evidence-use ablations
before integrating a checkpoint into active tools.
