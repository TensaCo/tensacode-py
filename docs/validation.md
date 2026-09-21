# Validation and scope

TensorCode provides operation, tool, persistence and training paths described below. Semantics still come from explicitly supplied models, training data, algorithms and policies; importing an operation does not install a cognitive ontology.

## Implemented surfaces

| Area | Active behavior | Important boundary |
|---|---|---|
| Vector operations | Space-tagged native tensors; text and spatial image encoding; supplied transforms/decoders; classification, candidate scoring, decision and retrieval | Random initialization supplies no semantics. Matching dimensions alone do not establish matching spaces. Discrete selection does not make its index differentiable. |
| Message operations | Immutable text/image messages, transforms, validated classification/decision/scoring/retrieval, explicit async and batch calls | Structured answers can fail validation. Missing confidence stays missing; model probabilities are not calibrated truth. |
| Providers | Explicit provider-neutral request/response protocol, HTTP adapters, local Transformers image/text model adapter | Remote credentials and model choice belong to the caller. Local models must be explicitly acquired; image URLs are not fetched by the local adapter. |
| Graph operations | Immutable identities, attributes, relations and source anchors; encode/decode, supplied graph queries/scoring/decisions; trainable graph-to-vector message passing | Graph structure and features are supplied. Relation strings do not acquire semantics from their names. No implicit ontology or graph grounding claim. |
| Tools | Replaceable decision pipeline; multimodal chatbot; objective revision hook; persistent retrieval memory; bounded supplied-action loop | Objective/retrieval/action policy is supplied. A configured action can have effects; history rollback does not undo an external effect. |
| Tracing | Identity-based dependency capture, explicit scalar handles, native live gradients, async boundaries, external roots and dependency closure | Ordinary Python between operations is not inferred as a differentiable operation. |
| Persistence/training | Data-only experiences, explicit target provenance, named operation/configuration bindings, intermediate release, recorded external boundaries, trainers and checkpoints | No executable-program deserialization. Remote outputs can be recorded constants, never differentiable remote calls. Custom codecs/configurations are trusted caller declarations. |

Start with the [documentation index](README.md) for current API guides and runnable examples.

## Actual learning across process restarts

[Banking77 restart results](results/banking77-restart.json) were produced by [the executable example](../examples/banking77_restart.py). Four subprocesses terminate in sequence: capture/save, baseline evaluation, reload/train/checkpoint, and final checkpoint evaluation. The run captures 79 experiences containing 9,997 explicitly labeled training rows. Six literal train/test overlaps are excluded. Vocabulary is built from training data only.

On 3,080 official held-out examples across 77 labels, accuracy changes from **0.97% to 89.48%** and cross-entropy from **4.3646 to 0.4566**. The model is a trainable 96-dimensional mean embedding and linear classifier, with fixed seed 7, Adam at 0.01 and 20 epochs. This demonstrates that persisted operation dependencies support actual supervised parameter updates after restart. It does not demonstrate autonomous target discovery, calibrated confidence, or arbitrary-program differentiation.

Dataset source: [PolyAI Banking77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data). The result file records source hashes, process IDs/exit status, settings and command. Data, experiences and weights stay outside the repository.

## Graph learning on real inputs

[The MUTAG example](../examples/mutag.py) reads the [official TU dataset](https://chrsmrrs.github.io/datasets/docs/datasets/) with a pinned archive hash. A fixed seed-7 split assigns 150 whole molecule graphs to training and 38 to evaluation. Training-only atom categories become supplied node features; adjacency drives two trainable message-passing steps, followed by mean pooling and classification. No node or graph appears in both splits.

Held-out cross-entropy changes from **0.68665 to 0.29464**. Accuracy changes from **33/38 to 34/38**, against a training-majority baseline of **26/38**. This is real parameter learning through the graph adapter, but the random initialization already classifies most test graphs correctly. One additional correct graph on this small split is weak accuracy evidence, not a benchmark result or evidence of inferred chemistry. Relation labels are preserved but not used by this neural adapter. [Full results and split IDs](results/mutag.json).

## Actual multimodal model behavior

[The local evaluation](../examples/local_multimodal.py) sends real image bytes and text through `ImageEncoder`, `Message`, `Transform`, and structured message operations. Models are loaded explicitly from locally downloaded, pinned revisions. The public [candy photograph](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG) has SHA256 `fc417c899e94f8df465b7541c5a70f0eebb85c414d06345f0b290c061eccc84c`. It contains five candies and visible fish-like printed symbols. This published model-card example is a smoke test, not unseen benchmark data.

| Supplied pretrained model | Observed behavior |
|---|---|
| [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct), revision `7e3e67edbbed1bf9888184d9df282b700a323964` | Described candies as gems, counted six, denied the visible animal symbols. Both structured classification and retrieval answers were rejected as invalid JSON. |
| [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), revision `89644892e4d85e24eaac8bacfd4f463576704203` | Recognized candies and counted five correctly; described the symbol inconsistently as a leaf/turtle. Both structured answers were rejected. An explicit no-Markdown/no-explanation format-instruction follow-up also failed JSON validation. |

[SmolVLM results](results/multimodal-smolvlm-256m.json) and [Qwen results, including the follow-up](results/multimodal-qwen3-vl-2b.json) preserve the actual outputs and failures. Final runs used the available NVIDIA GB10 device through the same optional local adapter. No semantic answer repair, confidence fabrication, parser relaxation or hidden fallback was applied. Token-budget exhaustion also raises an explicit error rather than returning a partial answer as complete.

The local adapter establishes a working supplied-model vision/language path. These runs **do not establish reliable visual understanding or reliable local structured decisions**. The current local adapter uses prompted JSON plus strict validation; it does not implement grammar-constrained decoding. A stronger or constrained backend is still needed for dependable use of these particular structured tasks. The tiny-model failures remain part of the evidence rather than being replaced by the stronger model's partial successes.

## Verification and operational limits

CI runs the complete suite and package build on Python 3.11, 3.12 and 3.13, plus a separate dependency-free wheel job. Core operations, training, tools and provider adapters import without PyTorch or Transformers. See the [test guide](../tests/README.md) for local verification commands.

The HTTP adapters are exercised through local servers, including real request bytes, refusal/truncation, redirects, timeouts and invalid outputs. No OpenAI or TypeSafe credentials were configured; live hosted-provider quality and account-specific behavior remain unverified. The Jev adapter supports only its documented typed choice/score operations, not chat, images or retrieval.

Saved tensor artifacts restore on CPU. Checkpoints cover supported module state and SGD/Adam/AdamW optimizer state; scheduler and RNG state are outside this checkpoint API. Data-only JSON avoids executable deserialization, but supplied codecs and configuration declarations are trusted application code, not a sandbox for arbitrary untrusted inputs. Replaying recorded external outputs holds them constant and does not rerun their effects.

Memory transactions and turn serialization cover one process; distributed locking is not implemented. Cancellation during commit settles that commit before releasing the turn lock. External effects from supplied callbacks cannot be undone by rolling back tool state. Graph neural processing currently consumes one graph per call and uses adjacency plus supplied numeric node features; it does not infer graph semantics from free-form evidence.

Models, targets, losses, objective updates, retrieval semantics and action authority remain caller-supplied.

## Earlier measurement

[The earlier in-process Banking77 run](results/banking77-in-process.json) measured held-out accuracy increasing from 0.97% to 89.38%, with cross-entropy decreasing from 4.3646 to 0.4628. It used the same official split and overlap exclusions. Its older `examples/banking77.py` entrypoint is available in source history at `7827ac0`; the current runnable path is [the restart example](../examples/banking77_restart.py). The newer restart measurement above is a separate run, not a replacement of the earlier evidence.
