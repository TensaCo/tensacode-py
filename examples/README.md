# Examples

These are small, complete programs for real input files. They use public
TensorCode operations and tools, expose their model/policy choices, and can be
adapted without adopting an application framework.

Install from the checkout with `python -m pip install -e .`. Run commands from the
repository root. The HTTP examples require your own running OpenAI-compatible
model; replace `your-served-model` with its actual model ID. Hosted credentials
come from `OPENAI_API_KEY`, or the variable named by `--api-key-env`. Selected file
contents are sent to the endpoint you configure.

| Build | Input and output | TensorCode concepts |
|---|---|---|
| [Hypothesis learning](hypothesis_learning.py) | Reviewed evidence sequences → revisable interpretations and saved weights | Upfront vector operations, sourced evidence, trace replay, checkpoint restoration |
| [Plan learning](plan_learning.py) | Observed plan outcomes → learned candidate rankings | Local outcome prediction, explicit feedback, MSE training, reloadable weights |
| [Support-ticket triage](support_triage.py) | Ticket JSONL + routing policy → routes, abstentions and supplied distributions | `llm.Classify`, explicit batch calls |
| [Document search and answers](document_search.py) | Text/Markdown directory + question → answer and cited excerpts | `llm.Retrieve`, message transforms, source IDs |
| [Image inspection](image_inspection.py) | Any supported image + question → model answer | `ImagePart`, multimodal `Chatbot`, explicit local/remote models |
| [Bounded research assistant](research_assistant.py) | Local document directory + question → answer, sources and action receipts | `llm.Decide`, `ActionLoop`, bounded file tools |
| [Banking77 learning](banking77_restart.py) | Labeled text CSVs → persisted traces, trained weights and held-out results across process restarts | `vec.TextEncoder`, `vec.Classify`, `Trainer`, checkpoints |
| [Molecular graph learning](mutag.py) | MUTAG graph dataset → trained graph encoder and held-out results | Graph encoding, message passing, gradients |
| [Vision model evaluation](local_multimodal.py) | Supplied image and model → recorded answers and failures | Multimodal operations, explicit model evaluation |
| [Dependency impact](dependency_impact.py) | Python package + changed file → affected import graph | `Graph`, source anchors, transforms, scoring and tracing |

## Initialize, collect, train, save and load

Install `python -m pip install -e '.[vec]'` for the learning examples. These run
locally with randomly initialized PyTorch models; no API key or pretrained weights
are required. `from tensorcode.ops import vec` is the public vector namespace.

Both learning-agent programs construct their operations in `bindings(manifest)`
before processing input. Their lifecycle is explicit:

1. **Collect:** build the vocabulary from training evidence, initialize all
   operations, save `initial.json`, and capture traces with sourced feedback.
2. **Train:** construct compatible operations again, load the initial weights and
   saved experiences, replay the DAG with gradients, and save `trained.json` with
   optimizer state.
3. **Predict:** construct fresh operations, load learned weights, and process new
   input. Each command can run in a separate process.

Keep artifacts outside the checkout and use a new directory for each collection.
The scripts' `train` commands fit the collected dataset from the initial checkpoint;
they do not resume an interrupted optimizer or automatically collect new feedback.
The [training API](../docs/training.md) also supports restoring optimizer state.
The [quickstart](../docs/quickstart.md) shows this lifecycle in one short program.

## Revise hypotheses as evidence arrives

[Hypothesis learning](hypothesis_learning.py) models a small interpretation workspace:
source evidence accumulates, a learned classifier revises its distribution over
supplied hypotheses, and an authored display renders the selected interpretation.
For example, use reviewed incident investigations, support conversations, or
research annotations where each evidence prefix has its own reviewed interpretation.

Collection JSONL has `case_id` and ordered `evidence` entries. This illustrative
record shows the schema; replace it with actual reviewed cases:

```json
{"case_id":"incident-17","evidence":[{"source_id":"ticket:17","text":"Requests are timing out.","target":"unresolved","reviewer":"review:17:1"},{"source_id":"log:17","text":"Service workers cannot connect to the database.","target":"service_fault","reviewer":"review:17:2"}]}
```

A target applies to the evidence available **at that step**. Do not copy the final
incident diagnosis onto earlier prefixes that could not support it. Hypothesis
names and reviewer identities are caller supplied, and never appended to the text
that the encoder learns from.

```bash
python examples/hypothesis_learning.py collect --input reviewed-cases.jsonl \
  --hypothesis unresolved --hypothesis service_fault --artifacts /tmp/hypothesis-model
python examples/hypothesis_learning.py train --artifacts /tmp/hypothesis-model --epochs 30
python examples/hypothesis_learning.py predict --input new-cases.jsonl \
  --artifacts /tmp/hypothesis-model
```

Prediction JSONL uses the same case/evidence structure with only `source_id` and
`text` in each evidence entry. Output retains the evidence prefix, full hypothesis
distribution, selected interpretation and whether it changed since the previous
step. The artifact directory retains original evidence and review provenance.

The learned mechanism is a mean-pooled text encoder and classifier. It learns
associations from reviewed text; it does not discover new hypotheses, reason about
causality, or reliably represent negation, evidence order or source reliability.
Distributions are uncalibrated. Revision is observable behavior, not a guarantee
that later evidence will produce a better interpretation.

## Learn which supplied plans tend to work

[Plan learning](plan_learning.py) encodes a task, its evidence and each candidate
plan, predicts a numeric outcome, then selects the highest-scoring candidate.
Possible applications include ranking troubleshooting procedures, experiment
plans, or job-recovery strategies from historical results.

Collection JSONL contains `id`, `task`, `evidence` (`id`, `text`) and observed
`plans` (`id`, `text`, `outcome`, `source`). This illustrative record describes one
observed action; it does not assign invented outcomes to untried alternatives:

```json
{"id":"incident-17","task":"restore checkout","evidence":[{"id":"log:17","text":"Errors began after deployment."}],"plans":[{"id":"rollback","text":"Restore the previous deployment.","outcome":1.0,"source":"incident:17:recovery-observation"}]}
```

Use a consistent numeric outcome scale where higher is better. Each training row
may contain just the plan actually tried. Include multiple outcomes only when
those observations exist; an unchosen plan does not receive an automatic zero.

```bash
python examples/plan_learning.py collect --input observed-plans.jsonl \
  --artifacts /tmp/plan-model
python examples/plan_learning.py train --artifacts /tmp/plan-model --epochs 100
python examples/plan_learning.py predict --input candidate-plans.jsonl \
  --artifacts /tmp/plan-model
```

Prediction uses the same task/evidence structure with all candidate plans, omitting
`outcome` and `source`. Every candidate is scored before selection. Output retains
the task, source evidence, candidate descriptions and scores; it never executes the
selected plan. Historical observations remain in the artifact directory.

To compare against initial weights, add `--checkpoint initial.json`. If prediction
input includes sourced outcomes for every candidate, output also includes held-out
MSE and rejects overlapping training IDs/inputs. Ordinary unlabeled prediction can
revisit a known task. Hold out whole tasks, not just different plan rows from the
same task, when measuring generalization.

The neural encoder and outcome predictor learn from feedback. Candidate generation,
the text representation and highest-score selection are authored policies. Scores
are uncalibrated estimates, not confidence or causal effects. Historical action
selection can bias them; this example does not estimate counterfactual outcomes or
learn a world model. Mechanism tests use explicitly authored outcomes, not evidence
of real-world planning quality.

## Route support tickets

Export tickets as UTF-8 JSONL, one object per line with `id` and `text` fields:

```json
{"id":"case-1042","text":"Our team cannot sign in after enabling SSO."}
```

Supply your own policy file defining the routes and when to abstain:

```bash
python examples/support_triage.py --input tickets.jsonl --policy routing-policy.txt \
  --label billing --label incident --label question \
  --base-url http://localhost:8000/v1 --model your-served-model \
  --output routes.jsonl
```

Output preserves ticket IDs and order. The script does not invent missing
confidence or replace invalid model answers with a default route. Limits on ticket
count, ticket length and policy length are available in `--help`.

## Search a handbook

```bash
python examples/document_search.py --directory ./handbook \
  --query 'How do I recover access after losing my MFA device?' --top-k 3 \
  --base-url http://localhost:8000/v1 --model your-served-model
```

The program chunks visible UTF-8 `.txt`/`.md` files, asks the model to retrieve
existing chunks, and generates an answer from selected excerpts. Output includes
relative file paths, character offsets and excerpt text. Unknown or missing
citation IDs are rejected. A valid citation identifies an excerpt; it does not
prove the claim follows from it.

This is intentionally for small collections, with explicit file, chunk and
request-size limits. It sends candidate excerpts to the model rather than building
a scalable embedding index. Hidden paths and symlinks are skipped.

## Inspect an image

Use an explicitly configured remote multimodal model:

```bash
python examples/image_inspection.py ./photos/equipment.jpg \
  'Describe the visible controls and any readable labels.' \
  --base-url http://localhost:8000/v1 --model your-served-model
```

Or install `.[local]` and use a model you have already downloaded:

```bash
python examples/image_inspection.py ./photos/equipment.jpg \
  'Describe the visible controls and any readable labels.' \
  --local-model Qwen/Qwen3-VL-2B-Instruct \
  --revision 89644892e4d85e24eaac8bacfd4f463576704203 --device cuda
```

Local loading is offline unless you explicitly pass `--allow-download`. MIME type
comes from the image filename, and the message retains its source reference.
Answers are model output, not independently verified visual facts.

## Let a model choose bounded research actions

```bash
python examples/research_assistant.py ./handbook \
  'What steps does our incident process require before closing an incident?' \
  --base-url http://localhost:8000/v1 --model your-served-model --max-steps 6
```

The model chooses among a lexical search action, fixed document-read actions and
finish. It cannot invent a path or execute a shell command. The result includes
its stop reason, read sources and receipts; exhausting the action budget does not
count as a finished answer. Search ranking is an authored term-count algorithm.
This demonstrates a bounded agent composition, not an unrestricted autonomous
researcher. Source IDs are checked; factual correctness still needs evaluation.

## Analyze a real Python package offline

This example requires neither a model nor PyTorch. Try it on this checkout:

```bash
python examples/dependency_impact.py src/tensorcode --changed ops/vec/latent.py
```

It parses static imports without executing the package, builds a source-anchored
graph, and traces a reverse-dependency transform and score. Output contains the
affected modules and import locations. Dynamic imports and runtime conditions are
not resolved, so this is partial dependency analysis, not a build guarantee.

## Durable text learning

Obtain the official train/test CSVs from
[PolyAI Banking77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data).
Choose a new artifact directory for each run:

```bash
python examples/banking77_restart.py \
  --train /path/to/train.csv --test /path/to/test.csv \
  --artifacts /tmp/banking77-run --output /tmp/banking77-results.json
```

This is the canonical Banking77 example. It replaces the earlier, redundant
in-process training script. Supervision comes from supplied dataset labels.

## Graph learning

Obtain the [official MUTAG archive](https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip):

```bash
python examples/mutag.py --data /path/to/MUTAG.zip \
  --output /tmp/mutag-results.json
```

The script verifies the archive hash and uses a fixed split. Atom categories and
adjacency are supplied dataset features, not inferred chemical knowledge.

## Multimodal smoke evaluation

Install `tensorcode[local]`. Explicitly download the model first, for example with
`hf download HuggingFaceTB/SmolVLM-256M-Instruct --revision 7e3e67edbbed1bf9888184d9df282b700a323964`.
Save the published
[candy photograph](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG)
locally, then run:

```bash
python examples/local_multimodal.py --image /path/to/candy.JPG \
  --source https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG \
  --output /tmp/multimodal-results.json
```

The prompts are specific to this image. This is a smoke test, not a general image
benchmark. Use `--device cuda` when available. To select another downloaded model,
supply **both** `--model` and its matching `--revision`; the defaults pin SmolVLM.
Model outputs can be wrong or fail JSON validation. These failures remain in the
report and must not be interpreted as successful decisions.

Each script supports `--help`. The package [README](../README.md) contains smaller
API examples; [documentation](../docs/README.md) explains contracts and limitations.
