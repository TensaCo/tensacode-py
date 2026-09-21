# Application examples

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
| [Support-ticket triage](support_triage.py) | Ticket JSONL + routing policy → routes, abstentions and supplied distributions | `llm.Classify`, explicit batch calls |
| [Document search and answers](document_search.py) | Text/Markdown directory + question → answer and cited excerpts | `llm.Retrieve`, message transforms, source IDs |
| [Image inspection](image_inspection.py) | Any supported image + question → model answer | `ImagePart`, multimodal `Chatbot`, explicit local/remote models |
| [Bounded research assistant](research_assistant.py) | Local document directory + question → answer, sources and action receipts | `llm.Decide`, `ActionLoop`, bounded file tools |
| [Dependency impact](dependency_impact.py) | Python package + changed file → affected import graph | `Graph`, source anchors, transforms, scoring and tracing |

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

## Learn from data and reproduce evaluations

The separate [evaluation scripts](evaluation/README.md) cover supervised
Banking77 training across process restarts, MUTAG graph learning and a fixed
multimodal smoke test. Their [recorded results](../docs/results/README.md) retain
both successes and failures. They serve a different purpose from the application
programs above.

All scripts support `--help`. The [quickstart](../docs/quickstart.md) demonstrates
small in-memory training; [API guides](../docs/README.md) explain extension and
composition contracts. Application tests use injected responses to check wiring,
source boundaries and failures; those fixtures are not model-quality evidence.
