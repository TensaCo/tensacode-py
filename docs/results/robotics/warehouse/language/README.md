# Observed-scene language interface

A real, frozen Qwen3-VL-2B-Instruct model translates requests into a compact
composition tree. Authored code expands that tree into the shared typed instruction
representation, validates it, resolves references against observed scene evidence,
and creates task dependencies. Questions query those observations; the model never
supplies answer facts. Model weights are unchanged.

## Measured results

| Evaluation | Model + adapter | Restricted authored grammar | Scope |
| --- | --- | --- | --- |
| Original v1 held-out requests | 7/12 | 5/12 | Verbose nested JSON; all failures retained |
| v2 reused development requests | 16/16 | 7/16 | Prompt/representation improved using v1 failures; **not held out** |
| New frozen v2 challenge | **9/12** | **3/12** | Different authored requests, frozen before first inference |
| New v2 supported actions/answers only | **6/7** | **0/7** | Excludes correct refusals and ambiguous/unknown references |

The comparison is against this repository's deliberately narrow English grammar,
not another language model or a robotics policy. These are small authored language
challenges with an explicitly **authored scene fixture**, not physical robot sensing
or evidence of general intelligence. Both parsers receive the same requests and
fixture. The model receives only language and its syntax prompt, never the scene.

The new challenge includes simultaneous and sequential transfers, an indirect
request, spatial references, location questions, unknown/ambiguous objects, and
unsupported conditions/deadlines/counting. Current v2 failures remain visible:

| Case | Actual model behavior | Admission boundary |
| --- | --- | --- |
| c5 | Turns “once that is done” into parallel execution | **Authored guard misses this sequencing paraphrase** |
| c10 | Drops “unless the destination is occupied” | Conditional-action guard rejects |
| c11 | Drops “in less than two seconds” | Timing-constraint guard rejects |

Passing schema validation or the cue guard does **not** prove preservation of
language meaning. Arbitrary model proposals are previews; the selected physical
mission has an explicitly reviewed object/destination mapping and dependency tree.
The guard never silently repairs a proposal.

`mission-v2-development-receipt.json` contains a correct reviewed model parse for
the parallel two-parcel mission. It is **development reuse**, not a fresh held-out
success. Two additional integration requests were then tried: the first emitted
malformed JSON and was rejected (`mission-integration-receipt.json`); the second
correctly parsed a different mapping of the challenge c1 sentence template
(`mission-integration2-receipt.json`). Both attempts remain visible. The latter
contains the raw response, pinned model revision, prompt hash and reviewed
interpretation for physical integration. Grounding against actual camera evidence
happens separately in the physical runner. `location-query-receipt.json` similarly supplies an actual
model-generated location query for subsequent sensor-grounded answering.

## What changed in v2

Instead of generating a verbose JSON object for every reference, the model emits:

```json
{"instruction":["parallel",["move","parcel_0","receiver_0"],["move","parcel_1","receiver_1"]]}
```

or `{"where":"parcel left of parcel_1"}`. A deterministic, authored adapter maps
box/carton/package/item to parcel and tray/platform to station, expands the compact
tree, and applies the shared strict validator. Explicit identifiers are preserved.
The improved prompt includes authored concurrency, sequencing and query examples.
These are prompt/interface improvements, **not learned TensorCode weights**.

On the same sixteen development requests, generated output falls from 849 to 200
tokens; observed inference time falls from 399.3 to 137.0 seconds. This is a local
measurement with the same frozen model and CPU allocation, not a general latency
benchmark. The new twelve-case challenge takes 118.7 seconds of inference.

Schema validation rejects extra fields, malformed references, duplicate keys,
nonfinite values and unsupported operations. Grounding rejects missing, ambiguous,
stale or insufficiently confident observations. Answers carry coordinates, age,
uncertainty, scene revision and source IDs. No generated text executes as code.

## Evidence and reproduction

Original v1 evidence remains in this directory. `v2-development/` contains reused
requests and `v2-challenge/` contains the new frozen challenge. Each includes
`cases.json`, `system_prompt.txt`, `freeze.json`, every raw prediction, and a summary.
The new challenge also includes `source.zip` and source hashes captured before its
first inference. V1's initial ambiguity-baseline scoring bug was corrected by
replaying unchanged raw outputs; the authored grammar receives its proper credit.

The pinned model revision is `89644892e4d85e24eaac8bacfd4f463576704203`. Loading is
local-cache-only, CPU bfloat16, two threads, greedy decoding, at most 384 generated
tokens. Every evaluation uses an 8 GiB RAM cap, GPU disabled, and a global headroom
guard. The new challenge's minimum available host RAM is **80.36%**; two of twenty
CPU cores are assigned to the model. Full resource samples are retained.

```bash
# From tensacode/python; choose two available CPU IDs on your host.
taskset -c 16,17 .venv/bin/python examples/robotics/g1/budget.py \
  --memory-gib 8 --output /tmp/warehouse-language-budget \
  --unit warehouse-language-eval -- "$PWD/.venv/bin/python" \
  -m examples.robotics.warehouse.language_eval --prompt-version v2 --challenge \
  --output /tmp/warehouse-language

# Replay grounding and scoring without loading a model.
.venv/bin/python -m examples.robotics.warehouse.language_eval \
  --prompt-version v2 --challenge \
  --output docs/results/robotics/warehouse/language/v2-challenge --replay
```

`model_language.py` also offers a single-instruction CLI (`--instruction`,
`--output`, optional `--prompt-version`). Its parse receipt is not a grounded task:
the executor still needs semantic admission and fresh sensor evidence.
